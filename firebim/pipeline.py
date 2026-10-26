# -*- coding: utf-8 -*-
"""공통 변환 엔진: 정보카드 PDF 1개 → 소방용 건물 모델(JSON)
사용: python -m firebim.pipeline card.pdf out.json"""
import json, sys, math, os
import numpy as np, cv2
from shapely.geometry import Point, box
from firebim.card import parse_card, extract_plans, extract_figures, redact, level_of
from firebim.vision import masks, detect_symbols, detect_fac, grid_lines, scale_from_dims, footprint_px_area, stair_regions
from firebim import geom
from firebim.align import align, align_real


def cluster_vertical(items, tol=1.6, key='xy'):
    """층마다 같은 자리에 반복되는 점들을 묶는다 → [{x,y,floors,members}]"""
    cl = []
    for it in items:
        for c in cl:
            if math.hypot(c['x'] - it['x'], c['y'] - it['y']) < tol:
                c['m'].append(it); n = len(c['m'])
                c['x'] += (it['x'] - c['x']) / n; c['y'] += (it['y'] - c['y']) / n
                break
        else:
            cl.append(dict(x=it['x'], y=it['y'], m=[it]))
    for c in cl:
        c['floors'] = sorted(set(m['floor'] for m in c['m']), key=level_of)
    return cl


BAY = {}
JAMBS = os.environ.get('FIREBIM_JAMBS', '1') != '0'
OCC = os.environ.get('FIREBIM_OCC', '1') != '0'          # 채색 기호가 가린 벽 틈 판별
FLAGTEXT = os.environ.get('FIREBIM_FLAGTEXT', 'real')  # 플래그에 쉬운 말 msg·기술 detail 을 JSON 에 붙이기(2026-10-03): real = 실제 도면 경로 층이 있을 때만(가상 카드 JSON 바이트 동일) · 1 = 항상 · 0 = 안 붙임.
#   JSON 에 msg 가 없어도 지휘화면(export)·검수 화면(review/make)은 그릴 때 flagtext.explain 으로 같은 쉬운 말을 씀


def grid_pitch(vl, tol=0.12):
    """등간격 그리드선(px) → 한 경간 px (빠진 선 허용, 전체 폭 최소제곱). 등간격이 아니면 None"""
    d = [b - a for a, b in zip(vl, vl[1:]) if b - a > 10]
    if len(vl) < 3 or not d: return None
    med = float(np.median(d))
    x = np.asarray(vl, float) - vl[0]
    k = np.round(x / med)
    ok = np.abs(x / med - k) < tol
    if ok.sum() < 3 or k[ok].max() - k[ok].min() < 2 or ok.mean() < 0.7: return None
    slope = float(np.polyfit(k[ok], x[ok], 1)[0])
    return slope if abs(slope / med - 1) < 0.05 else None


def refine_scale(plans, info=None):
    """치수 OCR 은 한두 자리 오독(8,100→8,500)이 흔하다 → 건물 전체에서 가장 많이 읽힌 경간(mm)을 합의값으로,
    등간격 그리드 도면은 축척 = 그리드 간격(전체 폭 최소제곱) ÷ 합의 경간. 각 도면 자체 추정과 6% 넘게 어긋나면 그대로 둔다."""
    BAY.clear()
    plans = [p for p in plans if not p.get('real')]          # 실제 도면 경로(firebim/realcard.py)는 자체 축척 합의 — 건드리지 않음
    def s_area(p):                                          # 정보카드 면적 ↔ 도면 윤곽 면적(±25% 수준, 배수 오독 판별용)
        if info is None: return None
        fa = [info['floors'][f]['area'] for f in p['floors'] if f in info['floors'] and info['floors'][f].get('area')]
        pa = footprint_px_area(p.get('M_raw', p['M'])) if fa else 0
        return math.sqrt(pa / fa[0]) if fa and pa > 0 else None
    for p in plans:
        p['_sa'] = s_area(p)
        if p['s'] and p['_sa'] and abs(math.log(p['s'] / p['_sa'])) > math.log(1.8):
            p['s'], p['scale_src'] = None, None                  # 치수 오독(3,400 ← 8,400 등)으로 면적과 2배 가까이 어긋남 → 다른 층 경간으로
    mms = [int(round(m / 50.0)) * 50 for p in plans for m in p.get('dim_mm', [])]
    if len(mms) < 4: return
    vals, cnt = np.unique(mms, return_counts=True)
    j = int(np.argmax(cnt))
    if cnt[j] < 3 or cnt[j] < 0.4 * len(mms) or (len(cnt) > 1 and sorted(cnt)[-2] == cnt[j]): return
    bay = vals[j] / 1000.0; BAY['m'] = bay
    for p in plans:
        pit = grid_pitch(p['vlines'])
        if not (p['s'] and pit): continue
        sf = pit / bay
        if abs(sf / p['s'] - 1) < 0.06: p['s'] = sf
        elif p['_sa'] and abs(math.log(sf / p['_sa'])) < math.log(1.3) and abs(math.log(p['s'] / p['_sa'])) > math.log(1.5):
            p['s'], p['scale_src'] = sf, 'grid'


def run(pdf, keep=False, daejang=None, site=None, ilramdo=None, model=None, dong=None):
    """daejang/site/ilramdo: 건축물대장(json|pdf)·부지(json)·경계구역 일람도(json) 경로. None 이면 card.pdf 옆 파일을 자동으로 찾는다.
    model: True 면 STR 분할 모델을 두 번째 추출기로 돌려 벽 교차검증·문 보충(firebim/xcheck.py). None → 환경변수 FIREBIM_MODEL=1. 기본은 규칙만."""
    auto = model is None and not os.environ.get('FIREBIM_MODEL')
    if model is None: model = os.environ.get('FIREBIM_MODEL', '').lower() in ('1', 'true', 'on', 'yes')
    info, doc = parse_card(pdf, dong)                # 표준양식이면 양식 항목명 키 + 기존 키(선택 동). 여러 동: dong= 또는 FIREBIM_DONG
    if auto and info.get('source') == 'ocr':         # 스캔본 카드: 모델 경로(교차검증·벽/문 보충)를 기본으로(모델·onnxruntime 있을 때). 끄기 FIREBIM_SCAN_MODEL=0 / model=False
        from firebim import scanplan as _SCP
        model = _SCP.model_default()
    info = redact(info)                              # ⑧ 관계인 성명·연락처 제외(기본) — FIREBIM_INCLUDE_CONTACTS=1 이면 유지
    raw = extract_plans(doc)
    if info.get('양식') == 'std':
        raw = [p for p in raw if p.get('dong', 0) == info.get('선택동', 0)]     # 선택 동의 평면도만(3D 는 한 동씩)
    plans, log = [], []
    from firebim import scanplan as SCP
    from firebim import realcard as RC
    real_raw = []
    PDFT = None
    for i, rp in enumerate(raw):
        img = rp['img']
        scan = rp.get('source') == 'ocr' and SCP.ENABLED          # 스캔본 카드에서 잘라낸 평면도만(글자층 카드 경로 불변)
        if scan:                                                   # 검출기별 정규화 영상(firebim/scanplan.py): geo=벽·문, sym=기호·계단, 원본=그리드·치수·면적
            img, img_sym = SCP.prep(rp['img']); M_raw = masks(rp['img'])
        M = masks(img)
        Ms = dict(masks(img_sym), blue_loose=SCP.blue_loose(img_sym)) if scan else M
        syms, leg, Z = detect_symbols(img_sym, Ms, scan=True) if scan else detect_symbols(img, M)
        fsyms, _ = detect_fac(img_sym if scan else img, leg, Ms)            # 소방시설 추가 기호(자홍) — 기존 symbols 와 분리(문·계단 판정에 쓰지 않음)
        tl = None
        if syms and not leg:                           # 범례 없이 '기호'만 있으면: 색 글자(공사범위 주기)인지 글자층으로도 확인
            if PDFT is None:
                from firebim import pdftext as _PT
                PDFT = _PT.load(doc)
            if PDFT:
                from firebim import pdftext as _PT
                tl = _PT.for_plan(PDFT, rp['caption'], rp['img'].shape)
        if RC.applicable(syms, leg, tl):               # 범례·기호 없는 실제 CAD 평면도
            if syms: syms, fsyms = [], []              # 색 글자(공사범위 주기)로 판정된 '기호'는 버림 — 승강기·소화전 점으로 쓰이지 않게 → 실제 도면 경로(아래 RC.process — 축척·벽·문·계단·승강기)
            plans.append(dict(i=i, floors=rp['floors'], caption=rp['caption'], s=None, scale_src=None, symbols=syms, vlines=[], hlines=[], legend=leg, Z=Z,
                              img=img, M=M, dim_mm=[], fac_symbols=fsyms, real=True, **({'img_src': rp['img']} if scan else {})))
            if PDFT is None:                                # 벡터 도면 글자층(review/attach.py 가 합본에 내장, firebim/pdftext.py) — 있으면 OCR 대신
                from firebim import pdftext as _PT
                PDFT = _PT.load(doc)
            tl = None
            if PDFT:
                from firebim import pdftext as _PT
                tl = _PT.for_plan(PDFT, rp['caption'], rp['img'].shape)
            real_raw.append((i, dict(rp, img=rp['img'], **({'text': tl} if tl else {}))))
            continue
        vl, hl = SCP.grid_lines(M_raw, img.shape) if scan and SCP.GRID else grid_lines(M, img.shape)
        mm = []
        if scan and SCP.DIMS:                          # 스캔본: 치수 줄 창별 다중 판독(scanplan.read_dims) + 카드 면적 축척으로 배수 오독 거름
            fa = [info['floors'][f]['area'] for f in rp['floors'] if f in info['floors'] and info['floors'][f].get('area')]
            pa = footprint_px_area(M_raw) if fa else 0
            dr = []
            s, vals = SCP.scale_from_dims(rp['img'], vl, math.sqrt(pa / fa[0]) if fa and pa > 0 else None, mm_out=mm, reads_out=dr)
        else:
            s, vals = scale_from_dims(img, M, vl, mm_out=mm)
        plans.append(dict(i=i, floors=rp['floors'], caption=rp['caption'], s=s, scale_src='dims' if s else None, symbols=syms,
                          vlines=vl, hlines=hl, legend=leg, Z=Z, img=img, M=M, dim_mm=mm, fac_symbols=fsyms,
                          **({'scan': True, 'img_src': rp['img'], 'M_sym': Ms, 'M_raw': M_raw, 'dim_reads': dr if SCP.DIMS else []} if scan else {})))
    if real_raw:                                       # 실제 도면: 층마다 축척 후보(시트 글자·치수·창호 태그·디딤판·문 폭·면적) → 층간 그리드 연결 합의 → 벽·문·계단·승강기
        RR = RC.process(real_raw, info)
        for p in plans:
            if p.get('real'):
                R = RR[p['i']]; p['real'] = R
                p['s'], p['scale_src'], p['vlines'], p['hlines'], p['bubbles'] = R['s'], R['scale_src'], R['vl'], R['hl'], R['bubbles']
    refine_scale(plans, info)
    if any(p.get('scan') for p in plans) and SCP.DIMS: SCP.fuse_scale(plans)       # 스캔본: 건물 치수 합의로 층 축척 다시(면적 축척보다 우선)
    # 치수를 못 읽은 층: 다른 층의 그리드 간격(m)을 빌려 축척 추정 → 그래도 없으면 정보카드 면적
    bays = []
    for p in plans:
        if p['s']:
            for L in (p['vlines'], p['hlines']):
                bays += [round(d / p['s'], 1) for d in np.diff(L) if d / p['s'] > 2.5]
    for p in plans:
        if p['s']: continue
        diffs = [d for L in (p['vlines'], p['hlines']) for d in np.diff(L) if d > 10]
        cand = []
        if bays and diffs:
            bm = max(set(bays), key=bays.count)
            dm = float(np.median([d for d in diffs if abs(d - np.median(diffs)) / np.median(diffs) < 0.1] or diffs))
            if 3 < dm / bm < 25: cand.append(dm / bm)
        sa = p.get('_sa')
        pit, bay_m = grid_pitch(p['vlines']), BAY.get('m')
        if pit and bay_m and 3 < pit / bay_m < 25 and (sa is None or 0.6 < (pit / bay_m) / sa < 1.6):
            p['s'], p['scale_src'] = pit / bay_m, 'grid'           # 건물 공통 경간(치수 합의) ÷ 이 도면 그리드 간격
        elif cand and (sa is None or 0.6 < cand[0] / sa < 1.6):
            p['s'], p['scale_src'] = cand[0], 'grid'
        else:
            fa = [info['floors'][f]['area'] for f in p['floors'] if f in info['floors'] and info['floors'][f].get('area')]
            px = footprint_px_area(p.get('M_raw', p['M']))
            p['s'], p['scale_src'] = (math.sqrt(px / fa[0]), 'area') if fa and px > 0 else (8.0, 'default')
    for p in plans:
        p['s'] = float(p['s']); s = p['s']
        if p.get('real'):                                  # 실제 도면: 벽 = STR 모델 벽(건물 덩어리로 자름), 계단·승강기 = SPA 하이브리드 → 정합 기준점
            R = p['real']
            p['walls_local'] = R['walls']; p['stairs_px'] = [q['box'] for q in R['stairs']]
            p['anchors'] = [dict(type='stair', px=(b[0] + b[2]) / 2, py=(b[1] + b[3]) / 2) for b in p['stairs_px']] + \
                           [dict(type='ev', px=(q['box'][0] + q['box'][2]) / 2, py=(q['box'][1] + q['box'][3]) / 2) for q in R['elevs']]
            continue
        H, W = p['img'].shape[:2]
        ex = np.zeros((H, W), np.uint8)
        if p['legend']:
            L = p['legend']; ex[max(0, L['y'] - 3):L['y'] + L['h'] + 3, max(0, L['x'] - 3):L['x'] + L['w'] + 3] = 1
        sb = [b for b in stair_regions(p.get('M_sym', p['M']), s) if not ex[int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)]]
        m = int(0.7 * s)
        for b in sb: ex[b[1] + m:b[3] - m, b[0] + m:b[2] - m] = 1          # 계단 디딤판은 벽이 아님(계단실 벽은 남김)
        wm = SCP.wall_mask(p['M'], s, p['vlines'], p['hlines'], exclude=ex) if p.get('scan') and SCP.GRIDWALL else geom.wall_mask(p['M'], s, p['vlines'], p['hlines'], exclude=ex)
        wl = geom.clean(geom.merge(geom.segments(wm, s)))
        if JAMBS:                                                             # 문틀 옆 짧은 벽 되살리기(계단실·승강기 문)
            wl = geom.rescue_jambs(wl, geom.merge(geom.segments(wm, s, min_len=0.45), min_len=0.45))
        p['walls_local'] = geom.split_thin(wl, p['M'], s)     # 문·셔터 자리(두께 사라짐) 끊기
        p['stairs_px'] = sb
    if any(p.get('real') for p in plans) and all(p.get('real') for p in plans): align_real(plans)     # 실제 도면: 계단·승강기·그리드 기준점
    else: align(plans)
    # 벽·문 (전역 좌표)
    out_floors = {}
    all_pts = []
    for p in plans:
        s, tx, ty = p['s'], p['tx'], p['ty']
        G = lambda px, py: (px / s + tx, -py / s + ty)
        walls = [dict(w) for w in p['walls_local']]
        for w in walls:
            if w['o'] == 'h': w['c'] += ty; w['a'] += tx; w['b'] += tx
            else: w['c'] += tx; w['a'] += ty; w['b'] += ty
        syms = []
        for q in p['symbols']:
            x, y = G(q['px'], q['py'])
            d = dict(type=q['type'], x=round(x, 2), y=round(y, 2))
            if 'box' in q:
                b = q['box']; (x0, y1), (x1, y0) = G(b[0], b[1]), G(b[2], b[3]); d['box'] = [x0, y0, x1, y1]
            if 'conf' in q: d['conf'] = q['conf']
            syms.append(d)
        fsy = []
        for q in p['fac_symbols']:
            x, y = G(q['px'], q['py']); fsy.append(dict(type=q['type'], x=round(x, 2), y=round(y, 2), conf=q.get('conf')))
        for fid in p['floors']:
            out_floors[fid] = dict(fac_symbols=fsy, plan=p['i'], s=round(s, 3), tx=round(tx, 3), ty=round(ty, 3), align=p['conf'], scale_src=p['scale_src'],
                                   walls=walls, symbols=syms, card=info['floors'].get(fid),
                                   _nosym=not p['symbols'] and not p['legend'])      # 범례·기호 없는 도면(실제 CAD 도면 등) → 기호 수량 비교 안 함
            for q in syms:
                if q['type'] in ('exit', 'ev'): all_pts.append(dict(q, floor=fid))
            if p.get('real'): real_floor(out_floors[fid], p, fid, info, all_pts)
    # 수직 동선: 계단 디딤판 영역 + 비상구 기호가 층마다 같은 자리 → 계단실 / 승강기 표기 반복 → 승강기 샤프트
    n_fl = len(out_floors)
    sitems = []
    for p in plans:
        s, tx, ty = p['s'], p['tx'], p['ty']
        for b in p['stairs_px']:
            x0, y1, x1, y0 = b[0] / s + tx, -b[1] / s + ty, b[2] / s + tx, -b[3] / s + ty
            for fid in p['floors']:
                sitems.append(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, box=[x0, y0, x1, y1], floor=fid, **({'real': True} if p.get('real') else {})))
        if p.get('real') and REAL_STAIR2:                            # 실제 도면 계단 근거(디딤판 수·UP/DN 글자·옥외) — 단일층·옥외 계단 판정용
            for it, q in zip(sitems[-len(p['stairs_px']) * len(p['floors']):] if p['stairs_px'] else [], [q for q in p['real']['stairs'] for _ in p['floors']]):
                it.update(treads=q.get('treads', 0), label=q.get('label', False), outdoor=q.get('outdoor', False), src=q.get('src'))
    stairs = cluster_vertical(sitems, 2.5)
    exits_cl = cluster_vertical([q for q in all_pts if q['type'] == 'exit'], 2.0)
    for c in stairs:
        c['box'] = [float(np.median([m['box'][k] for m in c['m']])) for k in range(4)]
        fl = set(c['floors'])
        for e in exits_cl:        # 계단 앞 비상구가 있는 층도 연결층으로
            if box(*c['box']).buffer(4.0).contains(Point(e['x'], e['y'])): fl |= set(e['floors'])
        lv = sorted(level_of(f) for f in fl)
        allf = sorted(out_floors, key=level_of)
        c['floors'] = [f for f in allf if lv[0] <= level_of(f) <= lv[-1]]      # 사이 층 채우기
    def near_exit(c):
        return any(box(*c['box']).buffer(4.0).contains(Point(e['x'], e['y'])) for e in exits_cl)
    # 실제 도면 계단(SPA·글자 확인)은 비상구 기호가 없으므로 '비상구 근접' 대신 층간 반복(2개 층 이상)으로 확정
    stairs = [c for c in stairs if (near_exit(c) or any(m.get('real') for m in c['m'])) and (len(set(m['floor'] for m in c['m'])) >= 2 or n_fl == 1
              or (REAL_STAIR2 and real_single_ok(c)))]
    if REAL_STAIR2 and any(p.get('real') for p in plans): real_stair_extend(stairs, plans, out_floors)
    elevs = [c for c in cluster_vertical([q for q in all_pts if q['type'] == 'ev'], 1.6) if len(c['floors']) >= min(2, n_fl)]
    if REAL_STAIR2 and any(p.get('real') for p in plans): real_ev_extend(elevs, plans, out_floors)
    for c in elevs:
        bxs = [m['box'] for m in c['m'] if 'box' in m]
        c['box'] = [float(np.median([b[k] for b in bxs])) for k in range(4)] if bxs else [c['x'] - 1.2, c['y'] - 1.2, c['x'] + 1.2, c['y'] + 1.2]
        lv = [level_of(f) for f in c['floors']]
        c['floors'] = [f for f in sorted(out_floors, key=level_of) if min(lv) <= level_of(f) <= max(lv)]   # 샤프트는 사이 층도 관통
    # 층마다 발자국·문
    for fid, F in out_floors.items():
        if F.get('_real') is not None:                     # 실제 도면: 건물 덩어리 외곽(대지선·주차·조경·옥상정원 벗김) + 모델 문(외부 출입문 판정 포함)
            fp = F['_fp']
            F['footprint_area'] = round(fp.area, 1) if fp is not None else None
            continue
        fp = geom.footprint_from_walls(F['walls'])
        if plans[F['plan']].get('scan') and SCP.MAINFP: fp = SCP.main_footprint(fp)
        ca = (F.get('card') or {}).get('area')
        if fp is not None and ca and 0.7 <= fp.area / ca <= 1.5:      # 외곽을 믿을 만할 때만 밖으로 삐진 선 제거
            F['walls'] = geom.clip_walls(F['walls'], fp)
        F['footprint_area'] = round(fp.area, 1) if fp is not None else None
        F['doors'] = geom.gap_doors(F['walls'], fp, occ=occ_fn(plans[F['plan']]) if OCC else None)
        F['_fp'] = fp
    xres = model_pass(plans, out_floors) if model else None      # 선택: 모델 교차검증(벽 보충 시 외곽·문 갱신 → 검사보다 먼저)
    flags = checks(info, out_floors, stairs, elevs)
    if REAL_STAIR2:
        for k, c in enumerate(stairs):
            if c.get('cert') in ('single', 'single+extended', 'extended'):
                flags.append(dict(kind='stair_partial', name=f'계단{k + 1}', floors=c['floors'], floors_ko=', '.join(ko_floor(f) for f in c['floors']),
                                  ext_floors=c.get('ext_floors') or [], outdoor=bool(c.get('outdoor')),
                                  why_ko=('옥외 계단' if c.get('outdoor') else '한 층에만 그려진 계단') if c['cert'].startswith('single') else '위·아래 층은 점선·흐린 디딤판으로만 이어짐',
                                  cert=c['cert']))
        for k, c in enumerate(elevs):
            if any(m.get('real') for m in c['m']) and ev_cert(c) != 'model':
                flags.append(dict(kind='elev_text', name=f'승강기 {chr(65 + k)}', floors=c['floors'], word='승강기 글자', cert=ev_cert(c)))
    from firebim import fac as FAC                           # 층별 소방시설 목록 + 위치 확실성(located/estimated/present)
    fac_res, fac_flags = FAC.build(info, out_floors, [dict(floors=c['floors'], box=c['box']) for c in stairs], [dict(floors=c['floors'], box=c['box']) for c in elevs], plans=plans)
    flags += fac_flags
    if xres: flags += xres.pop('_flags')
    ext = extensions(pdf, info, out_floors, stairs, elevs, flags, daejang, site, ilramdo)
    if os.environ.get('FIREBIM_EVAC', '1') != '0':           # 대피로·진입로(firebim/evac.py) — 결과는 res['evac'], 플래그는 res['evac']['flags']
        try:
            from firebim.evac import analyze as evac_analyze
            ext['evac'] = evac_analyze(info, out_floors, stairs, elevs, plans, ext)
        except Exception as e: ext['evac'] = dict(error=repr(e))
    res = dict(card=info, floors={k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in out_floors.items()},
               stairs=[dict(x=round(c['x'], 2), y=round(c['y'], 2), floors=c['floors'], box=[round(v, 2) for v in c['box']],
                            **({'cert': c['cert'], 'outdoor': bool(c.get('outdoor'))} if c.get('cert') else {}), **({'ext_floors': c['ext_floors']} if c.get('ext_floors') else {})) for c in stairs],
               elevs=[dict(x=round(c['x'], 2), y=round(c['y'], 2), floors=c['floors'], box=c['box'],
                           **({'cert': ev_cert(c)} if any(m.get('real') for m in c['m']) else {}), **({'ext_floors': c['ext_floors']} if c.get('ext_floors') else {})) for c in elevs],
               flags=flags, fac=fac_res, **ext, **({'xcheck': xres} if xres is not None else {}), plans=[dict(i=p['i'], caption=p['caption'], s=p['s'], scale_src=p['scale_src'], tx=p['tx'], ty=p['ty'], align=p['conf'],
                                                                                                                  **(real_info(p) if p.get('real') else {})) for p in plans])
    if info.get('양식') == 'std':                   # ⑪ 그림 목록(위치도·사진·진압작전도·평면도 — 캡션으로 구분, 진압작전도 인식 모듈 입력)
        figs = extract_figures(doc)
        res['figures'] = [{k: v for k, v in f.items() if k != 'img'} for f in figs]
        if keep: res['_figures'] = figs
    if FLAGTEXT == '1' or (FLAGTEXT == 'real' and any(p.get('real') for p in plans)):   # 확인사항 쉬운 말(msg) + 기술 정보(detail), 층별 축척 근거(scale_basis) — firebim/flagtext.py
        from firebim.flagtext import annotate
        annotate(res)
    if keep:      # 3D 출력용(도면 이미지·외곽선) — JSON 저장 안 함
        res['_plans'] = [dict(img=p.get('img_src', p['img']), s=p['s'], tx=p['tx'], ty=p['ty'], caption=p['caption']) for p in plans]
        res['_fp'] = {k: v['_fp'] for k, v in out_floors.items()}
    return res


REAL_STAIR2 = os.environ.get('FIREBIM_REAL_STAIR2', '1') != '0'      # 실제 도면: 한 층에만 그려진 계단·옥외 계단 · 위아래 층 이어짐(점선 디딤판) · 확실성 표시(2026-10-03)


def ko_floor(f):
    from firebim.flagtext import floor_ko
    return floor_ko(f)


def real_single_ok(c):
    """한 층에만 보이는 실제 도면 계단을 남길지: 디딤판 5단 이상 + (SPA 모델 또는 UP·DN·계단 글자), 또는 옥외(덩어리 밖) 디딤판 6단 이상"""
    m = [q for q in c['m'] if q.get('real')]
    return any((q.get('treads', 0) >= 5 and (q.get('src') == 'model' or q.get('label'))) or (q.get('outdoor') and q.get('treads', 0) >= 6) for q in m)


def real_stair_extend(stairs, plans, floors):
    """계단 묶음마다 확실성(cert)·종류(kind) 붙이고, 바로 위·아래 층 실제 도면의 같은 자리에 디딤판(점선 포함)이 있으면 그 층까지 잇기(cert 'extended')"""
    from firebim import realcard as RC
    by_floor = {f: p for p in plans for f in p['floors'] if p.get('real')}
    order = sorted(floors, key=level_of)
    for c in stairs:
        mem = [q for q in c['m'] if q.get('real')]
        if not mem: continue
        fl = sorted(set(q['floor'] for q in c['m']), key=level_of)
        c['outdoor'] = sum(1 for q in mem if q.get('outdoor')) >= max(1, len(mem) / 2)
        c['cert'] = 'confirmed' if len(fl) >= 2 else 'single'
        added = []
        for side in (-1, 1):
            k = order.index(fl[0] if side < 0 else fl[-1]) + side
            while 0 <= k < len(order):
                f = order[k]; p = by_floor.get(f)
                if p is None or f in fl: break
                s, tx, ty = p['s'], p['tx'], p['ty']
                x0, y0, x1, y1 = c['box']
                bpx = [(x0 - tx) * s, -(y1 - ty) * s, (x1 - tx) * s, -(y0 - ty) * s]
                if not RC.stair_at(p['img'], bpx, s): break
                added.append(f); k += side
        if added:
            c['ext_floors'] = sorted(added, key=level_of)
            lv = [level_of(f) for f in fl + added]
            c['floors'] = [f for f in order if min(lv) <= level_of(f) <= max(lv)]
            c['cert'] = 'extended' if c['cert'] == 'confirmed' else 'single+extended'


def real_ev_extend(elevs, plans, floors):
    """승강기 샤프트를 바로 위·아래 실제 도면 층으로 잇기: 그 층 같은 자리에 승강기 글자·X 표시가 보이면(작은 영역 OCR) 그 층 정차로 추가(ext_floors)"""
    from firebim import realcard as RC
    by_floor = {f: p for p in plans for f in p['floors'] if p.get('real')}
    order = sorted(floors, key=level_of)
    for c in elevs:
        if not any(m.get('real') for m in c['m']): continue
        bxs = [m['box'] for m in c['m'] if 'box' in m]
        if not bxs: continue
        box_ = [float(np.median([b[k] for b in bxs])) for k in range(4)]
        fl = sorted(set(m['floor'] for m in c['m']), key=level_of)
        added = []
        for side in (-1, 1):
            k = order.index(fl[0] if side < 0 else fl[-1]) + side
            while 0 <= k < len(order):
                f = order[k]; p = by_floor.get(f)
                if p is None or f in fl: break
                s, tx, ty = p['s'], p['tx'], p['ty']
                x0, y0, x1, y1 = box_
                if not RC.ev_at(p['img'], [(x0 - tx) * s, -(y1 - ty) * s, (x1 - tx) * s, -(y0 - ty) * s], s): break
                added.append(f); k += side
        if added:
            for f in added: c['m'].append(dict(type='ev', x=c['x'], y=c['y'], box=box_, floor=f, real=True, cert='extended'))
            c['floors'] = sorted(set(c['floors']) | set(added), key=level_of)
            c['ext_floors'] = sorted(added, key=level_of)


def real_floor(F, p, fid, info, all_pts):
    """실제 도면 평면도(p['real'] = realcard.process 결과) → 층 F 에 외곽·문·창·실명 붙이기(전역 좌표), 승강기 → all_pts(층간 묶음)"""
    from shapely import affinity
    R, s, tx, ty = p['real'], p['s'], p['tx'], p['ty']
    F['_real'] = R
    F['_fp'] = affinity.translate(R['fp'], tx, ty) if R['fp'] is not None else None
    etc = ((info.get('floors') or {}).get(fid) or {}).get('etc') or ''
    F['doors'] = [dict(d, x=round(d['x'] + tx, 2), y=round(d['y'] + ty, 2)) for d in R['doors']]
    if '피난층' in etc and fid != '1F': F['entry'] = True               # 정보카드 ⑩ '피난층'(경사지 등 1층이 아닌 지상 진입층) → 외부 출입문 = 진입구
    F['windows'] = [dict(w, x=round(w['x'] + tx, 2), y=round(w['y'] + ty, 2)) for w in R['windows']]
    F['room_labels'] = [dict(name=l['name'], x=round(l['px'] / s + tx, 2), y=round(-l['py'] / s + ty, 2), src=l['src']) for l in R['labels'] if l.get('name')]
    F['peeled_m2'] = R['peeled']
    for q in R['elevs']:
        b = q['box']; (x0, y1), (x1, y0) = (b[0] / s + tx, -b[1] / s + ty), (b[2] / s + tx, -b[3] / s + ty)
        all_pts.append(dict(type='ev', x=(x0 + x1) / 2, y=(y0 + y1) / 2, box=[x0, y0, x1, y1], floor=fid, real=True, cert=q.get('cert', 'model')))


def ev_cert(c):
    """승강기 묶음 확실성: 한 층이라도 모델·X 표시 검출이면 'model', 글자 칸이면 'word', 글자 중심 추정만이면 'text'"""
    cs = [m.get('cert', 'model') for m in c['m']]
    return 'model' if 'model' in cs else ('word' if 'word' in cs else 'text')


def real_info(p):
    R = p['real']
    return dict(real=dict(scale_conf=R['scale_conf'], scale_disagree=R['scale_disagree'], scale_cands=R['scale_cands'], model_s=R['model_s'], rerun=R['rerun'],
                          tags=len(R['tags']), tag_pairs=R['n_tag_pairs'], grid=[len(R['vl']), len(R['hl'])], bubbles=len(R['bubbles'] or {}),
                          stairs=len(R['stairs']), elevs=len(R['elevs']), doors=len(R['doors']), ext_doors=sum(1 for d in R['doors'] if d['ext']),
                          peeled_m2=R['peeled'], align=p.get('align_info'), t_s=R.get('t_s'), wall_style=R.get('wall_style')))


def occ_fn(p):
    """전역 좌표 벽 틈 → 도면 픽셀에서 채색 기호 가림·빈 길이"""
    tx, ty, s, M = p['tx'], p['ty'], p['s'], p['M']
    sy = [(q['type'], q['px'] / s, -q['py'] / s) for q in p['symbols']]
    def f(o, c, a, b, t):
        c, a, b = (c - ty, a - tx, b - tx) if o == 'h' else (c - tx, a - ty, b - ty)
        r = geom.gap_profile(M, s, o, c, a, b, t)
        near = []                                   # 틈을 덮은 기호: 벽 선에서 1.2 m 안, 틈 구간 ±0.3 m 안
        for typ, x, y in sy:
            u, v = (x, y) if o == 'h' else (y, x)
            if abs(v - c) < 1.2 and a - 0.3 < u < b + 0.3: near.append((abs(v - c), typ))
        return r + (min(near)[1] if near else None,)
    return f


def model_pass(plans, floors):
    """선택 경로: 도면마다 STR 모델 → 층마다 규칙↔모델 교차검증(firebim/xcheck.py).
    - 벽마다 conf = 모델 지지율(두 추출기 일치 → 1에 가까움)
    - 규칙만 본 벽 조각 / 모델만 본 벽 조각 → 의심도 p 가 붙은 불일치 목록(검수 항목의 원천)
    - 모델이 확신하는 누락 벽(p ≥ fill_p) 보충 → 외곽·문(틈) 다시 계산, 모델 문 보충(src='model')
    반환: 결과 JSON 'xcheck' 절 {도면 번호: {...}} + '_flags'(도면별 불일치 요약 플래그 kind='xcheck_wall')"""
    from firebim import xcheck as X
    C = X.CFG
    out, xflags = {}, []
    for p in plans:
        fl = sorted([f for f in floors if floors[f]['plan'] == p['i']], key=level_of)
        if not fl: continue
        if p.get('real'):                          # 실제 도면 경로는 벽·문이 이미 모델 결과(교차검증 대상 규칙 벽 없음)
            out[str(p['i'])] = dict(note='real_path: 벽·창·문 = STR 모델(firebim/realcard.py)'); continue
        if p.get('scan'):                          # 스캔본: 축척이 면적 대체로 크게 틀리면 모델 작업 영상이 수천만 화소 → 메모리 초과(hold 에서 7GB OOM) → 건너뜀
            from firebim.str_seg import WORK_PPM_1_100
            from firebim import scanplan as SCP
            npx = p['img'].shape[0] * p['img'].shape[1] * (WORK_PPM_1_100 / max(p['s'], 1e-3)) ** 2
            if npx > SCP.MODEL_MAX_PX:
                out[str(p['i'])] = dict(error=f'model_skipped: 작업 영상 {npx / 1e6:.0f}M 화소(축척 {p["scale_src"]})'); continue
        try:
            P = X.model_plan(p['img'], p['s']); P.update(sat=p['M']['sat'], s=p['s'])
        except Exception as e:                      # 모델 파일·onnxruntime 없음 → 규칙 결과 그대로
            out[str(p['i'])] = dict(error=repr(e)); continue
        rep = None
        for fid in fl:
            F = floors[fid]
            r = X.cross_floor(F, P, p['tx'], p['ty'], F['_fp'])
            for w, c in zip(F['walls'], r['sup']): w['conf'] = c
            add = X.fill_walls(F['walls'], r['model_only']) if C['wall_fill'] else []
            for m in r['model_only']: m['filled'] = m['p'] >= C['fill_p'] and not m.get('bridge')
            for w in add: w['conf'] = 0.0
            W, n_ext = X.extend_ends(F['walls'] + add, P, p['tx'], p['ty']) if C['corner'] else (F['walls'] + add, 0)
            if add or n_ext:
                fp = geom.footprint_from_walls(W)
                if p.get('scan'):
                    from firebim import scanplan as SCP
                    if SCP.MAINFP: fp = SCP.main_footprint(fp)
                ca = (F.get('card') or {}).get('area')
                if fp is not None and ca and 0.7 <= fp.area / ca <= 1.5: W = geom.clip_walls(W, fp)
                F['walls'], F['_fp'] = W, fp
                F['footprint_area'] = round(fp.area, 1) if fp is not None else None
                F['doors'] = geom.gap_doors(W, fp, occ=occ_fn(p) if OCC else None)
                r2 = X.cross_floor(F, P, p['tx'], p['ty'], fp)
                r['door_cand'] = r2['door_cand']
                # 새 외곽으로 잘려 나간 규칙 벽(표제·범례 상자 등)의 불일치 조각은 이미 해소 → 검수 항목에서 뺌
                keep = [x for x in r['rule_only'] if X.still_present(x, W)]
                r['resolved_rule_only'] = len(r['rule_only']) - len(keep); r['rule_only'] = keep
            dq = r.pop('door_cand')
            if C['door_add']:
                F['doors'] = F['doors'] + [dict(d, conf=round(1 - d['p'], 3)) for d in dq if 1 - d['p'] >= C['door_p']]
            r['doors_model'] = dq
            if rep is None: rep = dict(r, floor=fid, floors=fl)
        del rep['sup']
        out[str(p['i'])] = rep
        ro = [x for x in rep['rule_only'] if x['p'] >= C['item_p']]; mo = [x for x in rep['model_only'] if x['p'] >= C['item_p']]
        if ro or mo:
            xflags.append(dict(kind='xcheck_wall', floor=fl[0], floors=fl, rule_only=len(ro), model_only=len(mo), agree=rep['agree'],
                               len_rule_only=round(sum(x['b'] - x['a'] for x in ro), 1), len_model_only=round(sum(x['b'] - x['a'] for x in mo), 1),
                               p=round(max([x['p'] for x in ro + mo]), 2)))
    out['_flags'] = xflags
    return out


def _find(pdf, given, names):
    if given: return given if os.path.exists(given) else None
    d = os.path.dirname(os.path.abspath(pdf))
    return next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))), None)


def extensions(pdf, info, floors, stairs, elevs, flags, daejang=None, site=None, ilramdo=None):
    """기능 12(경계구역) · 18(건축물대장 교차검증) · 24(외부 부서 위치). 실패해도 기본 변환은 그대로 둔다."""
    from firebim import zones as Z, daejang as DJ, site as ST
    from firebim.export import spaces
    from firebim.sop import CFG
    out = {}
    pd = [dict(x=c['x'], y=c['y'], floors=c['floors'], box=c['box']) for c in stairs]
    pe = [dict(x=c['x'], y=c['y'], floors=c['floors'], box=c['box']) for c in elevs]
    fh = None
    try:
        p = _find(pdf, daejang, ['daejang.json', 'daejang.pdf'])
        if p:
            rep, fl = DJ.compare(DJ.load(p), info, floors, len(elevs))
            rep['file'] = os.path.basename(p); out['daejang'] = rep; flags += fl; fh = rep.get('floor_height')
    except Exception as e: out['daejang'] = dict(error=repr(e))
    try:
        cfg = dict(CFG, floor_height_m=fh) if fh else CFG
        p = _find(pdf, ilramdo, ['zones.json', 'ilramdo.json'])
        sp = {f: spaces(F['walls'], F['doors'], F['_fp']) for f, F in floors.items()}
        aref = {r['floor']: r['daejang'] for r in (out.get('daejang') or {}).get('table', []) if r.get('daejang')}
        zr, fl = Z.compute(floors, pd, pe, sp, Z.load_ilramdo(p) if p else None, cfg, aref)
        out['zones'] = zr; flags += fl
    except Exception as e: out['zones'] = dict(error=repr(e))
    try:
        p = _find(pdf, site, ['site.json'])
        has_fp = any(F.get('_fp') is not None for F in floors.values())
        ws = [dict(w, _f=f) for f, F in floors.items() if level_of(f) >= 0 for w in F['walls']] if has_fp else []
        # 진압작전도(정보카드 ⑪) 가 있으면 그림에서 부지(도로·소화전·송수구·출입구)를 읽어 기능 24 를 돌린다. 없거나 실패하면 site.json.
        if has_fp:
            try:
                rep, fl = sitemap_site(pdf, info, floors, fh, ws, ST.load(p) if p else None)
                if rep is not None:
                    rep['floor_h_src'] = '건축물대장 높이÷지상층수' if fh else '기본값(기관 SOP 설정 필요)'
                    out['site'] = rep; flags += fl
            except Exception as e: out['sitemap_error'] = repr(e)
        if 'site' not in out and p and has_fp:
            rep, fl = ST.analyze(ST.load(p), {f: F.get('_fp') for f, F in floors.items()}, [f for f in floors if level_of(f) >= 0], info, fh, walls=ws)
            rep['file'] = os.path.basename(p); rep['floor_h_src'] = '건축물대장 높이÷지상층수' if fh else '기본값(기관 SOP 설정 필요)'
            out['site'] = rep; flags += fl
    except Exception as e: out['site'] = dict(error=repr(e))
    return out


def sitemap_image_of(pdf, info):
    """진압작전도 이미지(BGR): 카드 해석기가 넘긴 것(info['sitemap_img']) → PDF 캡션 '진압작전도' 이미지 → 옆 파일 sitemap.png"""
    import cv2
    from firebim import sitemap as SM
    img = info.get('sitemap_img') if isinstance(info, dict) else None
    if img is None:                                              # 표준양식: ⑪ 그림 목록에서 '진압작전도'(첫 동)
        try:
            import pymupdf
            from firebim.card import extract_figures
            figs = [f for f in extract_figures(pymupdf.open(pdf)) if f['kind'] == '진압작전도' and f.get('dong', 0) == 0]
            if figs: img = figs[0]['img']
        except Exception:
            img = None
    if img is None: img = SM.find_in_pdf(pdf)
    if img is None:
        q = _find(pdf, None, ['sitemap.png', 'sitemap.jpg', '진압작전도.png', '진압작전도.jpg'])
        if q: img = cv2.imread(q)
    return img


def sitemap_site(pdf, info, floors, fh, ws, site_json=None):
    """[기능 24 · 작전도] 진압작전도 인식 → 1층 외곽에 정합 → 외부 부서 계산 + 교차검증. 작전도 없으면 (None, [])"""
    import math
    from firebim import sitemap as SM, site as ST
    img = sitemap_image_of(pdf, info)
    if img is None: return None, []
    rec = SM.recognize(img, legend=info.get('sitemap_legend'))
    fps = {f: F.get('_fp') for f, F in floors.items()}
    ref_id = '1F' if fps.get('1F') is not None else next((f for f in sorted(fps, key=level_of) if level_of(f) >= 0 and fps[f] is not None), None)
    ref = ST.regularize(fps[ref_id]) if ref_id else None
    area = ((info.get('floors') or {}).get('1F') or {}).get('area')
    fr = SM.to_frame(rec, ref_outline=ref, card_area=area)
    if fr is None: return dict(src='진압작전도', error=rec.get('error') or 'registration_failed'), [dict(kind='sitemap_fail', why=rec.get('error') or 'registration_failed')]
    F1 = floors.get(ref_id) or {}
    openings = [(d['x'], d['y']) for d in F1.get('doors', []) if d.get('ext')]
    ex_sym = [(q['x'], q['y']) for q in F1.get('symbols', []) if q['type'] == 'exit']
    plan_exits = [o for o in openings if any(math.dist(o, e) < 2.5 for e in ex_sym)]
    plan_fdc = [(q['x'], q['y']) for q in F1.get('fac_symbols', []) if q.get('type') == 'siamese']
    rep, fl = ST.analyze_drawing(fr, fps, [f for f in floors if level_of(f) >= 0], info, fh, walls=ws, site_json=site_json,
                                 openings=openings or None, plan_exits=plan_exits, plan_fdc=plan_fdc)
    rep['file'] = '진압작전도(정보카드 ⑪)' + (' + site.json(장애 구간)' if site_json else '')
    rep['drawing']['legend'] = {t: dict(bgr=v['bgr'], size=round(v['size'], 1)) for t, v in (rec.get('legend') or {}).get('icons', {}).items()}
    return rep, fl


def checks(info, floors, stairs, elevs):
    """정보카드 값 ↔ 도면 추출 값 교차검증 → 검수 대상"""
    flags = []
    for fid, F in floors.items():
        c = F.get('card') or {}
        if c.get('area') and F.get('footprint_area'):
            r = F['footprint_area'] / c['area']
            if not (0.8 <= r <= 1.2): flags.append(dict(kind='area', floor=fid, card=c['area'], drawing=F['footprint_area'], ratio=round(r, 2)))
        nh = sum(1 for q in F['symbols'] if q['type'] == 'hose')
        ch = (c.get('fac') or {}).get('옥내소화전')
        if ch is not None and ch != nh and not F.get('_nosym'): flags.append(dict(kind='hose_count', floor=fid, card=ch, drawing=nh))
        if F['align'] in ('low', 'none'): flags.append(dict(kind='align', floor=fid, conf=F['align']))
        R = F.get('_real')
        if R is not None and R.get('scale_disagree'):     # 실제 도면: 이 층 자체의 강한 축척 출처가 최종 축척과 10% 넘게 다름 → 검수
            flags.append(dict(kind='scale_disagree', floor=fid, s=round(R['s'], 2), src=R['scale_src'],
                              cands=[dict(src=c['src'], s=round(c['s'], 1)) for c in R['scale_cands'] if not c['src'].startswith('grid:')]))
    st = info.get('stairs') or {}
    n_card = sum(st.values()) if st else None
    if n_card is not None and n_card != len(stairs): flags.append(dict(kind='stair_count', card=n_card, drawing=len(stairs)))
    if info.get('양식') == 'std':                    # 표준양식 ⑦·⑨ 추가 검사(기존 근사 배치 카드에는 해당 칸 없음 → 기존 결과 불변)
        n_em = (info.get('elev') or {}).get('emergency') or 0
        if n_em and n_em > len(elevs): flags.append(dict(kind='emev_count', card=n_em, drawing=len(elevs)))
        n_all = n_em + ((info.get('elev') or {}).get('normal') or 0)          # ⑦ 승강기(일반)+(비상) 합 ↔ 도면 승강기 수(일반 승강기 포함, 2026-10-03)
        if ELEV_CHECK and (n_all or elevs) and n_all != len(elevs) and not (n_em and n_em > len(elevs) and n_all <= n_em):
            flags.append(dict(kind='elev_count', card=n_all, drawing=len(elevs)))
        for h in info.get('위험물') or []:
            flags.append(dict(kind='hazmat', cls=h['구분'], name=h.get('시설명'), item=h.get('품명'), qty=h.get('수량')))
    flags += ocr_flags(info)
    return flags


OCR_FLAG_MAX = 20
ELEV_CHECK = os.environ.get('FIREBIM_ELEV_CHECK', '1') != '0'      # 표준양식 ⑦ 승강기 합 ↔ 도면 승강기 수(elev_count)


def ocr_flags(info):
    """스캔본 정보카드(OCR) — 신뢰도 낮은 항목마다 검수 플래그 kind='card_ocr'(관계인은 값 없이 항목 이름만). 글자층 카드는 []"""
    if info.get('source') != 'ocr': return []
    from firebim.card import ocr_value
    oc = info.get('ocr') or {}
    if oc.get('error'): return [dict(kind='card_ocr_fail', why=oc['error'], pages=oc.get('pages'))]
    out = []
    for x in sorted(oc.get('low') or [], key=lambda x: x['conf'])[:OCR_FLAG_MAX]:
        v = None if x.get('private') else ocr_value(info, x['field'])
        if isinstance(v, dict): v = ' / '.join(f'{k} {w}' for k, w in v.items() if w is not None)
        elif isinstance(v, list): v = f'{len(v)}건'
        out.append(dict(kind='card_ocr', field=x['field'], conf=x['conf'], value=(str(v)[:80] if v is not None else None), private=bool(x.get('private'))))
    return out


if __name__ == '__main__':
    r = run(sys.argv[1])
    json.dump(r, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
    print('floors', len(r['floors']), 'stairs', len(r['stairs']), 'elevs', len(r['elevs']), 'flags', len(r['flags']))
