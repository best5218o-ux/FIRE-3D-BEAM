# -*- coding: utf-8 -*-
"""엔진 결과 → 3D 지휘화면 데이터(D, BIM) + IFC4 파일"""
import io, base64, math, json
import numpy as np, cv2
from PIL import Image
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from firebim.card import level_of

H_FLOOR = 4.6
SLAB = 0.3


def ko_name(fid):
    if fid.startswith('B'): return f'지하{fid[1:]}층'
    if fid == 'RF': return '옥탑층'
    if fid == 'PIT': return 'PIT층'
    return f'지상{fid[:-1]}층'


def _b64(arr, fmt='WEBP', q=74, maxw=1200):
    im = Image.fromarray(arr)
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, fmt, quality=q)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(b.getvalue()).decode()


def plan_texture(img, s, tx, ty, bb, fp, R=10):
    """도면을 전역 좌표 바닥 텍스처로 (어두운 화면용: 흑백 선은 밝게, 채색은 유지, 건물 밖 제거)"""
    x0, y0, x1, y1 = bb
    W, H = int((x1 - x0) * R), int((y1 - y0) * R)
    M = np.float32([[s / R, 0, (x0 - tx) * s], [0, s / R, (ty - y1) * s]])
    w = cv2.warpAffine(img, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    w = w[..., ::-1].astype(np.float32)          # RGB
    lum = w.mean(2); sat = w.max(2) - w.min(2)
    alpha = np.clip((205 - lum) * 2.2, 0, 255)
    alpha = np.where(sat > 60, np.clip(sat * 2.5, 0, 255), alpha)
    col = w.copy(); col[sat <= 60] = [226, 234, 244]
    if fp is not None:
        m = np.zeros((H, W), np.uint8)
        for g in ([fp] if fp.geom_type == 'Polygon' else list(fp.geoms)):
            pts = np.array([[(x - x0) * R, (y1 - y) * R] for x, y in g.exterior.coords], np.int32)
            cv2.fillPoly(m, [pts], 1)
        m = cv2.dilate(m, np.ones((31, 31), np.uint8))
        alpha = alpha * m
    red = ((w[..., 0] > 170) & (w[..., 1] < 110) & (w[..., 2] < 110)).astype(np.uint8)
    ev = np.zeros((H, W, 4), np.uint8); ev[..., 0] = 255; ev[..., 1] = 60; ev[..., 2] = 40; ev[..., 3] = cv2.dilate(red, np.ones((3, 3), np.uint8)) * 255
    return _b64(np.dstack([col, alpha]).astype(np.uint8)), (_b64(ev, q=70) if red.sum() > 50 else None)


def spaces(walls, doors, fp, res=5):
    """벽망으로 둘러싸인 영역 → 실(IfcSpace) 후보"""
    if fp is None: return []
    x0, y0, x1, y1 = fp.bounds
    W, H = int((x1 - x0) * res) + 4, int((y1 - y0) * res) + 4
    P = lambda x, y: (int((x - x0) * res) + 2, int((y1 - y) * res) + 2)
    inside = np.zeros((H, W), np.uint8); img = np.zeros((H, W), np.uint8)
    for g in ([fp] if fp.geom_type == 'Polygon' else list(fp.geoms)):
        cv2.fillPoly(inside, [np.array([P(x, y) for x, y in g.exterior.coords], np.int32)], 1)
    for w in walls:
        a, b = (P(w['a'], w['c']), P(w['b'], w['c'])) if w['o'] == 'h' else (P(w['c'], w['a']), P(w['c'], w['b']))
        cv2.line(img, a, b, 1, max(1, int(round(w['t'] * res))))
    for d in doors:
        if d['o'] == 'h': a, b = P(d['x'] - d['w'] / 2, d['y']), P(d['x'] + d['w'] / 2, d['y'])
        else: a, b = P(d['x'], d['y'] - d['w'] / 2), P(d['x'], d['y'] + d['w'] / 2)
        cv2.line(img, a, b, 1, 1)
    free = ((inside == 1) & (img == 0)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, 4)
    out = []
    for i in range(1, n):
        area = st[i][4] / res / res
        if area < 4 or area > 4000: continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ap = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.2, True)[:, 0, :]
        if len(ap) < 3: continue
        pg = Polygon([((px - 2) / res + x0, y1 - (py - 2) / res) for px, py in ap]).buffer(0)
        if pg.geom_type != 'Polygon' or pg.area < 3: continue
        out.append(dict(poly=[[round(x, 2), round(y, 2)] for x, y in pg.exterior.coords[:-1]], area=round(area, 1)))
    return out


def build(res, plans=None, name_hint=None):
    plans = plans or res['_plans']
    card = res['card']
    fids = sorted(res['floors'], key=level_of)
    # 지상 1층 기준 층 레벨
    D = dict(floors=[], markers=[], issues=[])
    BIM = dict(floors={}, stairs=[], elevs=[], equip=[], H=H_FLOOR)
    fps = {}
    for fid in fids:
        F = res['floors'][fid]
        fp = (res.get('_fp') or {}).get(fid)
        if fp is None:
            from firebim.geom import footprint_from_walls
            fp = footprint_from_walls(F['walls'])
        fps[fid] = fp
    ground = '1F' if '1F' in fids else fids[0]
    for fid in fids:
        F = res['floors'][fid]; fp = fps[fid]; lv = level_of(fid)
        p = plans[F['plan']]
        bb = [round(v, 1) for v in (fp.bounds if fp is not None else (-10, -10, 60, 40))]
        bb = [bb[0] - 4, bb[1] - 4, bb[2] + 4, bb[3] + 4]
        tex, evac = plan_texture(p['img'], p['s'], p['tx'], p['ty'], bb, fp)
        polys = [[[round(x, 2), round(y, 2)] for x, y in list(g.exterior.coords)[:-1]] for g in ([fp] if fp is not None and fp.geom_type == 'Polygon' else (list(fp.geoms) if fp is not None else []))]
        c = F.get('card') or {}
        ok, plan_img = cv2.imencode('.jpg', p['img'], [cv2.IMWRITE_JPEG_QUALITY, 70])
        KO = {'exit': '비상구', 'hose': '옥내소화전', 'gas': '가스밸브', 'panel': '수신기'}
        det = {v: 0 for v in KO.values()}
        for q in F['symbols']:
            if q['type'] in KO: det[KO[q['type']]] += 1
        D['floors'].append(dict(id=fid, name=ko_name(fid), level=lv, use=c.get('use') or '', area=c.get('area'), fac=c.get('fac') or {},
                                poly=polys, bb=bb, tex=tex, evac=evac, plan='data:image/jpeg;base64,' + base64.b64encode(plan_img.tobytes()).decode(),
                                det=det, group='main', grade=None))
        # 문: 틈 문 + 외벽 개구부(지상 진입층은 진입구)
        doors = []
        for d in F['doors']:
            if d.get('ext'):
                kind = 'entrance' if (fid == ground or F.get('entry')) else 'exit'      # entry: 정보카드 ⑩ '피난층'(실제 도면 경로, 1층이 아닌 지상 진입층)
                name = ('외부 출입구' if kind == 'entrance' else '외벽 개구부') + f" (폭 {d['w']:.1f}m, 추정)"
            else:
                kind, name = 'door', '문(추정)'
            if d.get('src') == 'model': name += ' · AI 모델 보충'
            doors.append(dict(kind=kind, o=d['o'], x=d['x'], y=d['y'], w=d['w'], name=name, ext=bool(d.get('ext'))))
        sp = spaces(F['walls'], [d for d in doors if d['kind'] != 'entrance'], fp)
        BIM['floors'][fid] = dict(walls=F['walls'], doors=doors, spaces=sp, fp=fp)
        GAS = '의료가스(산소 등) 차단밸브' if '의료' in (card.get('use') or '') else '가스 차단밸브'
        for q in F['symbols']:
            t = q['type']
            if t in ('hose', 'gas', 'panel'):
                BIM['equip'].append(dict(type=t, floor=fid, x=q['x'], y=q['y'], name={'hose': '옥내소화전', 'gas': GAS, 'panel': '자동화재탐지설비 수신기'}[t]))
            if t in ('exit', 'hose', 'gas', 'panel', 'ev'):
                D['markers'].append(dict(floor=fid, type=t, label={'exit': '비상구', 'hose': '옥내소화전', 'gas': GAS, 'panel': '수신기', 'ev': '엘리베이터'}[t], x=q['x'], y=q['y'], src='auto', conf=q.get('conf')))
        for d in doors:
            if d['kind'] == 'entrance':
                D['markers'].append(dict(floor=fid, type='entrance', label=d['name'], x=d['x'], y=d['y'], src='auto'))
    ST_NOTE = {'confirmed': '실제 도면: 디딤판·AI 계단 검출이 여러 층 같은 자리에서 반복(확실)', 'single': '실제 도면: 한 층 도면에만 보임(위·아래 연결 층 확인 필요)',
               'extended': '실제 도면: 여러 층 반복 + 위·아래 층은 점선·흐린 디딤판으로 이어짐(그 층은 확인 필요)',
               'single+extended': '실제 도면: 한 층 확실 + 이웃 층은 점선 디딤판으로만 이어짐(확인 필요)'}
    EV_NOTE = {'model': '승강기 표기·AI 검출 층간 반복(자동)', 'word': "도면 글자('E/V'·'N인승' 등)가 든 칸으로 찾음(X 표시 없음 — 위치 확인)",
               'text': "도면 글자('E/V'·'N인승')만 있어 글자 둘레 2 m 칸으로 추정(위치·크기 확인 필요)"}
    for i, S in enumerate(res['stairs']):
        BIM['stairs'].append(dict(id=f'ST{i + 1}', name=f'계단{i + 1}', box=S['box'], floors=S['floors'], kind='옥외계단' if S.get('outdoor') else '내부계단',
                                  note='검수자 추가(검수 화면)' if S.get('review') == 'added' else ST_NOTE.get(S.get('cert'), '디딤판 영역·비상구 기호의 층간 반복으로 자동 판정'),
                                  **({'cert': S['cert']} if S.get('cert') else {})))
    for i, E in enumerate(res['elevs']):
        BIM['elevs'].append(dict(id=f'EV{i + 1}', name=f'승강기 {chr(65 + i)}', box=E['box'], floors=E['floors'], cars=1,
                                 src='검수자 추가(검수 화면)' if E.get('review') == 'added' else EV_NOTE.get(E.get('cert'), '승강기 표기 층간 반복(자동)'),
                                 **({'cert': E['cert']} if E.get('cert') else {})))
    # 확인사항(교차검증)
    from firebim.flagtext import TITLE as _TT, explain as _explain0
    def _explain(f):
        try: return _explain0(f)
        except Exception: return None, None
    kindtxt = dict(fac_count='소방시설 수량 불일치', area='층 면적 불일치', hose_count='옥내소화전 수량 불일치', stair_count='계단 수 불일치', align='층 위치 맞춤 신뢰도 낮음',
                   elev_count='승강기 수 불일치', scale_disagree='도면 축척 확인 필요')
    SCALE_SRC_KO = dict(sheet='도면 표제란 축척', dims='치수선', doors='문 폭', tags='창호 폭 표기', tread='계단 디딤판', area='정보카드 면적', grid='다른 층 기준선')
    for f in res['flags']:
        if f['kind'] == 'fac_count': continue            # 아래(소방시설)에서 따로
        ext = ext_issue(f)                   # 기능 12·18·24 확인사항
        if ext: D['issues'].append(ext); continue
        if f['kind'] == 'area': d = f"{ko_name(f['floor'])}: 정보카드 {f['card']:,.0f}㎡ / 도면 추정 {f['drawing']:,.0f}㎡ (비율 {f['ratio']})"
        elif f['kind'] == 'hose_count': d = f"{ko_name(f['floor'])}: 정보카드 {f['card']}개 / 도면 표기 {f['drawing']}개"
        elif f['kind'] == 'stair_count': d = f"정보카드 {f['card']}개소 / 도면에서 찾은 계단 {f['drawing']}개소"
        elif f['kind'] == 'elev_count': d = f"정보카드 ⑦ 승강기(일반+비상) {f['card']}대 / 도면에서 찾은 승강기 {f['drawing']}대"
        elif f['kind'] == 'scale_disagree':       # 현장 문구(d) + 변환 기록(x, 화면에서 '상세'로 접힘)
            srcs = '·'.join(dict.fromkeys(SCALE_SRC_KO.get(t, '') for c in f.get('cands', []) for t in str(c['src']).replace('real:', '').replace(':', '+').split('+') if SCALE_SRC_KO.get(t)))
            d = f"{ko_name(f['floor'])}: 도면 축척의 근거가 서로 달라 거리·면적이 틀릴 수 있음({srcs or '여러 근거'} 기준이 10% 넘게 차이) → 출동 전 실제 치수 1곳 확인"
            x = f"채택 축척 {f['s']} px/m ({f['src']}) · 이 층 근거별 값: " + ', '.join(f"{c['src']} {c['s']}" for c in f.get('cands', []))
        else: d = f"{ko_name(f.get('floor', ''))}: 기호 부족으로 그리드선만으로 위치를 맞춤 → 확인 필요"
        x = x if f['kind'] == 'scale_disagree' else None
        msg, det = (f.get('msg'), f.get('detail')) if f.get('msg') else _explain(f)   # 쉬운 말(firebim/flagtext.py) — JSON 에 없으면 그릴 때 만듦 · 기술 정보는 상세(x)로 접힘
        if msg: d = msg
        if det and msg: x = det if not x else f"{det} · {x}"
        D['issues'].append(dict(lv='high' if f['kind'] in ('stair_count', 'align', 'scale_disagree') else 'mid', t=kindtxt.get(f['kind'], _TT.get(f['kind'], f['kind'])), d=d, cat='카드↔도면',
                                **({'x': x} if x else {})))
    # 기능 12·18·24 데이터(뷰어용)
    for k in ('zones', 'daejang', 'site'):
        if res.get(k) and 'error' not in res[k]: D[k] = res[k]
    if res.get('fac'):                                   # 소방시설 전체 목록 + 위치 확실성(firebim/fac.py)
        from firebim.fac_catalog import CATS
        D['fac'] = dict(res['fac'], cats=CATS)
    for f in res['flags']:
        if f['kind'] == 'fac_count':
            D['issues'].append(dict(lv='mid', t='소방시설 수량 불일치(정보카드 ↔ 도면 기호)', cat='소방시설',
                                    d=f"{ko_name(f['floor'])} {f['name']}: 정보카드 {f['card']}개 / 도면 기호 {f['drawing']}개"))
    if res.get('evac') and 'error' not in res['evac']:          # 대피로·진입로(firebim/evac.py) → 뷰어 데이터 + 확인사항
        from firebim.evac import to_viewer, issue as evac_issue
        D['evac'] = to_viewer(res['evac'])
        D['issues'] += [i for i in (evac_issue(f, ko_name) for f in res['evac'].get('flags', [])) if i]
    st = card.get('stairs') or {}; ev = card.get('elev') or {}
    D['building'] = dict(name=card.get('name') or name_hint or '대상물', use=card.get('use') or '', gfa=card.get('gfa'),
                         floors=f"지하{sum(1 for f in fids if f.startswith('B'))}층 / 지상{sum(1 for f in fids if f.endswith('F'))}층",
                         stairs={'일반계단': st.get('normal', 0), '피난계단': st.get('evac', 0), '특별피난계단': st.get('special', 0), '옥외계단': st.get('outdoor', 0),
                                 '승강기_일반': ev.get('normal', 0), '승강기_비상용': ev.get('emergency', 0)},
                         source='소방활동 정보카드(가상) 자동 변환', auto=dict(align=[F['align'] for F in res['floors'].values()]))
    D['building']['card'] = card_panel(card)          # 표준양식 ①~⑨(관계인 성명·연락처는 opt-in 때만)
    # 활동계획 기준값(sop.py SEARCH·ACT·TRAIN) + 실마다 인명검색 용도 범주(firebim/search.py) → 뷰어 search.js·훈련 채점
    from firebim.search import annotate
    from firebim.sop import ACT, TRAIN, SIM
    annotate(D); D['sop_act'] = ACT; D['sop_train'] = TRAIN; D['sop_sim'] = SIM
    return D, BIM


def card_panel(card):
    """정보카드 양식 항목(표준양식 키 그대로) → 지휘화면 오른쪽 '정보카드(건물개황)' 절. 관계인은 redact 결과를 그대로(기본: 역할·기재 여부만)"""
    if not card or card.get('양식') != 'std': return dict(양식=card.get('양식') if card else None)
    S = card.get('건물구조') or {}
    return dict(양식='std', **({'출처': f"스캔본 OCR({(card.get('ocr') or {}).get('pages')}쪽, 확인 필요 {len((card.get('ocr') or {}).get('low') or [])}항목)"} if card.get('source') == 'ocr' else {}),
                관리번호=card.get('관리번호'), 소재지=card.get('소재지'), 거리=card.get('거리'), 소방용수=card.get('소방용수') or [],
                건물구조={k: S.get(k) for k in ('양식', '구조', '지붕', '동명', '지하층수', '지상층수', '연면적', '바닥면적')}, 내화구조=card.get('내화구조'),
                관계인=card.get('관계인') or [], 개인정보=card.get('개인정보'), 위험물=card.get('위험물') or [],
                동목록=[dict(동명=(d.get('건물구조') or {}).get('동명'), 상호=d.get('상호'), 주용도=d.get('주용도'),
                          층수=f"지하{(d.get('건물구조') or {}).get('지하층수') or 0}/지상{(d.get('건물구조') or {}).get('지상층수') or 0}",
                          연면적=(d.get('건물구조') or {}).get('연면적'), 위험물=len(d.get('위험물') or []), 선택=(i == card.get('선택동', 0)))
                     for i, d in enumerate(card.get('동목록') or [])])


def ext_issue(f):
    """기능 12(경계구역)·18(건축물대장)·24(부지) 플래그 → 확인사항 문구. 해당 없으면 None"""
    from firebim.daejang import KIND_TXT, issue_text
    k = f['kind']
    if k == 'hazmat':                  # 표준양식 ⑨ 위험물 시설 등 → 진입 전 확인
        tip = {'위험물시설': '위험물 저장·취급 — 진입 전 위치·수량 확인, 위험물안전관리자 연락', '가스취급현황': '가스 취급 — 진입 전 가스 차단(밸브) 확인',
               '특수가연물': '특수가연물 적재 — 급격한 연소 확대·붕괴 위험, 진입 전 확인', '기타': '기타 위험 요인 — 진입 전 확인'}.get(f.get('cls'), '진입 전 확인')
        return dict(lv='high' if f.get('cls') in ('위험물시설', '가스취급현황') else 'mid', t=f"{f.get('cls')} 있음(정보카드 ⑨) → 진입 전 확인",
                    d=' · '.join(x for x in (f.get('name'), f.get('item'), f.get('qty')) if x) + f" — {tip}", cat='위험물')
    if k == 'card_ocr':                # 스캔본 정보카드 OCR — 신뢰도 낮은 항목(firebim/card_ocr.py)
        v = '(개인정보 — 값 표시 안 함)' if f.get('private') else (f"'{f.get('value')}'" if f.get('value') not in (None, '') else '(읽지 못함)')
        return dict(lv='mid', t='정보카드 OCR 확인(스캔본)', d=f"{f.get('field')}: {v} — OCR 신뢰도 {f.get('conf', 0) * 100:.0f}% → 원본 카드와 대조", cat='정보카드')
    if k == 'card_ocr_fail':
        return dict(lv='high', t='정보카드 OCR 실패(스캔본)', d=f"{f.get('why')} — 정보카드 항목(층·면적·소방시설 등)을 원본에서 직접 확인", cat='정보카드')
    if k == 'emev_count':
        return dict(lv='mid', t='비상용승강기 수 불일치(정보카드 ⑦ ↔ 도면)', d=f"정보카드 승강기(비상) {f['card']}대 / 도면에서 찾은 승강기 {f['drawing']}대", cat='카드↔도면')
    if k.startswith('daejang_'):
        return dict(lv='high' if k in ('daejang_floor_missing', 'daejang_floor_count') else 'mid', t=KIND_TXT.get(k, k), d=issue_text(f, ko_name), cat='건축물대장')
    if k == 'zone_count':
        d = (f"{ko_name(f['floor'])}: 정보카드 자동화재탐지기(회) {f['card']}회로 / NFTC 203 기준 최소 {f['required']}구역 추정 "
             f"(경계구역 면적 {f['area']:,.0f}㎡·최장변 {f['side']:.0f}m, 계단·승강로 제외)")
        if f.get('lv') == 'low': d += f" — 내부 전체가 보이는 대공간이면 1,000㎡ 단서로 {f['required_open']}구역 가능 → 현장 확인"
        return dict(lv=f.get('lv', 'mid'), t='경계구역 수 부족 의심(NFTC 203 2.1.1)', d=d, cat='경계구역')
    if k == 'zone_ilramdo':
        return dict(lv='mid', t='경계구역 일람도 ↔ 정보카드 회로 수 불일치', d=f"{ko_name(f['floor'])}: 정보카드 {f['card']}회로 / 일람도 {f['drawing']}구역", cat='경계구역')
    if k == 'water_dist':
        return dict(lv='mid', t='소방용수 거리 불일치(정보카드 ⑥ ↔ 부지)', d=f"소화전 {f['id']}: 정보카드 {f['card']}m / 부지 자료 계산 {f['site']}m → 소방용수 위치·번호 확인", cat='부지')
    if k.startswith('sitemap_'):                  # 진압작전도(정보카드 ⑪) 교차검증
        mk = dict(main='주출입구', exit='비상구')
        if k == 'sitemap_opening': d = f"작전도 {mk.get(f.get('mark'), f.get('mark'))} 표시({f.get('facade')}) 위치에 평면도 1층 외벽 개구부 없음" + (f" (가장 가까운 개구부 {f['d']}m)" if f.get('d') else '')
        elif k == 'sitemap_exit_unmarked': d = f"평면도 1층 외벽 비상구({f['x']:.0f}, {f['y']:.0f})가 진압작전도에 표시 안 됨"
        elif k == 'sitemap_fdc': d = f"작전도 연결송수구 위치가 평면도 송수구 기호와 {f['d']}m 다름"
        elif k == 'sitemap_dim': d = f"작전도 이격거리 글자 {f['text']:g}m / 그림에서 잰 거리 {f['measured']:g}m" + (f" ({f['src']})" if f.get('src') else '')
        elif k == 'sitemap_scale': d = f"작전도 축척 막대 {f['scalebar']} px/m ↔ 평면도 외곽 맞춤 {f['plan']} px/m"
        elif k == 'sitemap_vs_site': d = f"{f['what']}: 작전도 ↔ 부지 자료(site.json) 위치 {f['d']}m 차이"
        elif k == 'sitemap_fail':
            WHY = dict(legend_not_found='범례를 찾지 못함', footprint_not_found='대상 건물 윤곽을 찾지 못함', no_building='대상 건물 윤곽을 찾지 못함', no_image='그림이 없음', register_fail='1층 평면과 맞추지 못함')
            w = WHY.get(str(f.get('why')), '그림 판독 불가')
            d = f"정보카드의 진압작전도를 읽지 못함({w}) → 외부 진입·소화전 위치는 부지 자료 기준으로 표시, 출동 전 진압작전도 원본 확인"
        else: d = k
        return dict(lv='high' if k in ('sitemap_fail', 'sitemap_exit_unmarked') else 'mid', t='진압작전도 교차검증', d=d, cat='부지')
    if k == 'site_reg':
        return dict(lv='high', t='부지 자료 ↔ 도면 외곽 정합 실패', d='GIS 건물 외곽과 도면 외곽 크기가 달라 외부 부서 계산을 도면 외곽으로 대신함 → 확인 필요', cat='부지')
    if k == 'xcheck_wall':             # 선택 경로(모델): 두 독립 추출기(규칙·STR 모델) 벽 불일치 요약
        fl = f.get('floors') or [f['floor']]
        return dict(lv='high' if f.get('p', 0) >= 0.9 else 'mid', t='벽 인식 교차검증 불일치(규칙 ↔ AI 모델)',
                    d=f"{ko_name(fl[0])}" + (f" ~ {ko_name(fl[-1])}" if len(fl) > 1 else '') +
                      f": 규칙만 벽으로 본 곳 {f['rule_only']}곳({f['len_rule_only']:.0f}m, 오검출 의심) · 모델만 벽으로 본 곳 {f['model_only']}곳({f['len_model_only']:.0f}m, 누락 의심)"
                      + (f" · 벽 길이 일치율 {f['agree'] * 100:.0f}%" if f.get('agree') is not None else '') + " → 검수 화면에서 의심도 순으로 확인", cat='교차검증')
    if k == 'aerial_unreachable':
        fl = f['floors']
        return dict(lv='mid', t='고가사다리차 도달 불가 층', d=f"{ko_name(fl[0])}" + (f" ~ {ko_name(fl[-1])}" if len(fl) > 1 else '') + f" ({len(fl)}개 층): 외부 구조 불가 → 내부 진입·피난 유도 우선", cat='부지')
    return None


def export_ifc(D, BIM, path):
    import ifcopenshell, ifcopenshell.api as api
    F = api.run('project.create_file', version='IFC4')
    proj = api.run('root.create_entity', F, ifc_class='IfcProject', name=D['building']['name'] + ' 소방활동 BIM')
    api.run('unit.assign_unit', F)
    ctx = api.run('context.add_context', F, context_type='Model')
    body = api.run('context.add_context', F, context_type='Model', context_identifier='Body', target_view='MODEL_VIEW', parent=ctx)
    site = api.run('root.create_entity', F, ifc_class='IfcSite', name='부지')
    bldg = api.run('root.create_entity', F, ifc_class='IfcBuilding', name=D['building']['name'])
    api.run('aggregate.assign_object', F, products=[site], relating_object=proj)
    api.run('aggregate.assign_object', F, products=[bldg], relating_object=site)

    def pset(el, name, props):
        ps = api.run('pset.add_pset', F, product=el, name=name)
        api.run('pset.edit_pset', F, pset=ps, properties={k: v for k, v in props.items() if v is not None})
    B = D['building']
    pset(bldg, 'Pset_소방대상물', {'주용도': B['use'], '연면적_m2': B['gfa'], '층수': B['floors'], '출처': B['source'],
                               '계단_정보카드': ', '.join(f'{k} {v}' for k, v in B['stairs'].items()),
                               '모델정확도': '도면 자동 벡터화(LOD200 수준) — 검수 전'})
    C = B.get('card') or {}
    if C.get('양식') == 'std':                      # 표준양식 ①~⑨ (관계인 성명·연락처는 opt-in 때만)
        S = C.get('건물구조') or {}; L = C.get('소재지') or {}; G = C.get('거리') or {}
        pset(bldg, 'Pset_소방활동정보카드', {'관리번호': C.get('관리번호'), '소재지_지번': L.get('지번'), '소재지_도로명': L.get('도로명'),
                                          '구조': S.get('구조'), '지붕': S.get('지붕'), '동명': S.get('동명'), '주요구조부_내화_추정': C.get('내화구조'),
                                          '거리_소방서_km': G.get('소방서'), '거리_관할센터_km': G.get('관할센터'), '거리_최단센터_km': G.get('최단센터'),
                                          '소방용수': ' / '.join(f"{w['종류']}-{w['관서']}-{w['동']}-{w['번호']} {w['거리km']}km" for w in C.get('소방용수') or []) or None,
                                          '위험물_시설_등': ' / '.join(f"{h['구분']}: {h['시설명']} {h['품명']} {h['수량']}" for h in C.get('위험물') or []) or '없음',
                                          '관계인_역할': ', '.join(r['구분'] for r in C.get('관계인') or [] if r.get('기재') or r.get('성명')) or None,
                                          '개인정보': C.get('개인정보')})
        if any(r.get('성명') or r.get('연락처') for r in C.get('관계인') or []):      # FIREBIM_INCLUDE_CONTACTS=1 일 때만 값이 있음
            pset(bldg, 'Pset_관계인_내부용', {f"{r['구분']}": f"{r.get('성명') or ''} {r.get('연락처') or ''}".strip() for r in C['관계인']})
    styles = {}
    COL = dict(wall=(0.78, 0.78, 0.76), extwall=(0.55, 0.57, 0.62), slab=(0.62, 0.64, 0.68), door=(0.55, 0.38, 0.22), exit=(0.1, 0.75, 0.3),
               entrance=(0.1, 0.8, 0.9), stair=(0.2, 0.8, 0.35), ev=(0.25, 0.55, 0.95), hose=(0.9, 0.12, 0.12), valve=(1.0, 0.65, 0.0),
               panel=(0.6, 0.45, 0.95), space=(0.6, 0.8, 1.0))

    def style(key, alpha=0.0):
        k = (key, alpha)
        if k in styles: return styles[k]
        stl = api.run('style.add_style', F, name=key)
        r, g, b = COL[key]
        api.run('style.add_surface_style', F, style=stl, ifc_class='IfcSurfaceStyleShading',
                attributes={'SurfaceColour': {'Name': None, 'Red': r, 'Green': g, 'Blue': b}, 'Transparency': alpha})
        styles[k] = stl; return stl

    def place(el, x, y, z, rot=0.0):
        c, s = math.cos(rot), math.sin(rot)
        api.run('geometry.edit_object_placement', F, product=el, matrix=np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], float))

    def product(cls, name, storey, rep, key, x, y, z, rot=0.0, ptype=None, alpha=0.0):
        el = api.run('root.create_entity', F, ifc_class=cls, name=name, predefined_type=ptype) if ptype else api.run('root.create_entity', F, ifc_class=cls, name=name)
        if cls == 'IfcSpace': api.run('aggregate.assign_object', F, products=[el], relating_object=storey)
        else: api.run('spatial.assign_container', F, products=[el], relating_structure=storey)
        place(el, float(x), float(y), float(z), rot)
        api.run('geometry.assign_representation', F, product=el, representation=rep)
        api.run('style.assign_representation_styles', F, shape_representation=rep, styles=[style(key, alpha)])
        return el

    def boxrep(L, T, Hh, off=None):
        return api.run('geometry.add_wall_representation', F, context=body, length=float(L), height=float(Hh), thickness=float(T), offset=float(-T / 2 if off is None else off))
    storeys = {}
    for Fd in D['floors']:
        fid = Fd['id']; z = Fd['level'] * H_FLOOR
        st = api.run('root.create_entity', F, ifc_class='IfcBuildingStorey', name=fid)
        st.LongName = Fd['name']; st.Elevation = z
        api.run('aggregate.assign_object', F, products=[st], relating_object=bldg); place(st, 0, 0, z)
        pset(st, 'Pset_소방정보카드', dict({'용도': Fd['use'], '바닥면적_m2': Fd['area']}, **Fd['fac']))
        storeys[fid] = st
        fb = BIM['floors'][fid]
        for ring in Fd['poly']:
            rep = api.run('geometry.add_slab_representation', F, context=body, depth=SLAB, polyline=[(float(a), float(b)) for a, b in ring])
            product('IfcSlab', f'{fid} 바닥', st, rep, 'slab', 0, 0, z, ptype='FLOOR')
        wh = H_FLOOR - SLAB
        fp = fb['fp']
        for i, w in enumerate(fb['walls']):
            L = w['b'] - w['a']
            if L < 0.3: continue
            ext = fp is not None and ((LineStr(w)).distance(fp.boundary) < 0.6)
            rep = boxrep(L, w['t'], wh)
            if w['o'] == 'h': el = product('IfcWall', f'{fid} 벽 {i + 1}', st, rep, 'extwall' if ext else 'wall', w['a'], w['c'], z + SLAB)
            else: el = product('IfcWall', f'{fid} 벽 {i + 1}', st, rep, 'extwall' if ext else 'wall', w['c'], w['a'], z + SLAB, math.pi / 2)
            el.PredefinedType = 'STANDARD'
            pset(el, 'Pset_WallCommon', {'IsExternal': bool(ext)})
            w['gid'] = el.GlobalId; w['ext'] = bool(ext)
        for d in fb['doors']:
            dh = 2.1 if d['kind'] == 'door' else 2.6
            rep = boxrep(d['w'], 0.12, dh)
            key = {'door': 'door', 'exit': 'exit', 'entrance': 'entrance'}[d['kind']]
            if d['o'] == 'h': el = product('IfcDoor', f"{fid} {d['name']}", st, rep, key, d['x'] - d['w'] / 2, d['y'], z + SLAB)
            else: el = product('IfcDoor', f"{fid} {d['name']}", st, rep, key, d['x'], d['y'] - d['w'] / 2, z + SLAB, math.pi / 2)
            el.OverallWidth = d['w']; el.OverallHeight = dh; el.PredefinedType = 'DOOR'
            pset(el, 'Pset_DoorCommon', {'IsExternal': d['ext'], 'FireExit': d['kind'] != 'door'})
            d['gid'] = el.GlobalId
        for j, sp in enumerate(fb['spaces']):
            rep = api.run('geometry.add_slab_representation', F, context=body, depth=wh - 0.05, polyline=[(float(a), float(b)) for a, b in sp['poly']])
            el = product('IfcSpace', f'실 {j + 1}', st, rep, 'space', 0, 0, z + SLAB, alpha=0.85)
            el.LongName = sp.get('name') or f'실 {j + 1}'; el.PredefinedType = 'INTERNAL'
            pset(el, 'Pset_소방활동', {'면적_m2': sp['area'], '실명출처': '벽망 자동 분할(실명 미판독)'})
            sp['gid'] = el.GlobalId
    lvl = {Fd['id']: Fd['level'] for Fd in D['floors']}
    for S in BIM['stairs']:
        x0, y0, x1, y1 = S['box']; S['gids'] = []
        for fid in S['floors']:
            rep = boxrep(x1 - x0, y1 - y0, H_FLOOR - SLAB, off=0)
            el = product('IfcStair', f"{S['name']} {fid}", storeys[fid], rep, 'stair', x0, y0, lvl[fid] * H_FLOOR + SLAB, ptype='HALF_TURN_STAIR', alpha=0.35)
            pset(el, 'Pset_소방활동', {'계단ID': S['id'], '연결층': f"{S['floors'][0]} ~ {S['floors'][-1]}", '근거': S['note']})
            S['gids'].append(el.GlobalId)
    for E in BIM['elevs']:
        x0, y0, x1, y1 = E['box']; E['gids'] = []
        for fid in E['floors']:
            rep = boxrep(max(0.8, x1 - x0), max(0.8, y1 - y0), H_FLOOR - SLAB, off=0)
            el = product('IfcTransportElement', f"{E['name']} {fid}", storeys[fid], rep, 'ev', x0, y0, lvl[fid] * H_FLOOR + SLAB, ptype='ELEVATOR', alpha=0.35)
            pset(el, 'Pset_소방활동', {'승강기ID': E['id'], '운행층': f"{E['floors'][0]} ~ {E['floors'][-1]}", '출처': E.get('src') or '승강기 표기 층간 반복(자동)'})
            E['gids'].append(el.GlobalId)
    EQ = {'hose': ('IfcFireSuppressionTerminal', 'HOSEREEL', (0.9, 0.3, 1.6), 'hose'), 'panel': ('IfcUnitaryControlElement', 'ALARMPANEL', (1.0, 0.5, 1.8), 'panel'),
          'gas': ('IfcValve', 'ISOLATING', (0.4, 0.25, 0.4), 'valve')}
    for q in BIM['equip']:
        cls, pt, (L, T, Hh), key = EQ[q['type']]
        el = product(cls, f"{q['floor']} {q['name']}", storeys[q['floor']], boxrep(L, T, Hh), key, q['x'] - L / 2, q['y'], lvl[q['floor']] * H_FLOOR + SLAB + (1.0 if q['type'] == 'gas' else 0), ptype=pt)
        pset(el, 'Pset_소방활동', {'구분': q['name'], '출처': '평면도 범례기호 자동검출'})
        q['gid'] = el.GlobalId
    export_ifc_ext(F, api, D, BIM, storeys, bldg, site, product, boxrep, pset, lvl, body)
    export_ifc_fac(F, api, D, BIM, storeys, product, boxrep, pset, lvl)
    if D.get('evac'):
        from firebim.evac import export_ifc_evac
        export_ifc_evac(F, api, D, storeys, lvl, pset, H_FLOOR, SLAB)
    F.write(path)
    return path


def export_ifc_ext(F, api, D, BIM, storeys, bldg, site, product, boxrep, pset, lvl, body):
    """기능 12·18·24 → IFC: 경계구역 = IfcSpace(ObjectType 경계구역)+IfcZone(구역에 속한 실 묶음),
    송수구·소화전 = IfcFireSuppressionTerminal(BREECHINGINLET·FIREHYDRANT), 사다리차 부서 위치 = IfcAnnotation, 대장 = Pset"""
    Z = D.get('zones')
    if Z:
        for fid, rec in Z['floors'].items():
            if fid not in storeys: continue
            rooms = BIM['floors'][fid]['spaces']
            for z in rec['zones']:
                z['gids'] = []
                zone = api.run('root.create_entity', F, ifc_class='IfcZone', name=f"{fid} 경계구역 {z['no']}")
                zone.LongName = z['name']; zone.ObjectType = '자동화재탐지설비 경계구역'
                members = []
                for ring in z['poly']:
                    rep = api.run('geometry.add_slab_representation', F, context=body, depth=0.05,
                                  polyline=[(float(a), float(b)) for a, b in ring])
                    el = product('IfcSpace', f"{fid} 경계구역 {z['no']}", storeys[fid], rep, 'panel', 0, 0, lvl[fid] * H_FLOOR + SLAB + 0.02, alpha=0.8)
                    el.ObjectType = '경계구역'; el.PredefinedType = 'USERDEFINED'; el.LongName = z['name']
                    pset(el, 'Pset_경계구역', {'회로번호_층별': z['no'], '회로번호_일련': z['gno'], '면적_m2': z['area'], '최장변_m': z['side'],
                                            'NFTC203_적합': bool(z['ok']), '출처': '추정(정보카드 회로 수로 분할)' if z['est'] else rec['src'],
                                            '대표점': f"{z['rep'][0]},{z['rep'][1]}"})
                    members.append(el); z['gids'].append(el.GlobalId)
                zp = Polygon(z['poly'][0]) if z['poly'] else None
                for sp in rooms:                 # 대표점이 구역 안에 드는 실(IfcSpace)도 같은 IfcZone 으로 묶음
                    if zp is None or not sp.get('gid'): continue
                    c = Polygon(sp['poly']).representative_point()
                    if zp.contains(c): members.append(F.by_guid(sp['gid']))
                api.run('group.assign_group', F, products=members, group=zone)
    S = D.get('site')
    if S:
        for h in S.get('hydrants', []):
            rep = boxrep(0.5, 0.5, 1.0)
            el = api.run('root.create_entity', F, ifc_class='IfcFireSuppressionTerminal', name=f"소화전 {h['id']}", predefined_type='FIREHYDRANT')
            api.run('spatial.assign_container', F, products=[el], relating_structure=site)
            api.run('geometry.edit_object_placement', F, product=el, matrix=np.array([[1, 0, 0, h['x']], [0, 1, 0, h['y']], [0, 0, 1, 0], [0, 0, 0, 1]], float))
            api.run('geometry.assign_representation', F, product=el, representation=rep)
            pset(el, 'Pset_소방용수', {'종류': h['type'], '번호': h['id'], '출처': '진압작전도' if h.get('src') == '진압작전도' else '부지 자료(site.json)'})
        for q in S.get('fdc', []):
            g = '1F' if '1F' in storeys else list(storeys)[0]
            el = product('IfcFireSuppressionTerminal', q['name'], storeys[g], boxrep(0.6, 0.3, 0.8), 'hose', q['x'] - 0.3, q['y'], lvl[g] * H_FLOOR + SLAB + 0.5, ptype='BREECHINGINLET')
            pset(el, 'Pset_소방활동', {'구분': '연결송수관 송수구', '펌프차_거리_m': (S.get('pumper') or {}).get('fdc_d'), '출처': '진압작전도' if S.get('src') == '진압작전도' else '부지 자료'})
        for a in S.get('aerial', []):
            el = api.run('root.create_entity', F, ifc_class='IfcAnnotation', name=f"고가사다리차 부서 후보 {a['id']}")
            api.run('spatial.assign_container', F, products=[el], relating_structure=site)
            api.run('geometry.edit_object_placement', F, product=el, matrix=np.array([[1, 0, 0, a['x']], [0, 1, 0, a['y']], [0, 0, 1, 0], [0, 0, 0, 1]], float))
            pset(el, 'Pset_외부부서', {'도로': a['road'], '전개반경_m': a['radius'], '도달층': ', '.join(a['floors']), '도달면': ', '.join(a['facades']),
                                    '기준': '사다리 길이·전개 반경은 기관 SOP 값으로 설정 필요'})
    DJ = D.get('daejang')
    if DJ:
        t = DJ.get('title') or {}
        pset(bldg, 'Pset_건축물대장', {'연면적_m2': t.get('totArea'), '지상층수': t.get('grndFlrCnt'), '지하층수': t.get('ugrndFlrCnt'), '높이_m': t.get('heit'),
                                    '주용도': t.get('mainPurpsCdNm'), '사용승인일': t.get('useAprDay'), '출처': DJ.get('source'),
                                    '교차검증_불일치': sum(1 for r in DJ.get('table', []) if r['verdict'] != '일치')})


CERT_KO = {'located': '위치확인(도면 기호)', 'estimated': '위치추정(규칙)', 'present': '수량만(위치 없음)', 'mixed': '혼합'}


def export_ifc_fac(F, api, D, BIM, storeys, product, boxrep, pset, lvl):
    """소방시설 → IFC: 위치 있는 것(located·estimated)은 분류표 IFC 형식 요소 + 작은 상자, 수량만(present)은 형상 없는 요소(층에 소속)
    모든 요소에 Pset_FireBIM_소방시설(분류코드·법정명칭·대분류·중분류·위치확실성·출처·정보카드 원문/수량·추정반경·근거).
    옥내소화전·수신기 위치(located)는 기존 설비 요소로 이미 나가므로 중복 생성하지 않음. 비상용승강기 추정은 기존 승강기 요소에 Pset."""
    from firebim.fac_catalog import CATALOG
    FC = D.get('fac')
    if not FC: return
    COLK = {'소화설비': 'hose', '경보설비': 'panel', '피난구조설비': 'exit', '소화용수설비': 'ev', '소화활동설비': 'valve'}

    def mk(C, it, fid, name, x=None, y=None, cert='present', extra=None):
        cls, pt, ot = C['ifc'] if C else ('IfcDistributionElement', None, None)
        if cls is None: return None
        z = lvl[fid] * H_FLOOR + SLAB
        try:
            if x is None:
                el = api.run('root.create_entity', F, ifc_class=cls, name=name, predefined_type=pt) if pt else api.run('root.create_entity', F, ifc_class=cls, name=name)
                api.run('spatial.assign_container', F, products=[el], relating_structure=storeys[fid])
            else:
                zz = z + (0.5 if it['code'] in ('STANDPIPE', 'SIAMESE') else 0.8 if it['code'] == 'OUTLET' else 0)
                el = product(cls, name, storeys[fid], boxrep(0.5, 0.35, 0.6), COLK.get(it['cat'], 'panel'), x - 0.25, y, zz, ptype=pt, alpha=0.0 if cert == 'located' else 0.6)
        except Exception:
            el = api.run('root.create_entity', F, ifc_class='IfcDistributionElement', name=name)
            api.run('spatial.assign_container', F, products=[el], relating_structure=storeys[fid])
        if ot: el.ObjectType = ot
        props = {'분류코드': it['code'], '법정명칭': it['name'], '대분류': it['cat'], '중분류': it['group'], '위치확실성': cert, '위치확실성_설명': CERT_KO.get(cert),
                 '출처': ', '.join(it['src']), '정보카드원문': it['raw'], '정보카드수량': it['card'], '도면기호수': it['drawing'], '근거': it['note'] or None}
        props.update(extra or {})
        pset(el, 'Pset_FireBIM_소방시설', props)
        return el
    for fid, items in FC['floors'].items():
        if fid not in storeys: continue
        summ = {}
        for it in items:
            C = CATALOG.get(it['code'])
            key = f"{it['cat']}·{it['name']}" + (f"({it['raw']})" if it['code'] == '기타' else '')
            summ[key] = f"정보카드 {it['card'] if it['card'] is not None else '-'} · 도면 {it['drawing'] if it['drawing'] is not None else '-'} · {CERT_KO[it['cert']]}"
            if it['code'] not in ('HOSE', 'PANEL'):
                for k, q in enumerate(it['loc']):
                    el = mk(C, it, fid, f"{fid} {it['name']} {k + 1}", q['x'], q['y'], 'located', {'검출신뢰도': q.get('conf')})
                    if el is not None: q['gid'] = el.GlobalId
            if it['code'] == 'EMEV':
                for q in it['est']:
                    for E in BIM['elevs']:
                        b = E['box']
                        if abs((b[0] + b[2]) / 2 - q['x']) < 0.6 and abs((b[1] + b[3]) / 2 - q['y']) < 0.6 and fid in E['floors']:
                            g = E['gids'][E['floors'].index(fid)]
                            pset(F.by_guid(g), 'Pset_FireBIM_소방시설', {'분류코드': 'EMEV', '법정명칭': '비상용승강기', '위치확실성': 'estimated',
                                                                    '근거': it['note'], '추정근거_승강기': q.get('why') or None, '후보': bool(q.get('alt'))}); q['gid'] = g
            else:
                for k, q in enumerate(it['est']):
                    el = mk(C, it, fid, f"{fid} {it['name']} 추정 {k + 1}", q['x'], q['y'], 'estimated',
                            {'추정반경_m': q.get('r'), '기준점': f"{q.get('ax')},{q.get('ay')}", '후보중하나': bool(q.get('alt'))})
                    if el is not None: q['gid'] = el.GlobalId
            if it['present']:
                el = mk(C, it, fid, f"{fid} {it['name']} (수량 {it['present']})", cert='present', extra={'수량': it['present'], '비고': '위치 정보 없음 — 정보카드 수량만'})
                if el is not None: it['gid_present'] = el.GlobalId
        if summ: pset(storeys[fid], 'Pset_FireBIM_층별소방시설', summ)


def LineStr(w):
    from shapely.geometry import LineString
    return LineString([(w['a'], w['c']), (w['b'], w['c'])]) if w['o'] == 'h' else LineString([(w['c'], w['a']), (w['c'], w['b'])])


def to_viewer_bim(BIM):
    """뷰어용 압축 형식 (예전 지휘화면과 같은 구조)"""
    out = dict(floors={}, stairs=[], elevs=[], equip=[], H=BIM['H'])
    for fid, fb in BIM['floors'].items():
        out['floors'][fid] = dict(
            walls=[[w['o'], round(w['c'], 2), round(w['a'], 2), round(w['b'], 2), round(w['t'], 2), 1 if w.get('ext') else 0, w.get('gid', '')] for w in fb['walls']],
            doors=[[d['kind'], d['o'], d['x'], d['y'], d['w'], d['name'], 1 if d['ext'] else 0, d.get('gid', '')] for d in fb['doors']],
            spaces=[dict(poly=sp['poly'], area=sp['area'], name=sp.get('name', ''), crit=False, gid=sp.get('gid', '')) for sp in fb['spaces']])
    for S in BIM['stairs']: out['stairs'].append({k: S[k] for k in ('id', 'name', 'box', 'floors', 'kind', 'note', 'gids', 'cert') if k in S})
    for E in BIM['elevs']: out['elevs'].append(dict({k: E[k] for k in ('id', 'name', 'box', 'floors', 'cars', 'gids', 'cert') if k in E}, **({'note': E['src']} if E.get('cert') else {})))
    for q in BIM['equip']: out['equip'].append([q['type'], q['floor'], q['x'], q['y'], q['name'], q.get('gid', ''), 'auto'])
    return out
