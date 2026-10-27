# -*- coding: utf-8 -*-
"""진압작전도 인식 채점 — 가상 작전도(synth/sitemap.py) 정답 대비.
  python -m evalr.sitemap_eval dev  [--overlay DIR] [--only NAME]     개발 세트: data/* 건물 × 그림 난수 2개(r=0,1)
  python -m evalr.sitemap_eval held                                   보류 세트: data_hold/* 건물 × 다른 그림 난수(r=500,501) — 한 번만 보고
  python -m evalr.sitemap_eval dev_var | held_var                     치수 글자 변형 세트(dev 와 같은 그림·치수 글자만 변형 / 보류는 새 난수 502,503)
  python -m evalr.sitemap_eval dev_var2                               개발 변형 2(새 난수 40, 18장): '약'·'~'·소수 쉼표/점·미터·km·천 단위 쉼표(mm) 위주(dimvar='v2')
  python -m evalr.sitemap_eval dev_x                                  개발 추가(기본 형식, data/ × 난수 2~4, 54장) — 기본 그림 회귀 확인
  python -m evalr.sitemap_eval dev_y                                  개발 추가 2(기본 형식, data/ × 난수 5~9, 90장)
  python -m evalr.sitemap_eval dev_var3                               개발 변형 3(새 난수 41·42, 36장, dimvar='v2') — 치수선 찾기(끊긴 점선·기울어진 긴 글자·이어진 짧은 치수) 보강용
  python -m evalr.sitemap_eval pdf dev 3                              통합: 작전도를 붙인 시험용 카드 PDF → pipeline.run → 기능 24(작전도 경로)
지표: 범례 5종 찾음, 기호 종류별 P/R(주출입구·비상구·소화전·연결송수구·이격거리), 위치 오차(m, 정합 후), 거리 글자 OCR 정확도,
      정합 오차(방위°, 축척 %, 건물 모서리 m), 도로 폭, 기능24 결과(면별 펌프차 접근·펌프차→송수구 거리·사다리차 수·소방용수 플래그) vs 정답,
      심은 불일치 플래그(작전도↔평면도·치수) 찾음/오탐."""
import json, math, os, sys, glob, random, time
import numpy as np, cv2
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from synth.sitemap import sitemap_image, seed_of
from fire3dbeam import sitemap as SM, site as ST
from fire3dbeam.card import level_of

SETS = dict(dev=('data', (0, 1)), held=('data_hold', (500, 501)),
            dev_var=('data', (0, 1), dict(dimvar=True)),              # 치수 글자 변형(기울어진·세로·선 위 글자, 소수 쉼표, '약', '미터') — 개발 그림과 치수 글자만 다름
            held_var=('data_hold', (502, 503), dict(dimvar=True)),    # 변형 보류 세트(새 난수)
            dev_var2=('data', (40,), dict(dimvar='v2')),              # 개발 변형 2(새 난수 40): '약'·'~'·소수 쉼표/점·미터·km·천 단위 쉼표(mm) 위주
            dev_var3=('data', (41, 42), dict(dimvar='v2')),           # 개발 변형 3(새 난수 41·42, 36장, 2026-10-02 치수선 찾기 보강용): dev_var2 와 같은 형식, 다른 그림
            dev_x=('data', (2, 3, 4)),                                # 개발 추가(기본 형식, 난수 2~4, 54장 = 예전 devx_v6) — held 와 같은 기본 그림에서 회귀 확인용
            dev_y=('data', (5, 6, 7, 8, 9)))                          # 개발 추가 2(기본 형식, 난수 5~9, 90장) — 기본 형식 회귀 확인(스위치별 영향)
CARD_WATER = [dict(kind='소화전(지상식)', id='가상-1', km=0.15), dict(kind='소화전(지하식)', id='가상-2', km=0.32)]
CACHE_TAG = os.environ.get('SITEMAP_CACHE', '_v6')           # 생성 그림 캐시 폴더 꼬리표(생성기 바꾸면 새 꼬리표)
PLANT2FLAG = dict(sitemap_exit_extra='sitemap_opening', sitemap_exit_missing='sitemap_exit_unmarked', sitemap_dim='sitemap_dim')


def case(d, r, cache, opt=None):
    G = json.load(open(os.path.join(d, 'gt.json'), encoding='utf-8'))
    name = os.path.basename(d.rstrip('/'))
    ip, gp = os.path.join(cache, f'{name}_r{r}.png'), os.path.join(cache, f'{name}_r{r}_gt.json')
    if not os.path.exists(gp):
        img, gt = sitemap_image(G, random.Random(seed_of(G) * 31 + 7 + 1000 * r), dimvar=(opt or {}).get('dimvar') or False)
        img.save(ip); json.dump(gt, open(gp, 'w', encoding='utf-8'), ensure_ascii=False)
    return G, cv2.imread(ip), json.load(open(gp, encoding='utf-8')), f'{name}_r{r}'


def score_one(G, img, gt, overlay=None, tag=''):
    t0 = time.time()
    rec = SM.recognize(img)
    out = dict(name=tag, sec=round(time.time() - t0, 1), legend_words=len((rec.get('legend') or {}).get('words', {})),
               legend_icons=len((rec.get('legend') or {}).get('icons', {})), error=rec.get('error'))
    if overlay: SM.overlay(img, rec, os.path.join(overlay, tag + '.png'))
    Z = rec.get('Z', 24)
    # 1) 점 기호 P/R (화소, 허용 0.8·아이콘)
    det = {}
    for t in SM.PT_TYPES:
        g = [it['px'] for it in gt['items'] if it['type'] == t]
        e = [(q['px'], q['py']) for q in rec.get('points', []) if q['type'] == t]
        used, tp = set(), 0
        for a in g:
            j = min((j for j in range(len(e)) if j not in used), key=lambda j: math.dist(a, e[j]), default=None)
            if j is not None and math.dist(a, e[j]) < 0.8 * Z: used.add(j); tp += 1
        det[t] = [tp, len(g) - tp, len(e) - len(used)]
    # 이격거리 치수선: 가운데·방향·길이
    g = gt['dims']; e = rec.get('dims', [])
    used, tp, ocr_ok, ocr_n = set(), 0, 0, 0
    pairs = []
    for dm in g:
        pa, pb = dm['pa'], dm['pb']; mg = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2); Lg = math.dist(pa, pb); ag = math.atan2(pb[1] - pa[1], pb[0] - pa[0]) % math.pi
        best = None
        for j, q in enumerate(e):
            if j in used: continue
            me = ((q['p0'][0] + q['p1'][0]) / 2, (q['p0'][1] + q['p1'][1]) / 2)
            ae = math.atan2(q['p1'][1] - q['p0'][1], q['p1'][0] - q['p0'][0]) % math.pi
            da = abs(((ag - ae) + math.pi / 2) % math.pi - math.pi / 2)
            if da < math.radians(8) and math.dist(mg, me) < max(Z, 0.25 * Lg) and 0.6 < q['len_px'] / Lg < 1.4:
                sc = math.dist(mg, me)
                if best is None or sc < best[0]: best = (sc, j)
        if best:
            used.add(best[1]); tp += 1; ocr_n += 1; q = e[best[1]]; v = q.get('val')
            ocr_ok += v is not None and abs(v - dm['text_val']) < 0.3 and q.get('val_src') != 'geom'      # 글자 판독 값만(기하 추정값 제외)
            pairs.append((dm, best[1]))
    det['dim'] = [tp, len(g) - tp, len(e) - len(used)]
    out['dim_fp'] = [dict(p0=[round(v) for v in q['p0']], p1=[round(v) for v in q['p1']], val=q.get('val'), text=q.get('text')) for j, q in enumerate(e) if j not in used]   # 오검출 치수선(진단용)
    out['det'] = det; out['dim_ocr'] = [ocr_ok, ocr_n]
    # 2) 정합·위치 (평면도 외곽 = 정답 1층 외곽, 정보카드 1층 면적)
    fp = unary_union([box(*r) for r in G['floors']['1F']['rects']])
    fr = SM.to_frame(rec, ref_outline=fp, card_area=G['card']['areas']['1F']) if 'footprint_px' in rec else None
    out['reg_ok'] = fr is not None
    ne = abs(((rec.get('north_deg', 0) - gt['north_deg']) + 180) % 360 - 180) if 'north_deg' in rec else None
    out['north_err'] = round(ne, 2) if ne is not None else None
    out['north_src'] = rec.get('north_src')
    if rec.get('scalebar'): out['scalebar_err'] = round(rec['scalebar']['px_per_m'] / gt['px_per_m'] - 1, 4)
    if fr is None: return out, rec, None
    out['scale_err'] = round(fr['reg']['scale_px_per_m'] / gt['px_per_m'] - 1, 4); out['scale_src'] = fr['reg']['scale_src']
    out['north_err'] = round(abs(((fr['reg']['north_deg'] - gt['north_deg']) + 180) % 360 - 180), 2); out['north_src'] = fr['reg']['north_src']
    # 치수 최종 값(to_frame: 최종 축척으로 글자↔기하 길이 맞춘 뒤) — 출처별·확실도별
    fin = dict(n_gt=len(g), ocr_ok=0, ocr_bad=0, est_ok=0, est_bad=0, none=0, high_ok=0, high_bad=0, mism_flag=0)
    for dm, j in pairs:
        q = fr['dims'][j] if j < len(fr['dims']) else {}
        v, src = q.get('val'), q.get('val_src', 'ocr')
        if v is None: fin['none'] += 1
        elif src == 'geom': fin['est_ok' if abs(v - dm['text_val']) <= max(1.0, 0.1 * dm['text_val']) else 'est_bad'] += 1
        else:
            good = abs(v - dm['text_val']) < 0.3; fin['ocr_ok' if good else 'ocr_bad'] += 1
            if q.get('certainty') == 'high': fin['high_ok' if good else 'high_bad'] += 1
        fin['mism_flag'] += bool(q.get('geom_mismatch'))
        out.setdefault('dim_rows', []).append(dict(gt=dm['text'], tv=dm['text_val'], true=dm['val'], val=v, src=src, cert=q.get('certainty'), geom=q.get('geom_m'), text=q.get('text'), cands=q.get('cands'), tconf=q.get('tconf'), n_read=q.get('n_read')))
    out.setdefault('dim_rows', []).extend(dict(gt=dm['text'], tv=dm['text_val'], true=dm['val'], val='MISS') for dm in g if not any(dm is p_[0] for p_ in pairs))
    out['dim_final'] = fin
    M = fr['M']
    ce = [math.dist(M(*p), q) for p, q in zip(gt['footprint_px'], gt['footprint_m'])]
    out['reg_corner_m'] = round(float(np.sqrt(np.mean(np.square(ce)))), 2)
    pe = dict(main=[], exit=[], fdc=[], hyd=[], hyd_off=[])
    for it in gt['items']:
        t = it['type']
        if t in ('main', 'exit', 'fdc'):
            cand = fr['fdc'] if t == 'fdc' else [x for x in fr['entrances'] if x['kind'] == t]
            q = min(cand, key=lambda q: math.dist(q['px'], it['px']), default=None)
            if q and math.dist(q['px'], it['px']) < 0.8 * Z: pe[t].append(math.dist((q['x'], q['y']), (it['x'], it['y'])))
        else:
            q = min(fr['hydrants'], key=lambda q: math.dist(q['px'], it['px']), default=None)
            if not q or math.dist(q['px'], it['px']) >= 0.8 * Z: continue
            if it.get('offmap'):
                if q.get('dist_label') and q.get('dist_conf') is False: pe.setdefault('hyd_off_unread', []).append(1); continue
                h = next(h for h in gt['hydrants'] if h['id'] == it['id'])
                pe['hyd_off'].append(math.dist((q['x'], q['y']), (h['x'], h['y'])))
            else: pe['hyd'].append(math.dist((q['x'], q['y']), (it['x'], it['y'])))
    out['pos_err'] = {k: [round(v, 2) for v in vs] for k, vs in pe.items()}
    # 소화전 글자: 번호·거리
    hid, hdist = [0, 0], [0, 0]
    for it in gt['items']:
        if it['type'] != 'hyd': continue
        q = min(fr['hydrants'], key=lambda q: math.dist(q['px'], it['px']), default=None)
        if not q or math.dist(q['px'], it['px']) >= 0.8 * Z: continue
        hid[1] += 1; hid[0] += q.get('id') == it['id']
        if it.get('offmap'):
            hdist[1] += 1; hdist[0] += q.get('dist_label') is not None and abs(q['dist_label'] - it['dist']) <= 1.0
    out['hyd_id'] = hid; out['hyd_dist_ocr'] = hdist
    # 도로 폭(정답 도로 중심선에 가장 가까운 인식 도로)
    rw = []
    for r in gt['site_true']['roads']:
        from shapely.geometry import LineString
        (ax, ay), (bx, by) = r['line'][0], r['line'][-1]; Lr = math.hypot(bx - ax, by - ay)
        perp = lambda p: abs(-(p[0] - ax) * (by - ay) / Lr + (p[1] - ay) * (bx - ax) / Lr)          # 정답 도로 중심선(무한 직선)까지 거리
        c = [q for q in fr['roads'] if perp(np.mean(q['line'], axis=0)) < max(3, r['width'])
             and abs(abs(math.atan2(q['line'][1][1] - q['line'][0][1], q['line'][1][0] - q['line'][0][0])) % math.pi - abs(math.atan2(r['line'][1][1] - r['line'][0][1], r['line'][1][0] - r['line'][0][0])) % math.pi) < 0.2]
        rw.append(dict(true=r['width'], got=c[0]['width'] if c else None, src=c[0]['width_src'] if c else None))
    out['roads'] = rw
    # 3) 기능 24: 작전도 기반 vs 정답 부지
    ids = [f for f in G['floors'] if level_of(f) >= 0]
    fps = {f: fp for f in G['floors']}
    f1 = G['floors']['1F']; x0, y0, x1, y1 = fp.bounds
    ext = [q for q in f1['doors'] if min(abs(q['x'] - x0), abs(q['x'] - x1), abs(q['y'] - y0), abs(q['y'] - y1)) < 0.4]
    info = dict(water=CARD_WATER)
    try:
        rep, fl = ST.analyze_drawing(fr, fps, ids, info, openings=[(q['x'], q['y']) for q in ext], plan_exits=[(q['x'], q['y']) for q in ext if q['kind'] == 'exit'])
        tru, tfl = ST.analyze(gt['site_true'], fps, ids, info)
    except Exception as ex:
        import traceback; out['s24_error'] = repr(ex) + traceback.format_exc()[-400:]; return out, rec, fr
    Mf = {'남측면': 'S', '북측면': 'N', '동측면': 'E', '서측면': 'W'}
    fe = {}
    for f in rep['facades']:
        k = Mf.get(f['name'].split('-')[0]); fe[k] = fe.get(k, False) or f['engine']
    truf = gt['site_gt']['facade_engine']
    out['facade_ok'] = [sum(1 for k, v in truf.items() if fe.get(k, False) == v), len(truf)]
    if rep['pumper'] and tru['pumper']:
        out['pumper_fdc_err'] = round(abs(rep['pumper']['fdc_d'] - tru['pumper']['fdc_d']), 1)
        out['pumper_pos_err'] = round(math.dist((rep['pumper']['x'], rep['pumper']['y']), (tru['pumper']['x'], tru['pumper']['y'])), 1)
    out['aerial_n'] = [len(rep['aerial']), len(tru['aerial'])]
    out['aerial_pos_err'] = [round(min(math.dist((a['x'], a['y']), (b['x'], b['y'])) for b in tru['aerial']), 1) for a in rep['aerial']] if tru['aerial'] else []
    pw = [f['id'] for f in gt['site_gt']['flags'] if f['kind'] == 'water_dist']
    gw = [f['id'] for f in fl if f['kind'] == 'water_dist']
    out['water'] = [sum(1 for p in pw if p in gw), len(pw), sum(1 for g_ in gw if g_ not in pw)]
    # 심은 불일치
    exp = {}
    for p in gt['planted']: exp[PLANT2FLAG[p['kind']]] = exp.get(PLANT2FLAG[p['kind']], 0) + 1
    got = {}
    for f in fl:
        if f['kind'] in ('sitemap_opening', 'sitemap_exit_unmarked', 'sitemap_dim'): got[f['kind']] = got.get(f['kind'], 0) + 1
    out['plant'] = {k: [min(exp.get(k, 0), got.get(k, 0)), exp.get(k, 0), max(0, got.get(k, 0) - exp.get(k, 0))] for k in set(exp) | set(got)}
    out['flags'] = [f['kind'] for f in fl]
    return out, rec, fr


def summarize(rows):
    S = {}
    ok = [r for r in rows if 'det' in r]
    S['n'] = len(rows); S['legend_5of5'] = f"{sum(1 for r in ok if r['legend_icons'] == 5)}/{len(rows)}"
    for t in list(SM.PT_TYPES) + ['dim']:
        tp, fn, fp_ = [sum(r['det'][t][k] for r in ok) for k in range(3)]
        S[f'det_{t}'] = f"R{tp / max(1, tp + fn):.3f} P{tp / max(1, tp + fp_):.3f} (n={tp + fn})"
    a, b = sum(r['dim_ocr'][0] for r in ok), sum(r['dim_ocr'][1] for r in ok); S['dim_ocr'] = f"{a}/{b}"
    F = {}
    for r in ok:
        for k, v in (r.get('dim_final') or {}).items(): F[k] = F.get(k, 0) + v
    if F:
        ngt = sum(len(r['det']['dim']) and r['det']['dim'][0] + r['det']['dim'][1] for r in ok)
        S['dim_final'] = dict(read_ok=f"{F['ocr_ok']}/{ngt}", read_wrong=F['ocr_bad'], estimated=f"ok {F['est_ok']} / off {F['est_bad']}", none=F['none'],
                              high_conf=f"{F['high_ok']}/{F['high_ok'] + F['high_bad']}", geom_mismatch=F['mism_flag'])
    rg = [r for r in ok if r.get('reg_ok')]
    S['reg_ok'] = f"{len(rg)}/{len(rows)}"
    ne = [r['north_err'] for r in ok if r.get('north_err') is not None]
    S['north_err_deg'] = dict(median=round(float(np.median(ne)), 2), max=round(float(max(ne)), 2), within2=f"{sum(1 for v in ne if v <= 2)}/{len(rows)}") if ne else None
    se = [abs(r['scale_err']) * 100 for r in rg if 'scale_err' in r]
    if se: S['scale_err_pct'] = dict(median=round(float(np.median(se)), 2), max=round(float(max(se)), 2))
    sb = [abs(r['scalebar_err']) * 100 for r in ok if 'scalebar_err' in r]
    if sb: S['scalebar_err_pct'] = dict(n=len(sb), median=round(float(np.median(sb)), 2), max=round(float(max(sb)), 2))
    ce = [r['reg_corner_m'] for r in rg if 'reg_corner_m' in r]
    if ce: S['reg_corner_rms_m'] = dict(median=round(float(np.median(ce)), 2), max=round(float(max(ce)), 2))
    S['hyd_off_unread'] = sum(len(r.get('pos_err', {}).get('hyd_off_unread', [])) for r in rg)
    for k in ('main', 'exit', 'fdc', 'hyd', 'hyd_off'):
        v = [x for r in rg for x in r.get('pos_err', {}).get(k, [])]
        if v: S[f'pos_err_{k}_m'] = dict(n=len(v), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2), max=round(float(max(v)), 2))
    for k in ('hyd_id', 'hyd_dist_ocr'):
        S[k] = f"{sum(r.get(k, [0, 0])[0] for r in rg)}/{sum(r.get(k, [0, 0])[1] for r in rg)}"
    rw = [x for r in rg for x in r.get('roads', [])]
    S['road_width_ok'] = f"{sum(1 for x in rw if x['got'] is not None and abs(x['got'] - x['true']) <= 0.5)}/{len(rw)} (found {sum(1 for x in rw if x['got'] is not None)})"
    S['s24_facade_ok'] = f"{sum(r['facade_ok'][0] for r in rg if 'facade_ok' in r)}/{sum(r['facade_ok'][1] for r in rg if 'facade_ok' in r)}"
    pf = [r['pumper_fdc_err'] for r in rg if 'pumper_fdc_err' in r]
    if pf: S['s24_pumper_fdc_err_m'] = dict(n=len(pf), median=round(float(np.median(pf)), 2), max=round(float(max(pf)), 2))
    S['s24_aerial_n_equal'] = f"{sum(1 for r in rg if 'aerial_n' in r and r['aerial_n'][0] == r['aerial_n'][1])}/{sum(1 for r in rg if 'aerial_n' in r)}"
    ae = [x for r in rg for x in r.get('aerial_pos_err', [])]
    if ae: S['s24_aerial_pos_err_m'] = dict(median=round(float(np.median(ae)), 2), p90=round(float(np.percentile(ae, 90)), 2))
    S['s24_water_flag'] = f"found {sum(r['water'][0] for r in rg if 'water' in r)}/{sum(r['water'][1] for r in rg if 'water' in r)}, false {sum(r['water'][2] for r in rg if 'water' in r)}"
    pl = {}
    for r in rg:
        for k, v in r.get('plant', {}).items(): pl[k] = [a + b for a, b in zip(pl.get(k, [0, 0, 0]), v)]
    S['plant_flags'] = {k: f"found {v[0]}/{v[1]}, false {v[2]}" for k, v in pl.items()}
    S['sec_mean'] = round(float(np.mean([r['sec'] for r in rows])), 1)
    return S


def main(which, overlay=None, only=None):
    data, rs = SETS[which][:2]; opt = SETS[which][2] if len(SETS[which]) > 2 else None
    cache = os.path.join(ROOT, 'out_sitemap', which + CACHE_TAG); os.makedirs(cache, exist_ok=True)
    if overlay: os.makedirs(overlay, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, data, '*/'))):
        for r in rs:
            if only and only not in f"{os.path.basename(d.rstrip('/'))}_r{r}": continue
            G, img, gt, tag = case(d, r, cache, opt)
            try: o, rec, fr = score_one(G, img, gt, overlay, tag)
            except Exception as ex:
                import traceback; o = dict(name=tag, error=repr(ex), tb=traceback.format_exc()[-800:], sec=0); print(o['tb'])
            rows.append(o); print(json.dumps({k: v for k, v in o.items() if k not in ('roads', 'dim_rows')}, ensure_ascii=False), flush=True)
    S = summarize(rows)
    json.dump(dict(summary=S, rows=rows), open(os.path.join(ROOT, 'out_sitemap', f'{which}_eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))
    return S


# ───────────── 통합 시험: 진압작전도를 넣은 시험용 정보카드 PDF → fire3dbeam.pipeline.run → 기능 24(작전도 경로) 채점
def make_test_pdf(d, png, out_dir):
    """기존 card.pdf 뒤에 '0 진압작전도' 캡션 + 그림 + 범례(필수) 표 글자 쪽을 붙인 시험용 PDF(카드 생성기는 건드리지 않음)"""
    import pymupdf, shutil
    os.makedirs(out_dir, exist_ok=True)
    for f in ('gt.json', 'site.json', 'daejang.json'):
        if os.path.exists(os.path.join(d, f)): shutil.copy(os.path.join(d, f), out_dir)
    doc = pymupdf.open(os.path.join(d, 'card.pdf'))
    pg = doc.new_page(width=595, height=842)
    im = cv2.imread(png); H, W = im.shape[:2]
    pg.insert_text((40, 60), '0 진압작전도', fontname='korea', fontsize=11)
    r = pymupdf.Rect(40, 72, 555, 72 + 515 * H / W)
    pg.insert_image(r, filename=png)
    pg.insert_text((40, r.y1 + 22), '범례(필수)   주출입구   비상구   소화전   연결송수구   이격거리', fontname='korea', fontsize=10)
    out = os.path.join(out_dir, 'card.pdf'); doc.save(out); return out


def pdf_test(which='dev', n=3):
    from fire3dbeam.pipeline import run
    from evalr.score import score
    data, rs = SETS[which][:2]
    cache = os.path.join(ROOT, 'out_sitemap', which + CACHE_TAG); os.makedirs(cache, exist_ok=True)
    rows = []
    dirs = sorted(glob.glob(os.path.join(ROOT, data, '*/')))
    pick = [dirs[i] for i in np.linspace(0, len(dirs) - 1, n).astype(int)]
    for d in pick:
        G, img, gt, tag = case(d, rs[0], cache)
        td = os.path.join(ROOT, 'out_sitemap', 'pdf_' + which, tag)
        pdf = make_test_pdf(d, os.path.join(cache, tag + '.png'), td)
        t0 = time.time(); res = run(pdf); sec = time.time() - t0
        json.dump(res, open(os.path.join(td, 'res.json'), 'w'), ensure_ascii=False, default=str)
        sc = score(G, res)
        S = res.get('site') or {}
        # 정답 부지(site_true)로 같은 엔진 좌표에서 계산한 펌프차 → 비교
        T = G['transforms']; E = res['floors']
        from evalr.score import eng_map, gt_map
        fids = [f for f in G['floors'] if f in E and f in T]
        g = np.median([np.subtract(eng_map(E[f], T[f]['w'] / 2, T[f]['h'] / 2), gt_map(T[f], T[f]['w'] / 2, T[f]['h'] / 2)) for f in fids], axis=0)
        row = dict(name=tag, sec=round(sec, 1), src=S.get('src', 'site.json'), site_facade_ok=sc.get('site_facade_ok'), water_found=sc.get('water_found'),
                   water_false=sc.get('water_false'), flags=[f['kind'] for f in res['flags'] if f['kind'].startswith(('sitemap', 'water', 'site'))],
                   sitemap_error=res.get('sitemap_error'), reg=(S.get('drawing') or {}).get('reg'))
        if S.get('fdc') and 'error' not in S:
            f0 = gt['site_true']['fdc'][0]
            row['fdc_err_m'] = round(math.dist((S['fdc'][0]['x'] - g[0], S['fdc'][0]['y'] - g[1]), (f0['x'], f0['y'])), 2)
            row['pumper_fdc_d'] = (S.get('pumper') or {}).get('fdc_d')
            ents = [(e['x'] - g[0], e['y'] - g[1]) for e in S.get('entrances', [])]
            row['ent_err_m'] = [round(min(math.dist(p, (it['x'], it['y'])) for p in ents), 2) for it in gt['items'] if it['type'] in ('main', 'exit')] if ents else None
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(rows, open(os.path.join(ROOT, 'out_sitemap', f'pdf_{which}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return rows


if __name__ == '__main__':
    a = sys.argv[1:]
    ov = a[a.index('--overlay') + 1] if '--overlay' in a else None
    only = a[a.index('--only') + 1] if '--only' in a else None
    if a and a[0] == 'pdf': pdf_test(a[1] if len(a) > 1 else 'dev', int(a[2]) if len(a) > 2 else 3)
    else: main(a[0] if a else 'dev', ov, only)
