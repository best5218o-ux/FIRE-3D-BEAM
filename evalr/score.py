# -*- coding: utf-8 -*-
"""가상 건물 정답 vs 엔진 결과 자동 채점"""
import json, math, sys, os, glob
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


def eng_map(F, px, py): return (px / F['s'] + F['tx'], -py / F['s'] + F['ty'])
def gt_map(T, px, py): return (px / T['s'] + T['ox'], T['oy'] - py / T['s'])


def score(gt, res):
    out = dict(name=gt['name'], btype=gt['btype'])
    T = gt['transforms']; E = res['floors']
    fids = [f for f in gt['floors'] if f in E and f in T]
    out['floors_found'] = f"{len(fids)}/{len(gt['floors'])}"
    # 1) 층 정합: 전역 오프셋 g 를 뺀 잔차
    diffs = {}
    for f in fids:
        t = T[f]; pts = [(0, 0), (t['w'], 0), (0, t['h']), (t['w'], t['h']), (t['w'] / 2, t['h'] / 2)]
        diffs[f] = [np.subtract(eng_map(E[f], *p), gt_map(t, *p)) for p in pts]
    g = np.median([d[-1] for d in diffs.values()], axis=0) if diffs else np.zeros(2)
    resid = {f: float(max(np.hypot(*(d - g)) for d in ds)) for f, ds in diffs.items()}
    out['align_mean_m'] = round(float(np.mean(list(resid.values()))), 2) if resid else None
    out['align_ok'] = f"{sum(1 for v in resid.values() if v < 1.0)}/{len(resid)}"
    gx, gy = float(g[0]), float(g[1])
    # 2) 벽
    rec_n = rec_d = pre_n = pre_d = 0.0
    for f in fids:
        if resid[f] > 3: continue
        G = [LineString([(w['x0'], w['y0']), (w['x1'], w['y1'])]) for w in gt['floors'][f]['walls']]
        P = []
        for w in E[f]['walls']:
            if w['o'] == 'h': P.append(LineString([(w['a'] - gx, w['c'] - gy), (w['b'] - gx, w['c'] - gy)]))
            else: P.append(LineString([(w['c'] - gx, w['a'] - gy), (w['c'] - gx, w['b'] - gy)]))
        GU, PU = unary_union(G), unary_union(P) if P else None
        for l in G:
            rec_d += l.length
            if PU is not None: rec_n += l.intersection(PU.buffer(0.4)).length
        for l in P:
            pre_d += l.length; pre_n += l.intersection(GU.buffer(0.4)).length
    out['wall_recall'] = round(rec_n / rec_d, 3) if rec_d else None
    out['wall_precision'] = round(pre_n / pre_d, 3) if pre_d else None
    # 3) 소방기호 (종류별)
    sym = {}
    for typ in ('exit', 'hose', 'panel', 'gas', 'ev'):
        tp = fn = fp = 0
        for f in fids:
            if resid[f] > 3: continue
            gs = [(q['x'], q['y']) for q in gt['floors'][f]['symbols'] if q['type'] == typ]
            es = [(q['x'] - gx, q['y'] - gy) for q in E[f]['symbols'] if q['type'] == typ]
            used = set()
            for a in gs:
                j = min(range(len(es)), key=lambda j: math.dist(a, es[j]), default=None)
                if j is not None and j not in used and math.dist(a, es[j]) < 1.2: tp += 1; used.add(j)
                else: fn += 1
            fp += len(es) - len(used)
        if tp + fn + fp: sym[typ] = dict(recall=round(tp / max(1, tp + fn), 3), precision=round(tp / max(1, tp + fp), 3), n=tp + fn)
    out['symbols'] = sym
    # 4) 문
    tp = fn = npred = 0
    for f in fids:
        if resid[f] > 3: continue
        gd = [(d['x'], d['y']) for d in gt['floors'][f]['doors']]
        ed = [(d['x'] - gx, d['y'] - gy) for d in E[f]['doors']]
        npred += len(ed)
        for a in gd:
            if any(math.dist(a, b) < 0.9 for b in ed): tp += 1
            else: fn += 1
    ptp = 0
    for f in fids:
        if resid[f] > 3: continue
        gd = [(d['x'], d['y']) for d in gt['floors'][f]['doors']]
        ptp += sum(1 for d in E[f]['doors'] if any(math.dist((d['x'] - gx, d['y'] - gy), a) < 0.9 for a in gd))
    out['door_recall'] = round(tp / max(1, tp + fn), 3); out['door_precision'] = round(ptp / max(1, npred), 3)
    # 4-1) 진입구: 정답 = 외벽 개구부(주출입구·셔터·차량진입로 = entrance, 외벽 비상구 = exit).
    #      엔진 = 외곽선 위 개구부(ext=True). 넓은 셔터는 중심 허용오차를 폭/2 까지(최소 0.9m, 문과 동일)
    etp = efn = xtp = xfn = epred = eptp = 0
    for f in fids:
        if resid[f] > 3: continue
        gx_ = [d for d in gt['floors'][f]['doors'] if d['kind'] in ('entrance', 'exit')]
        ed = [(d['x'] - gx, d['y'] - gy) for d in E[f]['doors'] if d.get('ext')]
        epred += len(ed)
        tol = lambda d: max(0.9, d['w'] / 2)
        for d in gx_:
            hit = any(math.dist((d['x'], d['y']), b) < tol(d) for b in ed)
            if d['kind'] == 'entrance':
                etp += hit; efn += not hit
            xtp += hit; xfn += not hit
        eptp += sum(1 for b in ed if any(math.dist((d['x'], d['y']), b) < tol(d) for d in gx_))
    if etp + efn: out['entrance_found'] = f"{etp}/{etp + efn}"
    out['extopen_recall'] = round(xtp / max(1, xtp + xfn), 3) if xtp + xfn else None
    out['extopen_precision'] = round(eptp / max(1, epred), 3) if epred else None
    # 5) 계단·승강기 (수직 동선)
    def match_v(gl, el, pad):
        hit = 0
        for s in gl:
            b = box(*s['box']).buffer(pad)
            if any(b.contains(Point(e['x'] - gx, e['y'] - gy)) for e in el): hit += 1
        ok = sum(1 for e in el if any(box(*s['box']).buffer(pad).contains(Point(e['x'] - gx, e['y'] - gy)) for s in gl))
        return f"{hit}/{len(gl)}", f"{ok}/{len(el)}"
    out['stairs_found'], out['stairs_correct'] = match_v(gt['stairs'], res['stairs'], 2.5)
    out['elevs_found'], out['elevs_correct'] = match_v(gt['elevs'], res['elevs'], 1.0)
    # 6) 교차검증 플래그 (심어 둔 오류를 찾았나)
    planted = gt['flags']
    found = 0
    for p in planted:
        if any(e['kind'] == p['kind'] and e.get('floor') == p.get('floor') for e in res['flags']): found += 1
    out['flags_planted_found'] = f"{found}/{len(planted)}"
    out['flags_total'] = len(res['flags'])
    # 7) 정보카드 해석
    ok = sum(1 for f, a in gt['card']['areas'].items() if f in res['card']['floors'] and abs(res['card']['floors'][f]['area'] - a) < 0.05)
    out['card_floors_ok'] = f"{ok}/{len(gt['card']['areas'])}"
    out.update(score_ext(gt, res))
    try:                                                # 대피로(evalr/evac_score.py)
        from evalr.evac_score import score_evac
        out.update(score_evac(gt, res, gx, gy, resid))
    except Exception as e: out['ev_error'] = repr(e)
    out.update(score_fac(gt, res, fids, resid, gx, gy))
    return out


# 가상 정보카드 항목명 → 기대 분류 코드(분류표와 독립으로 손으로 적은 정답)
FAC_EXPECT = {'옥내소화전': 'HOSE', '유도등': 'GUIDE', '자동화재탐지기(감)': 'DET', '자동화재탐지기(회)': 'ZONE', '수동식소화기': 'EXT', '비상조명등': 'EMLIGHT',
              '스프링클러헤드': 'SP', 'S/P헤드': 'SP', '스프링클러설비(헤드)': 'SP', '스프링클러': 'SP', '포소화설비(헤드)': 'FOAM',
              '이산화탄소소화설비': 'CO2', '할로겐화합물및불활성기체소화설비': 'CLEAN', '청정소화약제소화설비': 'CLEAN', '자동확산소화기': 'EXT_AUTO',
              '주거용주방자동소화장치': 'AUTO_KITCHEN', '비상방송(스피커)': 'PA', '비상방송설비': 'PA', '시각경보기': 'VISUAL', '자동화재속보설비': 'AUTOCALL',
              '가스누설경보기': 'GASLEAK', '구조대': 'ESCAPE', '완강기': 'ESCAPE', '하향식피난구': 'ESCAPE', '공기호흡기': 'RESCUE',
              '제연설비(부속실)': 'SMOKE', '부속실제연': 'SMOKE', '연결송수관(방수구)': 'STANDPIPE', '방수구': 'STANDPIPE', '연결송수관설비(방수구)': 'STANDPIPE',
              '비상콘센트': 'OUTLET', '연결송수관(송수구)': 'SIAMESE', '무선통신보조설비': 'RADIO', '비상용승강기': 'EMEV', '방화셔터': '기타'}
FAC_SYM = {'EXT': 'ext', 'STANDPIPE': 'outlet', 'OUTLET': 'socket', 'SIAMESE': 'siamese'}


def score_fac(gt, res, fids, resid, gx, gy):
    """소방시설: 새 범례 기호 P/R, 정보카드 항목 해석, 규칙 추정 위치 오차(방수구·비상콘센트), 비상용승강기 추정, 수량 불일치 플래그"""
    if 'fac_gt' not in gt or 'fac' not in res: return {}
    out = {}
    FG = gt['fac_gt']
    # 1) 기호 P/R (종류별 [tp, fn, fp]) — 도면에 그린 것만 정답
    sym = {t: [0, 0, 0] for t in FAC_SYM.values()}
    for f in fids:
        if resid[f] > 3: continue
        es = [(q['type'], q['x'] - gx, q['y'] - gy) for q in res['floors'][f].get('fac_symbols', [])]
        used = set()
        for q in FG['pos'].get(f, []):
            if not q['drawn']: continue
            t = FAC_SYM[q['code']]
            c = [j for j, e in enumerate(es) if e[0] == t and j not in used and math.dist((q['x'], q['y']), e[1:]) < 1.2]
            if c: used.add(min(c, key=lambda j: math.dist((q['x'], q['y']), es[j][1:]))); sym[t][0] += 1
            else: sym[t][1] += 1
        for j, e in enumerate(es):
            if j not in used: sym[e[0]][2] += 1
    out['fac_sym'] = {t: v for t, v in sym.items() if any(v)}
    # 2) 정보카드 항목 해석: (층, 기대 코드, 수량) 일치
    ok = tot = 0
    for f, fac in gt['card']['fac'].items():
        got = {}
        for it in (res['card']['floors'].get(f) or {}).get('fac_items', []): got[it['code']] = got.get(it['code'], 0) + it['count']
        exp = {}
        for k, v in fac.items():
            base = k.replace('(쌍구형)', '') if k.endswith('(쌍구형)') else k
            c = FAC_EXPECT.get(base, '?'); exp[c] = exp.get(c, 0) + v
        tot += len(exp); ok += sum(1 for c, v in exp.items() if got.get(c) == v)
    out['fac_parse'] = f"{ok}/{tot}"
    # 3) 규칙 추정 위치 오차: 도면에 없는 방수구·비상콘센트 정답 ↔ 같은 층 추정 후보 중 가장 가까운 것
    errs = {'STANDPIPE': [], 'OUTLET': []}; ncand = []; inr = {'STANDPIPE': [0, 0], 'OUTLET': [0, 0]}
    for f in fids:
        if resid[f] > 3: continue
        items = {it['code']: it for it in res['fac']['floors'].get(f, [])}
        for q in FG['pos'].get(f, []):
            if q['code'] not in errs or q['drawn']: continue
            it = items.get(q['code'])
            est = [(e['x'] - gx, e['y'] - gy) for e in (it or {}).get('est', [])]
            if est: errs[q['code']].append(round(min(math.dist((q['x'], q['y']), e) for e in est), 2)); ncand.append(len(est))
            else: errs[q['code']].append(None)
            anc = [(e['ax'] - gx, e['ay'] - gy, e['r']) for e in (it or {}).get('est', []) if e.get('r')]
            inr[q['code']][1] += 1; inr[q['code']][0] += any(math.dist((q['x'], q['y']), a[:2]) <= a[2] for a in anc)     # 규칙 원(계단 출입구 5 m) 안에 정답이 드나
    out['fac_est_err'] = errs; out['fac_est_ncand'] = ncand; out['fac_est_inradius'] = inr
    # 4) 비상용승강기: 추정 승강기 중심 ↔ 정답 승강기 중심 1.5 m (정답이 여러 대(em['all'])면 층×정답 승강기마다 1건)
    em = FG.get('emev')
    if em:
        hit = n = wrong = 0
        for f in fids:
            if resid[f] > 3 or f not in em['floors']: continue
            it = next((i for i in res['fac']['floors'].get(f, []) if i['code'] == 'EMEV'), None)
            if not it: continue
            G = em.get('all') or [em]
            for g in G:
                n += 1; hit += any(math.dist((g['x'], g['y']), (e['x'] - gx, e['y'] - gy)) < 1.5 for e in it['est'])
            wrong += sum(1 for e in it['est'] if not any(math.dist((g['x'], g['y']), (e['x'] - gx, e['y'] - gy)) < 1.5 for g in G))
        out['fac_emev'] = f"{hit}/{n}"; out['fac_emev_wrong'] = wrong
    # 5) 수량 불일치 플래그(심은 것)
    pl = gt.get('fac_flags', []); got = [f for f in res['flags'] if f['kind'] == 'fac_count']
    out['fac_flags'] = f"{sum(1 for p in pl if any(g['floor'] == p['floor'] and g['code'] == p['code'] for g in got))}/{len(pl)}"
    out['fac_flags_false'] = sum(1 for g in got if not any(g['floor'] == p['floor'] and g['code'] == p['code'] for p in pl))
    return out


def score_ext(gt, res):
    """기능 12·18·24 채점(추가 항목) — gt 에 해당 정답이 있을 때만"""
    out = {}
    F = res.get('flags', [])
    # 18) 건축물대장 교차검증: 규칙상 나와야 할 플래그(심은 불일치 + 정보카드 오기 파생) 재현율·정밀도
    if 'daejang_flags' in gt and 'daejang' in res:
        exp = gt['daejang_flags']; got = [f for f in F if f['kind'].startswith('daejang_')]
        hit = sum(1 for e in exp if any(g['kind'] == e['kind'] and g.get('floor') == e.get('floor') for g in got))
        ok = sum(1 for g in got if any(g['kind'] == e['kind'] and g.get('floor') == e.get('floor') for e in exp))
        out['daejang_found'] = f"{hit}/{len(exp)}"; out['daejang_correct'] = f"{ok}/{len(got)}"
        out['daejang_planted'] = len(gt.get('daejang_planted', []))
    # 12) 경계구역 수 검증: 정답 도형(층 외곽−계단·승강로)으로 계산한 NFTC 최소 구역 수 < 카드 회로 수 인 층 = 플래그 정답
    if 'zones' in res and 'error' not in res['zones']:
        from fire3dbeam import zones as Z
        from fire3dbeam.sop import CFG
        agree = n = tp = fn = fp_ = 0
        for fid, f in gt['floors'].items():
            nc = gt['card']['fac'].get(fid, {}).get(Z.CIRCUIT_KEY)
            if nc is None: continue
            reg = unary_union([box(*r) for r in f['rects']])
            cut = [box(*x['box']) for x in gt['stairs'] + gt['elevs'] if fid in x['floors']]
            if cut: reg = reg.difference(unary_union(cut))
            walls = [dict(o='h', c=w['y0'], a=w['x0'], b=w['x1']) if abs(w['y1'] - w['y0']) < 1e-6 else dict(o='v', c=w['x0'], a=w['y0'], b=w['y1']) for w in f['walls']]
            need = Z.required_min(reg, walls, CFG) > nc
            got = any(g['kind'] == 'zone_count' and g.get('floor') == fid for g in F)
            n += 1; agree += need == got; tp += need and got; fn += need and not got; fp_ += got and not need
        out['zone_flag_agree'] = f"{agree}/{n}"; out['zone_flag_tp_fn_fp'] = [tp, fn, fp_]
        out['zones_total'] = res['zones'].get('total')
    # 24) 부지: 면별 펌프차 접근 가능 판정, 소방용수 거리 불일치
    if 'site_gt' in gt and 'site' in res and 'error' not in res['site']:
        M = {'남측면': 'S', '북측면': 'N', '동측면': 'E', '서측면': 'W'}
        fe = {}
        for f in res['site']['facades']:
            k = M.get(f['name'].split('-')[0]); fe[k] = fe.get(k, False) or f['engine']
        tru = gt['site_gt']['facade_engine']
        out['site_facade_ok'] = f"{sum(1 for k, v in tru.items() if fe.get(k, False) == v)}/{len(tru)}"
        pw = gt['site_gt']['flags']; gw = [f for f in F if f['kind'] == 'water_dist']
        out['water_found'] = f"{sum(1 for p in pw if any(g['id'] == p['id'] for g in gw))}/{len(pw)}"
        out['water_false'] = sum(1 for g in gw if not any(g['id'] == p['id'] for p in pw))
        out['aerial_n'] = len(res['site']['aerial']); out['windows_n'] = len(res['site']['windows'])
    return out


if __name__ == '__main__':
    rows = []
    for d in sorted(glob.glob('data/*/gt.json')):
        name = os.path.basename(os.path.dirname(d)); rp = f'out/{name}.json'
        if not os.path.exists(rp): continue
        r = score(json.load(open(d)), json.load(open(rp)))
        rows.append(r); print(json.dumps(r, ensure_ascii=False))
    json.dump(rows, open('out/score.json', 'w'), ensure_ascii=False, indent=1)
