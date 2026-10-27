# -*- coding: utf-8 -*-
"""층간 정합: 층마다 다른 축척·잘림을 한 좌표계(m)로
- 축척: 치수 OCR(우선) → 정보카드 층 면적(보조)
- 위치: 승강기 표기·비상구 기호가 층마다 같은 자리에 반복된다는 성질로 평행이동을 찾는다"""
import math, os
import numpy as np

W_TYPE = {'ev': 3.0, 'exit': 1.0, 'hose': 0.3}


def local_pts(plan):
    s = plan['s']
    return [(p['type'], p['px'] / s, -p['py'] / s) for p in plan['symbols'] if p['type'] in W_TYPE]


def score_shift(pts, ref, t, tol=1.2):
    sc, pairs = 0.0, []
    for typ, x, y in pts:
        best = None
        for rt, rx, ry in ref:
            if rt != typ: continue
            d = (x + t[0] - rx) ** 2 + (y + t[1] - ry) ** 2
            if d < tol * tol and (best is None or d < best[0]): best = (d, rx - x, ry - y)
        if best: sc += W_TYPE[typ]; pairs.append(best[1:])
    return sc, pairs


def grid_offset_score(a, b, t, tol=0.25):
    """두 층 그리드선(m) 집합이 이동 t 후 얼마나 겹치는지"""
    (vx_a, hy_a), (vx_b, hy_b) = a, b
    s = sum(1 for x in vx_a if any(abs(x + t[0] - y) < tol for y in vx_b))
    s += sum(1 for x in hy_a if any(abs(x + t[1] - y) < tol for y in hy_b))
    return s


def wall_raster(walls, t=(0, 0), R=2.0, bounds=None):
    """벽을 0.5m 격자 점 집합으로"""
    pts = set()
    for w in walls:
        L = w['b'] - w['a']; n = max(1, int(L * R))
        for k in range(n + 1):
            u = w['a'] + L * k / n
            x, y = ((u, w['c']) if w['o'] == 'h' else (w['c'], u))
            pts.add((int(round((x + t[0]) * R)), int(round((y + t[1]) * R))))
    return pts


def wall_overlap(pw, refset, t, R=2.0):
    P = wall_raster(pw, t, R)
    if not P: return 0.0
    hit = sum(1 for (x, y) in P if (x, y) in refset or (x + 1, y) in refset or (x - 1, y) in refset or (x, y + 1) in refset or (x, y - 1) in refset)
    return hit / len(P)


def align(plans):
    """plans: [{id, s, symbols, vlines, hlines}] → 각 plan에 tx, ty, conf 추가 (전역 = 지역 + t)"""
    for p in plans:
        p['_pts'] = local_pts(p)
        p['_grid'] = ([x / p['s'] for x in p.get('vlines', [])], [-y / p['s'] for y in p.get('hlines', [])])
    order = sorted(plans, key=lambda p: -(sum(W_TYPE[t] for t, _, _ in p['_pts'])))
    ref = order[0]; ref['tx'] = ref['ty'] = 0.0; ref['conf'] = 'ref'
    placed = [ref]
    for p in order[1:]:
        R = [(t, x + q['tx'], y + q['ty']) for q in placed for (t, x, y) in q['_pts']]
        cands = set()
        for typ, x, y in p['_pts']:
            if typ == 'hose': continue
            for rt, rx, ry in R:
                if rt == typ: cands.add((round(rx - x, 1), round(ry - y, 1)))
        refw = set()
        for q in placed: refw |= wall_raster(q.get('walls_local', []), (q['tx'], q['ty']))
        scored = []
        for t in cands:
            sc, pairs = score_shift(p['_pts'], R, t)
            if sc > 0: scored.append((sc, t, pairs))
        scored.sort(key=lambda z: -z[0])
        best = None
        top = scored[0][0] if scored else 0
        for sc, t, pairs in scored[:25]:
            if sc < 0.5 * top: break
            tt = (float(np.mean([a for a, b in pairs])), float(np.mean([b for a, b in pairs])))
            wo = wall_overlap(p.get('walls_local', []), refw, tt)
            key = sc * (0.3 + wo)                      # 기호 일치 × 벽 겹침(대칭 건물에서 헷갈림 방지)
            if best is None or key > best[0][0]:
                best = ((key, sc), tt, pairs)
        if best and best[0][1] >= 3.0:
            (key, sc), (tx, ty), pairs = best
            p['tx'], p['ty'], p['conf'] = tx, ty, 'high' if sc >= 6 else 'mid'
        else:
            # 기호가 부족하면 그리드선 겹침만으로 (기준층 그리드와 가장 많이 겹치는 이동, 외곽 중심 차이 최소)
            gs = []
            for x in p['_grid'][0]:
                for rx in ref['_grid'][0]:
                    for y in p['_grid'][1][:6]:
                        for ry in ref['_grid'][1][:6]:
                            t = (rx - x, ry - y)
                            gs.append((grid_offset_score(p['_grid'], ref['_grid'], t), -abs(t[0]) - abs(t[1]), t))
            if gs:
                g = max(gs); p['tx'], p['ty'] = g[2]; p['conf'] = 'low'
            else:
                p['tx'] = p['ty'] = 0.0; p['conf'] = 'none'
        placed.append(p)
    return plans


# ───────────────────────── 실제 도면(범례·기호 없음) 층 정합 (2026-10-03) ─────────────────────────
W_ANCHOR = {'ev': 3.0, 'stair': 2.0}


def _real_feats(p):
    s = p['s']
    an = [(a['type'], a['px'] / s, -a['py'] / s) for a in p.get('anchors', [])]
    gv = [x / s for x in p.get('vlines', [])]; gh = [-y / s for y in p.get('hlines', [])]
    bub = {}
    for (o, c), name in (p.get('bubbles') or {}).items():
        bub[(o, name)] = c / s if o == 'v' else -c / s
    return an, gv, gh, bub


NAMED_MEDIAN = os.environ.get('FIRE3DBEAM_ALIGN_NAMED', '1') != '0'


def align_real(plans, tol_a=1.0, tol_g=0.3):
    """실제 도면 층 정합: 기준점 = 계단·승강기 중심(층마다 같은 자리) + 그리드선(+ 그리드 버블 이름).
    후보 이동 = 같은 종류 기준점 짝 · 같은 이름 버블 · 그리드선 짝(가로×세로) → 점수 = 기준점 일치 가중 + 0.6×그리드선 일치 + 3×버블 일치 + 2×벽 겹침.
    옥탑·코어만 있는 층(부분 층)도 코어 기준점 + 그리드로 맞춤. plans 의 각 평면에 tx, ty, conf, align_info 추가."""
    F = {id(p): _real_feats(p) for p in plans}
    def weight(p):
        an, gv, gh, bub = F[id(p)]
        return sum(W_ANCHOR.get(t, 0) for t, _, _ in an) + 0.3 * (len(gv) + len(gh)) + len(bub)
    order = sorted(plans, key=lambda p: -weight(p))
    ref = order[0]; ref['tx'] = ref['ty'] = 0.0; ref['conf'] = 'ref'; ref['align_info'] = dict(ref=True)
    placed = [ref]
    for p in order[1:]:
        an, gv, gh, bub = F[id(p)]
        RA = [(t, x + q['tx'], y + q['ty']) for q in placed for t, x, y in F[id(q)][0]]
        RV = sorted({round(x + q['tx'], 3) for q in placed for x in F[id(q)][1]})
        RH = sorted({round(y + q['ty'], 3) for q in placed for y in F[id(q)][2]})
        RB = {}
        for q in placed:
            for (o, nm), c in F[id(q)][3].items(): RB.setdefault((o, nm), c + (q['tx'] if o == 'v' else q['ty']))
        if NAMED_MEDIAN:                       # 이름 같은 버블끼리의 이동이 축 안에서 0.35 m 넘게 흩어지면(층마다 버블 위치를 다르게 그린 도면) 그 축 이름은 안 씀
            bub = dict(bub)
            for o in ('v', 'h'):
                d = [RB[(oo, nm)] - c for (oo, nm), c in bub.items() if oo == o and (oo, nm) in RB]
                if len(d) >= 2 and max(d) - min(d) > 0.35: bub = {k_: v_ for k_, v_ in bub.items() if k_[0] != o}
        cx, cy = set(), set()
        cands = set()
        for t, x, y in an:
            for rt, rx, ry in RA:
                if rt == t: cands.add((round(rx - x, 2), round(ry - y, 2)))
        for (o, nm), c in bub.items():
            if (o, nm) in RB: (cx if o == 'v' else cy).add(round(RB[(o, nm)] - c, 2))
        for x in gv[:14]:
            for rx in RV[:14]: cx.add(round(rx - x, 2))
        for y in gh[:14]:
            for ry in RH[:14]: cy.add(round(ry - y, 2))
        def g1(vals, L, RL, o):                                  # 축 하나만의 그리드·버블 일치 수 → 상위 6개 이동만 조합
            sc = []
            for v in vals:
                n = sum(1 for x in L if any(abs(r - x - v) < tol_g for r in RL)) + 3 * sum(1 for (oo, nm), c in bub.items() if oo == o and (oo, nm) in RB and abs(RB[(oo, nm)] - c - v) < tol_g)
                sc.append((n, -abs(v), v))
            return [v for _, _, v in sorted(sc, reverse=True)[:6]]
        tx_top, ty_top = g1(cx, gv, RV, 'v'), g1(cy, gh, RH, 'h')
        for a, b in list(cands):                                   # 기준점 짝 이동의 한 축 + 다른 축 그리드 상위
            for bb in ty_top: cands.add((a, bb))
            for aa in tx_top: cands.add((aa, b))
        cands |= {(a, b) for a in tx_top for b in ty_top}
        refw = set()
        for q in placed: refw |= wall_raster(q.get('walls_local', []), (q['tx'], q['ty']))
        def score(t):
            sa, types, pa = 0.0, set(), []
            for typ, x, y in an:
                best = None
                for rt, rx, ry in RA:
                    if rt != typ: continue
                    d = math.hypot(x + t[0] - rx, y + t[1] - ry)
                    if d < tol_a and (best is None or d < best[0]): best = (d, rx - x, ry - y)
                if best: sa += W_ANCHOR[typ]; types.add(typ); pa.append(best[1:])
            gxm = [(rx - x) for x in gv for rx in [min(RV, key=lambda v: abs(v - x - t[0]))] if RV and abs(rx - x - t[0]) < tol_g]
            gym = [(ry - y) for y in gh for ry in [min(RH, key=lambda v: abs(v - y - t[1]))] if RH and abs(ry - y - t[1]) < tol_g]
            nb = sum(1 for (o, nm), c in bub.items() if (o, nm) in RB and abs(RB[(o, nm)] - c - (t[0] if o == 'v' else t[1])) < tol_g)
            return sa, types, pa, gxm, gym, nb
        best = None
        for t in cands:
            sa, types, pa, gxm, gym, nb = score(t)
            pre = sa + 0.6 * (len(gxm) + len(gym)) + 3 * nb
            if best is not None and pre + 2.0 < best[0]: continue
            wo = wall_overlap(p.get('walls_local', []), refw, t) if refw else 0.0
            key = pre + 2.0 * wo
            if best is None or key > best[0]: best = (key, t, sa, types, pa, gxm, gym, nb, wo)
        if best is None:
            p['tx'] = p['ty'] = 0.0; p['conf'] = 'none'; p['align_info'] = dict(); placed.append(p); continue
        key, t, sa, types, pa, gxm, gym, nb, wo = best
        # 다듬기: 일치한 기준점·그리드선 차이의 평균(축별)
        xs = [a for a, b in pa] + gxm; ys = [b for a, b in pa] + gym
        tx = float(np.mean(xs)) if xs else t[0]; ty = float(np.mean(ys)) if ys else t[1]
        if NAMED_MEDIAN:                       # 같은 이름 그리드 버블이 축마다 2개 이상 맞으면 그 축 이동 = 버블 이동의 중앙값(층마다 안쪽 보조 그리드가 달라도 바깥 축이 이김)
            nx = [RB[(o, nm)] - c for (o, nm), c in bub.items() if o == 'v' and (o, nm) in RB and abs(RB[(o, nm)] - c - t[0]) < 1.5]
            ny = [RB[(o, nm)] - c for (o, nm), c in bub.items() if o == 'h' and (o, nm) in RB and abs(RB[(o, nm)] - c - t[1]) < 1.5]
            if len(nx) >= 2: tx = float(np.median(nx))
            if len(ny) >= 2: ty = float(np.median(ny))
        ng = len(gxm) + len(gym)
        if (len(types) >= 2 and ng >= 3) or (nb >= 2 and ng >= 2): conf = 'high'
        elif (len(types) >= 1 and ng >= 2) or len(types) >= 2 or nb >= 2: conf = 'mid'
        else: conf = 'low'
        p['tx'], p['ty'], p['conf'] = tx, ty, conf
        p['align_info'] = dict(anchors=sorted(types), grid=ng, bubbles=nb, wall=round(wo, 2))
        placed.append(p)
    return plans
