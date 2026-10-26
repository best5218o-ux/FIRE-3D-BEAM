# -*- coding: utf-8 -*-
"""대피로 채점(가상 정답 기반) — 엔진 벽·문·계단으로 만든 격자 vs 정답 벽·문·계단으로 만든 격자.
정답 실(방)마다: (1) 엔진 경로 존재(벽 통과 없이) (2) 가장 가까운 목표(계단/옥외 출구) 일치
(3) 중심점 보행거리·최원점 보행거리 오차 (허용 max(2 m, 10%)) (4) 기준 초과 판정 일치.
층마다: 도면 대피방향 화살표 검출 재현율·정밀도(선 길이, 0.6 m 허용).
두 격자는 같은 규칙(firebim/evac.make_grid)·같은 보행거리 산정(시선 당김)을 쓴다 → 차이는 인식 오차(벽·문·계단·외곽)에서만 생긴다."""
import math
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from firebim.evac import make_grid, mcp, pull, _targets_cells, WALL

TOL_ABS, TOL_REL = 2.0, 0.10


def _field(G, targets):
    cells, own = [], {}
    for T in targets:
        cs = _targets_cells(G, T); cells += cs
        for c in cs: own[c] = T
    m, cum = mcp(G, cells)
    return m, cum, own


def _route(G, m, own, c):
    tc = m.traceback(c)[::-1]
    path, L, nw = pull(G, tc)
    return L, nw, own.get(tuple(tc[-1]))


def score_evac(gt, res, gx, gy, resid):
    ev = res.get('evac')
    if not ev or 'floors' not in ev: return {}
    ground = ev['ground']
    n = ok_route = ok_tgt = ok_d = ok_dm = 0
    gt_route = 0; errs = []; errs_m = []
    over = dict(tp=0, fn=0, fp=0, tn=0)
    arr_g = arr_gh = arr_e = arr_eh = 0.0
    for f, rec in ev['floors'].items():
        if f not in gt['floors'] or resid.get(f, 9) > 3: continue
        Fe, Fg = res['floors'][f], gt['floors'][f]
        # 엔진 격자
        ids = [f'ST{k + 1}' for k, S in enumerate(res['stairs']) if f in S['floors']]
        vbe = [(rec.get('stair_boxes') or {}).get(i, S['box']) for i, S in zip(ids, [S for S in res['stairs'] if f in S['floors']])]
        Ge = make_grid(rec['bb'], rec['fp'], Fe['walls'], vbe, sides=[(rec.get('stair_sides') or {}).get(i) for i in ids])
        te = []
        for T in rec['targets']:
            T = dict(T)
            if T['kind'] == 'stair': T['box'] = (rec.get('stair_boxes') or {}).get(T['id']) or next(S['box'] for k, S in enumerate(res['stairs']) if f'ST{k + 1}' == T['id'])
            te.append(T)
        # 정답 격자(정답 좌표 + 전역 오프셋 g → 엔진 좌표계)
        sh = lambda b: [b[0] + gx, b[1] + gy, b[2] + gx, b[3] + gy]
        wg = [dict(o='h', c=w['y0'] + gy, a=w['x0'] + gx, b=w['x1'] + gx, t=w['t']) if abs(w['y1'] - w['y0']) < 1e-6 else
              dict(o='v', c=w['x0'] + gx, a=w['y0'] + gy, b=w['y1'] + gy, t=w['t']) for w in Fg['walls']]
        reg = unary_union([box(*sh(r)) for r in Fg['rects']])
        polys = [list(g.exterior.coords) for g in ([reg] if reg.geom_type == 'Polygon' else list(reg.geoms))]
        vbg = [sh(S['box']) for S in gt['stairs'] + gt['elevs'] if f in S['floors']]
        Gg = make_grid(rec['bb'], polys, wg, vbg, soft=None)            # 정답은 문이 모두 있으므로 계단실 벽 통과 허용 안 함
        if f == ground:
            tg = [dict(id=f"G{k}", kind='exit', x=d['x'] + gx, y=d['y'] + gy) for k, d in enumerate(Fg['doors']) if d['kind'] in ('entrance', 'exit')]
        else:
            tg = [dict(id=S['id'], kind='stair', box=sh(S['box']), x=(S['box'][0] + S['box'][2]) / 2 + gx, y=(S['box'][1] + S['box'][3]) / 2 + gy)
                  for S in gt['stairs'] if f in S['floors'] and ground in S['floors']]
        # 화살표(도면 대피방향) 검출 채점
        GL = [LineString([(x + gx, y + gy) for x, y in ln]) for ln in Fg.get('evac', [])]
        EL = [LineString(a['pts']) for a in rec.get('arrows', [])]
        if GL:
            gu = unary_union(GL); eu = unary_union(EL) if EL else None
            arr_g += gu.length; arr_gh += gu.intersection(eu.buffer(0.6)).length if eu is not None else 0
        if EL:
            eu = unary_union(EL); arr_e += eu.length
            arr_eh += eu.intersection(unary_union(GL).buffer(0.6)).length if GL else 0
        if not tg: continue
        mg, cg, og = _field(Gg, tg)
        me, ce, oe = (_field(Ge, te) if te else (None, None, None))
        lim = rec['lim_room']
        for rm in Fg['rooms']:
            poly = [(x + gx, y + gy) for x, y in rm['poly']]
            cx, cy = np.mean([p[0] for p in poly]), np.mean([p[1] for p in poly])
            c_g = Gg.free(*Gg.cell(cx, cy))
            if not np.isfinite(cg[c_g]): continue
            dg, nwg, Tg = _route(Gg, mg, og, c_g)
            if nwg: continue                                      # 정답 격자에서도 벽을 지나야 하면 채점 제외
            n += 1
            msk = Gg.poly_mask(poly, erode=1) & (Gg.cost < WALL) & np.isfinite(cg)
            if msk.any():
                fcg = np.unravel_index(int(np.where(msk, cg, -1).argmax()), cg.shape); dmg = max(dg, _route(Gg, mg, og, fcg)[0])
            else: dmg = dg
            if me is None: over['fn' if dmg > lim else 'tn'] += 1; continue
            c_e = Ge.free(*Ge.cell(cx, cy))
            if not np.isfinite(ce[c_e]): over['fn' if dmg > lim else 'tn'] += 1; continue
            de, nwe, Te = _route(Ge, me, oe, c_e)
            if nwe == 0: ok_route += 1
            # 목표 일치: 엔진 목표가 정답 목표(계단 상자 +2.5 m / 출구 2 m) 안
            if Te is not None and Tg is not None:
                if Tg['kind'] == 'stair': hit = box(*Tg['box']).buffer(2.5).contains(Point(Te['x'], Te['y']))
                else: hit = math.dist((Te['x'], Te['y']), (Tg['x'], Tg['y'])) < max(2.0, 1.0)
                ok_tgt += hit
            me_msk = msk & (Ge.cost < WALL) & np.isfinite(ce)
            if me_msk.any():
                fce = np.unravel_index(int(np.where(me_msk, ce, -1).argmax()), ce.shape); dme = max(de, _route(Ge, me, oe, fce)[0])
            else: dme = de
            errs.append(abs(de - dg)); errs_m.append(abs(dme - dmg))
            ok_d += abs(de - dg) <= max(TOL_ABS, TOL_REL * dg)
            ok_dm += abs(dme - dmg) <= max(TOL_ABS, TOL_REL * dmg)
            gO, eO = dmg > lim, dme > lim
            over['tp' if gO and eO else 'fn' if gO else 'fp' if eO else 'tn'] += 1
    if not n: return {}
    out = dict(ev_rooms=n, ev_route=f"{ok_route}/{n}", ev_target=f"{ok_tgt}/{n}", ev_dist=f"{ok_d}/{n}", ev_dmax=f"{ok_dm}/{n}",
               ev_err_med=round(float(np.median(errs)), 2) if errs else None, ev_errmax_med=round(float(np.median(errs_m)), 2) if errs_m else None,
               ev_err_list=[round(e, 2) for e in errs], ev_over=[over['tp'], over['fn'], over['fp'], over['tn']])
    if arr_g: out['ev_arrow_R'] = round(arr_gh / arr_g, 3)
    if arr_e: out['ev_arrow_P'] = round(arr_eh / arr_e, 3)
    return out


def summarize(R):
    """run_all 요약용: 유형별 합계"""
    def ratio(k):
        t = [tuple(map(int, r[k].split('/'))) for r in R if k in r]
        return f"{sum(a for a, b in t)}/{sum(b for a, b in t)}" if t else None
    e = [x for r in R for x in r.get('ev_err_list', [])]
    ov = np.sum([r['ev_over'] for r in R if 'ev_over' in r], axis=0).tolist() if any('ev_over' in r for r in R) else None
    aR = [r['ev_arrow_R'] for r in R if 'ev_arrow_R' in r]; aP = [r['ev_arrow_P'] for r in R if 'ev_arrow_P' in r]
    return dict(ev_route=ratio('ev_route'), ev_target=ratio('ev_target'), ev_dist=ratio('ev_dist'), ev_dmax=ratio('ev_dmax'),
                ev_err_med=round(float(np.median(e)), 2) if e else None, ev_err_p90=round(float(np.percentile(e, 90)), 2) if e else None,
                ev_over_tp_fn_fp_tn=ov, ev_arrow_R=round(float(np.mean(aR)), 3) if aR else None, ev_arrow_P=round(float(np.mean(aP)), 3) if aP else None)
