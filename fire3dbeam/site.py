# -*- coding: utf-8 -*-
"""기능 24: 외부 부서 위치 — 건물 외곽 + 부지(도로·소방용수·송수구) → 면별 소방차 접근성, 고가사다리차 부서 후보,
층별 외부 구조(소방관 진입창) 후보, 펌프차 위치와 송수구·소화전 거리.

입력 site.json (좌표 m, 북쪽 +y):
  building_outline  GIS 건물 외곽(1층) — 도면 외곽과 맞춰 엔진 좌표로 옮긴다(평행이동, 크기 차는 검수 플래그)
  roads[]           {id,name,width,line:[[x,y],..]}   도로 중심선·폭(도로명주소 도로구간 형식)
  obstacles[]       {kind,poly}                        가로수·가공전선 등 사다리 전개 불가 구간
  hydrants[]        {id,type,x,y}                      소방용수시설(정보카드 ⑥ 의 번호와 같은 id)
  fdc[]             {name,x,y}                         연결송수관 송수구
기준값은 fire3dbeam.sop.CFG (사다리 길이·전개 반경·도로 폭 등은 '기관 SOP 값으로 설정 필요')."""
import json, math, heapq
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from fire3dbeam.sop import CFG
from fire3dbeam.card import level_of

DIRS = [(0, '북측면'), (90, '동측면'), (180, '남측면'), (270, '서측면')]


def load(path): return json.load(open(path, encoding='utf-8'))


def _main_poly(fp):
    if fp is None: return None
    if fp.geom_type == 'Polygon': return fp
    return max(fp.geoms, key=lambda g: g.area)


def register(site, fp, cfg=CFG):
    """GIS 건물 외곽 ↔ 도면 1층 외곽: bbox 중심을 맞추는 평행이동(둘 다 북쪽 위). 크기 차가 크면 플래그."""
    so = Polygon(site['building_outline'])
    sx0, sy0, sx1, sy1 = so.bounds; ex0, ey0, ex1, ey1 = fp.bounds
    dx, dy = (ex0 + ex1) / 2 - (sx0 + sx1) / 2, (ey0 + ey1) / 2 - (sy0 + sy1) / 2
    rw, rh = (ex1 - ex0) / max(1e-6, sx1 - sx0), (ey1 - ey0) / max(1e-6, sy1 - sy0)
    ok = abs(rw - 1) <= cfg['site_reg_scale_rel'] and abs(rh - 1) <= cfg['site_reg_scale_rel']
    return dx, dy, dict(dx=round(dx, 2), dy=round(dy, 2), scale_w=round(rw, 3), scale_h=round(rh, 3), ok=ok)


def _moved(site, dx, dy):
    T = lambda p: [round(p[0] + dx, 2), round(p[1] + dy, 2)]
    return dict(roads=[dict(r, line=[T(p) for p in r['line']]) for r in site.get('roads', [])],
                obstacles=[dict(o, poly=[T(p) for p in o['poly']]) for o in site.get('obstacles', [])],
                hydrants=[dict(h, x=T((h['x'], h['y']))[0], y=T((h['x'], h['y']))[1]) for h in site.get('hydrants', [])],
                fdc=[dict(f, x=T((f['x'], f['y']))[0], y=T((f['x'], f['y']))[1]) for f in site.get('fdc', [])])


def facades(fp, min_len=3.0):
    """외곽 → 외벽면(바깥 법선 방향이 같은 연속 변을 합침)"""
    p = orient(_main_poly(fp), 1.0)          # 반시계 → 바깥 법선 = (dy, -dx)
    cs = list(p.exterior.coords)[:-1]
    edges = []
    for i in range(len(cs)):
        a, b = cs[i], cs[(i + 1) % len(cs)]
        L = math.dist(a, b)
        if L < 0.3: continue
        n = ((b[1] - a[1]) / L, -(b[0] - a[0]) / L)
        ang = math.degrees(math.atan2(n[0], n[1])) % 360          # 0=북, 90=동
        if edges and abs(((edges[-1]['ang'] - ang + 180) % 360) - 180) < 12: edges[-1]['b'] = b; edges[-1]['len'] += L
        else: edges.append(dict(a=a, b=b, n=n, ang=ang, len=L))
    if len(edges) > 1 and abs(((edges[-1]['ang'] - edges[0]['ang'] + 180) % 360) - 180) < 12:
        edges[0]['a'] = edges[-1]['a']; edges[0]['len'] += edges[-1]['len']; edges.pop()
    out = []
    for e in edges:
        if e['len'] < min_len: continue
        nm = min(DIRS, key=lambda d: abs(((d[0] - e['ang'] + 180) % 360) - 180))[1]
        out.append(dict(name=nm, a=[round(v, 2) for v in e['a']], b=[round(v, 2) for v in e['b']], n=[round(v, 4) for v in e['n']], len=round(e['len'], 1)))
    cnt = {}
    for f in out: cnt[f['name']] = cnt.get(f['name'], 0) + 1
    seen = {}
    for f in out:
        if cnt[f['name']] > 1: seen[f['name']] = seen.get(f['name'], 0) + 1; f['name'] += f"-{seen[f['name']]}"
    return out


def _samples(a, b, step):
    L = math.dist(a, b); n = max(1, int(L // step))
    return [(a[0] + (b[0] - a[0]) * (k + 0.5) / n, a[1] + (b[1] - a[1]) * (k + 0.5) / n) for k in range(n)]


def _road_graph(roads, step):
    """도로 중심선을 step 간격 점으로 → 인접 점·교차점 연결 그래프(도로 거리 계산용)"""
    pts, adj = [], {}
    for r in roads:
        line = r['line']; prev = None
        for i in range(len(line) - 1):
            a, b = line[i], line[i + 1]; L = math.dist(a, b); n = max(1, int(L // step))
            for k in range(n + 1):
                if i > 0 and k == 0: continue
                p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                j = len(pts); pts.append(p); adj[j] = []
                if prev is not None: d = math.dist(p, pts[prev]); adj[j].append((prev, d)); adj[prev].append((j, d))
                prev = j
    # 교차: 서로 다른 도로의 가까운 점 연결
    from collections import defaultdict
    grid = defaultdict(list)
    for j, p in enumerate(pts): grid[(int(p[0] // step), int(p[1] // step))].append(j)
    for j, p in enumerate(pts):
        gx, gy = int(p[0] // step), int(p[1] // step)
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                for k in grid[(gx + ax, gy + ay)]:
                    if k > j and math.dist(p, pts[k]) < step * 1.2 and all(k != q for q, _ in adj[j]):
                        d = math.dist(p, pts[k]); adj[j].append((k, d)); adj[k].append((j, d))
    return pts, adj


def road_dist(G, a, b):
    pts, adj = G
    if not pts: return None
    na = min(range(len(pts)), key=lambda j: math.dist(pts[j], a)); nb = min(range(len(pts)), key=lambda j: math.dist(pts[j], b))
    dist = {na: 0.0}; pq = [(0.0, na)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == nb: break
        if d > dist.get(u, 1e18): continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, 1e18): dist[v] = nd; heapq.heappush(pq, (nd, v))
    if nb not in dist: return None
    return dist[nb] + math.dist(pts[na], a) + math.dist(pts[nb], b)


def regularize(fp):
    """도면에서 뽑은 외곽의 잔 요철(문 틈·기호) 정리: 거의 직사각형이면 최소 외접 사각형, 아니면 1m 단순화"""
    p = _main_poly(fp)
    r = p.minimum_rotated_rectangle
    if p.area / max(r.area, 1e-6) >= 0.9: return r
    return p.simplify(1.0).buffer(0)


def pick_reference(site, fps, cfg=CFG):
    """GIS 건물 외곽과 가장 잘 맞는 지상층 외곽(1층 우선) — 1층 외곽 추출 실패 대비"""
    best = None
    for fid in sorted(fps, key=lambda f: (f != '1F', level_of(f))):
        fp = fps[fid]
        if fp is None or level_of(fid) < 0: continue
        p = regularize(fp)
        _, _, reg = register(site, p, cfg)
        err = abs(reg['scale_w'] - 1) + abs(reg['scale_h'] - 1) + (0 if fid == '1F' else 0.03)
        if best is None or err < best[0]: best = (err, fid, p)
    return (best[1], best[2]) if best else (None, None)


def register_walls(site, walls, cfg=CFG):
    """GIS 건물 외곽 ↔ 도면 벽선 정합(평행이동): 외곽 경계에 겹치는 벽 길이가 최대가 되는 (dx, dy).
    외곽 다각형 추출이 실패한 층이 있어도 벽선만 있으면 된다."""
    from shapely.geometry import MultiLineString
    so = Polygon(site['building_outline']); cs = list(so.exterior.coords)
    ev = [(a[0], min(a[1], b[1]), max(a[1], b[1])) for a, b in zip(cs, cs[1:]) if abs(a[0] - b[0]) < 1e-6]
    eh = [(a[1], min(a[0], b[0]), max(a[0], b[0])) for a, b in zip(cs, cs[1:]) if abs(a[1] - b[1]) < 1e-6]
    wv = [w for w in walls if w['o'] == 'v' and w['b'] - w['a'] >= 3]; wh = [w for w in walls if w['o'] == 'h' and w['b'] - w['a'] >= 3]
    def top(ws, es):
        sc = {}
        for w in ws:
            for e in es:
                k = round((w['c'] - e[0]) / 0.25) * 0.25; sc[k] = sc.get(k, 0) + (w['b'] - w['a'])
        return [k for k, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:6]]
    if not (wv and wh and ev and eh): return None
    ml = MultiLineString([[(w['a'], w['c']), (w['b'], w['c'])] if w['o'] == 'h' else [(w['c'], w['a']), (w['c'], w['b'])] for w in wv + wh])
    best = None
    for dx in top(wv, ev):
        for dy in top(wh, eh):
            bd = Polygon([(x + dx, y + dy) for x, y in cs]).exterior
            L = ml.intersection(bd.buffer(0.6)).length
            if best is None or L > best[0]: best = (L, dx, dy)
    L, dx, dy = best
    nfl = max(1, len({w.get('_f') for w in walls}))
    cov = L / nfl / so.length                       # 층당 외곽 둘레 중 벽과 겹친 비율
    return dx, dy, dict(dx=round(dx, 2), dy=round(dy, 2), coverage=round(cov, 2), ok=cov >= 0.5, ref='도면 벽선 ↔ GIS 외곽 경계 정합')


def analyze(site, fps, floor_ids, info=None, floor_h=None, cfg=CFG, walls=None):
    """→ (report, flags). fps: {층: 외곽}, floor_ids: 지상 층 id 목록(1F, 2F, ...). 좌표는 엔진 전역."""
    ups_fp = [_main_poly(g) for f, g in fps.items() if g is not None and level_of(f) >= 0]
    U = unary_union(ups_fp).envelope if ups_fp else None           # 지상층 외곽 합집합(층별 외곽 추출 누락 보완)
    dx, dy, reg = register(site, U, cfg); reg['ref'] = '지상층 외곽 합집합'
    rw = register_walls(site, walls, cfg) if walls else None
    if rw and rw[2]['ok'] and (not reg['ok'] or math.hypot(rw[0] - dx, rw[1] - dy) < 3):
        dx, dy, r2 = rw; r2.update(scale_w=reg['scale_w'], scale_h=reg['scale_h'], bbox_dx=reg['dx'], bbox_dy=reg['dy']); reg = r2
    if not reg['ok']:
        ref, fp0 = pick_reference(site, fps, cfg)
        dx, dy, reg = register(site, fp0, cfg); reg['ref'] = ref
    else: fp0 = None
    # 외벽면은 GIS 건물 외곽(권위 자료)을 엔진 좌표로 옮겨 사용, 정합 실패 시 도면 외곽
    fp = Polygon([(x + dx, y + dy) for x, y in site['building_outline']]) if reg['ok'] else regularize(fp0)
    reg['facade_src'] = 'GIS 건물 외곽(정합 후)' if reg['ok'] else '도면 외곽(정합 실패)'
    S = _moved(site, dx, dy)
    fh = floor_h or cfg['floor_height_m']
    flags = []
    if not reg['ok']: flags.append(dict(kind='site_reg', **reg))
    roads = S['roads']
    rpoly = [(r, LineString(r['line']).buffer(r['width'] / 2, cap_style=2)) for r in roads]
    obst = unary_union([Polygon(o['poly']) for o in S['obstacles']]) if S['obstacles'] else None
    F = facades(fp)
    # 1) 면별 펌프차 접근: 외벽면에서 바깥 법선 방향 engine_max_setback 이내에 폭 기준을 넘는 도로가 닿는가
    reach = cfg['engine_max_setback_m']
    for f in F:
        pts = _samples(f['a'], f['b'], 2.0); hit = 0; best = None
        for (x, y) in pts:
            ray = LineString([(x + f['n'][0] * 0.3, y + f['n'][1] * 0.3), (x + f['n'][0] * reach, y + f['n'][1] * reach)])
            for r, rp in rpoly:
                if r['width'] < cfg['engine_min_road_width_m'] or not ray.intersects(rp): continue
                d = Point(x, y).distance(rp)
                hit += 1; best = (r, d) if best is None or d < best[1] else best
                break
        f['engine'] = hit / max(1, len(pts)) >= 0.3
        f['engine_frac'] = round(hit / max(1, len(pts)), 2)
        f['road'] = best[0]['name'] if best else None
        f['setback'] = round(best[1], 1) if best else None
    # 2) 고가사다리차 후보 위치(도로 위, 건물 쪽 차로, 장애 구간 제외)
    step = cfg['site_sample_m']
    cands = []
    for r, rp in rpoly:
        if r['width'] < cfg['aerial_min_road_width_m']: continue
        for i in range(len(r['line']) - 1):
            a, b = r['line'][i], r['line'][i + 1]; L = math.dist(a, b)
            if L < 1e-6: continue
            u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L); nrm = (-u[1], u[0])
            for (x, y) in _samples(a, b, step):
                if fp.distance(Point(x, y)) > cfg['aerial_ladder_len_m']: continue
                c = fp.centroid; sgn = 1 if (c.x - x) * nrm[0] + (c.y - y) * nrm[1] > 0 else -1
                px, py = x + sgn * nrm[0] * r['width'] / 4, y + sgn * nrm[1] * r['width'] / 4
                if obst is not None and obst.buffer(1.0).contains(Point(px, py)): continue
                cands.append(dict(x=round(px, 2), y=round(py, 2), road=r['name']))
    # 목표점: 2층 이상 각 층·각 외벽면 2m 간격, 높이 = 층 바닥 + 진입창 하단(0.8m)
    ups = sorted([f for f in floor_ids if level_of(f) >= 1], key=level_of)
    L, h0, R = cfg['aerial_ladder_len_m'], cfg['aerial_turntable_h_m'], cfg['aerial_max_outreach_m']
    targets = []
    for fid in ups:
        h = level_of(fid) * fh + cfg['rescue_window_sill_m']
        for fi, f in enumerate(F):
            for (x, y) in _samples(f['a'], f['b'], 2.0):
                targets.append((fid, fi, x + f['n'][0] * 0.3, y + f['n'][1] * 0.3, h))
    def reach_set(c):
        s = set()
        for ti, (fid, fi, x, y, h) in enumerate(targets):
            dz = h - h0
            if dz > L: continue
            n = F[fi]['n']
            if (c['x'] - x) * n[0] + (c['y'] - y) * n[1] <= 0.5: continue      # 외벽면 앞쪽에서만
            if math.hypot(c['x'] - x, c['y'] - y) <= min(R, math.sqrt(max(0.0, L * L - dz * dz))): s.add(ti)
        return s
    for c in cands: c['_s'] = reach_set(c)
    chosen, covered = [], set()
    pairs = lambda s: {(targets[t][0], targets[t][1]) for t in s}
    for _ in range(cfg['aerial_max_positions']):
        best, bv = None, (0, 0)
        for c in cands:
            if any(math.hypot(c['x'] - q['x'], c['y'] - q['y']) < cfg['aerial_min_gap_m'] for q in chosen): continue
            v = (len(pairs(c['_s']) - pairs(covered)), len(c['_s'] - covered))
            if v > bv: best, bv = c, v
        if best is None or bv[0] == 0: break
        chosen.append(best); covered |= best['_s']
    aerial = []
    for k, c in enumerate(chosen):
        pr = pairs(c['_s'])
        fl = sorted({p[0] for p in pr}, key=level_of)
        aerial.append(dict(id=f'A{k + 1}', x=c['x'], y=c['y'], road=c['road'], radius=R, floors=fl,
                           facades=sorted({F[p[1]]['name'] for p in pr}), top=fl[-1] if fl else None))
    # 3) 층별 외부 구조창(소방관 진입창) 후보: 사다리가 닿는 면마다 가장 가까운 점 + 40m 간격
    windows, per_floor = [], []
    for fid in ups:
        law = level_of(fid) + 1 <= cfg['rescue_window_max_floor']
        got = []
        for fi, f in enumerate(F):
            best = []
            for a in aerial:
                c = next(q for q in chosen if q['x'] == a['x'] and q['y'] == a['y'])
                ts = [targets[t] for t in c['_s'] if targets[t][0] == fid and targets[t][1] == fi]
                ts.sort(key=lambda t: math.hypot(t[2] - a['x'], t[3] - a['y']))
                pick = []
                for t in ts:
                    if all(math.hypot(t[2] - p[2], t[3] - p[3]) >= cfg['rescue_window_spacing_m'] for p in pick): pick.append(t)
                best += [(t, a) for t in pick]
            for t, a in best:
                if any(math.hypot(t[2] - w['x'], t[3] - w['y']) < cfg['rescue_window_spacing_m'] / 2 and w['floor'] == fid for w in windows): continue
                w = dict(floor=fid, facade=f['name'], x=round(t[2], 2), y=round(t[3], 2), h=round(t[4], 1), by=a['id'],
                         dist=round(math.hypot(t[2] - a['x'], t[3] - a['y']), 1), law=law)
                windows.append(w); got.append(w)
        per_floor.append(dict(floor=fid, h=round(level_of(fid) * fh + cfg['rescue_window_sill_m'], 1), n=len(got), law=law,
                              facades=sorted({w['facade'] for w in got}), by=sorted({w['by'] for w in got})))
    unreachable = [p['floor'] for p in per_floor if not p['n']]
    # 4) 펌프차: 송수구에 가장 가까운 도로 위 위치(사다리차 위치와 간격 유지) → 송수구·소화전 거리
    G = _road_graph(roads, step)
    pc = []
    for r, rp in rpoly:
        if r['width'] < cfg['engine_min_road_width_m']: continue
        for i in range(len(r['line']) - 1):
            for (x, y) in _samples(r['line'][i], r['line'][i + 1], step):
                if all(math.hypot(x - a['x'], y - a['y']) >= cfg['aerial_min_gap_m'] for a in aerial): pc.append((x, y, r))
    tgt = S['fdc'][0] if S['fdc'] else None
    if tgt is None:
        c = fp.exterior.interpolate(0); tgt = dict(name='(송수구 정보 없음) 건물 외벽', x=c.x, y=c.y)
    pumper = None
    if pc:
        x, y, r = min(pc, key=lambda p: math.hypot(p[0] - tgt['x'], p[1] - tgt['y']))
        d = math.hypot(x - tgt['x'], y - tgt['y'])
        hyd = []
        for h in S['hydrants']:
            rd = road_dist(G, (x, y), (h['x'], h['y']))
            hyd.append(dict(id=h['id'], type=h['type'], road_d=round(rd, 1) if rd else None, straight=round(math.hypot(x - h['x'], y - h['y']), 1),
                            hose=math.ceil((rd or math.hypot(x - h['x'], y - h['y'])) / cfg['hose_len_m'])))
        hyd.sort(key=lambda z: z['road_d'] if z['road_d'] is not None else z['straight'])
        pumper = dict(x=round(x, 2), y=round(y, 2), road=r['name'], fdc=tgt['name'], fdc_x=tgt['x'], fdc_y=tgt['y'], fdc_d=round(d, 1),
                      fdc_hose=math.ceil(d / cfg['hose_len_m']), hydrants=hyd)
    # 5) 정보카드 ⑥ 소방용수 거리 ↔ 부지 계산(건물 외곽~소화전 직선)
    water = []
    for w in (info or {}).get('water') or []:
        h = next((q for q in S['hydrants'] if q['id'] == w['id']), None)
        if h is None: water.append(dict(w, calc=None, ok=None)); continue
        calc = fp.distance(Point(h['x'], h['y']))
        ok = abs(calc - w['km'] * 1000) <= max(cfg['water_dist_abs_m'], cfg['water_dist_rel'] * w['km'] * 1000)
        water.append(dict(w, calc=round(calc, 1), ok=ok))
        if not ok: flags.append(dict(kind='water_dist', id=w['id'], card=round(w['km'] * 1000), site=round(calc)))
    no_eng = [f['name'] for f in F if not f['engine']]
    if unreachable: flags.append(dict(kind='aerial_unreachable', floors=unreachable, lv='mid'))
    rep = dict(reg=reg, roads=roads, obstacles=S['obstacles'], hydrants=S['hydrants'], fdc=S['fdc'], facades=F, aerial=aerial, windows=windows,
               per_floor=per_floor, unreachable=unreachable, pumper=pumper, water=water, floor_h=fh, no_engine=no_eng,
               cfg={k: cfg[k] for k in ('aerial_ladder_len_m', 'aerial_max_outreach_m', 'aerial_turntable_h_m', 'aerial_min_road_width_m',
                                        'engine_min_road_width_m', 'engine_max_setback_m', 'hose_len_m', 'rescue_window_max_floor', 'rescue_window_spacing_m')})
    return rep, flags


# ───────────────────────── 진압작전도(정보카드 ⑪ '0 진압작전도') → 부지 자료 ─────────────────────────
def from_drawing(fr, site_json=None):
    """fire3dbeam.sitemap.to_frame 결과(엔진 좌표) → analyze 입력 형식의 부지 dict.
    도로·소화전·송수구·주출입구/비상구는 작전도에서, 사다리 전개 장애(가로수·전선)는 GIS 부지 자료(site.json)가 있으면 그것에서."""
    hyd = []
    for k, h in enumerate(fr['hydrants']):
        if h.get('dist_label') and h.get('dist_conf') is False: continue          # 도면 밖 소화전인데 거리 글자가 불확실 → 위치 미상(계산에서 뺌)
        hyd.append(dict(id=h.get('id') or f'작전도-{k + 1}', type=h.get('type', '미상'), x=h['x'], y=h['y'],
                        dist_label=h.get('dist_label'), offmap=bool(h.get('offmap')), src='진압작전도'))
    return dict(source='진압작전도 인식(정보카드 ⑪)', crs='engine_m', building_outline=fr['footprint'], roads=fr['roads'],
                obstacles=(site_json or {}).get('obstacles', []), hydrants=hyd,
                fdc=[dict(name='연결송수구(작전도)', x=f['x'], y=f['y'], facade=f['facade']) for f in fr['fdc']],
                entrances=fr['entrances'], dims=fr['dims'])


def drawing_checks(fr, openings=None, plan_exits=None, plan_fdc=None, site_json=None, cfg=CFG):
    """작전도 ↔ 평면도 ↔ 정보카드 ↔ 부지 자료 교차검증 플래그.
    openings: 평면도 1층 외벽 개구부 [(x,y)], plan_exits: 그중 비상구(유도등 기호 붙은 것) [(x,y)], plan_fdc: 평면도 송수구 기호 [(x,y)]"""
    fl = []
    KO = dict(main='주출입구', exit='비상구')
    tol = cfg.get('sitemap_opening_tol_m', 4.0)
    if openings is not None:
        for e in fr['entrances']:
            d = min([math.dist((e['x'], e['y']), o) for o in openings] + [1e9])
            if d > tol: fl.append(dict(kind='sitemap_opening', mark=e['kind'], x=e['x'], y=e['y'], facade=e['facade'], d=round(d, 1) if d < 1e8 else None))
    if plan_exits:
        for (x, y) in plan_exits:
            if all(math.dist((x, y), (e['x'], e['y'])) > tol for e in fr['entrances']):
                fl.append(dict(kind='sitemap_exit_unmarked', x=round(x, 2), y=round(y, 2)))
    if plan_fdc and fr['fdc']:
        for f in fr['fdc']:
            d = min(math.dist((f['x'], f['y']), q) for q in plan_fdc)
            if d > cfg.get('sitemap_fdc_tol_m', 6.0): fl.append(dict(kind='sitemap_fdc', x=f['x'], y=f['y'], d=round(d, 1)))
    # 치수 글자 ↔ 도면 실측(소화전·송수구 사이 치수는 기호 중심 사이로 잰다)
    pts = [(h['x'], h['y']) for h in fr['hydrants'] if not h.get('offmap')] + [(f['x'], f['y']) for f in fr['fdc']]
    icons = [tuple(h.get('icon_m') or (h['x'], h['y'])) for h in fr['hydrants'] if not h.get('offmap')] + [tuple(f['icon']) for f in fr['fdc']]
    for dm in fr['dims']:
        if not dm.get('val'): continue
        a, b = dm['a'], dm['b']
        ja = min(range(len(icons)), key=lambda j: math.dist(a, icons[j]), default=None)
        jb = min(range(len(icons)), key=lambda j: math.dist(b, icons[j]), default=None)
        L = dm['len_m']
        if ja is not None and jb is not None and ja != jb and math.dist(a, icons[ja]) < 4 and math.dist(b, icons[jb]) < 4: L = math.dist(pts[ja], pts[jb])
        dm['measured_m'] = round(L, 1)
        dm['measured_src'] = 'icons' if L != dm['len_m'] else 'line'
        # 불일치 플래그는 양 끝이 기호(소화전·송수구)에 붙어 실측이 확실한 치수 + 두 번 읽어 같은 글자만
        if dm.get('val_src') == 'geom': continue                        # 글자를 못 읽어 기하 길이로 추정한 값 — 비교 대상 아님
        if dm['measured_src'] == 'icons' and dm.get('val_conf', True) and abs(dm['val'] - L) > max(3.0, 0.2 * dm['val']):
            fl.append(dict(kind='sitemap_dim', text=dm['val'], measured=round(L, 1)))
        elif dm.get('geom_mismatch') and dm.get('geom_m') and abs(dm['val'] - dm['geom_m']) > max(3.0, 0.2 * dm['val']):
            fl.append(dict(kind='sitemap_dim', text=dm['val'], measured=round(dm['geom_m'], 1), src='치수선 길이(축척)'))      # 판독이 뚜렷한데 선 길이와 안 맞음
    if site_json:                                           # 작전도 ↔ GIS 부지 자료
        for h in site_json.get('hydrants', []):
            q = next((x for x in fr['hydrants'] if x.get('id') == h['id']), None)
            if q is None or q.get('offmap'): continue
            if math.dist((q['x'], q['y']), (h['x'], h['y'])) > 10: fl.append(dict(kind='sitemap_vs_site', what=f"소화전 {h['id']}", d=round(math.dist((q['x'], q['y']), (h['x'], h['y'])), 1)))
    if fr.get('checks'): fl += fr['checks']
    return fl


def analyze_drawing(fr, fps, floor_ids, info=None, floor_h=None, cfg=CFG, walls=None, site_json=None, openings=None, plan_exits=None, plan_fdc=None):
    """작전도 기반 기능 24 → (report, flags). site_json(있으면)은 장애 구간·교차검증에만."""
    S = from_drawing(fr, site_json)
    rep, flags = analyze(S, fps, floor_ids, info, floor_h, cfg, walls=walls)
    # 작전도 거리 글자 ↔ 정보카드 ⑥ (위치 계산값과 별도로 글자 자체도 비교)
    ids = [q['id'] for q in S['hydrants']]
    keep = []
    for w in rep['water']:
        h = next((q for q in S['hydrants'] if q['id'] == w['id']), None)
        if h and h.get('dist_label'): w['label'] = h['dist_label']
        card_m = w['km'] * 1000; tol = max(cfg['water_dist_abs_m'], cfg['water_dist_rel'] * card_m)
        if w.get('ok') is False and (ids.count(w['id']) > 1 or (w.get('label') and abs(w['label'] - card_m) <= tol)):
            w['ok'] = None; w['note'] = '번호 중복 또는 작전도 거리 글자는 카드와 일치 → 판정 보류'
        elif w.get('ok') is False: keep.append(w['id'])
    flags = [f for f in flags if f['kind'] != 'water_dist' or f['id'] in keep]
    fl2 = drawing_checks(fr, openings, plan_exits, plan_fdc, site_json, cfg)
    rep.update(src='진압작전도', entrances=S['entrances'], dims=S['dims'], drawing=dict(reg=fr['reg'], n_hyd=len(S['hydrants']), n_fdc=len(S['fdc']),
                                                                                  n_ent=len(S['entrances'])))
    return rep, flags + fl2
