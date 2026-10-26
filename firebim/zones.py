# -*- coding: utf-8 -*-
"""기능 12: 정보카드 '자동화재탐지기(회)' 회로 수 → 층별 경계구역(추정) 분할 → 수신기 경보(층+회로) 시 화점 후보 구역.

근거: NFTC 203 2.1.1 (경계구역 600㎡·한 변 50m 이하, 층별, 500㎡ 이하 2개 층 단서, 내부 전체가 보이면 1,000㎡ 단서)
      NFTC 203 2.1.2 (계단·승강로 등 수직 부분은 별도 경계구역, 수직거리 45m 이하마다)
경계구역 일람도(JSON)가 주어지면 그 구역을 그대로 쓰고, 없으면 벽선을 따라 회로 수만큼 나눈 '추정' 구역을 만든다.
좌표는 엔진 전역 좌표(m, x 동쪽, y 북쪽)."""
import math, json
from shapely.geometry import Polygon, MultiPolygon, box, LineString
from shapely.ops import unary_union, polylabel
from firebim.sop import CFG
from firebim.card import level_of

CIRCUIT_KEY = '자동화재탐지기(회)'


def _parts(g):
    if g is None or g.is_empty: return []
    if g.geom_type == 'Polygon': return [g]
    return [p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon' and p.area > 0.5]


def floor_region(fp, fid, stairs, elevs):
    """층 외곽 − 계단실·승강로(수직 경계구역으로 따로 셈, NFTC 203 2.1.2)"""
    if fp is None: return None
    cut = [box(*s['box']) for s in list(stairs) + list(elevs) if fid in s['floors']]
    g = fp.buffer(0)
    if cut: g = g.difference(unary_union(cut))
    ps = _parts(g)
    if ps:                                          # 범례 상자 등 떨어진 작은 조각 제거
        big = max(q.area for q in ps); ps = [q for q in ps if q.area >= 0.1 * big]
    return unary_union(ps) if ps else None


def _cut(g, axis, c, lo):
    x0, y0, x1, y1 = g.bounds
    if axis == 'x': b = box(x0 - 1, y0 - 1, c, y1 + 1) if lo else box(c, y0 - 1, x1 + 1, y1 + 1)
    else: b = box(x0 - 1, y0 - 1, x1 + 1, c) if lo else box(x0 - 1, c, x1 + 1, y1 + 1)
    return g.intersection(b)


def _wall_cover(walls, axis, g):
    """자르는 선(x=c 또는 y=c)과 겹치는 벽 길이 → {c: 덮는 길이}"""
    cov = {}
    want = 'v' if axis == 'x' else 'h'
    for w in walls:
        if w['o'] != want or w['b'] - w['a'] < 1.0: continue
        c = round(w['c'] / 0.3) * 0.3
        cov[c] = cov.get(c, 0.0) + (w['b'] - w['a'])
    return cov


def _cut_at(g, axis, frac_target, walls, tol=0.12):
    """g 를 axis 방향 좌표 c 에서 잘라 아래쪽 면적 비율이 frac_target 이 되게. 가능하면 벽선 위치로 옮긴다."""
    x0, y0, x1, y1 = g.bounds
    lo_, hi_ = (x0, x1) if axis == 'x' else (y0, y1)
    A = g.area
    frac = lambda c: _cut(g, axis, c, True).area / A
    a, b = lo_, hi_
    for _ in range(40):
        m = (a + b) / 2
        if frac(m) < frac_target: a = m
        else: b = m
    best, bs = (a + b) / 2, -1e9
    span = (y1 - y0) if axis == 'x' else (x1 - x0)
    for c, L in _wall_cover(walls, axis, g).items():
        if not (lo_ + 1.5 < c < hi_ - 1.5): continue
        line = LineString([(c, y0 - 1), (c, y1 + 1)]) if axis == 'x' else LineString([(x0 - 1, c), (x1 + 1, c)])
        chord = line.intersection(g).length or span
        cover = min(1.0, L / chord)
        err = abs(frac(c) - frac_target) / min(frac_target, 1 - frac_target)
        if err > tol or cover < 0.3: continue
        sc = cover - 3 * err
        if sc > bs: best, bs = c, sc
    return _cut(g, axis, best, True), _cut(g, axis, best, False)


def _long_axis(g):
    x0, y0, x1, y1 = g.bounds
    return 'x' if (x1 - x0) >= (y1 - y0) else 'y'


def split(g, n, walls, tol=0.12):
    """전략 1(이분): 면적 비율(n1:n2)을 맞추며 긴 변에 수직으로 재귀 분할. 가능하면 벽선 위치에서 자른다."""
    if n <= 1 or g is None or g.area < 1: return [g]
    n1 = n // 2
    lo, hi = _cut_at(g, _long_axis(g), n1 / n, walls, tol)
    return split(lo, n1, walls, tol) + split(hi, n - n1, walls, tol)


def split_axis(g, counts, axis, walls, tol=0.12):
    """g 를 axis 방향으로 len(counts) 조각(면적 ∝ counts)으로 차례로 자른다"""
    out, rest, tot = [], g, sum(counts)
    for k, c in enumerate(counts[:-1]):
        lo, rest = _cut_at(rest, axis, c / tot, walls, tol)
        out.append(lo); tot -= c
    return out + [rest]


def split_strips(g, n, cols, walls):
    """전략 2(띠): 긴 변 방향으로 cols 개 띠 → 각 띠를 짧은 변 방향으로 나눔(격자형)"""
    if g is None or g.area < 1: return [g]
    cols = max(1, min(cols, n))
    counts = [n // cols + (1 if i < n % cols else 0) for i in range(cols)]
    ax = _long_axis(g); other = 'y' if ax == 'x' else 'x'
    out = []
    for strip, c in zip(split_axis(g, counts, ax, walls) if cols > 1 else [g], counts):
        out += split_axis(strip, [1] * c, other, walls) if c > 1 else [strip]
    return out


def partition(g, n, walls, cfg, limit=None):
    """기준(면적·한 변)을 만족하는 분할을 우선 선택. 없으면 이분 결과."""
    cands = [split(g, n, walls)] + [split_strips(g, n, c, walls) for c in range(1, min(n, 8) + 1)]
    cands += [split(g, n, [])] + [split_strips(g, n, c, []) for c in range(1, min(n, 8) + 1)]     # 벽선 무시(면적 균등) 대안
    for zs in cands:
        if all(zone_ok(z, cfg, limit) for z in zs): return zs, True
    return cands[0], False


def side_of(g):
    x0, y0, x1, y1 = g.bounds
    return max(x1 - x0, y1 - y0)


def zone_ok(g, cfg, limit=None):
    return g.area <= (limit or cfg['zone_max_area_m2']) + 1e-6 and side_of(g) <= cfg['zone_max_side_m'] + 1e-6


def required_min(region, walls, cfg, limit=None):
    """NFTC 203 2.1.1 을 만족하는 최소 경계구역 수(도면 외곽 기준 추정)"""
    lim = limit or cfg['zone_max_area_m2']
    k = max(1, math.ceil(region.area / lim))
    while k < 80:
        if partition(region, k, walls, cfg, lim)[1]: return k
        k += 1
    return k


def rep_point(g):
    """구역 대표점: 가장 큰 조각의 내부 최원점(polylabel) — 벽에서 가장 먼 곳"""
    p = max(_parts(g), key=lambda q: q.area)
    try: q = polylabel(p, 0.3)
    except Exception: q = p.representative_point()
    return [round(q.x, 2), round(q.y, 2)]


def _rings(g):
    return [[[round(x, 2), round(y, 2)] for x, y in list(p.exterior.coords)[:-1]] for p in _parts(g)]


def load_ilramdo(path):
    """경계구역 일람도 JSON:
    {"source":"경계구역 일람도", "frame":"floor_bbox",           # 좌표 기준: 층 외곽 bbox 왼쪽 아래 = (0,0), m
     "floors":{"1F":[{"no":1,"name":"1층 서측","poly":[[x,y],...]}, ...]}}
    frame 이 "global" 이면 엔진 전역 좌표 그대로."""
    return json.load(open(path, encoding='utf-8'))


def compute(floors, stairs, elevs, spaces=None, ilramdo=None, cfg=CFG, area_ref=None):
    """floors: {fid: {'_fp','walls','card'}} → (zones dict, flags)"""
    out, flags = {}, []
    fids = sorted(floors, key=level_of)
    gno = 0
    il = (ilramdo or {}).get('floors') or {}
    frame = (ilramdo or {}).get('frame', 'floor_bbox')
    for fid in fids:
        F = floors[fid]
        fp = F.get('_fp')
        card = F.get('card') or {}
        n_card = (card.get('fac') or {}).get(CIRCUIT_KEY)
        rec = dict(floor=fid, n_card=n_card, zones=[], src='추정')
        good = lambda G_: G_.get('_fp') is not None and (G_.get('card') or {}).get('area') and 0.85 <= G_['_fp'].area / G_['card']['area'] <= 1.15
        ca0 = card.get('area')
        if not good(F) and ca0:            # 외곽 추출 실패 층: 면적이 비슷하고 외곽을 믿을 만한 다른 층(반복층)의 외곽을 빌림
            alt = [g for g in fids if g != fid and good(floors[g]) and abs(floors[g]['card']['area'] - ca0) / ca0 <= 0.1 and (level_of(g) >= 0) == (level_of(fid) >= 0)]
            if alt:
                src = min(alt, key=lambda g: abs(level_of(g) - level_of(fid))); fp = floors[src]['_fp']; rec['fp_from'] = src
        region = floor_region(fp, fid, stairs, elevs)
        if region is None or region.area < 5:
            rec['note'] = '층 외곽을 찾지 못해 경계구역을 나눌 수 없음'; out[fid] = rec; continue
        # 대공간(가장 큰 실이 층의 70% 이상) → '주된 출입구에서 내부 전체가 보이는' 단서 해당 가능성
        big = max([s['area'] for s in (spaces or {}).get(fid, [])] or [0])
        open_hall = bool(big >= 0.7 * region.area)
        rec.update(area=round(region.area, 1), side=round(side_of(region), 1), open_hall=open_hall)
        n_req = required_min(region, F['walls'], cfg)
        n_req_hall = required_min(region, F['walls'], cfg, cfg['zone_open_hall_area_m2']) if open_hall else n_req
        ca = (area_ref or {}).get(fid) or card.get('area')      # 기준 면적: 건축물대장 층 면적(있으면) → 정보카드 면적
        if ca and not (0.8 <= region.area / ca <= 1.25):      # 도면 외곽을 못 믿음 → 서류 면적(계단·승강로 제외)으로 최소 수 산정
            rec['fp_unreliable'] = True; rec['area_src'] = '건축물대장' if (area_ref or {}).get(fid) else '정보카드'
            ca = ca - sum((s['box'][2] - s['box'][0]) * (s['box'][3] - s['box'][1]) for s in list(stairs) + list(elevs) if fid in s['floors'])
            n_req = max(n_req, math.ceil(ca / cfg['zone_max_area_m2']))
            n_req_hall = max(n_req_hall, math.ceil(ca / cfg['zone_open_hall_area_m2'])) if open_hall else n_req
        rec.update(n_req=n_req, n_req_open=n_req_hall)
        if fid in il and il[fid]:
            ox, oy = (fp.bounds[0], fp.bounds[1]) if frame == 'floor_bbox' else (0.0, 0.0)
            gs = [(z.get('no', i + 1), z.get('name'), Polygon([(x + ox, y + oy) for x, y in z['poly']]).buffer(0)) for i, z in enumerate(il[fid])]
            rec['src'] = (ilramdo or {}).get('source') or '경계구역 일람도'
        else:
            n = n_card if n_card else n_req
            zs, _ = partition(region, n, F['walls'], cfg, cfg['zone_open_hall_area_m2'] if open_hall and n < n_req else None)
            gs = [(i + 1, None, g) for i, g in enumerate(sorted(zs, key=lambda z: (round(z.centroid.y / 6), z.centroid.x)))]
        for no, nm, g in gs:
            if g is None or g.is_empty: continue
            gno += 1
            rec['zones'].append(dict(no=int(no), gno=gno, name=nm or f'{fid} 경계구역 {no}', poly=_rings(g), area=round(g.area, 1),
                                     side=round(side_of(g), 1), rep=rep_point(g), ok=zone_ok(g, cfg, cfg['zone_open_hall_area_m2'] if open_hall else None),
                                     est=rec['src'] == '추정'))
        if n_card is not None and n_card < n_req:
            lv = 'low' if (open_hall and n_card >= n_req_hall) else 'mid'
            flags.append(dict(kind='zone_count', floor=fid, card=n_card, required=n_req, required_open=n_req_hall, area=rec['area'],
                              side=rec['side'], lv=lv))
        if n_card is not None and rec['src'] != '추정' and len(rec['zones']) != n_card:
            flags.append(dict(kind='zone_ilramdo', floor=fid, card=n_card, drawing=len(rec['zones'])))
        out[fid] = rec
    # 수직 경계구역(계단·승강로): 수직거리 45m 이하마다 1구역, 지하층 계단은 별도(지하 1개 층뿐이면 제외)
    vert = []
    hf = cfg['floor_height_m']
    for kind, lst in (('계단', stairs), ('승강로', elevs)):
        for i, s in enumerate(lst):
            up = [f for f in s['floors'] if level_of(f) >= 0]; dn = [f for f in s['floors'] if level_of(f) < 0]
            n_up = max(1, math.ceil(len(up) * hf / cfg['zone_vertical_max_m'])) if up else 0
            n_dn = (1 if len(dn) >= 2 else 0) if kind == '계단' else 0
            vert.append(dict(kind=kind, idx=i + 1, box=[round(v, 2) for v in s['box']], floors=s['floors'], n=n_up + n_dn,
                             note=f"지상 {len(up)}개 층(층고 {hf}m 가정) → 수직 {n_up}구역" + (f" + 지하 계단 1구역" if n_dn else '')))
    return dict(floors=out, vertical=vert, total=gno,
                basis='NFTC 203 2.1.1(600㎡·한 변 50m·층별) / 2.1.2(계단·승강로 별도, 수직 45m)',
                numbering='층 낮은 순 → 층 안에서는 서→동, 남→북 분할 순(추정 — 수신기 회로 번호와 다를 수 있음)'), flags
