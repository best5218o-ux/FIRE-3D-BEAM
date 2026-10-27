# -*- coding: utf-8 -*-
"""대피로(피난 경로)·진입로 계산 — 실(벽망 자동 분할)마다 가장 가까운 직통계단/옥외 출구까지 보행 경로와 보행거리.

격자·비용은 뷰어(viewer/src/main.js gridFor)와 같은 규칙: 0.25 m 격자, 건물 안 1 · 밖 4 · 벽(두께/2+0.1 m) 80(막힘에 가까움),
계단실·승강기 둘레 0.6 m 안 벽은 6(계단실 문을 못 찾아도 통과), 문 = 벽 사이 틈. 최단 경로 = skimage MCP(8방향) → 시선(line-of-sight)
당기기로 꺾인 격자 경로를 곧게 편 길이를 보행거리로 보고한다.
도면의 빨간 '대피방향' 화살표는 검출해 약한 사전확률(화살표 ±0.75 m 칸 비용 0.8)로만 쓴다 → 표시 경로 선택에만 반영,
법정 보행거리는 사전확률 없는 최단 거리.

법적 기준(2026-09-30 국가법령정보센터 확인):
- 건축법 시행령 제34조①: 피난층 외의 층 — 거실의 각 부분에서 직통계단까지 보행거리 30 m 이하.
  주요구조부가 내화구조 또는 불연재료 → 50 m 이하(16층 이상 공동주택의 16층 이상 층은 40 m),
  자동식 소화설비 자동화 생산 공장 75 m(무인화 공장 100 m).
- 건축물의 피난·방화구조 등의 기준에 관한 규칙 제11조①(영 제39조 대상 건축물): 피난층 — 계단에서 옥외 출구까지
  영 제34조① 거리 이하, 거실 각 부분에서 옥외 출구까지 그 2배 이하.
결과는 도면 자동 추출 벽·문 기반 추정이며 인허가 판정이 아니라 현장 지휘 참고용이다."""
import math
import numpy as np, cv2
from shapely.geometry import Polygon, Point, box
from fire3dbeam.card import level_of

RES, WALL, OUT_COST, SOFT, PRIOR = 0.25, 80.0, 4.0, 20.0, 0.8     # SOFT: 계단실 둘레 벽(문 미검출 대비) — 뷰어는 6, 대피로는 실제 문을 더 우선
LAW_EVAC = {
    'stair_walk_m': 30.0,               # 건축법 시행령 제34조① 본문
    'stair_walk_fr_m': 50.0,            # 同 단서: 주요구조부 내화구조·불연재료
    'stair_walk_apt16_m': 40.0,         # 同 단서 괄호: 16층 이상 공동주택의 16층 이상 층
    'exit_room_factor': 2.0,            # 피난방화규칙 제11조①: 거실 → 옥외 출구 = 영 34조① 거리의 2배 이하
    'basis': '건축법 시행령 제34조①(직통계단 보행거리) · 건축물의 피난·방화구조 등의 기준에 관한 규칙 제11조①(피난층 옥외 출구)',
}
TOL_EVAC = {'common_path_m': 20.0}     # 두 방향 피난 경로가 갈라지기 전 공통 구간(내부 판단 기준, 법정값 아님)
FR_WORDS = ('철근콘크리트', '철골철근', '철골콘크리트', '콘크리트', '프리캐스트')


# ───────────────────────────── 격자
class Grid:
    def __init__(self, bb):
        self.x0, self.y0, self.x1, self.y1 = bb
        self.W = int(math.ceil((self.x1 - self.x0) / RES)) + 1
        self.H = int(math.ceil((self.y1 - self.y0) / RES)) + 1
        self.cost = np.full((self.H, self.W), OUT_COST, np.float64)

    def cell(self, x, y):
        return (min(self.H - 1, max(0, int(round((self.y1 - y) / RES)))), min(self.W - 1, max(0, int(round((x - self.x0) / RES)))))

    def pt(self, j, i): return (self.x0 + i * RES, self.y1 - j * RES)

    def free(self, j, i, lim=WALL, r=12):
        for k in range(r):
            for dj in range(-k, k + 1):
                for di in range(-k, k + 1):
                    a, b = j + dj, i + di
                    if 0 <= a < self.H and 0 <= b < self.W and self.cost[a, b] < lim: return (a, b)
        return (j, i)

    def rect_idx(self, ax0, ay0, ax1, ay1):
        i0, i1 = max(0, math.ceil((ax0 - self.x0) / RES)), min(self.W - 1, math.floor((ax1 - self.x0) / RES))
        j0, j1 = max(0, math.ceil((self.y1 - ay1) / RES)), min(self.H - 1, math.floor((self.y1 - ay0) / RES))
        return j0, j1, i0, i1

    def poly_mask(self, poly, erode=0):
        m = np.zeros((self.H, self.W), np.uint8)
        pts = np.array([[(x - self.x0) / RES, (self.y1 - y) / RES] for x, y in poly], np.float32)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if erode: m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        return m.astype(bool)


def make_grid(bb, polys, walls, vboxes, arrows=None, soft=SOFT, sides=None):
    """bb=(x0,y0,x1,y1) · polys=외곽 링들 · walls=[{o,c,a,b,t}] · vboxes=계단·승강기 상자 · arrows=대피방향 선(사전확률)"""
    G = Grid(bb)
    ins = np.zeros((G.H, G.W), np.uint8)
    for ring in polys:
        cv2.fillPoly(ins, [np.round(np.array([[(x - G.x0) / RES, (G.y1 - y) / RES] for x, y in ring], np.float32)).astype(np.int32)], 1)
    G.cost[ins > 0] = 1.0
    for w in walls:
        tt = max(0.2, w['t']) / 2 + 0.1
        r = (w['a'], w['c'] - tt, w['b'], w['c'] + tt) if w['o'] == 'h' else (w['c'] - tt, w['a'], w['c'] + tt, w['b'])
        j0, j1, i0, i1 = G.rect_idx(*r)
        if j1 >= j0 and i1 >= i0: G.cost[j0:j1 + 1, i0:i1 + 1] = WALL
    for k, bx in enumerate(vboxes):
        if soft is None: break
        sd = sides[k] if sides else None                      # 계단실 문 쪽(비상구 표지 방향)만 통과 허용, 모르면 네 변 모두
        x0, y0, x1, y1 = bx[0] - 0.6, bx[1] - 0.6, bx[2] + 0.6, bx[3] + 0.6
        r = {'W': (x0, y0, bx[0] + 0.6, y1), 'E': (bx[2] - 0.6, y0, x1, y1), 'S': (x0, y0, x1, bx[1] + 0.6), 'N': (x0, bx[3] - 0.6, x1, y1)}.get(sd, (x0, y0, x1, y1))
        j0, j1, i0, i1 = G.rect_idx(*r)
        sub = G.cost[j0:j1 + 1, i0:i1 + 1]; sub[sub >= WALL] = soft
    if arrows:
        m = np.zeros((G.H, G.W), np.uint8)
        for a in arrows:
            pts = np.round(np.array([[(x - G.x0) / RES, (G.y1 - y) / RES] for x, y in a['pts']], np.float32)).astype(np.int32)
            cv2.polylines(m, [pts], False, 1, thickness=int(round(1.5 / RES)))
        G.cost[(m > 0) & (G.cost == 1.0)] = PRIOR
    return G


def mcp(G, cells):
    from skimage.graph import MCP_Geometric
    m = MCP_Geometric(G.cost, fully_connected=True)
    cum, _ = m.find_costs([c for c in cells])
    return m, cum * RES


def los(G, a, b):
    """두 칸 사이 직선이 더 비싼 칸(벽·건물 밖)을 지나지 않는가"""
    n = int(max(abs(a[0] - b[0]), abs(a[1] - b[1])) * 2) + 2
    js = np.round(np.linspace(a[0], b[0], n)).astype(int); is_ = np.round(np.linspace(a[1], b[1], n)).astype(int)
    lim = max(G.cost[a], G.cost[b], 1.0)
    return bool((G.cost[js, is_] <= lim).all())


def pull(G, cells):
    """격자 경로 → 시선 당기기로 곧게 편 꺾은선(좌표) + 길이 + 벽 칸 수"""
    if not cells: return [], 0.0, 0
    ca = np.asarray(cells)
    nwall = int((G.cost[ca[:, 0], ca[:, 1]] >= WALL).sum())
    if len(cells) > 2:                                   # 방향이 바뀌는 칸만 남김(시선 검사 횟수 줄이기)
        d = np.diff(ca, axis=0); keep = np.r_[True, (d[1:] != d[:-1]).any(1), True]
        cells = [tuple(c) for c in ca[keep]]
    out, i, n = [cells[0]], 0, len(cells)
    while i < n - 1:
        k = i + 1
        while k + 1 < n and los(G, cells[i], cells[k + 1]): k += 1
        out.append(cells[k]); i = k
    pts = [G.pt(*c) for c in out]
    L = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
    return [[round(x, 2), round(y, 2)] for x, y in pts], L, nwall


def stair_sides(boxes, symbols):
    """계단 상자마다 문 쪽 변 추정: 4 m 안 가장 가까운 비상구 표지(계단실 문 앞에 붙음)가 가장 가까운 변 W/E/S/N, 없으면 None"""
    ex = [(q['x'], q['y']) for q in symbols if q['type'] == 'exit']
    out = []
    for b in boxes:
        bb = box(*b); c = [p for p in ex if bb.buffer(4.0).contains(Point(p))]
        if not c: out.append(None); continue
        x, y = min(c, key=lambda p: bb.distance(Point(p)))
        d = {'W': abs(x - b[0]), 'E': abs(x - b[2]), 'S': abs(y - b[1]), 'N': abs(y - b[3])}
        out.append(min(d, key=d.get))
    return out


def snap_box(b, sd, walls, reach=2.0):
    """계단 디딤판 상자의 문 쪽 변을 바깥 2 m 안 평행 벽(계단실 벽)까지 넓힘 → 계단실 입구 기준 보행거리"""
    if sd is None: return list(b)
    x0, y0, x1, y1 = b
    o = 'h' if sd in 'SN' else 'v'
    e = {'W': x0, 'E': x1, 'S': y0, 'N': y1}[sd]; out = -1 if sd in 'WS' else 1
    lo, hi = (x0, x1) if o == 'h' else (y0, y1)
    best = None
    for w in walls:
        if w['o'] != o: continue
        dd = (w['c'] - e) * out
        if not (0.05 < dd <= reach): continue
        if min(hi, w['b']) - max(lo, w['a']) < 0.3 * (hi - lo): continue
        if best is None or dd < best[0]: best = (dd, w['c'])
    if best is None: return list(b)
    nb = list(b); nb[{'W': 0, 'S': 1, 'E': 2, 'N': 3}[sd]] = best[1]
    return nb


# ───────────────────────────── 대피방향 화살표(도면 빨간 선) 검출
def detect_arrows(img, s, tx, ty, legend=None, fp=None):
    """빨간 선(R>120, R-G·R-B>70) 성분 중 길이 ≥ 2.5 m·폭 ≤ 0.8 m → 전역 좌표 선분 + 끝단 화살촉 유무(heads=[시작, 끝])"""
    b, g, r = [img[..., k].astype(int) for k in range(3)]
    red = ((r > 120) & (r - g > 70) & (r - b > 70) & (np.abs(g - b) < 60)).astype(np.uint8)   # 어두운 빨강(흐린 화살촉)까지
    if legend:
        L = legend; red[max(0, L['y'] - 4):L['y'] + L['h'] + 4, max(0, L['x'] - 4):L['x'] + L['w'] + 4] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red, 8)
    out = []
    G = lambda px, py: (px / s + tx, -py / s + ty)
    for k in range(1, n):
        if st[k][4] < 6: continue
        bx, by, bw, bh = st[k][:4]
        if math.hypot(bw, bh) < 2.5 * s: continue                     # 상자 대각선 < 2.5 m 면 길이도 < 2.5 m (결과 같음, 속도)
        ys, xs = np.nonzero(lab[by:by + bh, bx:bx + bw] == k); ys = ys + by; xs = xs + bx
        P = np.stack([xs, ys], 1).astype(float); c = P.mean(0)
        u, sv, vt = np.linalg.svd(P - c, full_matrices=False)
        d = vt[0]; t = (P - c) @ d; q = (P - c) @ vt[1]
        Lm = (t.max() - t.min()) / s
        if Lm < 2.5: continue
        mid = np.abs(t - (t.max() + t.min()) / 2) < (t.max() - t.min()) * 0.3
        w_mid = (np.percentile(q[mid], 95) - np.percentile(q[mid], 5) + 1) if mid.sum() > 3 else 2.0
        if w_mid / s > 0.8: continue                                   # 굵은 빨간 덩어리(기호 등)는 제외
        def head(sel): return (np.ptp(q[sel]) + 1) > 2.2 * w_mid + 1 if sel.sum() > 3 else False
        e0, e1 = c + d * t.min(), c + d * t.max()
        span = min(1.0 * s, (t.max() - t.min()) * 0.2)
        h0, h1 = head(t < t.min() + span), head(t > t.max() - span)
        p0, p1 = G(*e0), G(*e1)
        if fp is not None and not (fp.buffer(1.0).contains(Point(p0)) and fp.buffer(1.0).contains(Point(p1))): continue
        out.append(dict(pts=[[round(float(p0[0]), 2), round(float(p0[1]), 2)], [round(float(p1[0]), 2), round(float(p1[1]), 2)]], heads=[bool(h0), bool(h1)], len=round(float(Lm), 1)))
    return out


# ───────────────────────────── 기준
def card_structure(info):
    """정보카드(표준양식) ⑦ 건물구조 '구조' 글자(예 '철근콘크리트조') — 없으면 ''. 대장 주구조가 없을 때만 쓴다(대장 우선)"""
    return ((info or {}).get('건물구조') or {}).get('구조') or ''


def limits(info, title, fid):
    """(기준 m, 설명, 내화 여부 True/False/None)"""
    strct = (title or {}).get('strctCdNm') or ''
    src = '건축물대장'
    if not strct:                                   # 대장이 없으면 정보카드 ⑦ 건물구조 '구조' 칸(표준양식) — card_structure 훅
        strct, src = card_structure(info), '정보카드 ⑦'
    use = (info or {}).get('use') or (title or {}).get('mainPurpsCdNm') or ''
    fr = (any(w in strct for w in FR_WORDS)) if strct else None
    if fr:
        if '공동주택' in use and level_of(fid) >= 15:
            return LAW_EVAC['stair_walk_apt16_m'], f'주요구조부 내화({strct}) · 16층 이상 공동주택 → 40 m', fr
        return LAW_EVAC['stair_walk_fr_m'], f'주요구조부 내화({strct}, {src}) → 50 m', fr
    if fr is False: return LAW_EVAC['stair_walk_m'], f'구조 {strct} → 30 m', fr
    return LAW_EVAC['stair_walk_m'], '주요구조부 내화 여부 미확인(건축물대장·정보카드 구조 없음) → 30 m 적용, 내화이면 50 m', None


# ───────────────────────────── 층 분석
def _targets_cells(G, T):
    """목표(계단 상자 안 칸 / 출구 문 칸) → 칸 목록"""
    if T['kind'] == 'stair':
        x0, y0, x1, y1 = T['box']
        j0, j1, i0, i1 = G.rect_idx(x0 + 0.4, y0 + 0.4, x1 - 0.4, y1 - 0.4)
        cells = [(j, i) for j in range(j0, j1 + 1) for i in range(i0, i1 + 1) if G.cost[j, i] < WALL]
        return cells or [G.free(*G.cell(T['x'], T['y']))]
    j, i = G.free(*G.cell(T['x'], T['y']))
    return [(j, i)]


def floor_routes(G, Gp, targets, rooms, lim_room):
    """rooms=[{poly, area}] → 각 실: 표시 경로(화살표 사전확률)·법정 보행거리(중심·최원점)·도달 목표 수·공통 경로"""
    if not targets: return [dict(r, d=None, dmax=None, tgt=None, path=[], n_reach=0) for r in rooms]
    F = []
    for T in targets:
        cells = _targets_cells(G, T); T['_cells'] = cells
        m, cum = mcp(G, cells); F.append((m, cum))
    stack = np.stack([c for _, c in F]); best = stack.min(0); arg = stack.argmin(0)
    allc = [c for T in targets for c in T['_cells']]
    owner = {c: k for k, T in enumerate(targets) for c in T['_cells']}
    mp, _ = mcp(Gp, allc)
    out = []
    for r in rooms:
        pg = Polygon(r['poly']); rp = pg.representative_point()
        c = G.free(*G.cell(rp.x, rp.y))
        k = int(arg[c]); m, cum = F[k]
        path, d, nw = pull(G, m.traceback(c)[::-1])
        dp_cells = mp.traceback(c)[::-1]; disp, dlen, _ = pull(Gp, dp_cells)
        kd = owner.get(tuple(dp_cells[-1]), k)
        msk = G.poly_mask(r['poly'], erode=1) & (G.cost < WALL) & np.isfinite(best)
        if msk.any():
            vv = np.where(msk, best, -1); fc = np.unravel_index(int(vv.argmax()), vv.shape)
            kf = int(arg[fc]); _, dmax, nwf = pull(G, F[kf][0].traceback(fc)[::-1])
            far = G.pt(*fc)
        else: dmax, far, nwf = d, G.pt(*c), nw
        dmax = max(dmax, d)
        # 목표별 도달(벽 통과 없이) · 두 방향 공통 경로
        reach = []
        for kk, (mm, cc) in enumerate(F):
            if not np.isfinite(cc[c]): continue
            tc = mm.traceback(c)[::-1]; nwk = sum(1 for q in tc if G.cost[q] >= WALL)
            if nwk == 0: reach.append((cc[c], kk, tc))
        reach.sort(key=lambda z: z[0])
        common = None
        if len(reach) >= 2:
            ids = {targets[z[1]]['sid'] for z in reach if targets[z[1]].get('sid')}
            a, b = reach[0][2], reach[1][2]
            mb = np.zeros_like(G.cost, np.uint8); jj, ii = np.array(b).T; mb[jj, ii] = 1
            mb = cv2.dilate(mb, np.ones((9, 9), np.uint8))
            L = 0.0
            for p, q in zip(a, a[1:]):
                if not mb[q]: break
                L += math.dist(p, q) * RES
            common = round(L, 1)
        out.append(dict(poly=r['poly'], area=r['area'], c=[round(rp.x, 2), round(rp.y, 2)], far=[round(far[0], 2), round(far[1], 2)],
                        d=round(d, 1), dmax=round(dmax, 1), tgt=targets[k]['id'], disp_tgt=targets[kd]['id'], path=disp, plen=round(dlen, 1),
                        wall=bool(nw or nwf), n_reach=len(reach), common=common, **({'name': r['name']} if r.get('name') else {})))
    return out


def analyze(info, floors, stairs, elevs, plans, ext=None):
    """floors: pipeline out_floors(_fp 포함) · stairs/elevs: 층간 클러스터 · plans: 도면(img, s, tx, ty, legend)
    → {'floors': {fid: {...}}, 'entry': [...], 'law': {...}, 'flags': [...]}"""
    from fire3dbeam.export import spaces
    title = ((ext or {}).get('daejang') or {}).get('title')
    fids = sorted(floors, key=level_of)
    ground = '1F' if '1F' in floors else fids[0]
    arrows_plan = {}
    for p in plans:
        fp0 = next((floors[f]['_fp'] for f in p['floors'] if f in floors and floors[f].get('_fp') is not None), None)
        try: arrows_plan[p['i']] = detect_arrows(p['img'], p['s'], p['tx'], p['ty'], p.get('legend'), fp0)
        except Exception: arrows_plan[p['i']] = []
    ST = [dict(id=f'ST{k + 1}', sid=f'ST{k + 1}', name=f'계단{k + 1}', kind='stair', box=list(S['box']), x=S['x'], y=S['y'], floors=S['floors']) for k, S in enumerate(stairs)]
    vb = [S['box'] for S in stairs] + [E['box'] for E in elevs]
    res = dict(floors={}, entry=[], ground=ground, flags=[])
    cache = {}
    lim_note = None
    for fid in fids:
        F = floors[fid]; fp = F.get('_fp')
        if fp is None: continue
        polys = [list(g.exterior.coords) for g in ([fp] if fp.geom_type == 'Polygon' else list(fp.geoms))]
        x0, y0, x1, y1 = fp.bounds; bb = (round(x0, 1) - 4, round(y0, 1) - 4, round(x1, 1) + 4, round(y1, 1) + 4)
        vbf = [S['box'] for S in stairs if fid in S['floors']]                    # 대피 격자: 계단실만 문 미검출 대비(승강로는 벽 그대로)
        sides = stair_sides(vbf, F['symbols'])
        vbf = [snap_box(bx, sd, F['walls']) for bx, sd in zip(vbf, sides)]
        sbox = {T['id']: bx for T, bx in zip([T for T in ST if fid in T['floors']], vbf)}
        arr = arrows_plan.get(F['plan'], [])
        is_g = fid == ground
        doors = [d for d in F['doors'] if not (is_g and d.get('ext'))]            # export.build 과 같은 실 분할
        rooms = [r for r in spaces(F['walls'], doors, fp)
                 if not any(box(*b).buffer(0.3).contains(Polygon(r['poly']).representative_point()) for b in vbf + [E['box'] for E in elevs if fid in E['floors']])]
        if F.get('room_labels'): name_rooms(rooms, F['room_labels'])          # 실제 도면: 실명 OCR(fire3dbeam/realcard.py) → 실 이름(인명검색 용도 범주)
        if is_g:
            targets = [dict(id=f'X{k + 1}', kind='exit', name=f"옥외 출구 {k + 1} (폭 {d['w']:.1f}m)", x=float(d['x']), y=float(d['y']))
                       for k, d in enumerate([d for d in F['doors'] if d.get('ext')])]
        else:
            targets = [dict(T, box=sbox.get(T['id'], T['box'])) for T in ST if fid in T['floors'] and ground in T['floors']]
        lim, note, fr = limits(info, title, fid); lim_note = note
        lim_room = lim * (LAW_EVAC['exit_room_factor'] if is_g else 1.0)
        key = (F['plan'], tuple(T['id'] for T in targets), is_g, tuple(map(tuple, vbf)))
        G = make_grid(bb, polys, F['walls'], vbf, sides=sides); Gp = make_grid(bb, polys, F['walls'], vbf, arr, sides=sides)
        if key in cache and len(cache[key][0]) == len(rooms):
            rr = cache[key][1]
        else:
            rr = floor_routes(G, Gp, targets, rooms, lim_room); cache[key] = (rooms, rr)
        for r in rr:
            r = r
            if r['dmax'] is None: r['over'] = 'none'
            elif r['dmax'] > lim_room: r['over'] = 'high'
            elif fr is None and not is_g and r['dmax'] > LAW_EVAC['stair_walk_m'] and r['dmax'] <= LAW_EVAC['stair_walk_fr_m']: r['over'] = 'mid'
            else: r['over'] = None
        rec = dict(kind='피난층' if is_g else ('지하층' if level_of(fid) < 0 else '지상층'), lim=lim, lim_room=lim_room, lim_note=note, fr=fr,
                   targets=[{k: T[k] for k in ('id', 'kind', 'name', 'x', 'y')} for T in targets], rooms=rr, arrows=arr,
                   fp=[[[round(x, 2), round(y, 2)] for x, y in ring] for ring in polys], bb=list(bb),
                   stair_boxes={k: [round(v, 2) for v in bx] for k, bx in sbox.items()},
                   stair_sides={T['id']: sd for T, sd in zip([T for T in ST if fid in T['floors']], sides)})
        if is_g:
            # 피난층: 계단 → 옥외 출구(규칙 제11조①), 진입구 → 계단(진입로)
            ex = [T for T in targets]
            legs = {}
            if ex:
                tcs = {T['id']: _targets_cells(G, T) for T in ex}
                m, cum = mcp(G, [c for v in tcs.values() for c in v])
                own = {c: tid for tid, v in tcs.items() for c in v}
                for T in ST:
                    if fid not in T['floors'] or len(T['floors']) < 2: continue
                    c = G.free(*G.cell(T['x'], T['y']), lim=SOFT + 0.1)
                    tc = m.traceback(c)[::-1]; path, d, nw = pull(G, tc)
                    legs[T['id']] = dict(d=round(d, 1), path=path, exit=own.get(tuple(tc[-1])), over=d > lim, wall=nw > 0)
            rec['stair_exit'] = legs
            for k, d in enumerate([d for d in F['doors'] if d.get('ext')]):
                c0 = G.free(*G.cell(d['x'], d['y']))
                m, cum = mcp(G, [c0])
                rts = []
                for T in ST:
                    if fid not in T['floors'] or len(T['floors']) < 2: continue
                    c = G.free(*G.cell(T['x'], T['y']), lim=SOFT + 0.1)
                    if not np.isfinite(cum[c]): continue
                    path, dd, nw = pull(G, m.traceback(c))
                    rts.append(dict(stair=T['id'], name=T['name'], d=round(dd, 1), path=path, wall=nw > 0))
                rts.sort(key=lambda z: z['d'])
                res['entry'].append(dict(id=f'E{k + 1}', name=f"진입구 {k + 1} (폭 {d['w']:.1f}m)", x=d['x'], y=d['y'], floor=fid, routes=rts))
        res['floors'][fid] = rec
    # 계단 경유 전체 거리(실 → 계단 → 피난층 계단 → 옥외 출구)
    legs = (res['floors'].get(ground) or {}).get('stair_exit') or {}
    for fid, rec in res['floors'].items():
        for r in rec['rooms']:
            if fid != ground and r.get('tgt') in legs and r.get('d') is not None:
                r['total'] = round(r['d'] + legs[r['tgt']]['d'], 1); r['nfl'] = abs(level_of(fid) - level_of(ground))
        R = [r for r in rec['rooms'] if r.get('dmax') is not None]
        rec['summary'] = dict(n=len(rec['rooms']), reach=len(R), max=max([r['dmax'] for r in R], default=None),
                              over=sum(1 for r in R if r['over'] == 'high'), mid=sum(1 for r in R if r['over'] == 'mid'),
                              single=sum(1 for r in R if r['n_reach'] == 1 and len(rec['targets']) >= 2),
                              common=sum(1 for r in R if (r.get('common') or 0) > TOL_EVAC['common_path_m']),
                              wall=sum(1 for r in R if r['wall']), none=len(rec['rooms']) - len(R))
    res['law'] = dict(LAW_EVAC, note=lim_note, common_path_m=TOL_EVAC['common_path_m'])
    res['flags'] = make_flags(res)
    return res


def make_flags(res):
    fl = []
    for fid, rec in res['floors'].items():
        S = rec['summary']
        if S['over']:
            worst = max((r for r in rec['rooms'] if r['over'] == 'high'), key=lambda r: r['dmax'])
            fl.append(dict(kind='evac_over', floor=fid, n=S['over'], max=worst['dmax'], lim=rec['lim_room'], ground=rec['kind'] == '피난층', note=rec['lim_note']))
        if S['mid']:
            fl.append(dict(kind='evac_mid', floor=fid, n=S['mid'], lim=rec['lim'], note=rec['lim_note']))
        if S['single'] or S['common']:
            fl.append(dict(kind='evac_single', floor=fid, n=S['single'], common=S['common'], tol=TOL_EVAC['common_path_m']))
        if S['wall'] or S['none']:
            fl.append(dict(kind='evac_wall', floor=fid, n=S['wall'], none=S['none'], notarget=not rec['targets']))
        for sid, L in (rec.get('stair_exit') or {}).items():
            if L['over']: fl.append(dict(kind='evac_stair_exit', floor=fid, stair=sid, d=L['d'], lim=rec['lim']))
    return fl


def issue(f, ko_name):
    """evac_* 플래그 → 확인사항 문구(export.ext_issue 에서 호출)"""
    k = f['kind']; fl = ko_name(f['floor'])
    if k == 'evac_over':
        what = '거실 → 옥외 출구(피난방화규칙 제11조①, 2배)' if f.get('ground') else '거실 → 직통계단(건축법 시행령 제34조①)'
        return dict(lv='high', t='대피 보행거리 기준 초과 의심', d=f"{fl}: {what} 기준 {f['lim']:.0f}m 초과 실 {f['n']}곳(최대 {f['max']:.0f}m) — {f['note']} → 도면·현장 확인, 인명검색 우선 검토", cat='대피로')
    if k == 'evac_mid':
        return dict(lv='mid', t='대피 보행거리 30m 초과(내화구조면 적합)', d=f"{fl}: 30~50m 실 {f['n']}곳 — {f['note']}", cat='대피로')
    if k == 'evac_single':
        return dict(lv='mid', t='한 방향 피난(막다른 경로) 의심', d=f"{fl}: 직통계단 1곳에만 닿는 실 {f['n']}곳 · 두 방향 경로 공통 구간 {f['tol']:.0f}m 초과 실 {f['common']}곳 → 대피 유도·인명검색 참고", cat='대피로')
    if k == 'evac_wall':
        d = f"{fl}: 문을 못 찾아 벽을 통과한다고 가정한 실 {f['n']}곳" + (f" · 경로 없는 실 {f['none']}곳" if f['none'] else '')
        if f.get('notarget'): d += ' · 이 층에서 피난층까지 이어진 계단을 찾지 못함'
        return dict(lv='mid', t='대피로 계산 불확실(문·계단 인식)', d=d + ' → 도면으로 문 위치 확인', cat='대피로')
    if k == 'evac_stair_exit':
        return dict(lv='mid', t='피난층 계단 → 옥외 출구 보행거리 초과 의심', d=f"{fl}: {f['stair']} → 옥외 출구 {f['d']:.0f}m (기준 {f['lim']:.0f}m, 피난방화규칙 제11조①)", cat='대피로')
    return None


def name_rooms(rooms, labels):
    """실 다각형 안의 실명 글자(전역 좌표) → r['name'] (OCR 글자 우선, 여럿이면 '·' 로 이음)"""
    from shapely.geometry import Point
    for r in rooms:
        pg = Polygon(r['poly'])
        hit = [l for l in labels if pg.contains(Point(l['x'], l['y']))]
        if not hit: continue
        hit.sort(key=lambda l: 0 if l.get('src') == 'ocr' else 1)
        names = []
        for l in hit:
            if l['name'] not in names: names.append(l['name'])
        r['name'] = '·'.join(names[:3])


def to_viewer(ev):
    """뷰어용(작게): 실 폴리곤·거리·표시 경로, 화살표, 진입로"""
    if not ev or 'floors' not in ev: return None
    out = dict(law=ev['law'], ground=ev['ground'], entry=ev['entry'], floors={})
    for fid, rec in ev['floors'].items():
        out['floors'][fid] = dict(kind=rec['kind'], lim=rec['lim'], lim_room=rec['lim_room'], note=rec['lim_note'], targets=rec['targets'],
                                  arrows=rec['arrows'], summary=rec['summary'], stair_exit=rec.get('stair_exit') or {}, stair_sides=rec.get('stair_sides') or {},
                                  rooms=[dict({k: r.get(k) for k in ('poly', 'area', 'c', 'far', 'd', 'dmax', 'tgt', 'disp_tgt', 'path', 'plen', 'wall', 'n_reach', 'common', 'over', 'total', 'nfl')},
                                              **({'name': r['name']} if r.get('name') else {})) for r in rec['rooms']])
    return out


def export_ifc_evac(F, api, D, storeys, lvl, pset, H_FLOOR, SLAB):
    """대피로·진입로 → IfcAnnotation(ObjectType '대피로'|'진입로'|'피난층 계단→옥외 출구', 3D 꺾은선) + Pset_소방동선"""
    EV = D['evac']
    plan = api.run('context.add_context', F, context_type='Plan')
    ann = api.run('context.add_context', F, context_type='Plan', context_identifier='Annotation', target_view='PLAN_VIEW', parent=plan)

    def add(fid, name, otype, pts, props, dz=0.4):
        if fid not in storeys or len(pts) < 2: return
        z = lvl[fid] * H_FLOOR + SLAB
        pl = F.createIfcPolyline([F.createIfcCartesianPoint((float(x), float(y), float(dz))) for x, y in pts])
        rep = F.createIfcShapeRepresentation(ann, 'Annotation', 'Curve3D', [pl])
        el = api.run('root.create_entity', F, ifc_class='IfcAnnotation', name=name)
        el.ObjectType = otype
        api.run('spatial.assign_container', F, products=[el], relating_structure=storeys[fid])
        api.run('geometry.edit_object_placement', F, product=el, matrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]], float))
        api.run('geometry.assign_representation', F, product=el, representation=rep)
        pset(el, 'Pset_소방동선', props)
    law = EV['law']['basis']
    for fid, rec in EV['floors'].items():
        for k, r in enumerate(rec['rooms']):
            if not r.get('path'): continue
            add(fid, f"{fid} 대피로 실{k + 1}", '대피로', r['path'],
                {'종류': '대피로', '목표': r.get('tgt'), '보행거리_중심_m': r.get('d'), '보행거리_최원점_m': r.get('dmax'), '기준_m': rec['lim_room'],
                 '기준_초과': r.get('over') == 'high', '30m_초과_내화확인': r.get('over') == 'mid', '도달_계단수': r.get('n_reach'),
                 '문미검출_벽통과가정': bool(r.get('wall')), '실면적_m2': r.get('area'), '근거': law, '출처': '도면 자동 추출 벽·문 기반 추정(검수 전)'})
        for sid, L in (rec.get('stair_exit') or {}).items():
            add(fid, f"{fid} {sid} → 옥외 출구", '피난층 계단→옥외 출구', L['path'],
                {'종류': '대피로(피난층)', '계단': sid, '보행거리_m': L['d'], '기준_m': rec['lim'], '기준_초과': bool(L['over']), '근거': '건축물의 피난·방화구조 등의 기준에 관한 규칙 제11조①'}, 0.5)
    for e in EV.get('entry', []):
        for i, r in enumerate(e['routes'][:2]):
            add(e['floor'], f"{e['name']} → {r['name']}", '진입로', r['path'],
                {'종류': '진입로', '진입구': e['name'], '계단': r['stair'], '순위': i + 1, '보행거리_m': r['d'], '문미검출_벽통과가정': bool(r.get('wall'))}, 0.6)
