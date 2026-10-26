# -*- coding: utf-8 -*-
"""벽·문·실 벡터화 (축척 s px/m 을 알고 있는 한 장의 도면 기준, 결과는 도면 지역좌표 m: x=px/s, y=-py/s)"""
import numpy as np, cv2
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union


def wall_mask_scan(M, s, vlines, hlines, exclude=None):
    """(스캔 도면용) 벽 후보 잉크: 도면마다 다른 선 농도에 맞춰 자동 임계값(Otsu) + 채색 주석 제외"""
    lum = M['lum'].astype(np.uint8)
    t, _ = cv2.threshold(lum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t, 120, 205))
    ink = ((M['lum'] < thr) & (M['sat'] < 60)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = max(st[i][2], st[i][3]) >= 1.2 * s          # 글자·기호 조각 제거
    m = keep[lab].astype(np.uint8)
    # 그리드선 제거: 선 양옆(±3px)이 비어 있는 '얇은' 구간만 지운다 → 그리드 위에 놓인 굵은 벽은 보존
    Hh, Ww = m.shape
    for x in vlines:
        x = int(round(x))
        if x < 4 or x > Ww - 5: continue
        thin = (m[:, x - 4] == 0) & (m[:, x + 4] == 0)
        for dx in (-2, -1, 0, 1, 2): m[thin & (M['dark'][:, x + dx] == 0), x + dx] = 0
    for y in hlines:
        y = int(round(y))
        if y < 4 or y > Hh - 5: continue
        thin = (m[y - 4, :] == 0) & (m[y + 4, :] == 0)
        for dy in (-2, -1, 0, 1, 2): m[y + dy, thin & (M['dark'][y + dy, :] == 0)] = 0
    if exclude is not None: m[exclude > 0] = 0
    return m


GRID_KEEP_THICK = True


def wall_mask(M, s, vlines, hlines, exclude=None):
    """(CAD 출력 도면용, 기본) 진한 선 + 회색 채움 벽. 그리드선 위에서는 진한 픽셀만 남김"""
    dark = M['dark'].copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = max(st[i][2], st[i][3]) >= 1.6 * s          # 글자·기호 조각 제거
    dark = keep[lab].astype(np.uint8)
    # 2×2 열림: OpenCV morphologyEx 는 짝수 커널을 반사하지 않아 결과가 1px 오른쪽·아래로 밀린다
    # (그리드선 지우기 띠와 어긋나 문 틈을 메움) → 침식·팽창 기준점을 맞춰 제자리 열림
    k2 = np.ones((2, 2), np.uint8)
    gray = cv2.dilate(cv2.erode(M['gray'], k2, anchor=(0, 0)), k2, anchor=(1, 1))
    # 그리드선 제거: 그리드선 위에 놓인 외벽(회색 채움 + 윤곽, 3px 이상: 선 양옆 1px도 진함)은 남기고 1~2px 그리드선만 지운다
    Hh, Ww = gray.shape
    for x in vlines:
        x = int(x)
        if 3 <= x < Ww - 3:
            thick = (np.maximum(M['lum'][:, x - 1], M['lum'][:, x + 1]) < 195) & (np.maximum(M['sat'][:, x - 1], M['sat'][:, x + 1]) < 40) if GRID_KEEP_THICK else np.zeros(Hh, bool)
            gray[~thick, max(0, x - 1):x + 2] = 0
        else: gray[:, max(0, x - 1):x + 2] = 0
    for y in hlines:
        y = int(y)
        if 3 <= y < Hh - 3:
            thick = (np.maximum(M['lum'][y - 1, :], M['lum'][y + 1, :]) < 195) & (np.maximum(M['sat'][y - 1, :], M['sat'][y + 1, :]) < 40) if GRID_KEEP_THICK else np.zeros(Ww, bool)
            gray[max(0, y - 1):y + 2, ~thick] = 0
        else: gray[max(0, y - 1):y + 2, :] = 0
    m = (dark | gray).astype(np.uint8)
    if exclude is not None: m[exclude > 0] = 0
    return m


def segments(m, s, min_len=0.9, max_thick=1.3, bridge=0.25):
    L = max(5, int(min_len * s)); G = max(2, int(bridge * s))
    segs = []
    for ori in ('h', 'v'):
        kb = cv2.getStructuringElement(cv2.MORPH_RECT, (G, 1) if ori == 'h' else (1, G))
        mm = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kb)                 # 스캔으로 끊긴 선 잇기(문 폭보다 훨씬 작은 틈만)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if ori == 'h' else (1, L))
        o = cv2.morphologyEx(mm, cv2.MORPH_OPEN, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if ori == 'h':
                if h > max_thick * s:
                    if SPLIT_THICK: segs += _thin_parts(lab[y:y + h, x:x + w] == i, x, y, 'h', s, max_thick, L)
                    continue
                segs.append(dict(o='h', c=-(y + h / 2) / s, a=x / s, b=(x + w) / s, t=max(0.12, h / s)))
            else:
                if w > max_thick * s:
                    if SPLIT_THICK: segs += _thin_parts(lab[y:y + h, x:x + w] == i, x, y, 'v', s, max_thick, L)
                    continue
                segs.append(dict(o='v', c=(x + w / 2) / s, a=-(y + h) / s, b=-y / s, t=max(0.12, w / s)))
    return segs


SPLIT_THICK = True


def _thin_parts(sub, x0, y0, ori, s, max_thick, L):
    """두꺼운 덩어리(외벽 + 붙은 계단실·승강기 벽·뭉개진 디딤판)를 통째로 버리지 않고,
    축을 따라 두께가 max_thick 이하인 구간(외벽이 홀로 지나가는 부분)만 선분으로 되살린다."""
    if ori == 'v': sub = sub.T
    has = sub.any(0)
    top = np.argmax(sub, 0); bot = sub.shape[0] - 1 - np.argmax(sub[::-1], 0)
    ext = np.where(has, bot - top + 1, 10 ** 6)
    thin = ext <= max_thick * s
    out, j, n = [], 0, len(thin)
    while j < n:
        if thin[j]:
            k = j
            while k < n and thin[k]: k += 1
            if k - j >= L:
                cc = float(np.median((top[j:k] + bot[j:k]) / 2.0)) + 0.5; tt = float(np.median(ext[j:k]))
                if ori == 'h': out.append(dict(o='h', c=-(y0 + cc) / s, a=(x0 + j) / s, b=(x0 + k) / s, t=max(0.12, tt / s)))
                else: out.append(dict(o='v', c=(x0 + cc) / s, a=-(y0 + k) / s, b=-(y0 + j) / s, t=max(0.12, tt / s)))
            j = k
        else: j += 1
    return out


def merge(segs, tol_c=0.45, gap_join=0.45, min_len=0.9):
    out = []
    for o in ('h', 'v'):
        S = sorted([x for x in segs if x['o'] == o], key=lambda x: x['c'])
        groups = []
        for x in S:
            if groups and abs(x['c'] - groups[-1][-1]['c']) <= tol_c: groups[-1].append(x)
            else: groups.append([x])
        for gset in groups:
            cs = [x['c'] for x in gset]
            iv = sorted([(x['a'], x['b'], x['t']) for x in gset])
            cur = list(iv[0]); merged = []
            for a, b, t in iv[1:]:
                if a <= cur[1] + gap_join: cur[1] = max(cur[1], b); cur[2] = max(cur[2], t)
                else: merged.append(cur); cur = [a, b, t]
            merged.append(cur)
            th = float(np.clip(max(cs) - min(cs) + 0.15, 0.15, 0.5))
            cc = float(np.mean(cs))
            for a, b, t in merged:
                if b - a >= min_len: out.append(dict(o=o, c=round(cc, 3), a=round(a, 3), b=round(b, 3), t=round(th, 2)))
    return out


def _ends(w):
    return ((w['a'], w['c']), (w['b'], w['c'])) if w['o'] == 'h' else ((w['c'], w['a']), (w['c'], w['b']))


def _line(w):
    p, q = _ends(w); return LineString([p, q])


def clean(walls, keep_frac=0.12):
    """떨어진 작은 무리(제목상자·범례·문짝선) 제거 + 짧고 한쪽이 떠 있는 선 제거"""
    if not walls: return walls
    lines = [_line(w) for w in walls]
    n = len(walls); parent = list(range(n))
    def f(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = walls[i], walls[j]
            # 같은 선 위에서 개구부(출입구·셔터 ≤8m)로 끊긴 벽 조각도 한 무리로 본다
            coll = a['o'] == b['o'] and abs(a['c'] - b['c']) < 0.3 and max(a['a'], b['a']) - min(a['b'], b['b']) < 8.0
            if coll or lines[i].distance(lines[j]) < 0.8: parent[f(i)] = f(j)
    tot = {}
    for i in range(n): tot[f(i)] = tot.get(f(i), 0) + lines[i].length
    mx = max(tot.values())
    walls = [w for i, w in enumerate(walls) if tot[f(i)] >= keep_frac * mx]
    lines = [_line(w) for w in walls]
    out = []
    for i, w in enumerate(walls):
        if w['b'] - w['a'] < 1.3:
            p, q = [Point(e) for e in _ends(w)]
            others = [l for j, l in enumerate(lines) if j != i]
            if not (any(l.distance(p) < 0.35 for l in others) and any(l.distance(q) < 0.35 for l in others)):
                continue
        out.append(w)
    return out


def rescue_jambs(walls, cands, lmin=0.45, lmax=1.3, corner=0.35, gmin=0.6, gmax=2.2, tol_c=0.15):
    """문틀 옆 짧은 벽(모서리~문 틈 사이 0.45~1.3 m)은 짧아서 버려지기 쉽다(계단실·승강기·세대 현관 문 옆).
    되살리는 조건: 직교하는 벽과 만나고(모서리·T자), 한쪽 끝은 같은 선 위 벽과 문 폭(0.6~2.2 m) 틈을 두고 마주봄,
    같은 선 위 기존 벽과 겹치지 않음 → 벽 틈 문 검출이 가능해진다."""
    out = list(walls)
    for w in sorted(cands, key=lambda w: -(w['b'] - w['a'])):
        L = w['b'] - w['a']
        if not (lmin <= L <= lmax): continue
        same = [v for v in out if v['o'] == w['o'] and abs(v['c'] - w['c']) < tol_c + 0.5 * max(0.0, v.get('t', 0.2) - 0.2)]
        if any(min(v['b'], w['b']) - max(v['a'], w['a']) > 0.1 for v in same): continue       # 이미 있는 벽
        perp = [v for v in out if v['o'] != w['o'] and v['a'] - corner <= w['c'] <= v['b'] + corner]
        # 직교 벽과 만남: 끝(ㄱ자) 또는 몸통(ㅜ자 — 기호에 가려 끝이 조금 삐져나온 경우)
        if not any(w['a'] - corner <= v['c'] <= w['b'] + corner for v in perp): continue
        ok = False
        for v in same:                       # 한쪽 끝 너머 같은 선 벽과 문 폭 틈
            gap = max(v['a'] - w['b'], w['a'] - v['b'])
            if gmin <= gap <= gmax: ok = True; break
        if ok: out.append(dict(w, src='jamb'))
    return out


FP_EXT = 1.0


def footprint_from_walls(walls):
    """외곽 윤곽 추정: 벽을 굵게 칠한 뒤 닫고 외곽선"""
    if not walls: return None
    xs = [x for w in walls for e in _ends(w) for x in (e[0],)]; ys = [e[1] for w in walls for e in _ends(w)]
    x0, y0, x1, y1 = min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2
    R = 4
    W, H = int((x1 - x0) * R) + 1, int((y1 - y0) * R) + 1
    img = np.zeros((H, W), np.uint8)
    for w in walls:
        # 양 끝을 FP_EXT m 늘려 그림: 모서리에서 두 벽 끝이 조금씩 모자라(계단실·기호에 가려) ㄱ자 틈이 남으면 닫힘 연산으로도 안 닫힘.
        # 바깥으로 삐져나온 부분은 아래 열림 연산(2.5 m)으로 지워진다.
        e = FP_EXT
        (a, b), (c, d) = _ends(dict(w, a=w['a'] - e, b=w['b'] + e))
        cv2.line(img, (int((a - x0) * R), int((y1 - b) * R)), (int((c - x0) * R), int((y1 - d) * R)), 1, max(1, int(w['t'] * R)))
    for o in ('h', 'v'):                                    # 같은 선 위 벽 사이 틈(출입구·셔터 ≤8m)은 이어서 외곽을 닫는다
        S = sorted([w for w in walls if w['o'] == o], key=lambda w: (w['c'], w['a']))
        for p, q in zip(S, S[1:]):
            if abs(p['c'] - q['c']) < 0.3 and 0 < q['a'] - p['b'] <= 8.0:
                c = (p['c'] + q['c']) / 2
                a, b = ((p['b'], c), (q['a'], c)) if o == 'h' else ((c, p['b']), (c, q['a']))
                cv2.line(img, (int((a[0] - x0) * R), int((y1 - a[1]) * R)), (int((b[0] - x0) * R), int((y1 - b[1]) * R)), 1, 2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(img); cv2.drawContours(fill, cnts, -1, 1, -1)
    k = int(2.5 * R) | 1                                    # 외곽 밖으로 삐져나온 선(그리드·조경선) 제거
    op = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if op.sum() >= 0.6 * fill.sum(): fill = op             # 외곽이 안 닫힌 도면(대공간 등)은 깎지 않음
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 40 * R * R: continue
        ap = cv2.approxPolyDP(c, 2, True)[:, 0, :]
        if len(ap) >= 3:
            pg = Polygon([(x0 + px / R, y1 - py / R) for px, py in ap]).buffer(0)
            if pg.area > 20: polys.append(pg)
    return unary_union(polys) if polys else None


def clip_walls(walls, fp, pad=0.8):
    """건물 외곽 밖으로 나간 벽 조각 잘라내기"""
    if fp is None: return walls
    A = fp.buffer(pad); out = []
    for w in walls:
        g = _line(w).intersection(A)
        for seg in ([g] if g.geom_type == 'LineString' else list(getattr(g, 'geoms', []))):
            if seg.geom_type != 'LineString' or seg.length < 0.5: continue
            (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
            v = dict(w)
            if w['o'] == 'h': v['a'], v['b'] = min(x0, x1), max(x0, x1)
            else: v['a'], v['b'] = min(y0, y1), max(y0, y1)
            out.append(v)
    return out


def gap_profile(M, s, o, c, a, b, t=0.2):
    """(도면 지역좌표) 벽 선 위 구간 a..b 의 (채색 기호가 벽 축을 덮은 길이 m, 비어 있는(흰) 길이 m, 구간 양 끝 0.3 m 안이 가려졌나)"""
    lum, sat = M['lum'], M['sat']
    H, W = lum.shape
    hw = int(round(max(2.0, t * s / 2 + 1)))
    if o == 'h':
        y = int(round(-c * s)); x0, x1 = int(round(a * s)), int(round(b * s))
        if y - hw < 0 or y + hw >= H or x1 <= x0: return 0.0, 0.0, False, (0.0, b - a)
        L, S_ = lum[y - hw:y + hw + 1, max(0, x0):min(W, x1)], sat[y - hw:y + hw + 1, max(0, x0):min(W, x1)]
        ax = 0
    else:
        x = int(round(c * s)); y0, y1 = int(round(-b * s)), int(round(-a * s))
        if x - hw < 0 or x + hw >= W or y1 <= y0: return 0.0, 0.0, False, (0.0, b - a)
        L, S_ = lum[max(0, y0):min(H, y1), x - hw:x + hw + 1], sat[max(0, y0):min(H, y1), x - hw:x + hw + 1]
        ax = 1
    mid = S_[hw - 1:hw + 2] if ax == 0 else S_[:, hw - 1:hw + 2]      # 벽 축 ±1px: 기호가 벽 선 자체를 덮었나(옆에 붙은 표지는 제외)
    col = (mid >= 60).any(axis=ax)
    white = ((L > 212) & (S_ < 40)).mean(axis=ax) >= 0.7
    e = max(1, int(round(0.3 * s)))
    ends = bool(col[:e].any() or col[-e:].any())
    # 가리지 않은 가장 긴 구간(문 자리 추정) — 벽 축 좌표(a 기준 m, 화면 y 축은 뒤집힘)
    best, i, n = (0, 0), 0, len(col)
    while i < n:
        if not col[i]:
            j = i
            while j < n and not col[j]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j
        else: i += 1
    if o == 'h': free = ((max(0, x0) + best[0]) / s - a, (max(0, x0) + best[1]) / s - a)
    else: free = (-(max(0, y0) + best[1]) / s - a, -(max(0, y0) + best[0]) / s - a)
    return float(col.sum()) / s, float((white & ~col).sum()) / s, ends, free


def gap_doors(walls, fp, dmin=0.7, dmax=2.0, occ=None):
    """occ(o, c, a, b, t) → (가려진 m, 빈 m, 틈 양 끝 가림 여부, 덮은 기호 종류|None): 채색 기호(비상구·소화전 등)가 벽을 가린 틈 판별(전역 좌표). 없으면 틈 길이만으로."""
    doors = []
    for o in ('h', 'v'):
        S = sorted([w for w in walls if w['o'] == o], key=lambda w: (w['c'], w['a']))
        i = 0
        while i < len(S):
            j = i
            while j + 1 < len(S) and abs(S[j + 1]['c'] - S[i]['c']) < 0.1: j += 1
            line = sorted(S[i:j + 1], key=lambda w: w['a'])        # 같은 선(c 차 < 0.1) 안에서는 위치 순(c 순이면 틈 짝이 어긋남)
            for p, q in zip(line, line[1:]):
                gap = q['a'] - p['b']
                if fp is not None:
                    B = fp.boundary; mp, mq = _line(p).interpolate(0.5, True), _line(q).interpolate(0.5, True)
                    on_edge = B.distance(mp) < 0.8 and B.distance(mq) < 0.8     # 외벽선 위의 틈만(외곽에 닿는 칸막이벽 제외)
                else: on_edge = False
                lim = 7.0 if on_edge else dmax
                if occ is not None and dmin <= gap <= lim + 1.2 and (p['b'] - p['a']) >= 0.5 and (q['b'] - q['a']) >= 0.5:
                    cov, emp, ends, free, kind = occ(o, (p['c'] + q['c']) / 2, p['b'], q['a'], min(p['t'], q['t']))
                    if kind not in (None, 'exit') and cov >= 0.75 * gap and emp < 0.45: continue     # 소화전·수신기 등 기호가 벽 선을 덮은 것(문 아님)
                    if gap > lim and not (kind == 'exit' and ends and cov >= 0.3): continue   # 비상구 표지(= 문 자리)가 문틀을 덮어 틈이 넓어 보이는 문만 허용
                    fw = free[1] - free[0]
                    if kind == 'exit' and ends and cov >= 0.3 and dmin - 0.1 <= fw <= lim:     # 표지에 가리지 않은 부분 = 문 자리
                        mid, width = p['b'] + (free[0] + free[1]) / 2, fw
                    else: mid, width = (p['b'] + q['a']) / 2, gap
                elif not (dmin <= gap <= lim): continue
                else: mid, width = (p['b'] + q['a']) / 2, gap
                if (p['b'] - p['a']) >= 0.5 and (q['b'] - q['a']) >= 0.5:
                    x, y = (mid, p['c']) if o == 'h' else (p['c'], mid)
                    if fp is None or fp.buffer(0.5).contains(Point(x, y)):
                        doors.append(dict(o=o, x=round(x, 2), y=round(y, 2), w=round(width, 2), kind='opening' if on_edge else 'door', ext=bool(on_edge)))
            i = j + 1
    return doors


def split_thin(walls, M, s, min_gap=0.7, max_gap=7.0):
    """벽 가운데 '두께가 사라진' 구간(문·셔터 자리에 1px 그리드선만 지나가는 곳)을 찾아 벽을 끊는다.
    채색 기호·대피선(채도 높음)은 벽을 가린 것으로 보아 끊지 않는다. 도면 지역좌표(m) 벽 → 같은 형식."""
    lum, sat = M['lum'], M['sat']
    H, W = lum.shape
    ink = ((lum < 205) | (sat >= 60)).astype(np.uint8)
    out = []
    for w in walls:
        L = w['b'] - w['a']
        hw = int(round(max(2.0, w['t'] * s / 2 + 1.5)))
        if L < 2 * min_gap + 1.0:
            out.append(w); continue
        if w['o'] == 'h':
            y = int(round(-w['c'] * s)); x0, x1 = int(round(w['a'] * s)), int(round(w['b'] * s))
            if y - hw < 0 or y + hw >= H: out.append(w); continue
            prof = ink[y - hw:y + hw + 1, max(0, x0):min(W, x1)].sum(0); off = max(0, x0)
        else:
            x = int(round(w['c'] * s)); y0, y1 = int(round(-w['b'] * s)), int(round(-w['a'] * s))
            if x - hw < 0 or x + hw >= W: out.append(w); continue
            prof = ink[max(0, y0):min(H, y1), x - hw:x + hw + 1].sum(1); off = max(0, y0)
        if len(prof) < 5: out.append(w); continue
        med = float(np.median(prof))
        if med < 2.5: out.append(w); continue                 # 원래 얇은 선 벽은 그리드선과 구분 불가 → 그대로
        thin = (prof <= max(1, 0.4 * med)).astype(np.uint8)
        thin = cv2.morphologyEx(thin[None, :], cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]   # 문짝선 1~2px 끊김 무시
        runs, i = [], 0
        while i < len(thin):
            if thin[i]:
                j = i
                while j < len(thin) and thin[j]: j += 1
                if min_gap * s <= j - i <= max_gap * s and i > 0 and j < len(thin): runs.append((i, j))
                i = j
            else: i += 1
        if not runs: out.append(w); continue
        # 픽셀 구간 → 벽 좌표(a..b) 구간
        cuts = []
        for i, j in runs:
            p0, p1 = (off + i) / s, (off + j) / s
            cuts.append((p0, p1) if w['o'] == 'h' else (-p1, -p0))
        cur = w['a']
        for c0, c1 in sorted(cuts):
            if c0 - cur >= 0.5: out.append(dict(w, a=round(cur, 3), b=round(c0, 3)))
            cur = max(cur, c1)
        if w['b'] - cur >= 0.5: out.append(dict(w, a=round(cur, 3), b=w['b']))
    return out
