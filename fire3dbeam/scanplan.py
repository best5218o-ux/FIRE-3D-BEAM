# -*- coding: utf-8 -*-
"""스캔본(글자층 없는 그림 PDF) 정보카드에서 잘라낸 평면도 — 규칙 경로(vision·geom) 앞 전처리와 축척(치수) 판독 보강.
pipeline.run 이 평면도 출처가 OCR(그림 `source='ocr'`)일 때만 부른다 → 글자층 카드 결과는 바이트 단위로 그대로.
끄기: FIRE3DBEAM_SCANPLAN=0 (스캔본도 예전 규칙 경로 그대로)."""
import os
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np, cv2

ENABLED = os.environ.get('FIRE3DBEAM_SCANPLAN', '1') != '0'
GRID = os.environ.get('SCANPLAN_GRID', '1') != '0'
DIMS = os.environ.get('SCANPLAN_DIMS', '1') != '0'
GRIDWALL = os.environ.get('SCANPLAN_GRIDWALL', '0') != '0'       # 그리드선 잔여 지우기: dev 에서 벽 재현율·진입구 손해(칸막이벽이 그리드선 위) → 기본 끔
MAINFP = os.environ.get('SCANPLAN_MAINFP', '1') != '0'
MODEL_MAX_PX = float(os.environ.get('SCANPLAN_MODEL_MAX_PX', '2e7'))   # 모델 작업 영상(1 m≈29 px) 화소 상한 — 넘으면 그 평면도는 모델 건너뜀(보통 3~6M)


def model_default():
    """스캔본 카드의 모델 경로 기본값: FIRE3DBEAM_SCAN_MODEL(기본 1) 이고 STR 모델 파일·onnxruntime 이 있으면 True"""
    if os.environ.get('FIRE3DBEAM_SCAN_MODEL', '1') == '0' or not ENABLED: return False
    try:
        from fire3dbeam import str_seg
        if not os.path.exists(str_seg.MODEL): return False
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _env(k, d):
    v = os.environ.get(k)
    return type(d)(v) if v not in (None, '') else d


# ───────────────────────────── 그리드선(스캔: 남은 기울기·흐림으로 1px 열을 벗어나는 긴 선 허용)
def grid_lines(M, shape, seg=0.05, cover=0.4):
    """vision.grid_lines 의 스캔판: 그림을 거의 가로지르는 가는 회색 선 위치(px).
    (1) 가로 3px 팽창 뒤 짧은(그림 높이 5%) 세로 조각만 남기고 열마다 조각 화소를 더함 → 범례·나무·글자에 끊긴 선도 한 줄(합 ≥ 40%)
    (2) 남은 기울기로 선이 한 열 옆으로 옮겨 가도 한 줄 (3) 그림 가장자리 3px(잘라낸 틀 흔적) 제외. 반환 (세로선 x, 가로선 y)"""
    H, W = shape[:2]
    m = ((M['lum'] < 228) & (M['sat'] < 40)).astype(np.uint8)
    e = 3
    m[:e] = 0; m[-e:] = 0; m[:, :e] = 0; m[:, -e:] = 0
    def peaks(prof, thr):
        idx = np.where(prof > thr)[0]; out = []
        for i in idx:
            if out and i - out[-1][-1] <= 2: out[-1].append(i)
            else: out.append([i])
        return [float(np.average(g, weights=prof[g])) for g in out]
    d = cv2.dilate(m, np.ones((1, 3), np.uint8))
    o = cv2.morphologyEx(d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(H * seg)))))
    o = cv2.erode(o, np.ones((1, 3), np.uint8))
    vl = peaks(o.sum(0).astype(np.float32), H * cover)
    d = cv2.dilate(m, np.ones((3, 1), np.uint8))
    o = cv2.morphologyEx(d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(W * seg)), 1)))
    o = cv2.erode(o, np.ones((3, 1), np.uint8))
    hl = peaks(o.sum(1).astype(np.float32), W * cover)
    return vl, hl


# ───────────────────────────── 전처리
def blue_loose(img):
    from fire3dbeam.vision import blue_loose as bl
    return bl(img)


def _invgamma(img):
    """그림 자르기(card_ocr.fig_crop)가 무채색 화소에 건 감마(기본 1.5 — 옅어진 가는 선 살리기) 되돌리기 → float32.
    감마 영상에서는 회색 그리드선(원본 150)이 '진한 선'(lum<125)으로 바뀌어 벽으로 잡혔다."""
    from fire3dbeam.card_ocr import FIG_GAMMA, FIG_NORM
    c = img.astype(np.float32)
    if FIG_GAMMA and FIG_NORM:
        sat = c.max(2) - c.min(2)
        c = np.where((sat < 40)[..., None], 255.0 * (c / 255.0) ** (1.0 / FIG_GAMMA), c)
    return c


GEO_SHARP, GEO_SIG = 0.8, 1.0           # 벽·문용 윤곽 선명화(흐림 보정) — dev 에서만 고름(문 R .61→.64, 진입구·계단 ↑)


def prep(img):
    """잘라낸 스캔 평면도(BGR) → (geo, sym): 같은 크기·같은 좌표, 화소 값만 다름.
    geo = 감마 되돌림 + 윤곽 선명화 → 벽·문·외곽(회색 그리드선이 다시 '회색'이 되어 기존 그리드선 지우기가 동작)
    sym = 감마 되돌림 + 잉크 검정 수준(스캔에서 짙은 회색 10~45) → 0 늘이기 → 범례·소방기호·계단 디딤판(채도·명암 대비 회복)"""
    c = _invgamma(img)
    a = _env('SCANPREP_SHARP', GEO_SHARP)
    g8 = np.clip(c, 0, 255).astype(np.uint8)
    if a > 0:
        bl = cv2.GaussianBlur(g8, (0, 0), _env('SCANPREP_SIG', GEO_SIG))
        geo = cv2.addWeighted(g8, 1 + a, bl, -a, 0)
    else: geo = g8
    lum = c.mean(2)
    k = lum < 200
    b = float(np.percentile(lum[k], 1.0)) if k.sum() > 200 else 0.0
    b = min(max(b, 0.0), 80.0)
    sym = np.clip((c - b) / (255.0 - b) * 255.0, 0, 255).astype(np.uint8)
    return geo, sym


# ───────────────────────────── 치수 글자(그리드 경간 '8,400') — 스캔 흐림에서도 읽기
# 예전 판독(vision.scale_from_dims)은 도면 위쪽 절반 띠 전체를 2.5배로 키워 psm 11 로 한 번 읽는다 → 150~200 dpi 스캔의 6~7px 회색 글자는 거의 못 읽음.
# 여기서는 (1) 바탕 평탄화 영상에서 글자 덩어리를 찾아 '경간 가운데에 놓인 글자'가 2개 이상 같은 높이에 늘어선 줄 = 치수 줄로 보고,
# (2) 그 줄의 모든 경간 가운데에 같은 크기 창을 만들어(못 찾은 글자도 포함) (3) 창마다 명암 늘이기 + 4·5·6·8배 확대(+Otsu) 판독 —
# 설정마다 창들을 세로로 쌓은 한 장을 tesseract 1회(psm 6·4; 따로따로 psm 7 보다 작은 흐린 숫자를 훨씬 잘 읽음) + 창들을 겹쳐 평균한 한 장(잡음 감소)
# (4) 'N,NNN' 꼴만 투표. 면적 축척과 1.25배 넘게 어긋난 해석(8→4 첫 자리 오독)은 버리고, 100 mm 배수가 아닌 값은 가중을 낮춰 건물 합의(fuse_scale).
DIM_CONF = [(4, 0, 6), (6, 0, 6), (6, 1, 6), (8, 0, 6), (5, 0, 4)]       # (확대 배율, Otsu 이진화, tesseract psm)


def _flatten(g):
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(np.float32)
    return np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)


def _words(n, vl, thr=215):
    """평탄화 영상 → 경간 가운데에 놓인 글자 덩어리 [{box, bay, cy}] (긴 선 지운 뒤 글자 조각을 가로로 이어 붙임)"""
    ink = (n < thr).astype(np.uint8)
    lh = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 25), np.uint8))
    lv = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((25, 1), np.uint8))
    ink = ink & ~(lh | lv)
    nn, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    cs = sorted([list(st[i][:4]) for i in range(1, nn) if 1 <= st[i][3] <= 24 and st[i][2] <= 30 and st[i][4] >= 2], key=lambda c: c[0])
    W = []
    for c in cs:
        for w in W:
            ov = min(w[1] + w[3], c[1] + c[3]) - max(w[1], c[1])
            if (ov > 0.3 * min(w[3], c[3]) or (c[3] <= 4 and ov > 0)) and c[0] - (w[0] + w[2]) < 1.4 * max(w[3], c[3], 4):
                x0, y0 = min(w[0], c[0]), min(w[1], c[1]); x1, y1 = max(w[0] + w[2], c[0] + c[2]), max(w[1] + w[3], c[1] + c[3])
                w[:] = [x0, y0, x1 - x0, y1 - y0]; break
        else: W.append(list(c))
    out = []
    for x, y, w, h in W:
        if not (4 <= h <= 24 and 1.5 <= w / h <= 8): continue
        cx = x + w / 2
        for a, b in zip(vl, vl[1:]):
            if b - a > 10 and abs(cx - (a + b) / 2) < 0.12 * (b - a) and w < 0.8 * (b - a):
                out.append(dict(box=(x, y, w, h), bay=b - a, cy=y + h / 2)); break
    return out


def dim_windows(n, vl):
    """치수 줄(경간 가운데 글자 ≥2개가 같은 높이) → 그 줄의 모든 경간 가운데 창 [{win, bay}]. 줄이 여럿이면 글자가 많은 줄 순서로 최대 2줄"""
    W = _words(n, vl)
    if len(W) < 2: return []
    from collections import Counter
    rows = Counter(int(round(w['cy'] / 3)) for w in W)
    out, used = [], []
    for key, cnt in rows.most_common():
        cy = key * 3
        if any(abs(cy - u) < 8 for u in used): continue
        rw = [w for w in W if abs(w['cy'] - cy) < 4]
        if len(rw) < 2: continue
        used.append(cy)
        hh = float(np.median([w['box'][3] for w in rw])); ww = float(np.median([w['box'][2] for w in rw]))
        for a, b in zip(vl, vl[1:]):
            if b - a < max(10, ww): continue
            cx = (a + b) / 2
            x0, x1 = int(cx - ww / 2 - max(5, ww * 0.25)), int(cx + ww / 2 + max(5, ww * 0.25))
            y0, y1 = int(cy - hh / 2 - max(4, hh * 0.7)), int(cy + hh / 2 + max(4, hh * 0.7))
            if y0 < 0 or x0 < 0 or x1 > n.shape[1] or y1 > n.shape[0]: continue
            out.append(dict(win=(x0, y0, x1, y1), bay=b - a, a=a, b=b))
        if len(used) >= 2: break
    return out


def _parse_mm(t):
    import re
    t = (t or '').replace(' ', '').replace('.', ',')
    m = re.fullmatch(r'(\d{1,2}),?(\d{3})', t)
    if not m: return None
    v = int(m.group(1) + m.group(2))
    return v if 1500 <= v <= 20000 else None


def _stack_read(tiles, psm):
    """창 조각들을 세로로 쌓아(사이 흰 띠) tesseract 1회 → 조각별 글자. 같은 크기 글자 줄이 여럿 있어야 작은 흐린 숫자를 더 잘 읽는다"""
    from fire3dbeam.vision import ocr_tsv
    Wm = max(t.shape[1] for t in tiles) + 40
    parts, spans, y = [np.full((20, Wm), 255, np.uint8)], [], 20
    for t in tiles:
        row = np.full((t.shape[0], Wm), 255, np.uint8); row[:, 20:20 + t.shape[1]] = t
        gap = max(30, t.shape[0])
        parts += [row, np.full((gap, Wm), 255, np.uint8)]; spans.append((y, y + t.shape[0])); y += t.shape[0] + gap
    rs = ocr_tsv(np.vstack(parts), psm=psm)
    return [''.join(r['text'] for r in sorted(rs, key=lambda r: r['x']) if a - 5 <= r['y'] + r['h'] / 2 <= b + 5) for a, b in spans]


DIM_STACK_U, DIM_STACK_F = 4, (0.5, 0.75)


def _stack(crops, U=DIM_STACK_U):
    """명암 늘인 창 조각들 → U배 확대 후 첫 조각 기준 정렬(형판 맞춤)·평균(2회 반복). 반환 (평균 영상 uint8 | None, 겹친 창 번호)"""
    ups = [cv2.resize(c, None, fx=U, fy=U, interpolation=cv2.INTER_CUBIC).astype(np.float32) for c in crops]
    h = min(u.shape[0] for u in ups); w = min(u.shape[1] for u in ups)
    if h <= 6 * U or w <= 8 * U: return None, []
    ups = [u[:h, :w] for u in ups]
    ref, idx, good = ups[0], [], []
    for _ in range(2):
        T = ref[2 * U:h - 2 * U, 3 * U:w - 3 * U]
        if T.std() < 1: return None, []
        acc = []
        for i, u in enumerate(ups):
            r = cv2.matchTemplate(255 - u, 255 - T, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(r)
            acc.append((mv, i, u[ml[1]:ml[1] + T.shape[0], ml[0]:ml[0] + T.shape[1]]))
        good = [a for a in acc if a[0] > 0.6]
        if len(good) < 3: return None, []
        avg = np.mean([a[2] for a in good], 0)
        ref = np.pad(avg, ((2 * U, 2 * U), (3 * U, 3 * U)), constant_values=255)
        idx = [a[1] for a in good]
    return np.clip(avg, 0, 255).astype(np.uint8), idx


def read_dims(img, vl):
    """→ [{mm, bay, votes, a, b}] 창마다 투표 1위(판독 없으면 뺌)"""
    n = _flatten(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    wins = dim_windows(n, vl)
    if not wins: return []
    crops = []
    for w in wins:
        x0, y0, x1, y1 = w['win']
        c = n[y0:y1, x0:x1]
        lo, hi = np.percentile(c, 2), np.percentile(c, 75)
        crops.append(np.clip((c - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8))
    reads = [[] for _ in wins]
    for f, bz, psm in DIM_CONF:
        tiles = []
        for c in crops:
            up = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            if bz: up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            tiles.append(up)
        for i, t in enumerate(_stack_read(tiles, psm)): reads[i].append(t)
    if len(crops) >= 3:                        # 같은 줄 창들을 겹쳐 평균(부분 화소 정렬) → 잡음이 준 한 장을 더 읽어 맞춰진 창마다 표를 보탬
        st, idx = _stack(crops)
        if st is not None:
            got = []
            for f in DIM_STACK_F:
                t = cv2.resize(st, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                got.append(_stack_read([t], 7)[0])
            for i in idx: reads[i] += got
    out = []
    for w, rd in zip(wins, reads):
        vs = [v for v in (_parse_mm(t) for t in rd) if v]
        if not vs: continue
        best = max(set(vs), key=vs.count)
        out.append(dict(mm=best, votes=vs.count(best), bay=w['bay'], a=w['a'], b=w['b']))
    return out


def scale_from_dims(img, vl, s_prior=None, mm_out=None, prior_tol=1.25, reads_out=None):
    """스캔 평면도 축척(px/m): read_dims 판독 중 (정보카드 면적 축척 s_prior 와 prior_tol 배 안) 투표 1위 값의 창들로.
    반환 (s|None, vals[px/m]). mm_out 에 쓴 치수(mm)를 담는다(건물 합의 refine_scale 용)"""
    R = read_dims(img, vl)
    if s_prior:
        R = [r for r in R if abs(np.log((r['bay'] / (r['mm'] / 1000.0)) / s_prior)) < np.log(prior_tol)]
    if reads_out is not None: reads_out += R
    if not R: return None, []
    tally = {}
    for r in R: tally[r['mm']] = tally.get(r['mm'], 0) + r['votes'] * round_w(r['mm'])
    top = max(tally, key=tally.get)
    good = [r for r in R if r['mm'] == top]
    vals = [r['bay'] / (top / 1000.0) for r in good]
    if mm_out is not None: mm_out += [top] * len(good)
    if len(good) >= 2 or tally[top] >= 3:
        return float(np.median(vals)), vals
    return None, vals


def round_w(mm):
    """경간 값의 '반듯함' 가중: 100 mm 배수 1 · 50 mm 배수 0.7 · 그 밖 0.2 (오독 '7,218'·'7,388' 이 합의를 이기지 않게)"""
    return 1.0 if mm % 100 == 0 else (0.7 if mm % 50 == 0 else 0.2)


FUSE_TOL = 1.25        # 면적 축척과 이만큼(배) 넘게 어긋난 치수 해석은 버림(8,400 을 4,400 으로 읽는 등 첫 자리 오독)


def fuse_scale(plans, prior_tol=FUSE_TOL):
    """스캔본 건물 치수 합의: 모든 스캔 평면도의 창 판독 투표를 더해 1위 경간(mm)을 정하고(2위의 1.5배 이상·3표 이상),
    평면도마다 축척 = 등간격 그리드 간격(최소제곱) ÷ 경간 → 없으면 이 도면 창들의 경간 px 중앙값 ÷ 경간 → 없으면 면적 축척에 가까운 그리드 간격.
    정보카드 면적 축척(p['_sa'])과 prior_tol 배 넘게 어긋나면 바꾸지 않음. 도면 자체가 다른 값을 뚜렷이(창 2개 이상·건물 1위 표의 절반 이상) 읽었으면 그 값 유지
    (지하 주차장 경간이 다른 건물 등 — 첫 자리 오독 8→6·4 한두 창은 건물 합의가 이김)."""
    from fire3dbeam.pipeline import grid_pitch
    sc = [p for p in plans if p.get('scan')]
    tally = {}
    for p in sc:
        for r in p.get('dim_reads') or []: tally[r['mm']] = tally.get(r['mm'], 0) + r['votes'] * round_w(r['mm'])
    if not tally: return None
    rk = sorted(tally.items(), key=lambda kv: -kv[1])
    top, n1 = rk[0]; n2 = rk[1][1] if len(rk) > 1 else 0
    if n1 < 3 or n1 < 1.5 * n2: return None
    bay = top / 1000.0
    lt = np.log(prior_tol)
    for p in sc:
        sa = p.get('_sa'); R = p.get('dim_reads') or []
        own_t = {}
        for r in R: own_t[r['mm']] = own_t.get(r['mm'], 0) + r['votes'] * round_w(r['mm'])
        if own_t:
            ot = max(own_t, key=own_t.get)
            if ot != top and own_t[ot] >= max(3, 0.5 * n1) and sum(1 for r in R if r['mm'] == ot) >= 2 and p.get('scale_src') == 'dims': continue
        own = [r['bay'] for r in R if r['mm'] == top]
        pit = grid_pitch(p['vlines'])
        cand = None
        if own:
            so = float(np.median(own)) / bay
            cand = (pit / bay if pit and abs(pit / bay / so - 1) < 0.03 else so, 'dims')
        elif pit:
            cand = (pit / bay, 'grid')
        elif sa:
            ds = [d for d in np.diff(p['vlines']) if d > 10 and abs(np.log(d / bay / sa)) < lt]
            if ds: cand = (float(np.median(ds)) / bay, 'grid')
        if cand and (sa is None or abs(np.log(cand[0] / sa)) < lt):
            p['s'], p['scale_src'] = float(cand[0]), cand[1]
    return top


# ───────────────────────────── 벽 마스크: 그리드선 지우기(스캔 흐림으로 3~5px 로 번진 선)
GRID_INK = 210          # 그리드선 잉크(geo 영상 lum) 문턱
GRID_DARK = 8           # 선 가운데(±2px) 최저 밝기가 그 선 양 끝(여백)보다 이만큼 넘게 짙은 줄은 벽(회색 채움+윤곽)으로 남김


def _grid_runs(lum, xs, axis, half=8, ink=GRID_INK):
    """그리드선(세로 axis=0: 열 x / 가로 axis=1: 행 y)마다, 선을 가로지르는 잉크 구간(lum<ink) 길이 → 선 폭(중앙값)보다 1px 넘게 넓지 않은 줄의 구간 = 지울 화소 목록"""
    H, W = lum.shape
    out = []
    for x in xs:
        xi = int(round(x))
        if axis == 0:
            if xi - half < 0 or xi + half >= W: continue
            strip = lum[:, xi - half:xi + half + 1] < ink
        else:
            if xi - half < 0 or xi + half >= H: continue
            strip = (lum[xi - half:xi + half + 1, :] < ink).T
        n = strip.shape[0]; c = half
        L = np.zeros(n, int); R = np.zeros(n, int); on = strip[:, c].copy()
        if not on.any(): continue
        for k in range(1, half + 1):                       # 가운데에서 왼쪽·오른쪽으로 잉크가 이어지는 길이
            L += (L == k - 1) & strip[:, c - k]; R += (R == k - 1) & strip[:, c + k]
        run = np.where(on, L + R + 1, 0)
        band = (lum[:, xi - 2:xi + 3] if axis == 0 else lum[xi - 2:xi + 3, :].T).min(1)     # 줄마다 선 가운데 ±2px 최저 밝기
        idx = np.where(on)[0]
        if len(idx) < 0.6 * n: continue                                          # 그림을 거의 가로지르는 선만(건물 외벽만 긴 선은 그리드 아님)
        k = max(5, int(0.12 * (idx[-1] - idx[0])))
        ends = np.r_[idx[idx < idx[0] + k], idx[idx > idx[-1] - k]]              # 선 양 끝(도면 여백 = 건물 밖) → 그 선의 '그리드만 있는' 모습
        ref, wg = float(np.median(band[ends])), float(np.median(run[ends]))
        rows = np.where(on & (run <= wg + 1) & (L < half) & (R < half) & (band >= ref - GRID_DARK))[0]   # 끝보다 넓거나 짙으면 벽(그리드선 위 벽)
        out.append((xi, rows, L[rows], R[rows]))
    return out


def wall_mask(M, s, vlines, hlines, exclude=None):
    """geom.wall_mask(CAD 출력 도면용) 뒤에, 흐림으로 번진 그리드선 잔여(선 양 끝과 같은 폭·밝기의 잉크 줄)를 지운다. 그리드선 위 벽(더 넓거나 짙은 잉크)은 남김.
    기본 꺼짐(SCANPLAN_GRIDWALL=1 로 켬): 가상 도면은 칸막이벽(0.15 m 회색)이 그리드선 위에 있고 스캔 흐림 뒤 그리드선과 구별이 안 돼 dev 벽 재현율 .84→.78·진입구 34→24/36"""
    from fire3dbeam import geom
    m = geom.wall_mask(M, s, vlines, hlines, exclude)
    lum = M['lum']
    for xi, rows, L, R in _grid_runs(lum, vlines, 0):
        for y, l, r in zip(rows, L, R): m[y, xi - l:xi + r + 1] = 0
    for yi, cols, L, R in _grid_runs(lum, hlines, 1):
        for x, l, r in zip(cols, L, R): m[yi - l:yi + r + 1, x] = 0
    return m


def main_footprint(fp, frac=0.2):
    """외곽 추정이 여러 조각이면(제목 상자·나무·주변 선이 따로 닫힌 것) 가장 큰 조각의 frac 배 이상인 조각만 남김"""
    if fp is None or fp.geom_type != 'MultiPolygon': return fp
    from shapely.ops import unary_union
    ps = sorted(fp.geoms, key=lambda g: -g.area)
    return unary_union([g for g in ps if g.area >= frac * ps[0].area])
