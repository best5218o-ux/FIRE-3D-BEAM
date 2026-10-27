# -*- coding: utf-8 -*-
"""실제 도면 벽·창·문 분할 모델(ONNX) 추론 + 벡터화 — onnxruntime·opencv·numpy 만 필요(torch 불필요, 오프라인 Windows PC 용).

모델: models/str_seg.onnx (AI허브 「건축 도면 데이터」 STR 라벨로 학습한 U-Net, 입력 1ch 회색조 0..1, 출력 4클래스 로짓)
      = GPU 모델(2026-09-30, 4.46M, Colab T4). 이전 v3(1.12M) = models/str_seg_v3.onnx (STR_SEG_MODEL=… 또는 StrSeg(path) 로 비교·대체)
      모델마다 .json 에 벡터화 조정값 "vec_param"(VEC_PARAM 위에 덮음)·타일 크기 "tile" — StrSeg(path).vec_param / model_vec_param(path)
작업 해상도: 원본 300dpi 기준 ×0.2476 (≈74dpi, 1/100 도면이면 1 m ≈ 29 px). 입력 dpi(src_dpi) 또는 축척(px_per_m)으로 맞춘다.

사용:
  from fire3dbeam.str_seg import StrSeg, vectorize
  seg = StrSeg()                                  # models/str_seg.onnx
  lab, prob, k = seg.predict(gray, src_dpi=300)   # lab: 작업 해상도 0 배경·1 벽·2 창·3 문, k = 작업/입력 배율
  v = vectorize(lab, px_per_m=None, P=seg.vec_param)   # (P 생략 = 기본 모델 .json 의 vec_param) {walls:[{o,c,a,b,t}], walls_diag:[...], doors:[...], windows:[...]} (작업 px, 또는 m)
                                                  # = vectorize_v2 (2026-09-30). vectorize_v1 = 이전(짝수 커널 열기로 벽이 +1px 밀리던 버전, 비교용)
  python -m fire3dbeam.str_seg 도면.png 출력.png [--dpi 300]   # 겹침 그림
"""
import os, sys, json
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get('STR_SEG_MODEL', os.path.join(ROOT, 'models', 'str_seg.onnx'))
WORK_DPI = 300 * 0.2476          # 학습 캐시 해상도(AI허브 원본 300dpi × 0.6 × 0.4127)
WORK_PPM_1_100 = WORK_DPI / 25.4 * 10      # 1/100 도면의 작업 해상도 px/m (≈29.2)
NAMES = ['bg', 'wall', 'window', 'door']


class StrSeg:
    def __init__(self, path=MODEL, threads=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads: so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        if os.environ.get('STR_SEG_ARENA', '0') != '1': so.enable_cpu_mem_arena = False   # 실행 뒤 작업 메모리 반환(큰 타일 1회 ≈1.5 GB 가 계속 잡혀 있지 않게, 결과 동일)
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        mj = os.path.splitext(path)[0] + '.json'
        self.meta = json.load(open(mj)) if os.path.exists(mj) else {}
        self.path = path
        self.tile = int(self.meta.get('tile') or 1536)          # 겹침 타일 크기(큰 모델은 .json "tile" 로 줄여 메모리 제한)
        self.vec_param = dict(VEC_PARAM, **(self.meta.get('vec_param') or {}))   # 모델별 벡터화 조정값(모델 .json "vec_param")

    def _run(self, x):
        H, W = x.shape
        ph, pw = (-H) % 32, (-W) % 32
        xp = np.pad(x, ((0, ph), (0, pw)), constant_values=1.0)[None, None].astype(np.float32)
        return self.sess.run(None, {self.inp: xp})[0][0, :, :H, :W]

    def logits(self, g, tile=None, ov=96, even=False, skip_blank=False):
        """작업 해상도 회색조 g(uint8) → 로짓 (4,H,W). 큰 영상은 겹침 타일(기본 self.tile).
        even=True: 같은 타일 수로 영상을 고르게 나눔(마지막 타일이 앞 타일과 크게 겹쳐 같은 곳을 두세 번 돌리던 낭비 제거 — 실제 도면 경로, 2026-10-03)"""
        tile = tile or self.tile
        x = g.astype(np.float32) / 255.0
        H, W = x.shape
        if max(H, W) <= tile: return self._run(x)
        out = np.zeros((4, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
        step = tile - 2 * ov
        if even:
            def grid(N):
                n = 1 if N <= tile else -(-(N - 2 * ov) // step)
                t = min(N, -(-(N + (n - 1) * 2 * ov) // n))
                return [min(i * (t - 2 * ov), N - t) for i in range(n)], t
            ys_, th = grid(H); xs_, tw = grid(W)
            boxes = [(y0, x0, min(H, y0 + th), min(W, x0 + tw)) for y0 in ys_ for x0 in xs_]
        else:
            boxes = []
            for y0 in range(0, max(1, H - 2 * ov), step):
                for x0 in range(0, max(1, W - 2 * ov), step):
                    y1, x1 = min(H, y0 + tile), min(W, x0 + tile); boxes.append((max(0, y1 - tile), max(0, x1 - tile), y1, x1))
        for ya, xa, y1, x1 in boxes:
            if skip_blank and (g[ya:y1, xa:x1] < 200).mean() < 2e-4:
                continue                                 # 잉크가 거의 없는 빈 종이 타일(여러 평면 시트를 나눈 칸의 여백) — 배경으로(아래 wsum=0 칸)
            if True:
                o = self._run(x[ya:y1, xa:x1])
                wy = np.minimum(np.arange(y1 - ya) + 1, np.arange(y1 - ya)[::-1] + 1).clip(1, ov)[:, None]
                wx = np.minimum(np.arange(x1 - xa) + 1, np.arange(x1 - xa)[::-1] + 1).clip(1, ov)[None, :]
                w = (wy * wx).astype(np.float32)
                out[:, ya:y1, xa:x1] += o * w; wsum[ya:y1, xa:x1] += w
        if skip_blank and (wsum == 0).any():
            z = wsum == 0; out[0][z] = 10.0; out[1:, z] = 0.0; wsum[z] = 1.0
        return out / wsum

    def predict(self, g, src_dpi=None, px_per_m=None, out_shape=None, tile=None, even=False, skip_blank=False):
        """g: 회색조(uint8, 흰 바탕). 배율 k: px_per_m 이 있으면 1/100 작업 기준, 아니면 src_dpi(기본 300).
        반환 lab(작업 해상도 또는 out_shape), prob(4,…), k"""
        if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        k = WORK_PPM_1_100 / px_per_m if px_per_m else WORK_DPI / float(src_dpi or 300)
        gw = cv2.resize(g, (max(32, round(g.shape[1] * k)), max(32, round(g.shape[0] * k))),
                        interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR) if abs(k - 1) > 1e-3 else g
        lo = self.logits(gw, tile=tile, even=even, skip_blank=skip_blank)
        e = np.exp(lo - lo.max(0, keepdims=True)); p = e / e.sum(0, keepdims=True)
        if out_shape is not None:
            p = np.stack([cv2.resize(c, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR) for c in p])
        return p.argmax(0).astype(np.uint8), p, k


# ───────────────────────── 벡터화 ─────────────────────────
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton(m):
    try:
        return (cv2.ximgproc.thinning(m.astype(np.uint8) * 255) > 0).astype(np.uint8)
    except AttributeError:                                   # ximgproc 없는 opencv: 형태학 골격
        img = m.astype(np.uint8).copy(); sk = np.zeros_like(img); el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while img.any():
            er = cv2.erode(img, el); sk |= img & (1 - cv2.dilate(er, el)); img = er
        return sk


def _branches(sk):
    """골격 → 가지(순서 있는 점열) 목록. 교차점(이웃≥3)에서 끊고, 가지 양끝에 닿은 교차점 중심을 붙인다"""
    H, W = sk.shape
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = (sk > 0) & (nb >= 3)
    jl_n, jl = cv2.connectedComponents(junc.astype(np.uint8), connectivity=8)
    jc = [None] + [tuple(np.mean(np.argwhere(jl == i), 0)) for i in range(1, jl_n)]
    br = (sk > 0) & ~junc
    n, lab = cv2.connectedComponents(br.astype(np.uint8), connectivity=8)
    out = []
    pts_by = [[] for _ in range(n)]
    ys, xs = np.nonzero(lab)
    for y, x in zip(ys, xs): pts_by[lab[y, x]].append((y, x))
    for i in range(1, n):
        P = pts_by[i]
        S = set(P)
        deg = {p: sum((p[0] + dy, p[1] + dx) in S for dy, dx in _NB) for p in P}
        ends = [p for p in P if deg[p] <= 1]
        start = ends[0] if ends else P[0]
        seq = [start]; seen = {start}; cur = start
        while True:
            nxt = None
            for dy, dx in _NB:
                q = (cur[0] + dy, cur[1] + dx)
                if q in S and q not in seen: nxt = q; break
            if nxt is None: break
            seq.append(nxt); seen.add(nxt); cur = nxt
        # 양끝 교차점
        def jn(p):
            for dy, dx in _NB:
                y, x = p[0] + dy, p[1] + dx
                if 0 <= y < H and 0 <= x < W and jl[y, x]: return jl[y, x]
            return 0
        a, b = jn(seq[0]), jn(seq[-1])
        pts = ([jc[a]] if a else []) + [(float(y), float(x)) for y, x in seq] + ([jc[b]] if b else [])
        out.append(dict(pts=np.array(pts), ja=a, jb=b, n=len(seq)))
    return out, jc


def wall_vectors(wm, min_len=None, axis_tol_deg=6.0):
    """벽 마스크(작업 px) → 선분 [{x0,y0,x1,y1,t}] (중심선·두께). 짧은 잔가지(길이 < 두께·1.2, 한쪽 끝이 자유) 제거"""
    wm = (wm > 0).astype(np.uint8)
    if not wm.any(): return []
    dt = cv2.distanceTransform(wm, cv2.DIST_L2, 3)
    sk = skeleton(wm)
    tmed = float(np.median(dt[sk > 0])) * 2 if sk.any() else 3.0
    min_len = min_len or max(3.0, 0.8 * tmed)
    brs, _ = _branches(sk)
    segs = []
    for b in brs:
        P = b['pts']
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 0.0
        free = (b['ja'] == 0) + (b['jb'] == 0)
        if free >= 1 and L < 1.2 * tmed and not (free == 2 and L >= min_len): continue
        if L < 1.0: continue
        ap = cv2.approxPolyDP(P[:, ::-1].astype(np.float32).reshape(-1, 1, 2), max(1.0, 0.35 * tmed), False)[:, 0, :]
        for (x0, y0), (x1, y1) in zip(ap[:-1], ap[1:]):
            l = np.hypot(x1 - x0, y1 - y0)
            if l < min_len: continue
            n = max(2, int(l))
            xs = np.linspace(x0, x1, n).round().astype(int).clip(0, wm.shape[1] - 1)
            ys = np.linspace(y0, y1, n).round().astype(int).clip(0, wm.shape[0] - 1)
            t = float(max(1.0, 2 * np.median(dt[ys, xs]) - 1))
            ang = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
            if min(ang, 180 - ang) < axis_tol_deg: ym = (y0 + y1) / 2; y0 = y1 = ym          # 축 맞춤
            elif abs(ang - 90) < axis_tol_deg: xm = (x0 + x1) / 2; x0 = x1 = xm
            segs.append(dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), t=round(t, 2)))
    return segs


def wall_thickness(wm):
    wm = (wm > 0).astype(np.uint8)
    if not wm.any(): return 3.0
    dt = cv2.distanceTransform(wm, cv2.DIST_L2, 3)
    sk = skeleton(wm)
    return float(np.median(dt[sk > 0])) * 2 if sk.any() else 3.0


def _track_runs(op, L, tol_c, nf=1.5):
    """가로 벽 마스크 op 의 열(column)별 세로 run 을 열 방향으로 이어 궤적 → [(c, a, b, t)]
    (중심 변화 ≤ tol_c 이고 겹칠 때만 이어서, 계단식으로 꺾인 벽이 한 선분으로 뭉치지 않게)"""
    H, W = op.shape
    pad = np.zeros((H + 2, W), np.int8); pad[1:-1] = op
    d = np.diff(pad, axis=0)
    act = []; done = []
    cols = np.nonzero(op.any(0))[0]
    prev_x = -2
    for x in cols:
        st = np.nonzero(d[:, x] == 1)[0]; en = np.nonzero(d[:, x] == -1)[0]
        if x != prev_x + 1: done += act; act = []
        new = []; used = set()
        for y0, y1 in zip(st, en):
            c = (y0 + y1) / 2; best = None
            for j, tr in enumerate(act):
                if j in used: continue
                py0, py1 = tr['r'][-1]
                if y0 < py1 and py0 < y1 and abs(c - (py0 + py1) / 2) <= tol_c:
                    best = j; break
            if best is None: tr = dict(x0=x, r=[(y0, y1)])
            else: tr = act[best]; used.add(best); tr['r'].append((y0, y1))
            new.append(tr)
        done += [tr for j, tr in enumerate(act) if j not in used]
        act = new; prev_x = x
    done += act
    out = []
    for tr in done:
        n = len(tr['r'])
        if n < L: continue
        r = np.array(tr['r'], float)
        th = float(np.median(r[:, 1] - r[:, 0])); c = float(np.median((r[:, 0] + r[:, 1]) / 2))
        if n < nf * th: continue
        out.append((c, float(tr['x0']), float(tr['x0'] + n), th))
    return out


def axis_segments(wm, t=None, fixed=False, Lf=1.2, nf=1.5):
    """축정렬 벽: 방향별 열기(길이 ≥1.2t) → run 추적(중심 변화 ≤ max(1.5, 0.35t)) → [{o,c,a,b,t}] px"""
    wm = (wm > 0).astype(np.uint8)
    t = t or wall_thickness(wm)
    L = max(3, int(round(Lf * t)) + 1)
    tol = max(1.5, 0.35 * t)
    out = []
    for o in ('h', 'v'):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if o == 'h' else (1, L))
        op = (_open(wm, L, 1) if o == 'h' else _open(wm, 1, L)) if fixed else cv2.morphologyEx(wm, cv2.MORPH_OPEN, k)
        if o == 'v': op = np.ascontiguousarray(op.T)
        for c, a, b, th in _track_runs(op, L, tol, nf):
            out.append(dict(o=o, c=c, a=a, b=b, t=th))
    return out


def _merge_axis(segs, tol=2.0, gap=None, tmode='max'):
    """같은 축·같은 중심선(±tol)·겹치거나 (0.5t+1) 이내로 이어진 축 선분 병합. 입력·출력 [{o,c,a,b,t}] px"""
    out = []
    for o in ('h', 'v'):
        S = sorted((s['c'], s['a'], s['b'], s['t']) for s in segs if s['o'] == o)
        groups = []
        for x in S:
            if groups and abs(x[0] - groups[-1][-1][0]) <= tol: groups[-1].append(x)
            else: groups.append([x])
        for gs in groups:
            iv = sorted(gs, key=lambda z: z[1]); cur = list(iv[0])
            for c, a, b, t in iv[1:]:
                g_ = gap if gap is not None else 0.5 * max(t, cur[3])
                if a <= cur[2] + g_ + 1:
                    wa, wb = cur[2] - cur[1], b - a
                    cur[0] = (cur[0] * wa + c * wb) / max(1e-6, wa + wb); cur[2] = max(cur[2], b)
                    cur[3] = max(cur[3], t) if tmode == 'max' else (cur[3] * wa + t * wb) / max(1e-6, wa + wb)
                else: out.append(dict(o=o, c=cur[0], a=cur[1], b=cur[2], t=cur[3])); cur = [c, a, b, t]
            out.append(dict(o=o, c=cur[0], a=cur[1], b=cur[2], t=cur[3]))
    return out


def instances(m, min_area=6):
    """부류 마스크 → 개체 [{cx,cy,w(긴 변),d(짧은 변),ang,box:[x0,y0,x1,y1],area}] (작업 px)"""
    n, lab, st, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if st[i, 4] < min_area: continue
        ys, xs = np.nonzero(lab == i)
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        if w < h: w, h, ang = h, w, ang + 90
        x, y, bw, bh = st[i, :4]
        out.append(dict(cx=float(cx), cy=float(cy), w=float(w), d=float(h), ang=float(ang % 180),
                        box=[int(x), int(y), int(x + bw), int(y + bh)], area=int(st[i, 4])))
    return out


def filter_instances(L, min_area, min_w, rel):
    """작은 조각 제거: 면적 < max(min_area, rel×중앙값) 또는 긴 변 < min_w (작업 해상도 px; 1/100 도면 기준 문 0.4 m·창 0.3 m)"""
    if not L: return L
    med = float(np.median([d['area'] for d in L])) if len(L) >= 3 else 0.0
    thr = max(min_area, rel * med)
    return [d for d in L if d['area'] >= thr and d['w'] >= min_w]


def vectorize_v1(lab, px_per_m=None, min_open=None):
    """분할 라벨(작업 해상도) → 엔진 표현.
    walls: 축정렬 [{o,c,a,b,t}] — px_per_m 이 있으면 geom.py 와 같은 도면 지역좌표 m(x=px/s, y=-py/s), 없으면 px(화면 좌표)
    walls_diag: 사선 벽 [{x0,y0,x1,y1,t}], doors/windows: [{x,y,w,d,ang,box}] (w = 개구부 폭)"""
    wm = (lab == 1).astype(np.uint8)
    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    t = wall_thickness(wm)
    ax = axis_segments(wm, t)
    # 축 선분으로 설명 안 되는 벽(사선 벽) → 골격 선분
    cov = render_vectors(dict(unit='px', walls=ax, walls_diag=[], doors=[], windows=[]), wm.shape)
    res = wm & (1 - cv2.dilate(cov, np.ones((3, 3), np.uint8)))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(res, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n): keep[i] = st[i, 4] >= 2.5 * t * t and max(st[i, 2], st[i, 3]) >= 2.5 * t
    res = keep[lb].astype(np.uint8)
    dg = [s for s in wall_vectors(res, min_len=2.0 * t) if not (s['x0'] == s['x1'] or s['y0'] == s['y1'])]
    ax = _merge_axis(ax)
    dr = filter_instances(instances(lab == 3, 8), 30, 12, 0.12); wn = filter_instances(instances(lab == 2, 6), 24, 9, 0.10)
    s = px_per_m
    if s:
        for w in ax:
            if w['o'] == 'h': w.update(c=-w['c'] / s, a=w['a'] / s, b=w['b'] / s, t=w['t'] / s)
            else: w.update(c=w['c'] / s, a=-w['b'] / s, b=-w['a'] / s, t=w['t'] / s)
    def conv(d, kind):
        r = dict(x=d['cx'], y=d['cy'], w=d['w'], d=d['d'], ang=d['ang'], box=d['box'], kind=kind)
        if s: r.update(x=d['cx'] / s, y=-d['cy'] / s, w=d['w'] / s, d=d['d'] / s, unit='m')
        return r
    rnd = lambda D: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in D.items()}
    return dict(unit='m' if s else 'px', walls=[rnd(w) for w in ax], walls_diag=[rnd(w) for w in dg],
                doors=[rnd(conv(d, 'door')) for d in dr], windows=[rnd(conv(d, 'window')) for d in wn])


def _open(m, kx, ky):
    """이동 없는 열기(opening). cv2.morphologyEx 는 짝수 크기 커널에서 결과가 +1px 밀린다(침식·팽창이 같은 anchor 사용)"""
    k = np.ones((ky, kx), np.uint8)
    return cv2.dilate(cv2.erode(m, k, anchor=(kx // 2, ky // 2)), k, anchor=((kx - 1) // 2, (ky - 1) // 2))


_VP = {}


def model_vec_param(path=None):
    """모델 파일(.onnx) 옆 .json 의 "vec_param"(그 모델 출력에 맞춘 벡터화 조정값)을 VEC_PARAM 위에 덮은 dict. 없으면 VEC_PARAM"""
    path = os.path.abspath(path or MODEL)
    if path not in _VP:
        mj = os.path.splitext(path)[0] + '.json'
        try: vp = json.load(open(mj)).get('vec_param') or {}
        except Exception: vp = {}
        _VP[path] = dict(VEC_PARAM, **vp)
    return dict(_VP[path])


def vectorize(lab, px_per_m=None, min_open=None, prob=None, P=None, model=None):
    """기본 벡터화 = vectorize_v2 (min_open 은 옛 호출 호환용, 쓰지 않음). prob(4,H,W) 는 VEC_PARAM['pthr'] 쓸 때만.
    P: 조정값 dict(없으면 model — 기본 models/str_seg.onnx — 의 .json "vec_param"). StrSeg 객체가 있으면 P=seg.vec_param"""
    return vectorize_v2(lab, px_per_m, prob=prob, P=P if P is not None else model_vec_param(model))


# v2 매개변수(dev-val 254장에서만 선택): merge2 상자까지 다시 병합, box_grow 상자를 벽 마스크 안에서 1px 넓혀 옆 벽에 붙임,
# box_noopen 문·창 상자 안은 벽 상자 금지, pthr 벽 확률 하한(None=argmax; .45/.35 는 IoU↓), open2 2×2 이동 없는 열기,
# Lf·nf 축 열기 길이·궤적 최소 길이(두께 배), tmerge 병합 두께 'wmean'(길이가중)|'max'(v1), box_* 잔여 조각 → 축 상자 조건(두께 t 배)
# tadd 축 선분 두께 가산(작업 px), dx·dy 축 선분 이동(작업 px) — 모델별 보정용(기본 0 = v3 조정 그대로)
VEC_PARAM = dict(pthr=None, merge2=True, box_grow=True, box_noopen=True, open2=True, Lf=1.2, nf=1.5, tmerge='wmean', res_dil=1, box=True, box_amin=0.6, box_wmin=0.5, box_fill=0.6, box_lmin=1.0,
                 tadd=0.0, dx=0.0, dy=0.0)


def _residual_boxes(res, t, P, wm=None):
    """축 선분이 덮지 못한 벽 조각 중 네모난 것(짧은 벽기둥·창 사이 벽·짧은 칸막이) → 축 상자 선분. 나머지 마스크 반환.
    wm 이 있으면 상자를 벽 마스크 안에서 1px 넓혀(잔여 계산 때 뺀 여유) 옆 벽에 붙인다"""
    n, lb, st, _ = cv2.connectedComponentsWithStats(res, 8)
    out = []; rest = res.copy(); H, W = res.shape
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < P['box_amin'] * t * t or min(bw, bh) < max(2, P['box_wmin'] * t) or max(bw, bh) < P['box_lmin'] * t: continue
        if a < P['box_fill'] * bw * bh: continue
        rest[lb == i] = 0
        if wm is not None:
            y0, y1, x0, x1 = max(0, y - 1), min(H, y + bh + 1), max(0, x - 1), min(W, x + bw + 1)
            m = cv2.dilate((lb[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((3, 3), np.uint8)) & wm[y0:y1, x0:x1]
            ys, xs = np.nonzero(m)
            x, y, bw, bh = x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if bw >= bh: out.append(dict(o='h', c=y + bh / 2, a=float(x), b=float(x + bw), t=float(bh)))
        else: out.append(dict(o='v', c=x + bw / 2, a=float(y), b=float(y + bh), t=float(bw)))
    return out, rest


def vectorize_v2(lab, px_per_m=None, prob=None, P=None):
    """분할 라벨(작업 해상도) → 엔진 표현 (v2: 이동 없는 열기, 선분별 두께 길이가중, 짧은 벽 상자 보충).
    walls: 축정렬 [{o,c,a,b,t}] — px_per_m 이 있으면 geom.py 와 같은 도면 지역좌표 m(x=px/s, y=-py/s), 없으면 px(화면 좌표)
    walls_diag: 사선 벽 [{x0,y0,x1,y1,t}], doors/windows: [{x,y,w,d,ang,box}] (w = 개구부 폭)"""
    P = dict(VEC_PARAM, **(P or {}))
    wm = (lab == 1).astype(np.uint8)
    if prob is not None and P['pthr'] is not None:                 # 벽 확률 하한(문·창 argmax 는 제외)
        wm |= ((prob[1] >= P['pthr']) & (lab == 0)).astype(np.uint8)
    if P['open2']: wm = _open(wm, 2, 2)
    t = wall_thickness(wm)
    ax = axis_segments(wm, t, fixed=True, Lf=P['Lf'], nf=P['nf'])
    ax = _merge_axis(ax, tmode=P['tmerge'])
    cov = render_vectors(dict(unit='px', walls=ax, walls_diag=[], doors=[], windows=[]), wm.shape)
    res = wm & (1 - (cv2.dilate(cov, np.ones((3, 3), np.uint8)) if P['res_dil'] else cov))
    dr = filter_instances(instances(lab == 3, 8), 30, 12, 0.12); wn = filter_instances(instances(lab == 2, 6), 24, 9, 0.10)
    if P['box_noopen']:                                             # 문·창 상자 안은 벽 상자로 메우지 않음(개구부 유지)
        for d in dr + wn: b = d['box']; res[b[1]:b[3], b[0]:b[2]] = 0
    res = _open(res, 2, 2)
    if P['box']:
        bx, res = _residual_boxes(res, t, P, wm if P['box_grow'] else None); ax += bx
        if P['merge2']: ax = _merge_axis(ax, tol=max(2.0, 0.3 * t), tmode=P['tmerge'])
    # 축 선분으로 설명 안 되는 벽(사선 벽) → 골격 선분
    n, lb, st, _ = cv2.connectedComponentsWithStats(res, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n): keep[i] = st[i, 4] >= 2.5 * t * t and max(st[i, 2], st[i, 3]) >= 2.5 * t
    res = keep[lb].astype(np.uint8)
    dg = [s for s in wall_vectors(res, min_len=2.0 * t) if not (s['x0'] == s['x1'] or s['y0'] == s['y1'])]
    if P.get('tadd') or P.get('dx') or P.get('dy'):
        for w in ax:
            w['t'] = max(1.0, w['t'] + P.get('tadd', 0.0))
            dc, da = (P.get('dy', 0.0), P.get('dx', 0.0)) if w['o'] == 'h' else (P.get('dx', 0.0), P.get('dy', 0.0))
            w['c'] += dc; w['a'] += da; w['b'] += da
    s = px_per_m
    if s:
        for w in ax:
            if w['o'] == 'h': w.update(c=-w['c'] / s, a=w['a'] / s, b=w['b'] / s, t=w['t'] / s)
            else: w.update(c=w['c'] / s, a=-w['b'] / s, b=-w['a'] / s, t=w['t'] / s)
    def conv(d, kind):
        r = dict(x=d['cx'], y=d['cy'], w=d['w'], d=d['d'], ang=d['ang'], box=d['box'], kind=kind)
        if s: r.update(x=d['cx'] / s, y=-d['cy'] / s, w=d['w'] / s, d=d['d'] / s, unit='m')
        return r
    rnd = lambda D: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in D.items()}
    return dict(unit='m' if s else 'px', walls=[rnd(w) for w in ax], walls_diag=[rnd(w) for w in dg],
                doors=[rnd(conv(d, 'door')) for d in dr], windows=[rnd(conv(d, 'window')) for d in wn])


def render_vectors(v, shape, px_per_m=None, scale=1.0):
    """벡터 → 라벨 영상(평가·확인용). scale: 작업 px → 출력 px 배율(평가 해상도에 바로 그리기 — 라벨 영상을 최근접 축소하면
    화소 중심이 어긋나 ≈0.5px 밀린다). 좌표는 화소 경계 기준이라 그대로 곱한다"""
    out = np.zeros(shape, np.uint8); s = (px_per_m or 1.0) * scale
    for d in v['windows'] + v['doors']:
        b = [int(round(z * scale)) for z in d['box']]; out[b[1]:b[3], b[0]:b[2]] = 2 if d['kind'] == 'window' else 3
    H, W = shape
    for w in v['walls']:                                   # 축 선분: 좌표는 화소 경계 기준(run [a,b) · 중심 c · 두께 t) → 정확한 화소 범위
        if v['unit'] == 'm':
            if w['o'] == 'h': c, a, b, t = -w['c'] * s, w['a'] * s, w['b'] * s, w['t'] * s
            else: c, a, b, t = w['c'] * s, -w['b'] * s, -w['a'] * s, w['t'] * s
        else:
            c, a, b, t = w['c'] * s, w['a'] * s, w['b'] * s, w['t'] * s
        c0, c1 = int(round(c - t / 2)), int(round(c + t / 2)); a0, b0 = int(round(a)), int(round(b))
        c1 = max(c1, c0 + 1)
        if w['o'] == 'h': out[max(0, c0):max(0, c1), max(0, a0):max(0, b0)] = 1
        else: out[max(0, a0):max(0, b0), max(0, c0):max(0, c1)] = 1
    for w in v['walls_diag']:
        k = scale                                          # 사선 벽은 unit 과 무관하게 작업 px(변환 안 함)
        _thick_line(out, (w['x0'] * k, w['y0'] * k), (w['x1'] * k, w['y1'] * k), w['t'] * k)
    return out


def _thick_line(out, p, q, t, ext=True):
    """두께 t 의 사각 끝 선분 → 1(벽). ext: 끝을 t/2 연장(골격 선분의 모서리 채움). 축 선분은 범위가 이미 외곽 기준"""
    p, q = np.array(p, float), np.array(q, float)
    d = q - p; L = np.hypot(*d)
    u = d / L if L > 0 else np.array([1.0, 0.0]); n = np.array([-u[1], u[0]])
    h = max(0.5, t / 2)
    e = h if ext else 0.0
    pts = np.array([p - u * e + n * h, q + u * e + n * h, q + u * e - n * h, p - u * e - n * h])
    cv2.fillConvexPoly(out, np.round(pts).astype(np.int32), 1)


def overlay(g, lab, v=None, alpha=0.55):
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g.copy()
    if lab.shape != img.shape[:2]: lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    col = {1: (0, 0, 230), 2: (230, 80, 0), 3: (0, 170, 0)}
    for c, bgr in col.items():
        m = lab == c; img[m] = ((1 - alpha) * img[m] + alpha * np.array(bgr)).astype(np.uint8)
    return img


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('img'); ap.add_argument('out'); ap.add_argument('--dpi', type=float, default=300)
    ap.add_argument('--ppm', type=float, default=None, help='입력 px/m (알면)')
    a = ap.parse_args()
    g = cv2.imread(a.img, cv2.IMREAD_GRAYSCALE)
    seg = StrSeg()
    lab, p, k = seg.predict(g, src_dpi=a.dpi, px_per_m=a.ppm)
    gw = cv2.resize(g, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA)
    v = vectorize(lab)
    img = overlay(gw, lab)
    for w in v['walls']:
        p0, q0 = ((w['a'], w['c']), (w['b'], w['c'])) if w['o'] == 'h' else ((w['c'], w['a']), (w['c'], w['b']))
        cv2.line(img, tuple(int(round(z)) for z in p0), tuple(int(round(z)) for z in q0), (0, 255, 255), 1)
    for w in v['walls_diag']: cv2.line(img, (int(w['x0']), int(w['y0'])), (int(w['x1']), int(w['y1'])), (255, 0, 255), 1)
    cv2.imwrite(a.out, img)
    print(json.dumps(dict(k=round(k, 4), walls=len(v['walls']), diag=len(v['walls_diag']), doors=len(v['doors']), windows=len(v['windows']))))
