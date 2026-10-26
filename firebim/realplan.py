# -*- coding: utf-8 -*-
"""실제 도면(스캔·CAD 출력 PNG, A3@300dpi 흑백) 기하 전처리 — 가상 도면용 geom/vision 과 독립.

  sheet_layout(q)      도곽·표제란 제거 → 도면 영역(여러 세대 평면) 분할 + 캡션 후보
  wall_style / walls   벽 표현 분류(채움·해치·겹선·단선) → 벽 마스크·벽 선분·외곽(외벽 경계)
  dims_scale           치수선(가는 긴 선 + 눈금) + 치수 숫자 OCR(12,500 / 4530) → px/mm 합의 추정 + 신뢰도
  total_check          최외곽 전체치수(가로·세로) ↔ 외곽 범위(mm) 상대오차 = 자기지도 정확도

좌표: F = 원본(300dpi) px, 내부 해석은 H = 1/2(150dpi, 최소값 풀링 → 가는 선 보존), 배치는 Q = 1/4.
사용: python -m firebim.realplan sheet.png out_dir [--rules]   (기본 = 분할 모델 벽, --rules = 규칙 벽)
"""
import re, os, sys, json, time, subprocess, tempfile
import numpy as np, cv2

INK_T = 185          # 1/2 해상도(최소값 풀링) 잉크 임계
BLACK_T = 60         # 검정 채움 벽


def minpool(g, k):
    H, W = g.shape[0] // k * k, g.shape[1] // k * k
    return g[:H, :W].reshape(H // k, k, W // k, k).min(axis=(1, 3))


def K(w, h): return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w)), max(1, int(h))))


def opn(m, w, h): return cv2.morphologyEx(m, cv2.MORPH_OPEN, K(w, h))
def cls(m, w, h): return cv2.morphologyEx(m, cv2.MORPH_CLOSE, K(w, h))


def comps(m, min_area=1):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return [dict(i=i, x=int(st[i, 0]), y=int(st[i, 1]), w=int(st[i, 2]), h=int(st[i, 3]), a=int(st[i, 4])) for i in range(1, n) if st[i, 4] >= min_area], lab


# ───────────────────────── 1. 시트 배치 ─────────────────────────
def _n_ticks(ink, c, o):
    """선 옆에 붙은 짧은 수직 눈금(치수 눈금) 개수 — 치수선은 도곽선으로 지우지 않는다"""
    H, W = ink.shape
    if o == 'v':
        x = c['x'] + c['w'] // 2; y0, y1 = c['y'], c['y'] + c['h']
        sides = [ink[y0:y1, x + 2:x + 4].any(1) if x + 4 < W else None, ink[y0:y1, max(0, x - 3):max(0, x - 1)].any(1) if x >= 3 else None]
        far = [ink[y0:y1, x + 12:x + 14].any(1) if x + 14 < W else None, ink[y0:y1, max(0, x - 13):max(0, x - 11)].any(1) if x >= 13 else None]
    else:
        y = c['y'] + c['h'] // 2; x0, x1 = c['x'], c['x'] + c['w']
        sides = [ink[y + 2:y + 4, x0:x1].any(0) if y + 4 < H else None, ink[max(0, y - 3):max(0, y - 1), x0:x1].any(0) if y >= 3 else None]
        far = [ink[y + 12:y + 14, x0:x1].any(0) if y + 14 < H else None, ink[max(0, y - 13):max(0, y - 11), x0:x1].any(0) if y >= 13 else None]
    n = 0
    for sd, fa in zip(sides, far):
        if sd is None: continue
        t = sd & ~(fa if fa is not None else np.zeros_like(sd))
        n += int((np.diff(t.astype(np.int8)) == 1).sum())
    return n


def frame_lines(ink, frac=0.33, thin=3):
    """도곽·구획선·그리드선: 시트 길이의 frac 이상인 가는 직선 중 '짝(평행선)'이 없는 것.
    겹선 벽(바로 옆에 평행선)과 굵은 외벽은 남긴다"""
    H, W = ink.shape
    thick = cv2.dilate(opn(ink, thin + 1, thin + 1), K(5, 5))
    out = np.zeros_like(ink)
    for o in ('v', 'h'):
        ln = (opn(ink, 1, frac * H) if o == 'v' else opn(ink, frac * W, 1)) & (1 - thick)
        cs, lab = comps(ln, 20)
        for c in cs:
            if o == 'v':
                if c['w'] > thin: continue
                x = c['x'] + c['w'] // 2; y0, y1 = c['y'], c['y'] + c['h']
                near_edge = x < 0.08 * W or x > 0.92 * W
                L = ink[y0:y1, max(0, x - 8):max(0, x - 2)].any(1).mean() if x >= 3 else 0
                R_ = ink[y0:y1, x + 3:x + 9].any(1).mean() if x + 3 < W else 0
            else:
                if c['h'] > thin: continue
                y = c['y'] + c['h'] // 2; x0, x1 = c['x'], c['x'] + c['w']
                near_edge = y < 0.08 * H or y > 0.92 * H
                L = ink[max(0, y - 8):max(0, y - 2), x0:x1].any(0).mean() if y >= 3 else 0
                R_ = ink[y + 3:y + 9, x0:x1].any(0).mean() if y + 3 < H else 0
            long_ = (c['h'] if o == 'v' else c['w']) >= 0.6 * (H if o == 'v' else W)
            if near_edge or long_ or (max(L, R_) < 0.5 and _n_ticks(ink, c, o) < 3):
                out[lab == c['i']] = 1
    return out


def _merge_boxes(boxes, gap):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                if a[0] - gap <= c[2] and c[0] - gap <= a[2] and a[1] - gap <= c[3] and c[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
                    boxes.pop(j); changed = True; break
            if changed: break
    return boxes


def sheet_layout(q):
    """q: 1/4 해상도 회색조(75dpi). 반환 regions=[{box:[x0,y0,x1,y1] (Q px), kind, feats}], 캡션 후보 상자"""
    H, W = q.shape
    ink = (q < 190).astype(np.uint8)
    fr = frame_lines(ink)
    ink2 = ink & (1 - cv2.dilate(fr, K(3, 3)))
    # 가장자리 5px 는 도곽 잔재
    ink2[:6] = 0; ink2[-6:] = 0; ink2[:, :6] = 0; ink2[:, -6:] = 0
    # 구조(긴 선): 도면 코어 후보
    L = max(12, int(0.012 * W))
    struct = (opn(ink2, L, 1) | opn(ink2, 1, L))
    core = cls(struct, 21, 21)
    cs, _ = comps(core, 50)
    cores = [[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']] for c in cs if min(c['w'], c['h']) >= 40 and c['w'] * c['h'] >= 5000]
    cores = _merge_boxes(cores, 12)
    # 떨어져 그린 치수 띠(가늘고 긴 코어)는 옆의 큰 코어에 붙인다
    big = [c for c in cores if min(c[2] - c[0], c[3] - c[1]) >= 90]
    for c in [c for c in cores if c not in big]:
        w, h = c[2] - c[0], c[3] - c[1]
        if max(w, h) < 3 * min(w, h) or not big: continue
        def gap(b): return max(b[0] - c[2], c[0] - b[2], b[1] - c[3], c[1] - b[3], 0)
        b = min(big, key=gap)
        ov = (min(b[2], c[2]) - max(b[0], c[0])) / w if w > h else (min(b[3], c[3]) - max(b[1], c[1])) / h
        if gap(b) <= 0.07 * W and ov > 0.6:
            b[0], b[1], b[2], b[3] = min(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), max(b[3], c[3])
            cores.remove(c)
    cores = _merge_boxes(cores, 12)
    # 코어 주변 잉크 덩어리(치수·그리드 번호) 흡수, 먼 글자 덩어리는 캡션/주기 후보
    blob = cls(ink2, 9, 9)
    bs, _ = comps(blob, 6)
    bl = [[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']] for c in bs]
    regs = []
    for cb in cores:
        regs.append(dict(core=cb, box=list(cb)))
    near = int(0.035 * W)                         # 약 30mm(용지) 안의 덩어리 흡수
    changed = True; it = 0
    while changed and it < 6:
        changed = False; it += 1
        for r in regs:
            b = r['box']
            for x0, y0, x1, y1 in bl:
                if x0 >= b[0] and y0 >= b[1] and x1 <= b[2] and y1 <= b[3]: continue
                dx = max(b[0] - x1, x0 - b[2], 0); dy = max(b[1] - y1, y0 - b[3], 0)
                if max(dx, dy) > near: continue
                # 가장 가까운 코어에만
                own = min(regs, key=lambda r2: max(r2['core'][0] - x1, x0 - r2['core'][2], 0) + max(r2['core'][1] - y1, y0 - r2['core'][3], 0))
                if own is not r: continue
                if (x1 - x0) > 0.9 * W or (y1 - y0) > 0.9 * H: continue
                nb = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
                if nb != b: r['box'] = nb; b = nb; changed = True
    # 영역끼리 크게 겹치면 병합
    boxes = _merge_boxes([r['box'] for r in regs], -8)
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b
        sub = ink2[y0:y1, x0:x1]; st = struct[y0:y1, x0:x1]
        fe = dict(w=x1 - x0, h=y1 - y0, ink=float(sub.mean()), struct=float(st.mean()),
                  thick=float(opn(sub, 3, 3).mean()))
        out.append(dict(box=b, feats=fe))
    # 캡션 후보: 영역 밖 글자 덩어리(가로로 긴)
    caps = [b for b in bl if (b[2] - b[0]) > 2.5 * (b[3] - b[1]) and 5 <= (b[3] - b[1]) <= 40 and (b[2] - b[0]) >= 30]
    return out, caps, fr


# ───────────────────────── 2. 벽 표현 분류 · 벽 추출 ─────────────────────────
def run_gaps(Lh, lo=2, hi=60):
    """가로선 마스크에서 세로 방향 연속 선 사이 간격(px) 히스토그램 → 겹선 벽 두께 추정"""
    m = Lh.astype(np.int8)
    d = np.diff(m, axis=0)                       # +1 시작, -1 끝
    hist = np.zeros(hi + 1)
    cols = np.where(Lh.any(0))[0]
    for x in cols[::2]:
        e = np.where(d[:, x] == -1)[0]; s = np.where(d[:, x] == 1)[0]
        if len(e) == 0 or len(s) == 0: continue
        j = np.searchsorted(s, e)
        ok = j < len(s)
        g = s[j[ok]] - e[ok]
        g = g[(g >= lo) & (g <= hi)]
        np.add.at(hist, g, 1)
    return hist


def wall_range(s_h):
    """축척(1/2 해상도 px/mm)으로 벽 두께 범위(px). 없으면 None"""
    if not s_h: return None
    return max(2.0, 90 * s_h), 480 * s_h + 2


def wall_style(h, ink, s_h=None):
    """h: 1/2 해상도 영역. 반환 style, feats (t_pair = 겹선 벽 두께 px 추정)"""
    black = (h < BLACK_T).astype(np.uint8)
    b_th = opn(black, 5, 5)
    ink_th = opn(ink, 5, 5)
    big = cv2.dilate(opn(ink_th, 31, 31), K(3, 3))            # 바닥 채움(방 크기 덩어리) 제외
    ink_wall = ink_th & (1 - big)
    Lmin = 24
    Lh = opn(ink, Lmin, 1); Lv = opn(ink, 1, Lmin)
    thinL = (Lh | Lv) & (1 - ink_th)
    n_line = float(thinL.sum()) + 1
    d1 = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.eye(7, dtype=np.uint8))
    d2 = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.fliplr(np.eye(7, dtype=np.uint8)).copy())
    diag = (d1 | d2) & (1 - ink_th)
    hg = run_gaps(Lh) + run_gaps(Lv.T.copy())
    f = dict(black=float(b_th.sum()) / n_line, gray=float(ink_wall.sum()) / n_line, diag=float(diag.sum()) / n_line,
             big=float(big.mean()))
    wr = wall_range(s_h)
    lo, hi = (int(np.ceil(wr[0])), int(wr[1])) if wr else (4, 40)
    hi = max(hi, lo + 1)
    t = int(np.argmax(hg[lo:hi + 1]) + lo) if hg[lo:hi + 1].sum() > 0 else (int(np.mean(wr)) if wr else 10)
    f['t_pair'] = t
    f['pair_peak'] = float(hg[max(lo, t - 1):t + 2].sum() / max(1, hg[2:hi + 1].sum()))
    if f['black'] > 0.6: style = 'filled'
    elif f['gray'] > 0.8: style = 'filled_gray'
    elif f['diag'] > 0.25: style = 'hatched'
    else: style = 'double' if f['pair_peak'] > 0.12 else 'single'
    return style, f


def wall_mask(h, style, f, s_h=None):
    """스타일별 벽 마스크(1/2 해상도), 대표 두께 px"""
    ink = (h < INK_T).astype(np.uint8)
    t = f.get('t_pair', 10)
    wr = wall_range(s_h)
    Lmin = int(np.clip(700 * s_h, 12, 40)) if s_h else 24
    if style == 'filled':
        m = opn((h < BLACK_T).astype(np.uint8), 4, 4)
        tw = 4
    elif style == 'filled_gray':
        ith = opn(ink, 5, 5)
        m = ith & (1 - cv2.dilate(opn(ith, 31, 31), K(3, 3)))
        tw = 5
    else:
        if wr: dmin, dmax = int(np.ceil(wr[0])), int(np.ceil(wr[1]))
        else: dmax = int(np.clip(t * 1.35 + 2, 6, 44)); dmin = max(3, int(0.5 * t))
        dmin = max(dmin, 3)
        Lh = opn(ink, Lmin, 1); Lv = opn(ink, 1, Lmin)
        if style == 'hatched':                      # 해치 벽: 대각선까지 포함해 닫기
            Lh = Lh | (ink & cv2.dilate(Lh, K(1, dmax))); Lv = Lv | (ink & cv2.dilate(Lv, K(dmax, 1)))
        bh = opn(opn(cls(Lh, 1, dmax), 1, dmin), Lmin, 1)
        bv = opn(opn(cls(Lv, dmax, 1), dmin, 1), 1, Lmin)
        k3 = max(3, dmin)
        thick = opn(ink, k3, k3)                     # 단선(굵은 선) 벽
        th = opn(thick, Lmin, 1) | opn(thick, 1, Lmin)
        th = th & (1 - cv2.dilate(opn(thick, 25, 25), K(3, 3)))
        m = bh | bv | th
        tw = max(3, t)
    return m.astype(np.uint8), tw


def keep_main(m, tw, frac=0.08):
    """벽 마스크에서 주 구조(가장 큰 무리 대비 frac 이상) 만"""
    j = cv2.dilate(m, K(2 * tw + 3, 2 * tw + 3))
    cs, lab = comps(j, 30)
    if not cs: return m * 0, None
    mx = max(c['a'] for c in cs)
    keep = np.zeros(lab.max() + 1, bool)
    for c in cs:
        if c['a'] >= frac * mx: keep[c['i']] = True
    return (m & keep[lab]).astype(np.uint8), [c for c in cs if keep[c['i']]]


FILL_FIX = os.environ.get('REALPLAN_FILL_FIX', '1') != '0'          # (2026-10-02) 0 = 예전 동작


def outline(m, tw, s_h=None, close_mm=1300, open_mm=900):
    """벽 마스크 → 외벽 경계(채운 다각형). 문·창 틈은 닫기로 메움(축척 있으면 close_mm)"""
    H, W = m.shape
    ys, xs = np.where(m > 0)
    ext = max(xs.max() - xs.min(), ys.max() - ys.min()) if len(xs) else 100
    def closed_fill(k):
        c = cls(cv2.copyMakeBorder(m, k, k, k, k, cv2.BORDER_CONSTANT, 0), k, k)[k:-k, k:-k]
        inv = (1 - c).astype(np.uint8)
        lab_n, lab = cv2.connectedComponents(inv, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        if FILL_FIX: border.discard(0)                    # 0 = 벽(닫힌 마스크) 화소: 벽이 영역 가장자리에 닿아도 벽은 채움에 넣음
        return (~np.isin(lab, list(border))).astype(np.uint8)  # (예전: 벽이 가장자리에 닿으면 벽 전체가 '바깥'이 되어 외곽이 방 안쪽 선을 따라감)
    cm = sorted(close_mm) if isinstance(close_mm, (tuple, list)) else [close_mm]
    if s_h:
        ks = sorted({int(np.clip(v * s_h, 12, 90)) for v in cm})
    else:
        ks = [int(np.clip(max(6 * tw, 0.07 * ext), 18, 90))]
    fill = closed_fill(ks[-1])
    if len(ks) > 1:                                       # 여러 닫기: 채움 면적이 가장 큰 닫기의 FILL_KEEP 이상인 가장 작은 것
        A = float(fill.sum())                             # (작은 닫기 = 오목한 곳·테라스를 덜 메움, 벽 틈이 새면 큰 닫기로)
        for k in ks[:-1]:
            f = closed_fill(k)
            if f.sum() >= FILL_KEEP * A: fill = f; break
    ko = int(np.clip(open_mm * s_h, 7, 60)) if s_h else 2 * tw + 3
    ok = opn(fill, ko, ko)                                # 튀어나온 선(그리드·치수 연장선) 제거
    if ok.sum() > 0.5 * fill.sum(): fill = ok
    cs, lab2 = comps(fill, 100)
    if not cs: return None, None
    amax = max(c['a'] for c in cs)                        # 테라스 등으로 떨어진 동(棟)도 외곽에 포함(가장 큰 것의 20% 이상)
    keep = np.zeros(lab2.max() + 1, bool)
    for c in cs:
        if c['a'] >= 0.2 * amax: keep[c['i']] = True
    fm = keep[lab2].astype(np.uint8)
    cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [cv2.approxPolyDP(c, 2.0, True)[:, 0, :] for c in cnts if cv2.contourArea(c) > 50]
    if not polys: return None, None
    return polys, fm


def wall_segments(m, Lmin=20):
    """벽 마스크 → 축정렬 선분 [(o, c, a, b, t)] (1/2 해상도 px)"""
    segs = []
    for o in ('h', 'v'):
        mm = opn(m, Lmin, 1) if o == 'h' else opn(m, 1, Lmin)
        cs, _ = comps(mm, 10)
        for c in cs:
            if o == 'h' and c['h'] <= 3 * max(1, c['w']):
                segs.append(('h', c['y'] + c['h'] / 2, c['x'], c['x'] + c['w'], c['h']))
            elif o == 'v' and c['w'] <= 3 * max(1, c['h']):
                segs.append(('v', c['x'] + c['w'] / 2, c['y'], c['y'] + c['h'], c['w']))
    return segs


# ───────────────────────── 3. 치수 → 축척 ─────────────────────────
def ocr_tsv(img, psm=11, lang='eng', wl='0123456789,.'):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, img); fn = tf.name
    try:
        cmd = ['tesseract', fn, '-', '-l', lang, '--psm', str(psm), '-c', 'thresholding_method=0']
        if wl: cmd += ['-c', f'tessedit_char_whitelist={wl}']
        out = subprocess.run(cmd + ['tsv'], capture_output=True, text=True, timeout=120, env=dict(os.environ, OMP_THREAD_LIMIT='1')).stdout
    finally:
        os.unlink(fn)
    rows = []
    for ln in out.splitlines()[1:]:
        p = ln.split('\t')
        if len(p) == 12 and p[11].strip():
            rows.append(dict(x=int(p[6]), y=int(p[7]), w=int(p[8]), h=int(p[9]), conf=float(p[10]), text=p[11].strip()))
    return rows


def text_only(img, t=200, hmax=100, wmax=120):
    """OCR 입력 정리: 이진화 후 글자 크기 덩어리만(치수선·연장선·눈금 제거)"""
    b = (img < t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    w, h = st[:, 2], st[:, 3]
    keep = (h <= hmax) & (w <= wmax) & ~((h > 6 * w) & (h > 40)) & (st[:, 4] >= 3)
    keep[0] = False
    return np.where(keep[lab], 0, 255).astype(np.uint8)


def parse_mm(t):
    t = t.strip().strip('.,')
    m = re.fullmatch(r'(\d{1,2})[,.](\d{3})', t) or re.fullmatch(r'(\d{3,5})', t)
    if not m: return None
    v = int(''.join(m.groups()))
    if not (100 <= v <= 80000): return None
    if len(m.groups()) == 1 and len(t) == 5 and v > 60000: return None
    return v


FLIP_VERT = os.environ.get('REALPLAN_FLIP_VERT', '1') != '0'
MERGE_WORDS = os.environ.get('REALPLAN_MERGE_WORDS', '1') != '0'


def merge_split_words(words):
    """OCR 이 한 치수를 둘로 끊은 경우('12' + ',500', '1,2' + '00') → 붙인 후보 추가(둘 다 따로 치수로 읽히면 붙이지 않음)"""
    ws = sorted(words, key=lambda w: w['x0'])
    extra = []
    for i, a in enumerate(ws):
        for b in ws[i + 1:i + 4]:
            h_ = max(a['y1'] - a['y0'], b['y1'] - b['y0'], 1)
            if abs((a['y0'] + a['y1']) / 2 - (b['y0'] + b['y1']) / 2) > 0.4 * h_: continue
            gap = b['x0'] - a['x1']
            if gap < -2 or gap > 0.9 * h_: continue
            if parse_mm(a['text']) and parse_mm(b['text']): continue
            t = a['text'] + b['text']
            if parse_mm(t) is None: continue
            extra.append(dict(a, x0=a['x0'], y0=min(a['y0'], b['y0']), x1=b['x1'], y1=max(a['y1'], b['y1']), text=t, conf=min(a['conf'], b['conf'])))
    return extra


def dim_lines(ink, fm, wm):
    """가는 긴 가로선(외곽·벽 밖). ink: 1/2 해상도 잉크, fm: 외곽 채움, wm: 벽 마스크"""
    H, W = ink.shape
    Lh = opn(ink, 40, 1)
    if wm is not None: Lh[cv2.dilate(wm, K(5, 5)) > 0] = 0
    if fm is not None: Lh[cv2.dilate(fm, K(7, 7)) > 0] = 0
    Lh = cls(Lh, 25, 1)                                             # 점·원 눈금에 끊긴 치수선 잇기
    cs, _ = comps(Lh, 40)
    lines = []
    for c in cs:
        if c['h'] > 5 or c['w'] < 40: continue
        if c['y'] < 8 or c['y'] + c['h'] > H - 8: continue
        lines.append(dict(y=c['y'] + c['h'] // 2, top=c['y'], bot=c['y'] + c['h'], x0=c['x'], x1=c['x'] + c['w']))
    return lines


def line_ticks(ink, L, words):
    """치수선 L 위의 눈금(연장선·사선·점) x 위치. 선 위쪽 글자 칸은 위쪽 근거에서 뺀다"""
    x0, x1 = L['x0'], L['x1']
    up = ink[L['top'] - 6:L['top'] - 1, x0:x1].sum(0); dn = ink[L['bot'] + 1:L['bot'] + 6, x0:x1].sum(0)
    upok = up >= 2
    for w in words:
        if w['y1'] >= L['top'] - 8 and w['y0'] <= L['top'] and w['x1'] >= x0 and w['x0'] <= x1:
            upok[max(0, int(w['x0']) - x0 - 1):max(0, int(w['x1']) - x0 + 2)] = False
    dnok = dn >= 2
    for w in words:
        if w['y0'] <= L['bot'] + 8 and w['y1'] >= L['bot'] and w['x1'] >= x0 and w['x0'] <= x1:
            dnok[max(0, int(w['x0']) - x0 - 1):max(0, int(w['x1']) - x0 + 2)] = False
    tk = (upok | dnok).astype(np.uint8)
    ticks, i = [], 0
    while i < len(tk):
        if tk[i]:
            j = i
            while j < len(tk) and tk[j]: j += 1
            if j - i <= 14: ticks.append(x0 + (i + j - 1) / 2)
            i = j
        else: i += 1
    return ticks


def dims_one(hF, h, fm, wm, rot):
    """한 방향(rot=0 가로 치수 / rot=1 세로 치수를 90° 돌려 가로로) 치수 읽기.
    hF: 원본 해상도 영역, h: 1/2 해상도 영역(같은 방향으로 돌린 것). 반환 [{mm, seg(F px), a, b, y}]"""
    ink = (h < INK_T).astype(np.uint8)
    lines = dim_lines(ink, fm, wm)
    if not lines: return [], lines
    H, W = h.shape
    strip = np.zeros((H, W), np.uint8)
    TH = 44                                                    # 글자 띠 높이(1/2 해상도, 분양 평면의 큰 숫자까지)
    for L in lines:
        strip[max(0, L['top'] - TH):max(0, L['top']), max(0, L['x0'] - 10):L['x1'] + 10] = 1
        strip[L['bot']:min(H, L['bot'] + TH), max(0, L['x0'] - 10):L['x1'] + 10] = 1   # 선 아래 숫자(분양 평면 등)
    if fm is not None: strip[cv2.dilate(fm, K(5, 5)) > 0] = 0
    for L in lines: strip[L['top']:L['bot'], L['x0']:L['x1']] = 0
    sF = cv2.resize(strip, (hF.shape[1], hF.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = np.where(sF > 0, hF, 255).astype(np.uint8)
    ys, xs = np.where(sF > 0)
    if len(ys) == 0: return [], lines
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    to = text_only(crop)
    n_, _, st_, _ = cv2.connectedComponentsWithStats((to == 0).astype(np.uint8), 8)
    hs = st_[1:, 3][st_[1:, 3] >= 8]
    up = 2.0 if len(hs) >= 5 and np.median(hs) < 20 else 1.0          # 작은 글자(1/200 도면 등)는 2배 확대 후 OCR
    def read(img_):
        if up > 1:
            cu = cv2.resize(img_, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            rr = ocr_tsv(text_only(cu, hmax=100, wmax=160))
            for r in rr:
                for k in ('x', 'y', 'w', 'h'): r[k] = int(round(r[k] / up))
            return rr
        return ocr_tsv(text_only(img_))
    rows = read(crop)
    if rot == 1 and FLIP_VERT:                                  # 세로 치수 글자가 위→아래로 쓰인 도면: 돌린 띠에서 거꾸로 보임 → 180° 뒤집어 한 번 더
        n0 = sum(1 for r in rows if parse_mm(r['text']))
        if n0 < max(3, 0.5 * len(lines)):
            Hc, Wc = crop.shape
            rf = read(cv2.rotate(crop, cv2.ROTATE_180))
            if sum(1 for r in rf if parse_mm(r['text'])) > n0:
                for r in rf: r['x'], r['y'] = Wc - r['x'] - r['w'], Hc - r['y'] - r['h']
                rows = rf
    words = [dict(x0=(x0 + r['x']) / 2, y0=(y0 + r['y']) / 2, x1=(x0 + r['x'] + r['w']) / 2, y1=(y0 + r['y'] + r['h']) / 2, **r) for r in rows]
    for L in lines: L['ticks'] = line_ticks(ink, L, words)
    if MERGE_WORDS: words = words + merge_split_words(words)
    out = []
    for r in words:
        v = parse_mm(r['text'])
        if v is None: continue
        cx = (r['x0'] + r['x1']) / 2; by, ty = r['y1'], r['y0']
        tw_ = r['x1'] - r['x0']; th_ = r['y1'] - r['y0']
        if th_ < 4 or th_ > 44: continue
        # 후보: 숫자 위·아래 가까운 치수선마다 (선 위 숫자가 기본, 선 아래는 분양 평면식). 어느 선인지는 축척 합의로 고른다
        wid = len(out)
        cands = []
        for L in lines:
            if not (L['x0'] - 5 <= cx <= L['x1'] + 5): continue
            d_up = L['top'] - by; d_dn = ty - L['bot']
            if -2 <= d_up <= 1.2 * th_ + 3: sc = d_up
            elif -2 <= d_dn <= 1.2 * th_ + 3: sc = 3 + d_dn
            else: continue
            T = sorted(L['ticks'])
            a = max([t for t in T if t < cx], default=None); b = min([t for t in T if t > cx], default=None)
            if a is None or b is None: continue
            seg = b - a
            if seg < 0.8 * tw_: continue                           # 숫자가 구간보다 넓음 → 밖에 쓴 짧은 치수
            cands.append((sc, dict(mm=v, seg=float(seg * 2), a=float(a * 2), b=float(b * 2), y=float(L['y'] * 2), conf=r['conf'], rot=rot,
                                   text=r['text'], lt=[float(T[0] * 2), float(T[-1] * 2)], wid=(rot, id(r)), rank=0, _L=L)))
        cands.sort(key=lambda c: c[0])
        for k, (sc, c) in enumerate(cands[:2]):
            c['rank'] = k; out.append(c)
    return out, lines


def consensus(cands, tol=0.03, force=None, prior=None):
    """px/mm 합의: 숫자마다 후보(치수선 1~2개) → ±tol 안에 가장 많은 숫자(구간 길이 가중)가 모이는 값.
    force: 합의값을 이 값으로 고정(시트 공통 축척 적용) — 그 값에 맞는 숫자만 inlier.
    반환 s, 신뢰도, 숫자별 채택 후보 목록(inlier 표시, 채택된 선에 used 표시)"""
    if not cands: return (force if force else None), 0.0, []
    r = np.array([d['seg'] / d['mm'] for d in cands]); w = np.array([np.sqrt(d['seg']) for d in cands])
    wids = [d['wid'] for d in cands]
    best = None
    for i, v in enumerate([force] if force else r):
        inl = np.abs(r / v - 1) < tol
        per = {}
        for j in np.where(inl)[0]: per[wids[j]] = max(per.get(wids[j], 0), w[j])
        sc = sum(per.values())
        if prior and abs(np.log(v / prior)) > np.log(PRIOR_TOL): sc *= 0.25     # 문 폭 축척과 크게 어긋난 합의값은 불리하게
        if best is None or sc > best[0]: best = (sc, v, len(per))
    groups = {}
    for d in cands: groups.setdefault(d['wid'], []).append(d)
    n_words = len(groups)
    s = None
    if best[2] >= (1 if force else 2):
        v = best[1]
        pick = []
        for g in groups.values():
            ok = [d for d in g if abs(d['seg'] / d['mm'] / v - 1) < tol]
            pick.append(min(ok, key=lambda d: d['rank']) if ok else min(g, key=lambda d: d['rank']))
            pick[-1]['inlier'] = bool(ok)
        segs = np.array([d['seg'] for d in pick if d['inlier']]); rr = np.array([d['seg'] / d['mm'] for d in pick if d['inlier']])
        s = float(np.average(rr, weights=segs))
        conf = float(best[2] / n_words) * min(1.0, best[2] / 5)
    else:
        pick = [min(g, key=lambda d: d['rank']) for g in groups.values()]
        for d in pick: d['inlier'] = False
        conf = 0.0
        if force: s = float(force)
    for d in pick:
        L = d.pop('_L', None)
        if L is not None and (d['inlier'] or s is None): L['used'] = L.get('used', 0) + 1
        d.pop('wid', None); d.pop('rank', None)
    for d in cands: d.pop('_L', None)
    return s, round(conf, 3), pick


# ───────────────────────── 4. 한 영역 해석 ─────────────────────────
def read_dims(h, hF, wm, fm):
    dims, lines_all = [], []
    for rot in (0, 1):
        if rot == 0: hh, hhF, ff, ww = h, hF, fm, wm
        else:
            R = lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE) if a is not None else None
            hh, hhF, ff, ww = R(h), R(hF), R(fm), R(wm)
        d, L = dims_one(hhF, hh, ff, ww, rot)
        dims += d; lines_all.append(L)
    return dims, lines_all


def exterior_lines(lines_all, m, frac=0.03):
    """숫자가 붙은 치수선 중 바깥 선(선 너머 벽 질량 < frac) → [(rot, line)]"""
    H, W = m.shape
    tot = float(m.sum()) + 1
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return []
    cy, cx = np.median(ys), np.median(xs)
    out = []
    for rot, L in enumerate(lines_all):
        for ln in L:
            if not ln.get('used'): continue
            if rot == 0:
                y = ln['y']; a0, a1 = ln['x0'], ln['x1']
                side = m[:y, a0:a1].sum() if y < cy else m[y + 1:, a0:a1].sum()
            else:
                x = ln['y']; a0, a1 = H - ln['x1'], H - ln['x0']
                side = m[a0:a1, :x].sum() if x < cx else m[a0:a1, x + 1:].sum()
            if side < frac * tot: out.append((rot, ln))
    return out


def dim_bounds(ext_lines, m):
    """각 변에서 가장 안쪽 바깥 치수선으로 건물 범위 제한. 반환 (x0, y0, x1, y1) 1/2 해상도 또는 None"""
    H, W = m.shape
    ys, xs = np.where(m > 0)
    if len(ys) == 0 or not ext_lines: return None
    cy, cx = np.median(ys), np.median(xs)
    b = [0, 0, W - 1, H - 1]
    for rot, ln in ext_lines:
        v = ln['y']
        if rot == 0:
            if v < cy: b[1] = max(b[1], v)
            else: b[3] = min(b[3], v)
        else:
            if v < cx: b[0] = max(b[0], v)
            else: b[2] = min(b[2], v)
    if b[2] - b[0] < 0.25 * W or b[3] - b[1] < 0.25 * H: return None
    return b


def dim_clip_mask(ext_lines, m, wide=0.6):
    """dim_bounds 의 국소판: 바깥 치수선마다 '선 너머' 를 지우되, 선이 벽 범위(그 축)의 wide 이상을 덮으면 영역 전체 폭에서,
    아니면 그 선의 구간(+선 길이 10%)에서만 지운다 — 한 영역에 크기가 다른 평면이 나란히 있으면 한 평면의 치수선이 옆 평면을 자르지 않게.
    반환 남길 곳 마스크(uint8) 또는 None"""
    H, W = m.shape
    ys, xs = np.where(m > 0)
    if len(ys) == 0 or not ext_lines: return None
    cy, cx = np.median(ys), np.median(xs)
    span = {0: float(xs.max() - xs.min() + 1), 1: float(ys.max() - ys.min() + 1)}
    keep = np.ones((H, W), np.uint8)
    b = dim_bounds(ext_lines, m)
    if b is None: return None                                  # 전역판이 너무 좁다고 본 경우(25% 미만)는 자르지 않음 — 이전과 같음
    for rot, ln in ext_lines:
        v = ln['y']
        if rot == 0: a0, a1 = ln['x0'], ln['x1']
        else: a0, a1 = H - ln['x1'], H - ln['x0']
        L = a1 - a0
        if L >= wide * span[rot]: a0, a1 = 0, (W if rot == 0 else H)
        else: a0, a1 = max(0, int(a0 - 0.1 * L)), int(a1 + 0.1 * L)
        if rot == 0:
            if v < cy: keep[:max(0, v), a0:a1] = 0
            else: keep[v + 1:, a0:a1] = 0
        else:
            if v < cx: keep[a0:a1, :max(0, v)] = 0
            else: keep[a0:a1, v + 1:] = 0
    return keep


def region_prep(gF, box_q, frame_q=None, want_dims=True):
    """영역 자르기(원본 hF·1/2 h) + 치수 후보 읽기(합의 전). 시트 공통 축척을 쓰려고 분리"""
    x0, y0, x1, y1 = [v * 4 for v in box_q]
    pad = 40
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(gF.shape[1], x1 + pad), min(gF.shape[0], y1 + pad)
    x1 = x0 + (x1 - x0) // 4 * 4; y1 = y0 + (y1 - y0) // 4 * 4
    hF = gF[y0:y1, x0:x1].copy()
    if frame_q is not None:                                   # 도곽·구획선 지우기
        fq = cv2.dilate(frame_q[y0 // 4:y1 // 4, x0 // 4:x1 // 4], K(3, 3))
        hF[cv2.resize(fq, (hF.shape[1], hF.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 255
    h = minpool(hF, 2)
    ink = (h < INK_T).astype(np.uint8)
    # 1차: 치수 → 축척 (벽 모름: 굵은 채움만 제외)
    thick0 = opn(ink, 7, 7)
    dims, lines_all = read_dims(h, hF, thick0, None) if want_dims else ([], [[], []])
    return dict(hF=hF, h=h, x0=x0, y0=y0, raw=dims, lines_all=lines_all)


def analyze_region(gF, box_q, frame_q=None, want_dims=True, model_lab=None, pre=None):
    """gF: 원본 시트(회색조), box_q: 1/4 좌표 영역, frame_q: 1/4 도곽선 마스크. 반환 dict(결과), 그리기용 자료
    model_lab: (선택) 시트 전체 1/2 해상도 분할 라벨(0 배경·1 벽·2 창·3 문, firebim.str_seg) — 있으면 규칙 벽 추출 대신 사용
    pre: (선택) region_prep 결과 + 합의(s, conf, dims, scale_src) — 없으면 여기서 구함"""
    if pre is None:
        pre = region_prep(gF, box_q, frame_q, want_dims)
        pre['s'], pre['conf'], pre['dims'] = consensus(pre['raw'])
    hF, h, x0, y0, lines_all = pre['hF'], pre['h'], pre['x0'], pre['y0'], pre['lines_all']
    s, conf, dims = pre['s'], pre['conf'], pre['dims']
    ink = (h < INK_T).astype(np.uint8)
    s_h = s / 2 if s else None                                # 1/2 해상도 px/mm
    style, f = wall_style(h, ink, s_h)
    if model_lab is not None:
        return _analyze_region_model(gF, hF, h, ink, x0, y0, dims, lines_all, s, conf, s_h, style, f, model_lab)
    m0, tw0 = wall_mask(h, style, f, s_h)
    h_orig = h
    # 숫자가 붙은 '바깥' 치수선(선 너머에 벽이 거의 없음)은 벽 후보에서 지운다(겹쳐 그린 치수 사슬이 겹선 벽처럼 보임)
    ext_lines = exterior_lines(lines_all, m0)
    H2, W2 = h.shape
    dm = np.zeros_like(ink)
    for rot, ln in ext_lines:
        if rot == 0: dm[max(0, ln['top'] - 1):ln['bot'] + 1, ln['x0']:ln['x1']] = 1
        else: dm[max(0, H2 - ln['x1']):H2 - ln['x0'], max(0, ln['top'] - 1):ln['bot'] + 1] = 1
    if dm.any():
        h = np.where(dm > 0, 255, h).astype(h.dtype); ink = (h < INK_T).astype(np.uint8)
        style, f = wall_style(h, ink, s_h)
        m, tw = wall_mask(h, style, f, s_h)
    else:
        m, tw = m0, tw0
    bnd = dim_bounds(ext_lines, m)
    if bnd is not None:                                       # 바깥 치수선 너머(범례·상세·연장선)는 건물이 아님
        x0b, y0b, x1b, y1b = bnd
        m[:max(0, y0b)] = 0; m[y1b + 1:] = 0; m[:, :max(0, x0b)] = 0; m[:, x1b + 1:] = 0
    m, parts = keep_main(m, tw)
    poly, fm = outline(m, tw, s_h) if m.sum() > 0 else (None, None)
    res = dict(offset_F=[int(x0), int(y0)], style=style, style_feats={k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()},
               wall_t_px=int(tw), wall_px=int(m.sum()))
    if poly is not None:
        bx, by, bw, bh = cv2.boundingRect(np.vstack(poly).astype(np.int32))
        res['outline'] = [(p * 2).tolist() for p in poly]; res['outline_bbox_F'] = [int(bx * 2), int(by * 2), int((bx + bw) * 2), int((by + bh) * 2)]
        res['outline_parts'] = len(poly)
        res['outline_area_frac'] = round(float(fm.mean()), 3)
    segs = wall_segments(m)
    res['n_wall_segs'] = len(segs)
    res['dims'] = dims; res['n_dims'] = len(dims)
    res['px_per_mm'] = s; res['scale_conf'] = conf
    if s: res['scale_1_to'] = round(300 / 25.4 / s, 1)          # 300dpi 출력 기준 도면 축척 1/N
    res['total_check'] = total_check(res, fm, tw, h.shape)
    viz = dict(h=h_orig, m=m, fm=fm, poly=poly, segs=segs, lines=lines_all)
    return res, viz


M_CLOSE_MM = tuple(float(v) for v in os.environ.get('REALPLAN_M_CLOSE', '300,600,1300').split(','))  # 모델 경로 외곽 닫기 후보(개구부는 이미 창·문으로 막힘)
FILL_KEEP = float(os.environ.get('REALPLAN_FILL_KEEP', 0.85))
M_OPEN_MM = float(os.environ.get('REALPLAN_M_OPEN', 900))
M_DIMCLIP = os.environ.get('REALPLAN_M_DIMCLIP', 'local')    # 모델 경로 바깥 치수선 자르기: local(2026-10-02 기본) | global(이전) | off


def _analyze_region_model(gF, hF, h, ink, x0, y0, dims, lines_all, s, conf, s_h, style, f, model_lab):
    """analyze_region 의 모델 경로: 벽 = 모델 벽, 외곽 = 벽 ∪ 창 ∪ 문(개구부가 외벽을 닫음). 치수·축척·검사는 규칙 경로와 같음"""
    from firebim import str_seg as SS
    H2, W2 = h.shape
    ml = np.zeros((H2, W2), np.uint8)
    sub = model_lab[y0 // 2:y0 // 2 + H2, x0 // 2:x0 // 2 + W2]
    ml[:sub.shape[0], :sub.shape[1]] = sub
    m = (ml == 1).astype(np.uint8)
    op = ((ml == 2) | (ml == 3)).astype(np.uint8)
    tw = int(max(3, round(SS.wall_thickness(m)))) if m.any() else 3
    ext_lines = exterior_lines(lines_all, m)
    if M_DIMCLIP == 'local':                                  # 바깥 치수선 너머 지우기를 그 선이 덮는 구간에서만(한 영역에 평면 여럿)
        keep = dim_clip_mask(ext_lines, m)
        if keep is not None: m &= keep; op &= keep
    elif M_DIMCLIP != 'off':
        bnd = dim_bounds(ext_lines, m)
        if bnd is not None:
            x0b, y0b, x1b, y1b = bnd
            for a in (m, op):
                a[:max(0, y0b)] = 0; a[y1b + 1:] = 0; a[:, :max(0, x0b)] = 0; a[:, x1b + 1:] = 0
    mo, parts = keep_main(m | op, tw)
    m = m & mo; op = op & mo
    poly, fm = outline(mo, tw, s_h, M_CLOSE_MM, M_OPEN_MM) if mo.sum() > 0 else (None, None)
    res = dict(offset_F=[int(x0), int(y0)], style=style, style_feats={k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()},
               wall_t_px=int(tw), wall_px=int(m.sum()), wall_source='model')
    if poly is not None:
        bx, by, bw, bh = cv2.boundingRect(np.vstack(poly).astype(np.int32))
        res['outline'] = [(p * 2).tolist() for p in poly]; res['outline_bbox_F'] = [int(bx * 2), int(by * 2), int((bx + bw) * 2), int((by + bh) * 2)]
        res['outline_parts'] = len(poly)
        res['outline_area_frac'] = round(float(fm.mean()), 3)
    segs = wall_segments(m)
    res['n_wall_segs'] = len(segs)
    def inst(mask, kind):                                  # 1/2 해상도 개체 → 원본(F) 좌표 (작은 조각 제거: 작업 해상도 기준값 ×2 길이·×4 면적)
        out = []
        L = SS.instances(mask, 12)
        L = SS.filter_instances(L, 120, 24, 0.12) if kind == 'door' else SS.filter_instances(L, 96, 18, 0.10)
        for d in L:
            b = d['box']
            out.append(dict(kind=kind, x=round(d['cx'] * 2 + x0, 1), y=round(d['cy'] * 2 + y0, 1), w=round(d['w'] * 2, 1), d=round(d['d'] * 2, 1),
                            ang=round(d['ang'], 1), box_F=[b[0] * 2 + x0, b[1] * 2 + y0, b[2] * 2 + x0, b[3] * 2 + y0],
                            w_mm=round(d['w'] * 2 / s) if s else None))
        return out
    res['doors'] = inst((ml == 3) & (mo > 0), 'door'); res['windows'] = inst((ml == 2) & (mo > 0), 'window')
    res['dims'] = dims; res['n_dims'] = len(dims)
    res['px_per_mm'] = s; res['scale_conf'] = conf
    if s: res['scale_1_to'] = round(300 / 25.4 / s, 1)
    res['total_check'] = total_check(res, fm, tw, h.shape)
    viz = dict(h=h, m=m, fm=fm, poly=poly, segs=segs, lines=lines_all)
    return res, viz


_SEG = None
_LAST_ML = None


def model_labels(gF, threads=None, src=None):
    """시트 전체 → 1/2 해상도(H) 분할 라벨 (firebim.str_seg ONNX, 원본 300dpi 가정).
    src: (선택) dict(img=다른 해상도 원본 영상, dpi=그 dpi) — 축소본(예: 124dpi)을 되돌려 키운 gF 대신 원래 축소본을 모델에 넣음
         또는 dict(lab=미리 구한 1/2 해상도 라벨)"""
    global _SEG
    from firebim import str_seg as SS
    H2, W2 = gF.shape[0] // 2, gF.shape[1] // 2
    if src is not None and src.get('lab') is not None:
        return cv2.resize(src['lab'], (W2, H2), interpolation=cv2.INTER_NEAREST) if src['lab'].shape != (H2, W2) else src['lab']
    if _SEG is None: _SEG = SS.StrSeg(threads=threads)
    if src is not None and src.get('img') is not None:
        lab, p, k = _SEG.predict(src['img'], src_dpi=src['dpi'])
    else:
        lab, p, k = _SEG.predict(gF, src_dpi=300)
    pu = np.stack([cv2.resize(c, (W2, H2), interpolation=cv2.INTER_LINEAR) for c in p])
    return pu.argmax(0).astype(np.uint8)


def total_check(res, fm, tw, shape):
    """자기지도 정확도: 축별로 (1) 치수 사슬 합집합 범위(최외곽 치수 눈금 사이) 와 (2) 가장 긴 단일 치수(전체치수)를
    외곽 범위와 비교. mm 환산은 합의 축척 s. 오차 = |외곽 mm − 치수 mm| / 치수 mm"""
    s = res.get('px_per_mm')
    if not s or fm is None or 'outline_bbox_F' not in res: return None
    bx0, by0, bx1, by1 = res['outline_bbox_F']
    ox, oy = res['offset_F']
    Hh = shape[0] * 2                                     # 돌린 좌표(세로 치수)는 x' = H-1-y
    out = {}
    for rot, (lo, hi) in ((0, (bx0, bx1)), (1, (Hh - by1, Hh - by0))):
        ext = hi - lo
        D = [d for d in res['dims'] if d['rot'] == rot and d.get('inlier') and d['a'] > lo - 0.12 * ext and d['b'] < hi + 0.12 * ext]
        if not D: continue
        u0, u1 = min(d['a'] for d in D), max(d['b'] for d in D)
        iv = sorted((d['a'], d['b']) for d in D); cov, cur = 0.0, list(iv[0])
        for a_, b_ in iv[1:]:
            if a_ <= cur[1] + 0.004 * ext: cur[1] = max(cur[1], b_)
            else: cov += cur[1] - cur[0]; cur = [a_, b_]
        cov = (cov + cur[1] - cur[0]) / max(1e-6, u1 - u0)
        big = max(D, key=lambda d: d['seg'])
        # 검사 가능: 끊김 없는 치수 사슬(합집합 95% 이상)이 그 치수선의 첫 눈금~끝 눈금을 90% 이상 덮음(사슬을 다 읽음)
        lt = max((d['lt'] for d in D), key=lambda t: t[1] - t[0])
        full = (min(u1, lt[1]) - max(u0, lt[0])) / max(1e-6, lt[1] - lt[0])
        if cov < 0.95 or full < 0.9: continue
        uni_mm = (u1 - u0) / s
        o_mm = ext / s; o_cl = (ext - 2 * tw) / s
        e_u = min(abs(o_mm - uni_mm), abs(o_cl - uni_mm)) / uni_mm
        r = dict(union_mm=round(uni_mm), outline_mm=round(o_mm), err=round(e_u, 4), err_raw=round(abs(o_mm - uni_mm) / uni_mm, 4),
                 edge_mm=[round((lo - u0) / s), round((hi - u1) / s)], n=len(D))
        if big['seg'] >= 0.6 * ext:
            r['total_mm'] = big['mm']; r['err_total'] = round(min(abs(o_mm - big['mm']), abs(o_cl - big['mm'])) / big['mm'], 4)
        out['x' if rot == 0 else 'y'] = r
    if not out: return None
    out['err'] = round(float(np.mean([v['err'] for k, v in out.items() if k in ('x', 'y')])), 4)
    et = [v['err_total'] for k, v in out.items() if k in ('x', 'y') and 'err_total' in v]
    out['err_total'] = round(float(np.mean(et)), 4) if et else None
    return out


# ───────────────────────── 5. 시트 전체 ─────────────────────────
CAP_KEYS = ('평면', '세대', 'PLAN', 'plan', 'Plan', '㎡', 'm²', '형', '층')


def caption_for(gF, reg, caps):
    """영역 아래(또는 위) 가까운 가로 글자 덩어리 → OCR(한글+영문)"""
    b = reg['box']; best = None
    for c in caps:
        if b[0] - 20 <= (c[0] + c[2]) / 2 <= b[2] + 20:
            if c[1] >= b[3] - 5: d = c[1] - b[3]
            elif c[3] <= b[1] + 5: d = b[1] - c[3] + 10
            else: continue
            if d <= 90 and (best is None or d < best[0]): best = (d, c)
    if best is None: return None
    c = best[1]
    crop = gF[max(0, c[1] * 4 - 6):c[3] * 4 + 6, max(0, c[0] * 4 - 6):c[2] * 4 + 6]
    if crop.size == 0: return None
    rows = ocr_tsv(crop, psm=7, lang='kor+eng', wl=None)
    txt = ' '.join(r['text'] for r in rows if r['conf'] > 20)
    return txt if any(k in txt for k in CAP_KEYS) else None


SHEET_POOL = os.environ.get('REALPLAN_SHEET_POOL', '1') != '0'
DOOR_MM = 1040.0          # 모델 문 개체 폭(문틀 포함) 중앙값 ≈ 1040 mm (obj2 개발 60장, 치수 축척 신뢰도≥0.6 영역 44개, 10~90% ±11%)
PRIOR_TOL = 1.45          # 치수 합의값이 문 폭 축척과 이 배 이상 다르면 불리하게
DOOR_PRIOR = os.environ.get('REALPLAN_DOOR_PRIOR', '1') != '0'


def door_scale(ml, pre):
    """모델 문 개체 폭 중앙값 → px/mm(원본 F) 추정 (문 3개 이상). ml: 시트 1/2 해상도 라벨"""
    from firebim import str_seg as SS
    H2, W2 = pre['h'].shape
    sub = ml[pre['y0'] // 2:pre['y0'] // 2 + H2, pre['x0'] // 2:pre['x0'] // 2 + W2]
    L = SS.filter_instances(SS.instances((sub == 3).astype(np.uint8), 12), 120, 24, 0.12)
    if len(L) < 3: return None
    return float(np.median([d['w'] * 2 for d in L])) / DOOR_MM


def sheet_scale(pres, tol=0.03):
    """영역별 축척 합의 + 시트 공통 축척: 한 시트의 여러 평면은 대개 같은 축척 →
    영역 자체 합의가 없거나 약하면(신뢰도<0.5) 시트 전체 숫자를 모은 합의값을 쓴다
    (그 영역 숫자 중 공통값에 맞는 것이 자체 합의 수 이상일 때만 — 1/50 상세 평면은 자기 값 유지). pres 를 고침"""
    pool, own = [], []
    for k, p in enumerate(pres):
        raw = p['raw']
        keep = [dict(d) for d in raw]                          # 고정값 재합의용(치수선 _L 연결 유지)
        r = np.array([d['seg'] / d['mm'] for d in raw]) if raw else np.zeros(0)
        wids = [d['wid'] for d in raw]
        for d in raw:
            c = {kk: v for kk, v in d.items() if kk != '_L'}; c['wid'] = (k,) + tuple(d['wid']); pool.append(c)
        p['s'], p['conf'], p['dims'] = consensus(raw, tol, prior=p.get('door_s'))
        p['scale_src'] = 'region' if p['s'] else None
        n_own = len({w for w, rr in zip(wids, r) if abs(rr / p['s'] - 1) < tol}) if p['s'] else 0
        own.append((keep, r, wids, n_own))
    S, C = None, 0.0
    if SHEET_POOL and len(pres) >= 2:
        ds = [p['door_s'] for p in pres if p.get('door_s')]
        S, C, _ = consensus(pool, tol, prior=float(np.median(ds)) if ds else None)
    if not S or C < 0.3: S = None
    for p, (keep, r, wids, n_own) in zip(pres, own):
        if S is None: continue
        if p['s'] and p['conf'] >= 0.5: continue
        agree = len({w for w, rr in zip(wids, r) if abs(rr / S - 1) < tol})
        if p['s'] and agree < n_own: continue
        for L in p['lines_all']:
            for ln in L: ln.pop('used', None)
        _, c_, p['dims'] = consensus(keep, tol, force=S)
        p['s'] = float(S); p['conf'] = round(max(c_, float(C) * 0.8), 3); p['scale_src'] = 'sheet'
    for p in pres:                                             # 치수를 못 읽은 영역: 문 폭 축척(낮은 신뢰도)
        if not p['s'] and p.get('door_s'):
            p['s'] = p['door_s']; p['conf'] = 0.2; p['scale_src'] = 'doors'
    return pres


def model_available():
    """실제 도면 기본 경로 = 분할 모델(models/str_seg.onnx + onnxruntime 이 있을 때). 없으면 규칙 경로"""
    from firebim import str_seg as SS
    if not os.path.exists(SS.MODEL): return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def process_sheet(path, want_dims=True, want_caption=True, model=None):
    """model=None(기본): 모델 파일·onnxruntime 있으면 벽·창·문을 분할 모델(firebim/str_seg.py)로, False = 규칙 벽.
    배치·치수·축척·외곽 검사는 같음"""
    gF = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gF is None: raise IOError(path)
    return process_gray(gF, os.path.basename(path), want_dims, want_caption, model)


def analyze(img, model=None, want_dims=True, want_caption=True):
    """편의 입구: img = 경로 또는 회색조 배열(300dpi 원본 기준). model=True 면 벽·창·문 = 분할 모델. 반환 결과 dict (process_sheet 의 첫 값)"""
    if isinstance(img, str): return process_sheet(img, want_dims, want_caption, model)[0]
    return process_gray(img, 'array', want_dims, want_caption, model)[0]


def process_gray(gF, name, want_dims=True, want_caption=True, model=None, model_src=None):
    """model: None=있으면 모델(기본)·True·False(규칙). model_src: model_labels 의 src (축소본 영상+dpi, 또는 미리 구한 라벨)"""
    global _LAST_ML
    if model is None: model = model_available()
    t0 = time.time()
    q = minpool(gF, 4)
    regs, caps, fr = sheet_layout(q)
    out = dict(file=name, shape=list(gF.shape), regions=[], wall_source='model' if model else 'rules')
    ml = model_labels(gF, src=model_src) if model else None
    _LAST_ML = ml
    vizs = []
    pres = [region_prep(gF, r['box'], fr, want_dims) for r in regs]
    if ml is not None and DOOR_PRIOR:
        for pre in pres: pre['door_s'] = door_scale(ml, pre)
    sheet_scale(pres)
    for r, pre in zip(regs, pres):
        res, viz = analyze_region(gF, r['box'], fr, want_dims, model_lab=ml, pre=pre)
        res['scale_src'] = pre.get('scale_src')
        res['box_Q'] = r['box']; res['layout_feats'] = {k: round(v, 3) for k, v in r['feats'].items()}
        ob = res.get('outline_bbox_F')
        nmin = 4 if res.get('wall_source') == 'model' else 8          # 모델 벽은 이어진 덩어리라 선분 수가 적음(단순한 작은 평면)
        res['kind'] = 'plan' if ob and min(ob[2] - ob[0], ob[3] - ob[1]) >= 240 and res['n_wall_segs'] >= nmin else 'other'
        if want_caption and res['kind'] == 'plan':
            try: res['caption'] = caption_for(gF, r, caps)
            except Exception: res['caption'] = None
        out['regions'].append(res); vizs.append(viz)
    out['runtime_s'] = round(time.time() - t0, 2)
    return out, vizs, q, fr


# ───────────────────────── 그리기 ─────────────────────────
def draw_region(res, viz, maxw=1400):
    h = viz['h']
    img = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    img = (0.55 * img + 0.45 * 255).astype(np.uint8)
    m = viz['m'] > 0
    img[m] = (0.4 * img[m] + 0.6 * np.array([0, 0, 230])).astype(np.uint8)
    if viz['poly'] is not None:
        cv2.polylines(img, [p.astype(np.int32) for p in viz['poly']], True, (0, 170, 0), 2)
    H, W = h.shape
    for rot, L in enumerate(viz['lines']):
        for ln in L:
            if rot == 0:
                cv2.line(img, (ln['x0'], ln['y']), (ln['x1'], ln['y']), (255, 150, 0), 1)
                for t in ln['ticks']: cv2.circle(img, (int(t), ln['y']), 3, (255, 0, 0), 1)
            else:   # 돌린 좌표 → 원래: (x', y') in rotated CW:  x = y', y = H-1-x'
                cv2.line(img, (ln['y'], H - 1 - ln['x0']), (ln['y'], H - 1 - ln['x1']), (255, 150, 0), 1)
                for t in ln['ticks']: cv2.circle(img, (ln['y'], int(H - 1 - t)), 3, (255, 0, 0), 1)
    for d in res['dims']:
        col = (0, 140, 255) if not d.get('inlier') else (200, 0, 200)
        a, b, y = d['a'] / 2, d['b'] / 2, d['y'] / 2
        if d['rot'] == 0: p, q_ = (int(a), int(y) - 3), (int(b), int(y) - 3); tp = (int((a + b) / 2) - 20, int(y) - 8)
        else: p, q_ = (int(y) - 3, int(H - 1 - a)), (int(y) - 3, int(H - 1 - b)); tp = (int(y) - 60, int(H - 1 - (a + b) / 2))
        cv2.line(img, p, q_, col, 2)
        cv2.putText(img, str(d['mm']), tp, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    ox, oy = res['offset_F']
    for d in res.get('doors', []) + res.get('windows', []):        # 모델 경로: 문(초록)·창(파랑) 상자
        b = [(d['box_F'][0] - ox) // 2, (d['box_F'][1] - oy) // 2, (d['box_F'][2] - ox) // 2, (d['box_F'][3] - oy) // 2]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 160, 0) if d['kind'] == 'door' else (230, 90, 0), 2)
    tc = res.get('total_check') or {}
    txt = f"{res.get('wall_source', 'rules')} {res['style']} t={res['wall_t_px']} s={res.get('px_per_mm') and round(res['px_per_mm'],4)} conf={res.get('scale_conf')} err={tc.get('err')}"
    cv2.putText(img, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(img, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    if img.shape[1] > maxw:
        f = maxw / img.shape[1]; img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def draw_sheet(out, q, fr):
    img = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    img[fr > 0] = (0, 200, 255)
    for r in out['regions']:
        b = r['box_Q']; col = (0, 180, 0) if r['kind'] == 'plan' else (160, 160, 160)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2)
        cv2.putText(img, f"{r['kind']}:{r['style']}", (b[0] + 3, b[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img


if __name__ == '__main__':
    path, od = sys.argv[1], sys.argv[2]
    os.makedirs(od, exist_ok=True)
    out, vizs, q, fr = process_sheet(path, model=False if '--rules' in sys.argv else None)
    stem = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(od, stem + '_sheet.png'), draw_sheet(out, q, fr))
    for i, (r, v) in enumerate(zip(out['regions'], vizs)):
        cv2.imwrite(os.path.join(od, f'{stem}_r{i}.png'), draw_region(r, v))
    for r in out['regions']:
        print(r['kind'], r['style'], 'box', r['box_Q'], 's', r['px_per_mm'] and round(r['px_per_mm'], 4), 'conf', r['scale_conf'], 'ndims', r['n_dims'],
              'total', json.dumps(r['total_check']), 'cap', r.get('caption'))
    print('runtime', out['runtime_s'])
