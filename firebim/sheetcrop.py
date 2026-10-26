# -*- coding: utf-8 -*-
"""CAD 도면 시트 → 평면도 그림 영역 (2026-10-03 2차 강건화, review/attach.crop_sheet 가 부름).

예전 crop_sheet 는 '가장 큰 도면 영역'(realplan.sheet_layout) 하나만 골랐는데, 표제란이 도곽 아래·오른쪽에 붙어 있거나(공백 없음)
범례·주기 표가 평면 가까이 있으면 그것까지 한 영역으로 묶여 잘라내지 못했다(공개 도면 Z·K).
  1) 도곽: 시트 가장자리 쪽 긴 직선(가로·세로 50% 이상)의 바깥 상자.
  2) 표제란 띠: 도곽 안에서 도곽 한 변과 평행하게 도곽 길이의 80% 이상 이어지는 긴 선이 그 변에서 30% 안에 있고,
     그 선과 도곽 변 사이 띠가 '표'(띠 폭의 40% 넘게 가로지르는 칸 선 4개 이상) + 글자 밀도가 있으면 표제란 → 잘라냄. 여러 선이면 가장 안쪽.
  3) 범례·주기·일람 표: 남은 영역에서 얇은 직선으로 닫힌 상자 안에 칸 줄(상자 폭 90% 이상 가로선) 3개 이상, 줄 간격 중앙값 ≤ 용지 12 mm,
     줄의 절반 이상에 글자 → 표로 보고 흰색으로 지움(좌표는 그대로).
  4) 남은 그림에서 realplan.sheet_layout 의 도면 영역 중 가장 큰 것(+ 같은 크기 수준의 다른 평면은 split 에서).
캡션 나누기(split_regions): 한 시트에 평면도가 여럿(캡션 '1층 평면도'·'2층 평면도')이면 캡션마다 가장 가까운(주로 위쪽) 도면 영역을 대응시켜 층별로 자름.
끄기: FIREBIM_SHEET_CROP2=0 (예전 crop_sheet)."""
import os
import numpy as np, cv2

ENABLED = os.environ.get('FIREBIM_SHEET_CROP2', '1') != '0'
THICK_FRAME = os.environ.get('FIREBIM_THICK_FRAME', '1') != '0'


def _K(w, h): return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w)), max(1, int(h))))


def _opn(m, w, h): return cv2.morphologyEx(m, cv2.MORPH_OPEN, _K(w, h))


def _runs(v, min_len):
    """1차원 불리언 → [(시작, 끝)] (길이 ≥ min_len)"""
    v = np.r_[False, v.astype(bool), False]
    d = np.diff(v.astype(np.int8))
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(a, b) for a, b in zip(s, e) if b - a >= min_len]


def _merge_close(v, d):
    out = []
    for x in sorted(v):
        if out and x - out[-1][-1] <= d: out[-1].append(x)
        else: out.append([x])
    return [int(round(sum(g) / len(g))) for g in out]


def frame_box(ink):
    """도곽 상자 [x0,y0,x1,y1] (q px) — 가장자리 쪽 긴 선. 없으면 그림 전체"""
    H, W = ink.shape
    hl = _opn(ink, int(0.5 * W), 1); vl = _opn(ink, 1, int(0.5 * H))
    rows = np.nonzero(hl.sum(1) > 0.5 * W)[0]; cols = np.nonzero(vl.sum(0) > 0.5 * H)[0]
    x0, x1 = (cols.min(), cols.max()) if len(cols) >= 2 and cols.max() - cols.min() > 0.5 * W else (0, W - 1)
    y0, y1 = (rows.min(), rows.max()) if len(rows) >= 2 and rows.max() - rows.min() > 0.5 * H else (0, H - 1)
    return [int(x0), int(y0), int(x1), int(y1)]


def _table_rows(ink, box, o):
    """상자 안 칸 선(o='h': 상자 폭 40% 이상 가로선 줄 수 / 'v': 세로선)"""
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    if sub.size == 0: return 0, []
    if o == 'h':
        L = max(3, int(0.4 * (x1 - x0))); m = _opn(sub, L, 1); prof = m.any(1)
    else:
        L = max(3, int(0.4 * (y1 - y0))); m = _opn(sub, 1, L); prof = m.any(0)
    rr = _runs(prof, 1)
    return len(rr), [(a + b) / 2 for a, b in rr]


def title_strip(ink, fb, mm_px):
    """표제란 띠 상자 | None. mm_px = 용지 1 mm 의 q px"""
    x0, y0, x1, y1 = fb
    FW, FH = x1 - x0, y1 - y0
    best = None
    sub = ink[y0:y1 + 1, x0:x1 + 1]
    vl = _opn(sub, 1, int(0.8 * FH)); hl = _opn(sub, int(0.8 * FW), 1)
    vcols = [int(round((a + b) / 2)) for a, b in _runs(vl.any(0), 1)]
    hrows = [int(round((a + b) / 2)) for a, b in _runs(hl.any(1), 1)]
    edge = 2.5 * mm_px
    cands = []
    for c in vcols:
        if edge < c < 0.3 * FW: cands.append(('L', [x0, y0, x0 + c, y1], c))
        if FW - 0.3 * FW < c < FW - edge: cands.append(('R', [x0 + c, y0, x1, y1], FW - c))
    for r in hrows:
        if edge < r < 0.3 * FH: cands.append(('T', [x0, y0, x1, y0 + r], r))
        if FH - 0.3 * FH < r < FH - edge: cands.append(('B', [x0, y0 + r, x1, y1], FH - r))
    for side, b, width in cands:
        o = 'h' if side in 'LR' else 'v'                 # 세로 띠(오른쪽 표제란) = 가로 칸 선이 많음
        n, _ = _table_rows(ink, b, o)
        n2, _ = _table_rows(ink, b, 'v' if o == 'h' else 'h')
        dens = float(ink[b[1]:b[3], b[0]:b[2]].mean()) if (b[3] - b[1]) * (b[2] - b[0]) > 0 else 0
        if n >= 4 and dens >= 0.01:
            # 가장 안쪽(띠가 넓은) 선 — 단 표 꼴은 유지
            if best is None or width > best[2]: best = (side, b, width, n, n2)
    return best


TABLE_TEXT = os.environ.get('FIREBIM_TABLE_TEXT', '1') != '0'   # 표 칸 줄에 '글자'가 있어야 표(2026-10-04: 타일 줄눈 그린 옥상 평면을 표로 보고 지우던 것 — 새 세트 E 지붕층)


def _has_chars(band, mm_px):
    """칸 줄 띠 안에 글자 꼴 잉크(긴 가로·세로선 지운 뒤 용지 0.6~8 mm 크기 성분 2개 이상)가 있나"""
    b = band.astype(np.uint8)
    h, w = b.shape
    lines = _opn(b, max(3, int(0.3 * w)), 1) | _opn(b, 1, max(3, int(0.8 * h)))
    r = b & (1 - cv2.dilate(lines, _K(3, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(r, 8)
    k = 0
    for i in range(1, n):
        cw, ch = st[i, 2], st[i, 3]
        if 0.6 * mm_px <= max(cw, ch) <= 8 * mm_px and st[i, 4] >= 3: k += 1
        if k >= 2: return True
    return False


def table_boxes(ink, area_box, mm_px):
    """영역 안 범례·주기·일람 표 상자 목록(q px)"""
    x0, y0, x1, y1 = area_box
    sub = ink[y0:y1, x0:x1]
    if sub.size == 0: return []
    L = max(8, int(12 * mm_px))
    lines = _opn(sub, L, 1) | _opn(sub, 1, L)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(lines, _K(3, 3)), 8)
    out = []
    A = sub.shape[0] * sub.shape[1]
    for i in range(1, n):
        bx, by, bw, bh, _ = st[i]
        if bw < 20 * mm_px or bh < 8 * mm_px or bw * bh > 0.3 * A: continue
        box = [bx, by, bx + bw, by + bh]
        nr, ys = _table_rows(sub, box, 'h')
        if nr < 3: continue
        # 칸 줄: 상자 폭 60% 이상 가로선. 겹선(1.5 mm 안 평행선 짝)이 대부분이면 벽(방) → 표 아님
        m = _opn(sub[by:by + bh, bx:bx + bw], int(0.6 * bw), 1)
        raw = [int(round((a + b) / 2)) for a, b in _runs(m.any(1), 1)]
        rows = _merge_close(raw, 2.2 * mm_px)
        vm = _opn(sub[by:by + bh, bx:bx + bw], 1, int(0.9 * bh))
        rawc = [int(round((a + b) / 2)) for a, b in _runs(vm.any(0), 1)]
        cols = _merge_close(rawc, 2.2 * mm_px)
        if len(rows) < 3 or len(cols) < 2: continue
        if (len(raw) - len(rows)) + (len(rawc) - len(cols)) >= 0.3 * (len(rows) + len(cols)): continue      # 겹선(2 mm 안 평행선 짝) = 벽 → 방이지 표가 아님
        gaps = np.diff(rows)
        if not len(gaps) or float(np.median(gaps)) > 15 * mm_px: continue
        # 줄 사이 글자 있는 칸 줄: 2개 이상(3개 미만이면 칸 높이가 고르게) — 방 하나(글자 한 줄)는 표가 아님
        tb = []
        for a_, b_ in zip(rows, rows[1:]):
            band = sub[by + a_ + 2:by + b_ - 1, bx + 2:bx + bw - 2]
            if b_ - a_ >= 2.5 * mm_px and band.size and band.mean() > 0.01 and (not TABLE_TEXT or _has_chars(band, mm_px)): tb.append(b_ - a_)
        if len(tb) < 2 or (len(tb) < 3 and max(tb) > 2.5 * min(tb)): continue
        if TABLE_TEXT:                                   # 표 아님: 폭이 용지 35 mm 미만(치수선·버블 줄 띠) · 칸 줄마다 세로 획이 15개 넘음(타일 줄눈·격자 그린 평면)
            if bw < 35 * mm_px: continue
            vs = []
            for a_, b_ in zip(rows, rows[1:]):
                band = sub[by + a_ + 2:by + b_ - 1, bx + 2:bx + bw - 2]
                if band.size and b_ - a_ >= 2.5 * mm_px:
                    vs.append(len(_runs(_opn(band, 1, max(3, int(0.8 * band.shape[0]))).any(0), 1)))
            if vs and float(np.median(vs)) > 15: continue
        if len(tb) < 0.5 * (len(rows) - 1): continue
        out.append([bx + x0, by + y0, bx + bw + x0, by + bh + y0])
    return out


def analyze(img):
    """시트 그림 → dict(scale f(원본/q), frame, title, tables, regions, crop[원본 px]) | None(시트 아님)"""
    from firebim import realplan as RP
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    f = max(1, int(round(max(H, W) / 1240)))
    q = RP.minpool(g, f) if f > 1 else g
    Hq, Wq = q.shape
    ink = (q < 190).astype(np.uint8)
    fr = RP.frame_lines(ink)
    bx, by = max(2, int(0.08 * Wq)), max(2, int(0.08 * Hq))
    sides = [fr[:, :bx].sum() > 0.5 * Hq, fr[:, -bx:].sum() > 0.5 * Hq, fr[:by].sum() > 0.5 * Wq, fr[-by:].sum() > 0.5 * Wq]
    if sum(sides) < 3 and THICK_FRAME:                 # 굵은 펜 도곽(두께 > 3 q px)은 frame_lines 가 굵은 벽으로 보고 뺌 → 가장자리 띠 안 긴 직선(길이 50% 이상)으로 다시 봄(2026-10-04, 새 세트 E)
        hl, vl = _opn(ink, int(0.5 * Wq), 1), _opn(ink, 1, int(0.5 * Hq))
        sides = [s_ or b_ for s_, b_ in zip(sides, [vl[:, :bx].any(), vl[:, -bx:].any(), hl[:by].any(), hl[-by:].any()])]
    if sum(sides) < 3: return None
    fb = frame_box(ink)
    mm_px = max(Wq, Hq) / 420.0                          # A 계열 시트 긴 변 = 용지 긴 변(배율 무관 비례) — A3 기준 1 mm
    ts = title_strip(ink, fb, mm_px)
    area = list(fb)
    if ts:
        side, b = ts[0], ts[1]
        if side == 'R': area[2] = b[0]
        elif side == 'L': area[0] = b[2]
        elif side == 'B': area[3] = b[1]
        else: area[1] = b[3]
    pad = int(1.5 * mm_px)
    area = [area[0] + pad, area[1] + pad, area[2] - pad, area[3] - pad]
    tabs = table_boxes(ink, area, mm_px)
    q2 = np.full_like(q, 255)
    q2[area[1]:area[3], area[0]:area[2]] = q[area[1]:area[3], area[0]:area[2]]
    for t in tabs: q2[max(0, t[1] - 2):t[3] + 3, max(0, t[0] - 2):t[2] + 3] = 255
    regs, caps, _ = RP.sheet_layout(q2)
    regs = [r for r in regs if (r['box'][2] - r['box'][0]) * (r['box'][3] - r['box'][1]) >= 0.02 * Wq * Hq]
    return dict(f=f, q_shape=(Hq, Wq), q2=q2, frame=fb, title=ts[1] if ts else None, title_side=ts[0] if ts else None, area=area, tables=tabs,
                regions=[r['box'] for r in regs], caps=caps, mm_px=mm_px)


def crop(img, A=None):
    """→ (잘린 그림, [x0,y0,x1,y1] 원본 px | None, 지운 표 상자들(잘린 그림 px)). 가장 큰 도면 영역 + 여백. 표는 흰색으로 지움"""
    A = A or analyze(img)
    if not A or not A['regions']: return img, None, []
    f = A['f']; Hq, Wq = A['q_shape']
    b = max(A['regions'], key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    frac = (b[2] - b[0]) * (b[3] - b[1]) / float(Wq * Hq)
    if not 0.03 < frac < 0.92: return img, None, []
    # 평면 하나: 도곽 안 표제란 밖 전체(그리드 버블·치수·캡션 포함 — 버블·치수는 축척·층 정합에 씀) 중 잉크가 있는 범위
    b = ink_box(A)
    if BUBBLE_KEEP and A.get('q2') is not None: b = bubble_extend(A, b)
    return cut(img, A, b)


BUBBLE_KEEP = os.environ.get('FIREBIM_CROP_BUBBLES', '1') != '0'     # 도면 영역 밖으로 떨어져 한 줄로 선 그리드 버블 원까지 자르기 범위 넓힘(2026-10-03 2차 — 안전장치: 시험 세트 7종에서는 버블이 이미 영역 안이라 발동 0회)


def bubble_extend(A, box, reach_mm=30):
    """도면 영역 상자 바깥 띠(reach_mm 안)에서 한 줄로 선 원(지름 용지 3~16 mm, 고리 꼴) 2개 이상 → 그 원들까지 상자 넓힘 (q px)"""
    q = A['q2']; mm = A['mm_px']; ar = A['area']
    ink = (q < 190).astype(np.uint8)
    x0, y0, x1, y1 = box
    R = int(reach_mm * mm)
    strips = dict(T=[x0, max(ar[1], y0 - R), x1, y0], B=[x0, y1, x1, min(ar[3], y1 + R)],
                  L=[max(ar[0], x0 - R), y0, x0, y1], Rt=[x1, y0, min(ar[2], x1 + R), y1])
    nb = list(box)
    for side, (a, b_, c, d) in strips.items():
        if c - a < 3 or d - b_ < 3: continue
        sub = ink[b_:d, a:c]
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
        circ = []
        for i in range(1, n):
            bx, by, bw, bh, ar_ = st[i]
            if not (3 * mm <= bw <= 16 * mm and 3 * mm <= bh <= 16 * mm) or abs(bw - bh) > 0.25 * max(bw, bh): continue
            if not 0.08 <= ar_ / float(bw * bh) <= 0.6: continue
            m = (lab[by:by + bh, bx:bx + bw] == i)
            cy_, cx_ = bh // 2, bw // 2
            if m[max(0, cy_ - bh // 6):cy_ + bh // 6 + 1, max(0, cx_ - bw // 6):cx_ + bw // 6 + 1].mean() > 0.3: continue      # 가운데가 찬 것(점·글자) 아님
            circ.append((bx + a + bw / 2, by + b_ + bh / 2, bw / 2))
        if len(circ) < 2: continue
        k = 1 if side in ('T', 'B') else 0                         # 줄 방향에 수직인 좌표가 같아야(한 줄)
        vals = sorted(c_[k] for c_ in circ)
        med = vals[len(vals) // 2]
        row = [c_ for c_ in circ if abs(c_[k] - med) <= 3 * mm]
        if len(row) < 2: continue
        pad = 1.5 * mm
        if side == 'T': nb[1] = int(min(nb[1], min(c_[1] - c_[2] for c_ in row) - pad))
        elif side == 'B': nb[3] = int(max(nb[3], max(c_[1] + c_[2] for c_ in row) + pad))
        elif side == 'L': nb[0] = int(min(nb[0], min(c_[0] - c_[2] for c_ in row) - pad))
        else: nb[2] = int(max(nb[2], max(c_[0] + c_[2] for c_ in row) + pad))
    return nb


def ink_box(A):
    """표제란·표를 뺀 도곽 안 잉크 범위(q px) — 도면 영역(regions)과 캡션 후보를 합친 상자"""
    bs = list(A['regions']) + [c for c in A.get('caps') or [] if c[0] >= A['area'][0] and c[2] <= A['area'][2] and c[1] >= A['area'][1] and c[3] <= A['area'][3]]
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def cut(img, A, b):
    f = A['f']; H, W = img.shape[:2]
    pad = int(2.0 * A['mm_px'])
    ar = A['area']
    x0, y0 = max(0, max(ar[0], b[0] - pad) * f), max(0, max(ar[1], b[1] - pad) * f)
    x1, y1 = min(W, min(ar[2], b[2] + pad) * f), min(H, min(ar[3], b[3] + pad) * f)
    out = img[y0:y1, x0:x1].copy()
    wiped = []
    for t in A['tables']:
        tx0, ty0, tx1, ty1 = t[0] * f - x0, t[1] * f - y0, t[2] * f - x0, t[3] * f - y0
        if tx1 <= 0 or ty1 <= 0 or tx0 >= out.shape[1] or ty0 >= out.shape[0]: continue
        out[max(0, ty0 - 2 * f):max(0, ty1 + 3 * f), max(0, tx0 - 2 * f):max(0, tx1 + 3 * f)] = 255
        wiped.append([tx0, ty0, tx1, ty1])
    return out, [int(x0), int(y0), int(x1), int(y1)], wiped


SPLIT_TIGHT = os.environ.get('FIREBIM_SPLIT_TIGHT', '0') == '1'   # (시험용, 기본 끔) 칸을 평면 잉크 범위로 좁힘 — ZM 1층 약한 치수 판독이 37.3→41.3 으로 바뀌어 채택 안 함. 빈 종이는 STR 빈 타일 건너뛰기로 대신


def split_regions(img, captions, A=None):
    """captions: [(층 목록, [x0,y0,x1,y1] 원본 px)] (시트 글자층·OCR 의 '…층 평면도' 줄). 서로 다른 층 캡션이 2개 이상이고 도면 영역도 2개 이상이면
    → [(층 목록, 잘린 그림, 상자)] (캡션마다 가장 가까운 영역: 캡션 위쪽 영역 우선). 아니면 None"""
    A = A or analyze(img)
    if not A or len(A['regions']) < 2: return None
    caps = [(fl, c) for fl, c in captions if fl]
    if len({tuple(fl) for fl, _ in caps}) < 2: return None
    f = A['f']
    regs = [r for r in A['regions']]
    big = max((r[2] - r[0]) * (r[3] - r[1]) for r in regs)
    regs = [r for r in regs if (r[2] - r[0]) * (r[3] - r[1]) >= 0.15 * big]          # 평면 크기 수준 영역만
    if len(regs) < 2: return None
    pick = []
    used = set()
    for fl, c in caps:
        cx, cy = (c[0] + c[2]) / 2 / f, (c[1] + c[3]) / 2 / f
        def dist(r):
            dx = max(r[0] - cx, cx - r[2], 0); dy = max(r[1] - cy, cy - r[3], 0)
            below = cy >= r[3] - 0.1 * (r[3] - r[1])                # 캡션이 영역 아래(보통) → 가산 없음, 위·옆이면 1.5배
            return (dx + dy) * (1.0 if below else 1.5)
        k = min((j for j in range(len(regs)) if j not in used), key=lambda j: dist(regs[j]), default=None)
        if k is None: continue
        used.add(k); pick.append((fl, list(regs[k])))
    # 캡션 없는 영역(버블·치수·범례 조각)은 가장 가까운 평면에 붙임
    for j, r in enumerate(regs):
        if j in used or not pick: continue
        gap = lambda b: max(b[0] - r[2], r[0] - b[2], 0) + max(b[1] - r[3], r[1] - b[3], 0)
        q = min(pick, key=lambda x: gap(x[1]))[1]
        q[0], q[1], q[2], q[3] = min(q[0], r[0]), min(q[1], r[1]), max(q[2], r[2]), max(q[3], r[3])
    out = []
    ar = A['area']
    capq = [[v / f for v in c] for _, c in caps] + list(A.get('caps') or [])
    for k, (fl, r) in enumerate(pick):
        # 그 평면의 칸 = 도곽 안 영역에서 다른 평면 쪽 변을 두 영역 사이 틈 가운데로 자름(버블·치수·캡션 포함)
        b = list(ar)
        for j, (_, r2) in enumerate(pick):
            if j == k: continue
            if r2[0] >= r[2] - 2: b[2] = min(b[2], (r[2] + r2[0]) / 2)
            elif r2[2] <= r[0] + 2: b[0] = max(b[0], (r2[2] + r[0]) / 2)
            elif r2[1] >= r[3] - 2: b[3] = min(b[3], (r[3] + r2[1]) / 2)
            elif r2[3] <= r[1] + 2: b[1] = max(b[1], (r2[3] + r[1]) / 2)
        if SPLIT_TIGHT:                                  # 칸 안에서도 그 평면 잉크(영역 + 칸 안 캡션) 범위 + 10 mm 로 좁힘 — 빈 종이를 모델에 넣지 않게(층당 모델 타일 7~8 → 3~4)
            e = list(r)
            for c in capq:
                if c[0] >= b[0] and c[2] <= b[2] and c[1] >= b[1] and c[3] <= b[3]:
                    e = [min(e[0], c[0]), min(e[1], c[1]), max(e[2], c[2]), max(e[3], c[3])]
            m = 10 * A['mm_px'] if A.get('mm_px') else 0
            b = [max(b[0], e[0] - m), max(b[1], e[1] - m), min(b[2], e[2] + m), min(b[3], e[3] + m)]
        im, bx, _ = cut(img, dict(A, mm_px=0), [int(v) for v in b])
        out.append((fl, im, bx))
    return out if len(out) >= 2 else None
