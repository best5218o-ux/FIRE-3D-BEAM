# -*- coding: utf-8 -*-
"""도면 이미지 해석 (픽셀 좌표): 범례·소방기호·승강기 표기·그리드선·치수 OCR → 축척"""
import re, subprocess, tempfile, os
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np, cv2


def masks(img):
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    lum = (b + g + r) / 3; sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    return dict(
        green=((g > 85) & (r < 110) & (b < 140) & (g - r > 45) & (g - b > 20)).astype(np.uint8),
        blue=((b > 125) & (r < 140) & (g > 60) & (g < 200) & (b - r > 45) & (b - g > 25)).astype(np.uint8),
        red=((r > 170) & (g < 110) & (b < 110)).astype(np.uint8),
        orange=((r > 200) & (g > 100) & (g < 190) & (b < 110) & (r - b > 110)).astype(np.uint8),
        dark=((lum < 125) & (sat < 70)).astype(np.uint8),
        gray=((lum >= 110) & (lum <= 205) & (sat < 35)).astype(np.uint8),
        lum=lum, sat=sat)


def blue_loose(img):
    """(스캔본 범례 테두리 찾기 전용) 흐림·JPEG 으로 옅어진 파란 테두리까지 — masks()['blue'] 보다 느슨한 문턱"""
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    return ((b > 120) & (r < 175) & (g < 215) & (b - r > 28) & (b - g > 15)).astype(np.uint8)


def comps(m, min_area=12):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    return [dict(x=int(st[i][0]), y=int(st[i][1]), w=int(st[i][2]), h=int(st[i][3]), a=int(st[i][4]),
                 cx=float(cen[i][0]), cy=float(cen[i][1]), i=i) for i in range(1, n) if st[i][4] >= min_area], lab


def find_legend(M, shape, scan=False):
    """파란 둥근 테두리 범례 상자: 위쪽에 있는 가장 큰 파란 윤곽 (JPEG로 테두리가 끊겨도 병합)
    scan=True(스캔본 평면도): 흐림으로 둥근 모서리가 끊겨 테두리가 변마다 따로 떨어진 경우 — 6px 안에 붙은 파란 조각들
    (또는 위·아래 변 높이가 같고 20px 안에 이웃한 조각들)을 한 묶음으로 본 상자도 후보"""
    H, W = shape[:2]
    b = cv2.morphologyEx(M['blue_loose'] if scan and 'blue_loose' in M else M['blue'], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = comps(b, 30)
    thin = [c for c in cs if c['w'] > W * 0.08 and c['a'] < 0.35 * c['w'] * c['h'] and c['y'] < H * 0.4]
    boxes = [[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']] for c in thin]
    if scan:
        top = [c for c in cs if c['y'] < H * 0.4]
        grp = [[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h'], c['a']] for c in top]
        merged = True
        while merged:
            merged = False
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, c = grp[i], grp[j]
                    near = a[0] <= c[2] + 6 and c[0] <= a[2] + 6 and a[1] <= c[3] + 6 and c[1] <= a[3] + 6
                    row = (abs(a[1] - c[1]) <= 4 or abs(a[3] - c[3]) <= 4) and a[0] <= c[2] + 20 and c[0] <= a[2] + 20   # 같은 위·아래 변(끊긴 테두리)
                    if near or row:
                        grp[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3]), a[4] + c[4]]; grp.pop(j); merged = True; break
                if merged: break
        boxes += [g[:4] for g in grp if g[2] - g[0] > W * 0.08 and g[4] < 0.35 * (g[2] - g[0]) * (g[3] - g[1])]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                if a[0] <= c[2] + 4 and c[0] <= a[2] + 4 and a[1] <= c[3] + 4 and c[1] <= a[3] + 4:
                    boxes[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]; boxes.pop(j); merged = True; break
            if merged: break
    boxes = [bx for bx in boxes if (bx[2] - bx[0]) > 1.8 * (bx[3] - bx[1]) and (bx[3] - bx[1]) > 12]
    if not boxes:
        # 흐린 스캔: 테두리 세로변이 끊겨 위·아래 가로변만 남은 경우 → 같은 줄 조각을 잇고 위·아래 변을 짝지음
        hs = [c for c in cs if c['y'] < H * 0.4 and c['h'] <= 8 and c['w'] >= 20]
        rows = []
        for c in sorted(hs, key=lambda c: (c['y'], c['x'])):
            for r in rows:
                if abs(r[1] - c['y']) <= 3 and c['x'] <= r[2] + 40 and r[0] <= c['x'] + c['w'] + 40:
                    r[0], r[2], r[3] = min(r[0], c['x']), max(r[2], c['x'] + c['w']), max(r[3], c['y'] + c['h']); break
            else: rows.append([c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']])
        rows = [r for r in rows if r[2] - r[0] > W * 0.08]
        for i, a in enumerate(rows):
            for c in rows[i + 1:]:
                ov = min(a[2], c[2]) - max(a[0], c[0])
                top, bot = (a, c) if a[1] < c[1] else (c, a)
                if ov >= 0.7 * min(a[2] - a[0], c[2] - c[0]) and 12 < bot[3] - top[1] < H * 0.25:
                    boxes.append([min(a[0], c[0]), top[1], max(a[2], c[2]), bot[3]])
        boxes = [bx for bx in boxes if (bx[2] - bx[0]) > 1.8 * (bx[3] - bx[1])]
    if not boxes: return None
    bx = max(boxes, key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]))
    return dict(x=bx[0], y=bx[1], w=bx[2] - bx[0], h=bx[3] - bx[1])


def classify_green(sub, w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    lr = (xx / max(w, 1) + yy / max(h, 1)) > 1.0
    lrf = sub[lr].mean() if lr.any() else 0
    ulf = sub[~lr].mean() if (~lr).any() else 0
    fill = sub.mean(); asp = w / max(h, 1)
    if lrf > 0.62 and ulf < 0.42: return 'hose'
    if asp > 1.2 and fill > 0.45: return 'panel'
    if 0.7 < asp < 1.4 and 0.3 < fill < 0.97: return 'exit'
    if asp >= 1.2 and lrf > 0.5 and ulf < 0.5: return 'hose'
    return None


LABELS = [('비상구', 'exit'), ('피난구', 'exit'), ('소화전', 'hose'), ('수신기', 'panel'), ('가스', 'gas'), ('밸브', 'gas'), ('대피', 'arrow'), ('승강기', 'ev')]


def legend_words(img, leg):
    crop = cv2.cvtColor(img[leg['y']:leg['y'] + leg['h'], leg['x']:leg['x'] + leg['w']], cv2.COLOR_BGR2GRAY)
    k = 3.0
    up = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    out = []
    for r in ocr_tsv(up, psm=11, lang='kor', wl=None):
        out.append(dict(x=leg['x'] + r['x'] / k, y=leg['y'] + r['y'] / k, w=r['w'] / k, h=r['h'] / k, text=r['text']))
    return out


def label_for_icon(c, words, Z):
    """범례 아이콘 오른쪽의 글자로 종류를 정한다 (OCR이 글자를 쪼개거나 한두 자 틀려도 맞춘다)"""
    row = [w for w in words if c['x'] + c['w'] - 2 <= w['x'] <= c['x'] + c['w'] + 7 * Z
           and not (w['y'] > c['y'] + c['h'] or w['y'] + w['h'] < c['y'])]
    row.sort(key=lambda w: w['x'])
    if not row or row[0]['x'] > c['x'] + c['w'] + 3.5 * Z: return None
    txt, end = '', None
    for w in row:                               # 아이콘 바로 옆 낱말부터 붙여 읽기(다음 칸 글자 전까지)
        if end is not None and w['x'] - end > 1.2 * Z: break
        txt += w['text']; end = w['x'] + w['w']
    for k, t in LABELS:
        if k in txt: return t
    best = None
    for k, t in LABELS:                         # 느슨한 일치: 낱말 글자가 절반 이상 겹침
        hit = sum(1 for ch in set(k) if ch in txt)
        sc = hit / len(set(k))
        if hit >= 2 or (len(k) == 2 and hit >= 1 and len(txt) <= 2):
            if best is None or sc > best[0]: best = (sc, t)
    return best[1] if best and best[0] >= 0.5 else None


def _shape_feats(sub):
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lr = (xx / max(w, 1) + yy / max(h, 1)) > 1.0
    lrf = sub[lr].mean() if lr.any() else 0
    ulf = sub[~lr].mean() if (~lr).any() else 0
    return w / max(h, 1), lrf - ulf, sub.mean()


def _shape_score(t, sub):
    a, d, fill = _shape_feats(sub)
    if t == 'hose': return 3 * d - abs(a - 1.67)
    if t == 'panel': return -3 * abs(d) - 1.5 * abs(a - 1.43) + fill
    if t == 'exit': return -3 * abs(d) - 2 * abs(a - 1.0) + fill
    return -9


def assign_legend(icons, subs, words, Z):
    """범례 초록 아이콘 → 종류. 글자(OCR) 우선, 못 읽은 아이콘은 남은 종류 중 모양 점수가 가장 높은 배정"""
    import itertools
    lab = {}
    for i, c in enumerate(icons):
        t = label_for_icon(c, words, Z)
        if t in ('exit', 'hose', 'panel') and t not in lab.values(): lab[i] = t
    rest = [i for i in range(len(icons)) if i not in lab]
    free = [t for t in ('exit', 'hose', 'panel') if t not in lab.values()]
    if rest and free:
        best = None
        k = min(len(rest), len(free))
        for idx in itertools.permutations(rest, k):
            for ts in itertools.permutations(free, k):
                sc = sum(_shape_score(t, subs[i]) for i, t in zip(idx, ts))
                if best is None or sc > best[0]: best = (sc, idx, ts)
        for i, t in zip(best[1], best[2]):
            if _shape_score(t, subs[i]) > -2.5: lab[i] = t
    return lab


DEBUG = None
SPLIT_THR = 0.6
SYM_W_NCC, SYM_W_AR, SYM_W_PAD = 1.0, 2.5, 1.0     # 점수 = IoU + 무늬상관 - 2.5·|ln 가로세로비| + 배율형판 상관


def green_map(img):
    """초록 농도(0~1): 기호 속 흰 무늬를 살린 채 비교하기 위한 연속값"""
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    return np.clip((g - np.maximum(r, b) - 10) / 80, 0, 1)


def _trim(sub):
    """이진 아이콘 가장자리에 붙은 1px 잡선(그리드선 조각 등)만 떼어낸 경계"""
    h, w = sub.shape; x0, y0, x1, y1 = 0, 0, w, h
    r, c = sub.mean(1), sub.mean(0)
    if h > 4 and r[0] < 0.25 and r[1] > 0.6: y0 = 1
    if h > 4 and r[-1] < 0.25 and r[-2] > 0.6: y1 = h - 1
    if w > 4 and c[0] < 0.25 and c[1] > 0.6: x0 = 1
    if w > 4 and c[-1] < 0.25 and c[-2] > 0.6: x1 = w - 1
    return x0, y0, x1, y1


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def pad_scores(gm, bx, tpls, k, Z):
    """후보 상자 둘레에서, 도면 배율 k 로 키운 여백 포함 형판의 최고 상관(종류별)"""
    H, W = gm.shape; pad = int(0.4 * Z) + 3
    x0, y0, x1, y1 = max(0, bx[0] - pad), max(0, bx[1] - pad), min(W, bx[2] + pad), min(H, bx[3] + pad)
    G = gm[y0:y1, x0:x1].astype(np.float32); out = {}
    for t, tp in tpls.items():
        for sc in (0.85 * k, 0.95 * k, 1.05 * k, 1.15 * k):
            T = cv2.resize(tp['gpad'], None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            if T.shape[0] >= G.shape[0] or T.shape[1] >= G.shape[1] or T.std() < 1e-3: continue
            out[t] = max(out.get(t, -1.0), float(cv2.matchTemplate(G, T, cv2.TM_CCOEFF_NORMED).max()))
    return out


def split_blob(gm, gb, bx, tpls, k, Z, thr=None, max_n=3):
    """겹친 초록 기호 덩어리 → [기호]: 가장 잘 맞는 기호부터 찾고, 찾은 자리는 가려 가며 반복
    (가려진 기호는 보이는 부분만으로 상관). 같은 종류 기호가 크게 겹쳐 두 번 잡히지 않게 한다."""
    thr = SPLIT_THR if thr is None else thr
    H, W = gm.shape
    pad = int(0.6 * Z) + 3
    x0, y0, x1, y1 = max(0, bx[0] - pad), max(0, bx[1] - pad), min(W, bx[2] + pad), min(H, bx[3] + pad)
    G = gm[y0:y1, x0:x1].astype(np.float32)
    B = gb[y0:y1, x0:x1] > 0
    K = np.zeros(G.shape, bool)
    TT = []
    for t, tp in tpls.items():
        for sc in (0.9 * k, k, 1.1 * k):
            T = cv2.resize(tp['gpad'], None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            if T.shape[0] < G.shape[0] and T.shape[1] < G.shape[1] and T.std() > 1e-3: TT.append((t, T, int(round(2 * sc))))
    if not TT: return []
    amin = min((T.shape[0] - 2 * p) * (T.shape[1] - 2 * p) for _, T, p in TT)
    out = []
    def ovl(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    for it in range(max_n):
        R = (B & ~K).astype(np.float32)
        if R.sum() < 0.3 * amin: break
        RI = cv2.integral(R)
        best = None
        for t, T, p in TT:
            h, w = T.shape; ih, iw = h - 2 * p, w - 2 * p
            full = cv2.matchTemplate(G, T, cv2.TM_CCOEFF_NORMED) if not K.any() else None
            Tm = T - T.mean()
            for y in range(0, G.shape[0] - h + 1):
                for x in range(0, G.shape[1] - w + 1):
                    # 새 기호 안쪽에 아직 설명 안 된 초록이 충분해야 함
                    res = RI[y + p + ih, x + p + iw] - RI[y + p, x + p + iw] - RI[y + p + ih, x + p] + RI[y + p, x + p]
                    if res < 0.3 * ih * iw: continue
                    box_ = (x + p, y + p, x + p + iw, y + p + ih)
                    if any(q[0] == t and ovl(q[1], box_) > 0.1 for q in out): continue     # 같은 기호가 겹쳐 두 번 그려지진 않음
                    if full is not None: v = float(full[y, x])
                    else:
                        m = ~K[y:y + h, x:x + w]
                        if m.mean() < 0.35: continue
                        a_ = G[y:y + h, x:x + w][m]; b_ = T[m]
                        a_ = a_ - a_.mean(); b_ = b_ - b_.mean(); d = np.sqrt((a_ * a_).sum() * (b_ * b_).sum())
                        v = float((a_ * b_).sum() / d) if d > 1e-6 else 0.0
                    if best is None or v > best[0]: best = (v, t, box_)
        if best is None or best[0] < thr: break
        v, t, (bx0, by0, bx1, by1) = best
        K[by0:by1, bx0:bx1] = True
        out.append((t, (bx0, by0, bx1, by1), v))
    return [dict(type=t, px=x0 + (b[0] + b[2]) / 2, py=y0 + (b[1] + b[3]) / 2, w=b[2] - b[0], h=b[3] - b[1], conf=round(v, 2), split=True) for t, b, v in out]


def gas_template(img, M, leg, Z, words=None, cols=None):
    """범례 안의 검은 원형 아이콘(의료가스밸브) → 회색조 형판(여백 2px) 또는 None.
    words(범례 OCR)가 있으면 아이콘 옆 글자가 가스밸브일 때만(없는 건물의 글자 'ㅇ' 등을 형판으로 쓰지 않게)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lx0, ly0 = leg['x'], leg['y']
    dk = M['dark'][ly0:ly0 + leg['h'], lx0:lx0 + leg['w']]
    for c in comps(dk, 10)[0]:
        s = max(c['w'], c['h'])
        if not (0.35 * Z <= s <= 1.3 * Z and 0.75 < c['w'] / max(c['h'], 1) < 1.33): continue
        # 범례 글자와 구분: 원형도(윤곽이 원에 가까움) 확인
        sub = (dk[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea); A = cv2.contourArea(cnt); P = cv2.arcLength(cnt, True)
        if P == 0 or 4 * np.pi * A / (P * P) < 0.72: continue
        # 글자 'ㅇ'·'0'(가운데가 빈 고리)과 구분: 살(spoke)이 만나는 가운데가 어두워야 함
        cy_, cx_ = c['h'] // 2, c['w'] // 2
        if sub[max(0, cy_ - 1):cy_ + 2, max(0, cx_ - 1):cx_ + 2].sum() == 0: continue
        if words:
            lab_ = label_for_icon(dict(c, x=c['x'] + lx0, y=c['y'] + ly0), words, Z)
            if lab_ not in (None, 'gas'): continue
        if cols:                                   # 범례는 표 형식: 아이콘 열(초록 아이콘 열 간격의 정수배)에 있어야 함
            gx = lx0 + c['x'] + c['w'] / 2
            x0c = min(cols); pitch = (max(cols) - x0c) if len(cols) > 1 and max(cols) - x0c > Z else None
            n_ = (gx - x0c) / pitch if pitch else 0
            if not (pitch and abs(n_ - round(n_)) * pitch < 0.5 * Z and round(n_) >= 0): continue
        pad = 2
        return gray[ly0 + c['y'] - pad: ly0 + c['y'] + c['h'] + pad, lx0 + c['x'] - pad: lx0 + c['x'] + c['w'] + pad].copy()
    return None


GAS_NCC, GAS_CEN, GAS_RING = 0.62, 40, 0.7      # 가상 18동 기준 양성 최저 0.66 · 음성 최고 0.57 의 중간
GAS_NCC_SCAN = float(os.environ.get('FIREBIM_GAS_NCC_SCAN', '0.75'))   # 스캔본: 흐림으로 그리드선·벽 교차점이 원형 기호와 0.62~0.69 상관 → 문턱 올림(dev 에서만 정함)


def _ring_feats(gray, x, y, r):
    """원형 기호 검증: 둘레(반지름 r±1)의 어두운 비율, 중심(3×3) 최솟값"""
    H, W = gray.shape
    def at(px, py):
        px, py = int(round(px)), int(round(py))
        return 255 if not (0 <= px < W and 0 <= py < H) else int(gray[py, px])
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = max(float(np.mean([min(at(x + rr * np.cos(a), y + rr * np.sin(a)) for rr in (q - 1, q, q + 1)) < 130 for a in ang]))
               for q in (0.75 * r, 0.9 * r, 1.05 * r))                      # 반지름 추정 오차 허용
    cen = min(at(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    return ring, cen


def detect_gas(img, M, leg, Z, k=None, words=None, cols=None, ncc=None):
    """의료가스밸브(검은 원 + Y자 살): 원 안쪽만 보는 가림 형판 상관 → 둘레·중심 어두움 검증.
    벽선에 닿아 그려진 기호도 원 밖 여백을 비교하지 않으므로 잡힌다. 배율은 초록 기호에서 구한 k 근처만 본다."""
    tpl = gas_template(img, M, leg, Z, words, cols)
    if tpl is None: return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lx0, ly0, lx1, ly1 = leg['x'], leg['y'], leg['x'] + leg['w'], leg['y'] + leg['h']
    r_leg = (min(tpl.shape) - 4) / 2
    scales = [k * f for f in (0.85, 0.95, 1.05, 1.15, 1.25)] if k else list(np.arange(0.85, 1.6, 0.1))
    best = np.full(gray.shape, -1.0, np.float32); bsc = np.ones(gray.shape, np.float32)
    for sc in scales:
        t = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        hh, ww = t.shape
        if hh >= H or ww >= W or t.std() < 1: continue
        yy, xx = np.mgrid[0:hh, 0:ww]
        r = (min(hh, ww) - 4 * sc) / 2
        m = (((xx - (ww - 1) / 2) ** 2 + (yy - (hh - 1) / 2) ** 2) <= (r + 1) ** 2).astype(np.float32)
        res = np.nan_to_num(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED, mask=m), nan=-1, posinf=-1, neginf=-1)
        oy, ox = hh // 2, ww // 2
        sub = best[oy:oy + res.shape[0], ox:ox + res.shape[1]]; sb = bsc[oy:oy + res.shape[0], ox:ox + res.shape[1]]
        upd = res > sub; sub[upd] = res[upd]; sb[upd] = sc
    best[max(0, ly0 - 5):ly1 + 5, max(0, lx0 - 5):lx1 + 5] = -1
    dil = cv2.dilate(best, np.ones((5, 5), np.uint8))
    ys, xs = np.where((best >= dil) & (best > (GAS_NCC if ncc is None else ncc)))
    found = []
    for x, y in zip(xs, ys):
        ring, cen = _ring_feats(gray, x, y, r_leg * bsc[y, x])
        if DEBUG is not None: DEBUG.append(dict(gas=(float(x), float(y), float(best[y, x]), ring, cen)))
        if cen < GAS_CEN and ring >= GAS_RING: found.append((float(best[y, x]), float(x), float(y)))
    found.sort(reverse=True); pts = []
    for sco, x, y in found:
        if all((x - p[0]) ** 2 + (y - p[1]) ** 2 > (0.5 * Z) ** 2 for p in pts): pts.append((x, y, sco))   # 붙어 있는 밸브 2개도 따로
    return [dict(type='gas', px=x, py=y, conf=round(sco, 2)) for x, y, sco in pts]


def detect_symbols(img, M=None, scan=False):
    """반환: symbols(list), legend(dict|None), Z(기호 크기 px). scan=True: 스캔본 평면도(범례 테두리 조각 묶기)"""
    M = M or masks(img)
    H, W = img.shape[:2]
    leg = find_legend(M, img.shape, scan)
    g = cv2.morphologyEx(M['green'], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, lab = comps(g, 15)
    inleg = lambda x, y: leg is not None and leg['x'] - 3 <= x <= leg['x'] + leg['w'] + 3 and leg['y'] - 3 <= y <= leg['y'] + leg['h'] + 3
    leg_icons = [c for c in cs if inleg(c['cx'], c['cy'])]
    Z = float(np.median([max(c['w'], c['h']) for c in leg_icons])) if leg_icons else None
    if Z is None:
        sizes = sorted(max(c['w'], c['h']) for c in cs if c['a'] > 40)
        Z = float(np.median(sizes)) if sizes else 16.0
    out = []
    tpls = {}
    words = legend_words(img, leg) if leg is not None else []
    subs = [(lab[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['i']).astype(float) for c in leg_icons]
    gm = green_map(img)
    for i, t in assign_legend(leg_icons, subs, words, Z).items():
        c = leg_icons[i]; x0, y0, x1, y1 = _trim(subs[i])
        X0, Y0 = c['x'] + x0, c['y'] + y0
        tpls[t] = dict(bin=subs[i][y0:y1, x0:x1] > 0.5, g=gm[Y0:c['y'] + y1, X0:c['x'] + x1].astype(np.float32), ar=(x1 - x0) / max(1, y1 - y0),
                       gpad=cv2.copyMakeBorder(gm[Y0:c['y'] + y1, X0:c['x'] + x1].astype(np.float32), 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0))
    # 후보: 범례 밖 초록 덩어리. 빨간 대피선에 쪼개진 조각은 합친다
    cand = [[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']] for c in cs if not inleg(c['cx'], c['cy']) and c['a'] >= 0.06 * Z * Z]
    merged = True
    while merged:
        merged = False
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                A, C = cand[i], cand[j]
                gap = max(C[0] - A[2], A[0] - C[2], C[1] - A[3], A[1] - C[3])
                U = [min(A[0], C[0]), min(A[1], C[1]), max(A[2], C[2]), max(A[3], C[3])]
                if gap <= max(3, 0.25 * Z) and max(U[2] - U[0], U[3] - U[1]) <= 1.7 * Z:
                    cand[i] = U; cand.pop(j); merged = True; break
            if merged: break
    hard, ks, pend = [], [], []
    for bx in cand:
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        if max(w, h) < 0.5 * Z: continue
        if max(w, h) > 2.2 * Z:
            if max(w, h) <= 4.0 * Z: hard.append((bx, None))          # 기호 2~3개가 겹친 덩어리 → 뒤에서 나눔
            continue
        patch = g[bx[1]:bx[3], bx[0]:bx[2]].astype(np.float32)
        gp = gm[bx[1]:bx[3], bx[0]:bx[2]]
        best = None; feats = {}
        for t, tp in tpls.items():
            ct = tp['bin']
            if ct.shape[0] < 3 or ct.shape[1] < 3: continue
            # 같은 크기로 맞춰 겹침(IoU) + 초록 농도 무늬 상관(흰 줄무늬·삼각형 구분) + 가로세로비
            ar_t, ar_p = tp['ar'], w / max(h, 1)
            pr = cv2.resize(patch, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
            iou = (pr & ct).sum() / max(1, (pr | ct).sum())
            ncc = _ncc(cv2.resize(gp, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_AREA), tp['g'])
            arp = abs(np.log(ar_p / ar_t))
            feats[t] = (float(iou), float(ncc), float(arp))
            if not (0.7 < ar_p / ar_t < 1.43): continue
            sc = iou + SYM_W_NCC * ncc - SYM_W_AR * arp
            if best is None or sc > best[0]: best = (sc, t, iou)
        if best and best[2] >= 0.42:                                      # 1차 판정 → 도면 기호 배율 k 추정용
            tb = tpls[best[1]]['bin']; ks.append((w / tb.shape[1] + h / tb.shape[0]) / 2)
        if not tpls:
            t = classify_green(patch > 0.5, w, h)
            if t: out.append(dict(type=t, px=(bx[0] + bx[2]) / 2, py=(bx[1] + bx[3]) / 2, w=w, h=h))
        else: pend.append((bx, feats))
    ksym = float(np.median(ks)) if len(ks) >= 2 else None       # 도면 기호 / 범례 아이콘 크기 비
    if DEBUG is not None: DEBUG.append(dict(ksym=ksym, tpls=tpls, gm=gm, g=g))
    # 2차 판정: 도면 배율로 키운 형판(여백 포함 → 바깥 윤곽까지 비교)의 상관을 더함. 가로세로비는 벌점으로만(소화전 흰 반쪽이 잘려 비율이 흔들림)
    for bx, feats in pend:
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        pads = pad_scores(gm, bx, tpls, ksym or 1.25, Z)
        best = None
        for t, (iou, ncc, arp) in feats.items():
            sc = iou + SYM_W_NCC * ncc - SYM_W_AR * arp + SYM_W_PAD * pads.get(t, 0)
            if best is None or sc > best[0]: best = (sc, t, iou)
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        if DEBUG is not None: DEBUG.append(dict(px=cx, py=cy, w=w, h=h, feats=feats, pads=pads, bx=bx))
        if best and best[2] >= 0.42:
            out.append(dict(type=best[1], px=cx, py=cy, w=w, h=h, conf=round(float(best[2]), 2), _bx=bx))
        else:
            hard.append((bx, None))
    # 겹친 기호 나누기: 도면 기호/범례 아이콘 배율 k 로 형판을 키워, 이미 설명된 화소를 가린 채 차례로 맞춤
    if tpls and hard + [q for q in out if '_bx' in q]:
        k = float(np.median(ks)) if ks else 1.25
        for q in list(out):
            bx = q.get('_bx')
            if bx is None: continue
            tb = tpls[q['type']]['bin']
            if (bx[2] - bx[0]) > 1.3 * k * tb.shape[1] or (bx[3] - bx[1]) > 1.3 * k * tb.shape[0]:
                out.remove(q); hard.append((bx, q))
        for bx, prev in hard:
            got = split_blob(gm, g, bx, tpls, k, Z)
            if got: out += got
            elif prev is not None: out.append(prev)
    for q in out: q.pop('_bx', None)
    # 승강기 표기(파란 사각 윤곽) — 범례 제외
    bl = cv2.morphologyEx(M['blue'], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for c in comps(bl, 20)[0]:
        if leg is not None and c['x'] >= leg['x'] - 2 and c['y'] >= leg['y'] - 2 and c['x'] + c['w'] <= leg['x'] + leg['w'] + 2 and c['y'] + c['h'] <= leg['y'] + leg['h'] + 2: continue
        if 0.6 * Z <= max(c['w'], c['h']) <= 6 * Z and c['a'] < 0.88 * c['w'] * c['h'] and min(c['w'], c['h']) >= 0.4 * Z:
            out.append(dict(type='ev', px=c['cx'], py=c['cy'], w=c['w'], h=c['h'], box=[c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']]))
    # 의료가스밸브: 범례 안의 검은 원형 아이콘을 형판으로
    if leg is not None: out += detect_gas(img, M, leg, Z, ksym, words, sorted(set(round(c['cx']) for c in leg_icons)), ncc=GAS_NCC_SCAN if scan else None)
    return out, leg, Z


def grid_lines(M, shape):
    """도면 전체를 가로지르는 얇은 회색 그리드선 위치(px)"""
    H, W = shape[:2]
    m = ((M['lum'] < 228) & (M['sat'] < 40)).astype(np.uint8)
    v = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(H * 0.45)))))
    h = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(W * 0.45)), 1)))
    def peaks(prof, thr):
        idx = np.where(prof > thr)[0]; out = []
        for i in idx:
            if out and i - out[-1][-1] <= 2: out[-1].append(i)
            else: out.append([i])
        return [float(np.mean(g)) for g in out]
    vl, hl = peaks(v.sum(0), H * 0.4), peaks(h.sum(1), W * 0.4)
    if len(vl) < 3:
        # 그리드선이 짧은 흰 틈(글자 바탕 등)으로 끊겨 한 토막이 문턱보다 짧은 경우 → 세로로 작은 틈을 메우고 다시 찾는다
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(H * 0.04)))))
        v2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(H * 0.45)))))
        vl2 = peaks(v2.sum(0), H * 0.4)
        if len(vl2) >= 3: vl = vl2
    if len(hl) < 2:
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(W * 0.04)), 1)))
        h2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(W * 0.45)), 1)))
        hl2 = peaks(h2.sum(1), W * 0.4)
        if len(hl2) >= 2: hl = hl2
    return vl, hl


def ocr_tsv(img_gray, psm=11, lang='eng', wl='0123456789,'):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, img_gray); fn = tf.name
    try:
        cmd = ['tesseract', fn, '-', '-l', lang, '--psm', str(psm)]
        if wl: cmd += ['-c', f'tessedit_char_whitelist={wl}']
        out = subprocess.run(cmd + ['tsv'], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.unlink(fn)
    rows = []
    for ln in out.splitlines()[1:]:
        p = ln.split('\t')
        if len(p) == 12 and p[11].strip():
            rows.append(dict(x=int(p[6]), y=int(p[7]), w=int(p[8]), h=int(p[9]), conf=float(p[10]), text=p[11].strip()))
    return rows


def scale_from_dims(img, M, vlines, mm_out=None):
    """그리드선 사이 치수 글자(예 7,500)를 읽어 px/m 추정. mm_out(list) 을 주면 읽은 치수(mm)를 담아 준다(건물 단위 합의용)"""
    DIM_MM = mm_out if mm_out is not None else []
    if len(vlines) < 3: return None, []
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    band = gray[: int(H * 0.5)]
    up = cv2.resize(band, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    vals = []
    for r in ocr_tsv(up):
        t = r['text'].replace(' ', '')
        m = re.fullmatch(r'(\d{1,2}),?(\d{3})', t)
        if not m: continue
        mm = int(m.group(1) + m.group(2))
        if not (1500 <= mm <= 20000): continue
        cx = (r['x'] + r['w'] / 2) / 2.5
        for a, b in zip(vlines, vlines[1:]):
            if a < cx < b and (b - a) > 10:
                vals.append((b - a) / (mm / 1000.0)); DIM_MM.append(mm); break
    if not vals: return None, []
    med = float(np.median(vals))
    good = [v for v in vals if abs(v - med) / med < 0.04]
    return (float(np.mean(good)) if len(good) >= 2 else None), vals


def footprint_px_area(M):
    ink = ((M['dark'] > 0) | (M['gray'] > 0)).astype(np.uint8)
    k = max(3, int(min(ink.shape) * 0.01))
    cl = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((cv2.contourArea(c) for c in cnts), default=0)


def stair_regions(M, s):
    """계단 디딤판(촘촘한 평행선) 또는 뭉개진 조밀 영역 → 계단 후보 상자(px)"""
    ink = ((M['lum'] < 205) & (M['sat'] < 40)).astype(np.uint8)
    H, W = ink.shape
    acc = np.zeros((H, W), np.uint8)
    win = max(5, int(2.0 * s))
    for ori in ('h', 'v'):
        L = max(4, int(0.7 * s))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if ori == 'h' else (1, L))
        ln = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k).astype(np.int16)
        if ori == 'h':
            tr = np.zeros_like(ln, np.float32); tr[1:] = (np.abs(np.diff(ln, axis=0)) > 0)
            cnt = cv2.boxFilter(tr, -1, (1, win), normalize=False)
            reg = (cnt >= 12).astype(np.uint8)
            reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(1.0 * s)), 1)))
        else:
            tr = np.zeros_like(ln, np.float32); tr[:, 1:] = (np.abs(np.diff(ln, axis=1)) > 0)
            cnt = cv2.boxFilter(tr, -1, (win, 1), normalize=False)
            reg = (cnt >= 12).astype(np.uint8)
            reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(1.0 * s)))))
        acc |= reg
    # 뭉개진 디딤판(저해상도): 회색 조밀 영역
    dens = cv2.boxFilter(ink.astype(np.float32), -1, (max(3, int(1.2 * s)),) * 2)
    solid = cv2.boxFilter((M['lum'] < 80).astype(np.float32), -1, (max(3, int(1.2 * s)),) * 2)
    acc |= ((dens > 0.55) & (solid < 0.35)).astype(np.uint8)
    acc = cv2.morphologyEx(acc, cv2.MORPH_CLOSE, np.ones((max(3, int(0.6 * s)),) * 2, np.uint8))
    out = []
    for c in comps(acc, 10)[0]:
        w, h = c['w'] / s, c['h'] / s
        if 1.2 <= min(w, h) and max(w, h) <= 10 and 1.8 <= max(w, h) and c['a'] > 0.35 * c['w'] * c['h']:
            out.append([c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h']])
    return out


# ───────────────── 소방시설 추가 기호(자홍 단색: 소화기·방수구·비상콘센트·송수구) — 범례 적응형(범례 아이콘 = 형판, 옆 글자 = 종류)
FAC_LABELS = [('소화기', 'ext'), ('방수구', 'outlet'), ('콘센트', 'socket'), ('송수구', 'siamese')]


def magenta_map(img):
    """자홍 농도 0~1: min(R,B) − G (빨강·파랑·초록·회색은 0에 가까움)"""
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    return np.clip((np.minimum(r, b) - g - 40) / 100, 0, 1)


def _jamo(ch):
    o = ord(ch) - 0xAC00
    return (o // 588, (o % 588) // 28, o % 28) if 0 <= o < 11172 else (ch,)


def _jsim(a, b):
    """두 글자열(같은 길이)의 자모 일치율 — OCR 오독('승수구'↔'송수구', '빙수구'↔'방수구') 허용"""
    tot = hit = 0
    for x, y in zip(a, b):
        jx, jy = _jamo(x), _jamo(y); tot += 3
        hit += sum(1 for u, v in zip(jx, jy) if u == v) * (3 // len(jx) if len(jx) == len(jy) else 0)
    return hit / max(1, tot)


def fac_label(c, words, Z, xmax=None):
    xm = min(c['x'] + c['w'] + 7 * Z, xmax if xmax is not None else 1e9)       # 같은 줄 다음 아이콘 전까지만
    row = [w for w in words if c['x'] + c['w'] - 2 <= w['x'] and w['x'] + 0.5 * w['w'] < xm and not (w['y'] > c['y'] + c['h'] or w['y'] + w['h'] < c['y'])]
    row.sort(key=lambda w: w['x'])
    if not row or row[0]['x'] > c['x'] + c['w'] + 3.5 * Z: return None
    txt, end = '', None
    for w in row:
        if end is not None and w['x'] - end > 1.2 * Z: break
        txt += w['text']; end = w['x'] + w['w']
    txt = ''.join(ch for ch in txt if '가' <= ch <= '힣')
    for k, t in FAC_LABELS:
        if k in txt: return t
    sc = []
    for k, t in FAC_LABELS:                   # 글자 창을 밀며 자모 일치율 최고값
        best = max([_jsim(txt[i:i + len(k)], k) for i in range(0, max(1, len(txt) - len(k) + 1))] or [0])
        sc.append((best, t))
    sc.sort(reverse=True)
    if sc[0][0] >= 0.6 and (len(sc) < 2 or sc[0][0] - sc[1][0] >= 0.15): return sc[0][1]
    return None


def _fac_feats(sub):
    """자홍 농도 조각(최댓값 1로) → (채움비, 위−아래 채움 차, 네 귀퉁이 채움, 가운데 채움)"""
    sub = sub.astype(np.float32); sub = sub / max(1e-3, float(sub.max()))
    h, w = sub.shape
    fill = sub.mean()
    top, bot = sub[:max(1, h // 2)].mean(), sub[h - max(1, h // 2):].mean()
    k = max(1, min(h, w) // 4)
    corner = np.mean([sub[:k, :k].mean(), sub[:k, -k:].mean(), sub[-k:, :k].mean(), sub[-k:, -k:].mean()])
    cy, cx, r = h // 2, w // 2, max(1, min(h, w) // 6)
    center = sub[cy - r:cy + r + 1, cx - r:cx + r + 1].mean()
    return np.array([fill, top - bot, corner, center])


PROTO = {'ext': np.array([0.55, -0.4, 0.22, 0.9]), 'outlet': np.array([0.55, 0.0, 0.15, 0.2]), 'socket': np.array([0.85, 0.0, 0.8, 1.0]),
         'siamese': np.array([0.55, 0.0, 0.03, 1.0])}
FAC_FW = 1.5          # 모양 특징 거리 가중(형판 상관에서 뺌)


def detect_fac(img, leg, M=None):
    """→ (symbols [{type, px, py, w, h, conf}], legend_types {type: n_icons}). 범례에 자홍 아이콘이 없으면 도면에서도 찾지 않는다(범례 적응형)"""
    mm = magenta_map(img); mb = (mm > 0.3).astype(np.uint8)
    cs, lab = comps(mb, 6)
    H_, W_ = mm.shape
    # 소방시설 범례: 자홍 테두리 상자(기존 파란 범례 옆·아래). 없으면 파란 범례 안의 자홍 아이콘
    fr = [c for c in cs if c['w'] > 30 and c['h'] > 12 and c['a'] < 0.2 * c['w'] * c['h']]
    if fr:
        c = max(fr, key=lambda c: c['w'] * c['h']); leg = dict(x=c['x'], y=c['y'], w=c['w'], h=c['h'])
        cs = [q for q in cs if q is not c]
    if leg is None: return [], {}
    inleg = lambda c: leg['x'] - 3 <= c['cx'] <= leg['x'] + leg['w'] + 3 and leg['y'] - 3 <= c['cy'] <= leg['y'] + leg['h'] + 3
    icons = [c for c in cs if inleg(c) and c['a'] >= 10]
    if not icons: return [], {}
    Z = float(np.median([max(c['w'], c['h']) for c in icons]))
    words = legend_words(img, leg)
    subs = [(lab[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['i']).astype(float) for c in icons]
    typ = {}
    for i, c in enumerate(icons):
        nxt = [o['x'] for o in icons if o is not c and o['x'] > c['x'] + c['w'] and abs(o['cy'] - c['cy']) < 0.6 * Z]
        t = fac_label(c, words, Z, min(nxt) if nxt else None)
        if t and t not in typ.values(): typ[i] = t
    rest = [i for i in range(len(icons)) if i not in typ]
    free = [t for t in PROTO if t not in typ.values()]
    if rest and free:                                   # 글자를 못 읽은 아이콘: 남은 종류와 모양(채움·위아래·귀퉁이) 거리 합이 최소인 배정
        import itertools
        k = min(len(rest), len(free)); best = None
        for idx in itertools.permutations(rest, k):
            for ts in itertools.permutations(free, k):
                d = sum(np.abs(_fac_feats(mm[icons[i]['y']:icons[i]['y'] + icons[i]['h'], icons[i]['x']:icons[i]['x'] + icons[i]['w']]) - PROTO[t]).sum() for i, t in zip(idx, ts))
                if best is None or d < best[0]: best = (d, idx, ts)
        for i, t in zip(best[1], best[2]): typ[i] = t
    tpl = {}
    for i, t in typ.items():
        c = icons[i]
        crop = mm[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']]
        tpl[t] = (cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0), _fac_feats(crop))
    H, W = mm.shape
    out = []
    for c in cs:
        if inleg(c): continue
        s = max(c['w'], c['h'])
        if not (0.55 * Z <= s <= 2.2 * Z) or c['a'] < 0.2 * Z * Z or not (0.5 < c['w'] / max(1, c['h']) < 2.0): continue
        pad = int(0.5 * Z) + 3
        x0, y0, x1, y1 = max(0, c['x'] - pad), max(0, c['y'] - pad), min(W, c['x'] + c['w'] + pad), min(H, c['y'] + c['h'] + pad)
        G = mm[y0:y1, x0:x1]
        k = s / Z                                      # 도면 기호 / 범례 아이콘 크기 비
        best = None
        fp_ = _fac_feats(mm[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']])
        for t, (tp, tf) in tpl.items():
            v = -1.0
            for sc in (0.9 * k, k, 1.1 * k):
                Tm = cv2.resize(tp, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                if Tm.shape[0] >= G.shape[0] or Tm.shape[1] >= G.shape[1] or Tm.std() < 1e-3: continue
                v = max(v, float(cv2.matchTemplate(G, Tm, cv2.TM_CCOEFF_NORMED).max()))
            sc_ = v - FAC_FW * float(np.abs(fp_ - tf).sum())       # 종류 고르기: 상관 − 모양 특징 거리
            if best is None or sc_ > best[0]: best = (sc_, t, v)
        if best and best[2] > 0.5:                               # 받아들이기: 고른 종류 형판과의 상관
            out.append(dict(type=best[1], px=c['cx'], py=c['cy'], w=c['w'], h=c['h'], conf=round(float(best[2]), 2)))
    return out, {t: 1 for t in tpl}
