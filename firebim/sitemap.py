# -*- coding: utf-8 -*-
"""진압작전도(정보카드 ⑪ '0 진압작전도') 인식 — 범례 적응형.

표준양식의 범례(필수) 5종: 주출입구 · 비상구 · 소화전 · 연결송수구 · 이격거리.
  1) 글자 줄 찾기(작은 어두운/색 성분을 줄로 묶음) → 줄마다 OCR(3배 확대)
  2) 범례: 5개 범례 낱말을 자모 유사도로 찾고, 낱말 왼쪽의 색 아이콘을 그 종류의 형판으로(색 + 회전 불변 모양 특징)
  3) 도면 안 같은 색 성분 → 모양 특징 거리로 종류 판정(작전도마다 모양·색이 달라도 범례만 맞으면 됨)
  4) 이격거리: 범례 선 색의 긴 선분(점선 이음, 기호에서 나눔·잘라 냄 / 아래 보강은 SITEMAP_DIM* 스위치·기본 끔, SITEMAP_DIMSET=A 로 묶어 켬: 옅은 점선은 약한 색상 마스크로 따라감·같은 직선 위 끊긴 조각 잇기·
     이름표에 가린 끝은 기호까지·글자 상자에 가려 양 끝 조각만 보이는 선은 조각 짝짓기, 틱 끝은 빗금 가운데) → 글자 후보 상자(글자 줄·둘레 덩어리·
     선 방향 띠 — 기울어진 선은 처음부터) 다중 판독 투표(상자 고르기에 OCR 신뢰도, 애매한 첫 글자 '약'은 한 글자 한글 OCR 로 확인)
     → 기하 길이(축척 × 선 길이, 끝이 기호면 기호 중심까지)와 맞춰 값·출처(ocr / ocr+geom / geom)·확실도 결정(resolve_dim)
  5) 방위표('N' 글자 + 화살표), 축척 막대(0 … 20m), 대상 건물 외곽(주출입구·송수구 기호가 둘레에 붙은 닫힌 윤곽), 도로(회색 띠 + 폭 글자)
  6) to_frame(): 화소 → 건물 좌표(m). 회전 = 방위표를 건물 외곽 축에 맞춤, 축척 = 축척 막대·도면(평면도) 외곽 크기·정보카드 면적 합의,
     평행이동 = 평면도 외곽(엔진 좌표)과 외곽 중심 맞춤 → 주출입구·비상구·송수구는 외벽 위로 붙임.
사용: rec = recognize(img_bgr); fr = to_frame(rec, ref_outline=Polygon(엔진 1층 외곽), card_area=㎡)"""
import math, re, subprocess, tempfile, os
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np, cv2
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import nearest_points

WORDS = dict(main='주출입구', exit='비상구', hyd='소화전', fdc='연결송수구', dim='이격거리')
# 2026-10-02 치수 보강 스위치(SITEMAP_DIM*)는 모두 기본 끔 = 예전 동작과 같음(보류 세트 held 에서 글자 맞음이 내려가 기본으로 켜지 않음).
# SITEMAP_DIMSET=A: 기본 형식 개발 세트(dev·dev_x·dev_y)에서 맞음 ≥ 예전·틀림 ≤ 예전을 지킨 묶음(아래 8개 켬, RES2·TICK 은 끔) — 개별 스위치로 덮어쓸 수 있음.
DIMSET_A = os.environ.get('SITEMAP_DIMSET', '').upper() == 'A'
PT_TYPES = ('main', 'exit', 'hyd', 'fdc')
DEBUG = None


# ─────────────────────────────── 기본
def _lum_sat(img):
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    return (b + g + r) / 3.0, (np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)).astype(np.float32)


def _ocr(gray, lang='kor+eng', psm=7, wl=None):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, gray); fn = tf.name
    try:
        cmd = ['tesseract', fn, '-', '-l', lang, '--psm', str(psm)]
        if wl: cmd += ['-c', f'tessedit_char_whitelist={wl}']
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    finally: os.unlink(fn)
    return ' '.join(out.split())


def _jamo(ch):
    o = ord(ch) - 0xAC00
    return (o // 588, (o % 588) // 28, o % 28) if 0 <= o < 11172 else (ch,)


def _jsim(a, b):
    tot = hit = 0
    for x, y in zip(a, b):
        jx, jy = _jamo(x), _jamo(y); tot += 3
        hit += sum(1 for u, v in zip(jx, jy) if u == v) * (3 // len(jx) if len(jx) == len(jy) else 0)
    return hit / max(1, tot)


def word_score(txt, w):
    t = ''.join(ch for ch in txt if '가' <= ch <= '힣')
    if w in t: return 1.0
    if not t: return 0.0
    return max([_jsim(t[i:i + len(w)], w) for i in range(0, max(1, len(t) - len(w) + 1))] + [0])


def parse_dist(txt):
    """'38m' '0.150km' 'L=9.5M' '150 m' → m (float) 또는 None"""
    t = re.sub(r'(\d)\s*[.,]\s*(\d)', r'\1.\2', txt.replace('O', '0').replace('o', '0'))
    m = None
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s?(k[mM]|K[mM]|[mM])', t): pass
    if not m: return None
    v = float(m.group(1)); return v * 1000 if m.group(2).lower() == 'km' else v


# ─────────────────────────────── 1) 글자 줄
def text_lines(img, lum, sat, bg):
    ink = ((lum < bg - 75) | ((sat > 60) & (lum < bg - 25))).astype(np.uint8)
    # 글자가 긴 선(도로 가장자리·외곽선) 위에 얹힌 경우: 긴 직선을 먼저 지워 글자를 떼어 낸다
    dk = ((lum < bg - 75) & (sat < 60)).astype(np.uint8)
    segs = cv2.HoughLinesP(dk, 1, np.pi / 360, threshold=60, minLineLength=80, maxLineGap=3)
    if segs is not None:
        lm = np.zeros_like(ink)
        for x0_, y0_, x1_, y1_ in segs[:, 0, :]: cv2.line(lm, (int(x0_), int(y0_)), (int(x1_), int(y1_)), 1, 3)
        ink = ink & (1 - lm)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    cs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not ((4 <= h <= 26 and w <= 3 * h + 6) or (2 <= h < 4 and 4 <= w <= 16)) or a < 6: continue      # 가로획(ㅡ)도 글자 조각
        col = float(sat[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean()) > 45
        if col and max(w, h) >= 19: continue                               # 색 아이콘은 글자가 아님
        cs.append([x, y, x + w, y + h, col])
    cs.sort(key=lambda c: c[0])
    par = list(range(len(cs)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i, a in enumerate(cs):
        ha = a[3] - a[1]
        for j in range(i + 1, len(cs)):
            b = cs[j]
            if b[0] - a[2] > 1.2 * max(ha, 10): break
            hb = b[3] - b[1]
            ov = min(a[3], b[3]) - max(a[1], b[1])
            if ov >= 0.45 * min(ha, hb) and b[0] - a[2] <= 0.9 * max(ha, hb) and max(ha, hb) / max(1, min(ha, hb)) <= 3.0:
                par[f(j)] = f(i); continue
            xo = min(a[2], b[2]) - max(a[0], b[0])                 # 위아래로 쌓인 자모(ㅈ+ㅜ)
            vg = max(a[1], b[1]) - min(a[3], b[3])
            if xo >= 0.5 * min(a[2] - a[0], b[2] - b[0]) and vg <= 3 and max(a[3], b[3]) - min(a[1], b[1]) <= 20:
                par[f(j)] = f(i)
    groups = {}
    for i, c in enumerate(cs): groups.setdefault(f(i), []).append(c)
    out = []
    for g in groups.values():
        x0, y0 = min(c[0] for c in g), min(c[1] for c in g); x1, y1 = max(c[2] for c in g), max(c[3] for c in g)
        if x1 - x0 < 5: continue
        out.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, h=y1 - y0, n=len(g), col=sum(c[4] for c in g) > len(g) / 2, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2))
    return out


def ocr_lines(gray, lines, k=4):
    H, W = gray.shape
    for ln in lines:
        p = 4
        x0, y0, x1, y1 = max(0, ln['x0'] - p), max(0, ln['y0'] - p), min(W, ln['x1'] + p), min(H, ln['y1'] + p)
        crop = cv2.resize(gray[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        if ln['n'] == 1 and (ln['x1'] - ln['x0']) < 1.3 * ln['h']:                  # 한 글자(방위 N 등): 6배 확대가 잘 읽힘
            c6 = cv2.resize(gray[y0:y1, x0:x1], None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
            _, c6 = cv2.threshold(c6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ln['text'] = _ocr(cv2.copyMakeBorder(c6, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255), 'eng', 13)
        elif ln['col']:
            ln['text'] = ln['text_num'] = _ocr(crop, 'eng', 7, '0123456789.kmM-=() L')
        else:
            ln['text'] = _ocr(crop, 'kor', 7)                                       # 한글 우선(kor+eng 는 짧은 한글을 영문으로 오독)
            if re.search(r'\d', ln['text']):                                       # 숫자 부분은 영문·숫자 전용으로 다시(m·km 오독 줄이기)
                ln['text_num'] = _ocr(crop, 'eng', 7, '0123456789.kmM-=() L')
    return lines


def _num(ln):
    v = parse_dist(ln.get('text_num', '') or '')
    return v if v is not None else parse_dist(ln.get('text', ''))


# ─────────────────────────────── 2) 범례
def _shape_feats(m):
    """이진 조각 → 회전 불변 모양 특징"""
    m = (m > 0).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    outer = [i for i in range(len(cnts)) if hier[0][i][3] < 0]
    c = max((cnts[i] for i in outer), key=cv2.contourArea)
    A = float(m.sum()); Ac = max(1.0, cv2.contourArea(c)); P = max(1.0, cv2.arcLength(c, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hull = max(1.0, cv2.contourArea(cv2.convexHull(c)))
    holes = sum(1 for i in range(len(cnts)) if hier[0][i][3] >= 0 and cv2.contourArea(cnts[i]) > 0.03 * Ac)
    hu = cv2.HuMoments(cv2.moments(m, True)).ravel()
    hl = -np.sign(hu[:2]) * np.log10(np.abs(hu[:2]) + 1e-12)
    return np.array([A / max(1.0, rw * rh), A / hull, min(1.0, 4 * math.pi * Ac / (P * P)), min(holes, 2) / 2.0,
                     max(rw, rh) / max(1.0, min(rw, rh)) / 3.0, hl[0] / 2.0, hl[1] / 6.0]), max(rw, rh)


FW = np.array([2.0, 2.5, 2.0, 0.5, 1.0, 1.5, 0.6])


def _lab(img): return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _cmask(LAB, c, thr):
    d = LAB - np.array(c, np.float32)
    dist = np.sqrt(0.35 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    m = dist < thr
    ch = math.hypot(c[1] - 128, c[2] - 128)
    if ch > 20: m &= np.hypot(LAB[..., 1] - 128, LAB[..., 2] - 128) > 0.45 * ch     # 유채색 형판: 회색·검정 선 제외
    return m.astype(np.uint8)


def _hmask(LAB, c, ang_deg=25, frac=0.25, cmin=11.0, lmax=222):
    """색상각 마스크(가는 선·흐린 점선용): 밝기 무시, (a,b) 방향이 형판과 비슷하고 채도가 조금이라도 있으면"""
    ta, tb = c[1] - 128, c[2] - 128; tch = math.hypot(ta, tb)
    a, b = LAB[..., 1] - 128, LAB[..., 2] - 128; ch = np.hypot(a, b)
    cosd = (a * ta + b * tb) / np.maximum(1e-3, ch * tch)
    return ((ch > max(cmin, frac * tch)) & (cosd > math.cos(math.radians(ang_deg))) & (LAB[..., 0] < lmax)).astype(np.uint8)   # 연한 바탕색(건물 채움) 제외


DIM_LEGWEAK = os.environ.get('SITEMAP_DIMLEGWEAK', '1' if DIMSET_A else '0') != '0'   # 이격거리 범례 선을 못 찾으면 낮은 채도 문턱으로 다시(기본 끔, 1 = 켬)


def find_legend(img, lines, lum, sat, bg, LAB):
    """범례 낱말 5개 → 낱말 왼쪽 색 아이콘 = 형판. 반환 dict(box, icons{type: dict(lab, feats, size, bbox)}, words{type: line})"""
    cand = {}
    for t, w in WORDS.items():
        sc = sorted(((word_score(ln.get('text', ''), w), k) for k, ln in enumerate(lines) if not ln['col']), reverse=True)
        cand[t] = [(s, k) for s, k in sc if s >= 0.55][:4]
    found = {t: v for t, v in cand.items() if v}
    if len(found) < 3: return None
    # 서로 가까이 모인 조합(범례 상자) 고르기: 가장 확실한 낱말들의 중앙값 근처
    anchors = [lines[v[0][1]] for v in found.values() if v[0][0] >= 0.8] or [lines[v[0][1]] for v in found.values()]
    mx, my = np.median([a['cx'] for a in anchors]), np.median([a['cy'] for a in anchors])
    words = {}
    used = set()
    # 범례 칸은 한 줄(아래 범례: 같은 높이) 또는 한 열(오른쪽·안쪽 범례: 왼쪽 끝이 같음)에 늘어섬 → 가장 많은 종류의 낱말이 맞춰 선 줄·열을 기준으로.
    # (가로로 긴 아래 범례의 양 끝 낱말이 가운데 가까운 다른 글자에 밀리던 문제)
    allc = [(t, s, k) for t, v in found.items() for s, k in v]
    def _al(a, b): return abs(a['cy'] - b['cy']) <= 6 or abs(a['x0'] - b['x0']) <= 12
    def _cover(c): return {t2: max(s2 for t2_, s2, k2 in allc if t2_ == t2 and _al(lines[c[2]], lines[k2])) for t2 in {t2 for t2, s2, k2 in allc if _al(lines[c[2]], lines[k2])}}
    refc = max(allc, key=lambda c: (len(_cover(c)), sum(_cover(c).values()))) if allc else None
    aligned = {t: [(s, k) for s, k in v if _al(lines[refc[2]], lines[k])] for t, v in found.items()} if refc and len(_cover(refc)) >= 3 else {}
    for t, v in sorted(found.items(), key=lambda kv: -kv[1][0][0]):
        va = [(s, k) for s, k in aligned.get(t, []) if k not in used]
        if va: best = max(va, key=lambda sk: (sk[0], -math.hypot(lines[sk[1]]['cx'] - lines[refc[2]]['cx'], lines[sk[1]]['cy'] - lines[refc[2]]['cy'])))[1]
        else: best = min((k for s, k in v if k not in used), key=lambda k: math.hypot(lines[k]['cx'] - mx, lines[k]['cy'] - my) - 300 * [s for s, kk in v if kk == k][0], default=None)
        if best is None: continue
        if math.hypot(lines[best]['cx'] - mx, lines[best]['cy'] - my) > 0.6 * max(img.shape[:2]): continue
        words[t] = lines[best]; used.add(best)
    if len(words) < 3: return None
    miss = [t for t in WORDS if t not in words]
    if len(miss) == 1 and len(words) == 4:                  # 한 낱말만 못 읽음: 범례 칸 간격으로 빈 칸 위치를 추정
        ws_ = list(words.values())
        vert = np.std([w['x0'] for w in ws_]) < np.std([w['cy'] for w in ws_])
        key = (lambda w: w['cy']) if vert else (lambda w: w['x0'])
        vs = sorted(key(w) for w in ws_); d_ = np.diff(vs); pitch = float(np.min(d_)) if len(d_) else 0
        if pitch > 8:
            cands = [vs[0] - pitch, vs[-1] + pitch] + [a + pitch for a, b in zip(vs, vs[1:]) if b - a > 1.5 * pitch]
            ref = ws_[0]
            for c_ in cands:
                ln = dict(ref); 
                if vert: ln.update(cy=c_, y0=c_ - ref['h'] / 2, y1=c_ + ref['h'] / 2)
                else: ln.update(x0=c_, x1=c_ + (ref['x1'] - ref['x0']), cx=c_ + (ref['x1'] - ref['x0']) / 2)
                if 0 <= ln['y0'] and ln['y1'] < img.shape[0] and 0 <= ln['x0'] < img.shape[1]:
                    ln['inferred'] = True; words.setdefault('_cand', []).append(ln)
    cand_rows = words.pop('_cand', [])
    col = ((sat > 60) & (lum < bg - 20)).astype(np.uint8)
    icons = {}
    rows_ = list(words.items())
    if miss and len(miss) == 1: rows_ += [(miss[0], ln) for ln in cand_rows]
    for t, ln in rows_:
        if t in icons: continue
        h = max(10, ln['h']); zg = 2.4 * h
        x0, x1 = int(max(0, ln['x0'] - 3.2 * zg)), int(ln['x0'] - 1)
        y0, y1 = int(max(0, ln['cy'] - 1.3 * zg)), int(min(img.shape[0], ln['cy'] + 1.3 * zg))
        if x1 <= x0 + 3: continue
        n, lab, st, cen = cv2.connectedComponentsWithStats(col[y0:y1, x0:x1], 8)
        best = None
        for i in range(1, n):
            x, y, w, hh, a = st[i]
            if a < 15 or max(w, hh) < 6: continue
            if abs((y + hh / 2) - (ln['cy'] - y0)) > 0.9 * zg: continue
            gap = (x1 - x0) - (x + w)
            sc = a - 3 * gap
            if best is None or sc > best[0]: best = (sc, i, x, y, w, hh)
        weak_ = False
        if best is None and t == 'dim' and DIM_LEGWEAK:              # 가늘고 옅은 이격거리 범례 선(JPEG 로 채도가 빠짐): 낮은 채도 문턱으로 다시 — 가로로 긴 조각만
            colw0 = ((sat > 22) & (lum < bg - 8)).astype(np.uint8)[y0:y1, x0:x1]
            colw = cv2.morphologyEx(colw0, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))       # 점선 조각·틱을 한 덩어리로
            n, lab, st, cen = cv2.connectedComponentsWithStats(colw, 8)
            lab = lab * (colw0 > 0)                                                            # 색은 원래 먹 화소에서만
            for i in range(1, n):
                x, y, w, hh, a = st[i]
                if a < 10 or w < max(12, 2.5 * hh): continue
                if abs((y + hh / 2) - (ln['cy'] - y0)) > 0.9 * zg: continue
                gap = (x1 - x0) - (x + w)
                sc = a - 3 * gap
                if best is None or sc > best[0]: best = (sc, i, x, y, w, hh); weak_ = True
        if best is None: continue
        _, i, x, y, w, hh = best
        # 같은 색 조각(점선 이격거리 등) 합치기: 아이콘 색 마스크로 다시
        pix = img[y0:y1, x0:x1][lab == i]
        c = np.median(LAB[y0:y1, x0:x1][lab == i], axis=0)
        if weak_:                                                  # 옅은 선: 채도 높은 화소(선 가운데)의 색만
            sv = sat[y0:y1, x0:x1][lab == i]; hi_ = sv >= np.percentile(sv, 70)
            c = np.median(LAB[y0:y1, x0:x1][lab == i][hi_], axis=0); pix = pix[hi_]
        bx = [x0 + x, y0 + y, x0 + x + w, y0 + y + hh]
        if t == 'dim' and not weak_:
            m = _cmask(LAB[y0:y1, x0:x1], c, 40)
            ys, xs = np.where(m[max(0, y - 6):y + hh + 6, :] > 0)
            if len(xs): bx = [x0 + xs.min(), y0 + max(0, y - 6) + ys.min(), x0 + xs.max() + 1, y0 + max(0, y - 6) + ys.max() + 1]
        sub = _cmask(LAB[bx[1]:bx[3], bx[0]:bx[2]], c, 38)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        ff = _shape_feats(sub)
        if ff is None: continue
        icons[t] = dict(lab=[float(v) for v in c], bgr=[int(v) for v in np.median(pix, axis=0)], feats=ff[0], size=ff[1], bbox=[int(v) for v in bx])
    # 범례 상자: 낱말·아이콘을 감싸는 가장 작은 어두운 사각 윤곽(없으면 감싼 상자 + 여백)
    xs = [ln['x0'] for ln in words.values()] + [i['bbox'][0] for i in icons.values()]
    ys = [ln['y0'] for ln in words.values()] + [i['bbox'][1] for i in icons.values()]
    xe = [ln['x1'] for ln in words.values()] + [i['bbox'][2] for i in icons.values()]
    ye = [ln['y1'] for ln in words.values()] + [i['bbox'][3] for i in icons.values()]
    inner = [min(xs), min(ys), max(xe), max(ye)]
    dk = (lum < bg - 100).astype(np.uint8)
    cnts, _ = cv2.findContours(dk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lb = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if x <= inner[0] - 2 and y <= inner[1] - 2 and x + w >= inner[2] + 2 and y + h >= inner[3] + 2 and w * h < 0.3 * img.shape[0] * img.shape[1]:
            if lb is None or w * h < lb[2] * lb[3]: lb = (x, y, w, h)
    if lb is None or lb[2] * lb[3] > 6 * (inner[2] - inner[0] + 40) * (inner[3] - inner[1] + 60):
        lb = (inner[0] - 20, inner[1] - 45, inner[2] - inner[0] + 40, inner[3] - inner[1] + 60)
    return dict(box=[int(lb[0]), int(lb[1]), int(lb[0] + lb[2]), int(lb[1] + lb[3])], icons=icons, words={t: [ln['x0'], ln['y0'], ln['x1'], ln['y1']] for t, ln in words.items()})


# ─────────────────────────────── 3) 점 기호
DET_THR = 1.6          # 모양 거리 받아들임 문턱(개발 세트 조정)
COL_THR = 34


def detect_points(img, LAB, leg, excl):
    H, W = img.shape[:2]
    out = []
    Zs = [leg['icons'][t]['size'] for t in PT_TYPES if t in leg['icons']]
    if not Zs: return out, 20.0
    Z = float(np.median(Zs))
    for t in PT_TYPES:
        if t not in leg['icons']: continue
        tp = leg['icons'][t]
        m = _cmask(LAB, tp['lab'], COL_THR)
        m[excl > 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        parts = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            s = max(w, h)
            if s > 1.9 * tp['size'] and a >= 0.12 * tp['size'] ** 2 and s < 30 * tp['size']:
                # 같은 색 선(치수선)에 붙은 기호: 가는 선을 열림 연산으로 떼어 내고 다시
                sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
                for kk in (3, 5):
                    so = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
                    n2, l2, s2, c2 = cv2.connectedComponentsWithStats(so, 8)
                    got = [(x + s2[j][0], y + s2[j][1], s2[j][2], s2[j][3], s2[j][4], (l2[s2[j][1]:s2[j][1] + s2[j][3], s2[j][0]:s2[j][0] + s2[j][2]] == j).astype(np.uint8),
                            (x + c2[j][0], y + c2[j][1])) for j in range(1, n2) if 0.55 * tp['size'] <= max(s2[j][2], s2[j][3]) <= 1.9 * tp['size']]
                    if got: parts += got; break
                continue
            parts.append((x, y, w, h, a, m[y:y + h, x:x + w].copy(), tuple(cen[i])))          # 상자 안 같은 색 전부(가운데 점 등 떨어진 부분 포함 — 범례 형판과 같게)
        for (x, y, w, h, a, msk, cc) in parts:
            s = max(w, h)
            if s < 0.55 * tp['size'] or s > 1.9 * tp['size'] or a < 0.12 * tp['size'] ** 2: continue
            ff = _shape_feats(msk)
            if ff is None: continue
            # 모든 범례 형판과 비교(색이 비슷한 종류끼리 헷갈리지 않게) — 색 거리도 벌점
            best = None
            for t2 in PT_TYPES:
                if t2 not in leg['icons']: continue
                tp2 = leg['icons'][t2]
                cd = float(np.linalg.norm(np.array(tp2['lab']) - np.array(tp['lab'])))
                if cd > 2 * COL_THR: continue
                dsh = float((FW * np.abs(ff[0] - tp2['feats'])).sum()) + 2.0 * abs(math.log(ff[1] / tp2['size'])) + cd / 40
                if best is None or dsh < best[0]: best = (dsh, t2)
            if DEBUG is not None: DEBUG.append(dict(t=t, x=float(cc[0]), y=float(cc[1]), d=best))
            if best and best[0] < DET_THR:
                out.append(dict(type=best[1], px=float(cc[0]), py=float(cc[1]), size=float(ff[1]), d=round(best[0], 2)))
    # 같은 자리 중복 제거
    out.sort(key=lambda q: q['d'])
    keep = []
    for q in out:
        if all(math.hypot(q['px'] - k['px'], q['py'] - k['py']) > 0.5 * Z for k in keep): keep.append(q)
    return keep, Z


# ─────────────────────────────── 4) 이격거리 치수선
# 선 찾기(범례 선 색상각 마스크 → Hough → 같은 직선 조각 합치기) → 끝점 다듬기(화살촉·틱 포함, 같은 색 기호에 붙은 끝은 잘라 냄)
# → 기하 길이(px, 끝이 소화전·송수구 기호에 붙으면 기호 중심까지) → 글자 후보 상자(글자 줄 + 선 방향으로 돌린 띠)
# → 상자마다 여러 번 판독(배율 3·4, Otsu, 지나가는 긴 선 지움, psm 7·8, 숫자+단위 화이트리스트, kor) → 값 투표
# → resolve_dim(): 축척(축척 막대·도로 폭 글자·다른 치수 합의 / to_frame 에서는 최종 축척)으로 기하 길이와 맞춰 값·출처·확실도 결정.
DIM_WL = '0123456789.,mMkK=L '
DIM_TOL_ABS, DIM_TOL_REL = 1.5, 0.10          # 기하 길이 ↔ 글자 허용(m, 비율) — 끝이 기호에 붙은 쪽마다 0.35·기호 크기 더함
DIMDBG = None                                  # 디버그: 리스트를 넣으면 상자·판독 기록
DIM_MAX_M = 300.0                              # 이격거리(건물 간격·소화전→송수구·도로) 그럴듯한 범위 상한(m) — 넘는 판독은 다른 해석(소수점 빠짐·앞 글자 숫자 오독)을 찾음


def _ocr_batch(imgs, lang='eng', psm=7, wl=None, conf=False):
    """여러 조각을 tesseract 한 번 호출로(목록 파일 입력 → 쪽마다 \\f 구분).
    conf=True: tsv 출력으로 (글자들, 쪽마다 낱말 신뢰도 평균(0~100, 글자 없으면 -1)) — 글자는 같은 판독 결과의 낱말을 이어 붙인 것."""
    if not imgs: return ([], []) if conf else []
    import shutil
    d = tempfile.mkdtemp(prefix='fbocr')
    try:
        names = []
        for i, im in enumerate(imgs):
            fn = os.path.join(d, f'{i:03d}.png'); cv2.imwrite(fn, im); names.append(fn)
        lst = os.path.join(d, 'list.txt')
        with open(lst, 'w') as f: f.write('\n'.join(names) + '\n')
        cmd = ['tesseract', lst, '-', '-l', lang, '--psm', str(psm)]
        if wl: cmd += ['-c', f'tessedit_char_whitelist={wl}']
        if conf: cmd += ['tsv']
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60 + 5 * len(imgs)).stdout
    finally: shutil.rmtree(d, ignore_errors=True)
    if conf:
        W_ = {}
        for row in out.split('\n')[1:]:
            c = row.split('\t')
            if len(c) < 12 or c[0] != '5': continue
            t = c[11].strip()
            if not t: continue
            try: W_.setdefault(int(c[1]) - 1, []).append((t, float(c[10])))
            except ValueError: continue
        txt = [' '.join(' '.join(w for w, _ in W_.get(k, [])).split()) for k in range(len(imgs))]
        cf = [float(np.mean([c_ for _, c_ in W_[k]])) if W_.get(k) else -1.0 for k in range(len(imgs))]
        return txt, cf
    pages = [' '.join(p.split()) for p in out.split('\f')][:len(imgs)]
    return pages + [''] * (len(imgs) - len(pages))


def parse_dim_vals(txt):
    """치수 글자 → [(값 m, 단위 있음, '약')] — '12.5m' '12,5 m'(소수 쉼표) 'L=9.5M' '약 12m' '12미터' '0.15km' '12 m'.
    단위 없는 숫자도 (값, False, …) 로 돌려준다(가중치를 낮춰 씀)."""
    if not txt: return []
    # 쉼표: 뒤 숫자 1~2자리 = 소수 쉼표('42,0m'·'12,5m'), 3자리 = 천 단위 → mm 로 쓴 치수('12,500' = 12.5 m — 소수로 읽어도 m 값이 같음).
    #       단위 m 이 붙은 'N,NNN m'(1,200 m)도 소수(1.2 m)로 읽힘 — 이격거리는 0.5~300 m 라 천 m 단위는 없다고 봄.
    t = txt.replace('O', '0').replace('o', '0')
    t = re.sub(r'(?<=\d)\s*[미비마이메][\s]*[터타더]|[미마]\s*[터타더]|meters?|METERS?', 'm', t)
    approx = bool(re.search(r'[약악]|~|±|≒|≈', t))
    t = re.sub(r'(?<=\d)\s+(?=\d)', '', t)                                  # '1 2.5' → '12.5'
    t = re.sub(r'(\d)\s*[.,]\s*(\d)', r'\1.\2', t)
    out = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s?(k\s?[mM]|K\s?[mM]|[mM])?', t):
        try: v = float(m.group(1))
        except ValueError: continue
        u = m.group(2)
        if u and u.replace(' ', '').lower() == 'km': v *= 1000
        elif not u and '.' not in m.group(1) and len(m.group(1)) >= 4: v /= 1000      # 단위 없는 4자리 이상 정수 = mm('43,000' 의 쉼표가 빠진 '43000')
        elif v < 0.3 and '.' in m.group(1) and re.match(r'\s*[,.]?\s*[kK]|\s*[,.]\s*[mM]', t[m.end():]):   # '0.016km' 의 k 를 ',' 로 읽거나 m 이 빠짐 → m 로는 너무 작음 = km
            v *= 1000; u = 'km'
        out.append((v, bool(u), approx))
    return out


_KNUM = re.compile(r'(\d+(?:\s*[.,]\s*\d+)?)\s*[미비마이메]\s*[터타더]')


def _trim_kunit(t, knums):
    """한글 판독이 'NN미터' 로 읽은 상자: 화이트리스트 판독은 '미터' 를 숫자로 읽어 뒤에 붙임('53미터' → '530164') →
    판독 숫자열이 한글 판독 숫자열 NN 으로 시작하고 더 길면 'NNm' 으로 자름."""
    m = re.search(r'\d[\d.,]*', t or '')
    if not m or not knums: return t
    ds = re.sub(r'[.,]', '', m.group(0))
    for k in knums:
        kd = re.sub(r'\D', '', k)
        if kd and ds.startswith(kd) and len(ds) > len(kd): return t[:m.start()] + k.replace(' ', '') + 'm'
    return t


def _vote_texts(texts, unit_hint=False, knums=()):
    """판독 글자 여러 개 → {값: [직접 표, 단위 있음, 약, 약한 표]}. 단위 붙은 숫자 1표, 단위 없는 숫자 0.5표.
    약한 표(0.25): 앞·뒤 한 글자 뺀 값·소수점 빠진 값('125' → 12.5) — 화살촉·'='·틱을 숫자로 읽은 경우용, 기하 길이와 맞을 때만 쓰임."""
    V = {}
    def add(v, w, u, a, weak=False):
        if not (0.3 <= v <= 2000): return
        k = round(v, 2); e = V.setdefault(k, [0.0, False, False, 0.0]); e[3 if weak else 0] += w; e[1] |= u; e[2] |= a
    for t in texts:
        vs = parse_dim_vals(_trim_kunit(t, knums))
        if not vs: continue
        prim = [x for x in vs if x[1]] or vs
        for v, u, a in prim[-1:]:                                           # 한 판독에서 값 하나(단위 붙은 마지막)
            u = u or unit_hint                                              # 한글 판독에 '미터'가 보이면 영문 판독(화이트리스트로 한글이 빠진 것)도 단위 있음으로
            add(v, 1.0 if u else 0.5, u, a)
            s = f'{v:g}'
            if len(s.replace('.', '')) >= 2:
                for s2 in {s[1:], s[:-1]} | ({s[:-1] + '.' + s[-1]} if '.' not in s else set()):
                    s2 = s2.strip('.')
                    if s2 and s2 != s:
                        try: add(float(s2), 0.25, u, a, weak=True)
                        except ValueError: pass
    return V


def _bgval(patch):
    return float(np.percentile(patch, 85))


def _flat(patch, k=9):
    """바탕 평탄화: 닫힘 연산(가는 어두운 획 지움)으로 바탕을 추정해 나눔 → 도로 회색·흰 바탕 모두 255 근처, 글자·선만 어둡게"""
    if min(patch.shape[:2]) < 3: return patch.copy()
    bg = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    return np.clip(patch.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)


def _clean_lines(patch, h, keep_mask=None):
    """(평탄화한) 조각 안을 지나가는 긴 가는 직선(도로 가장자리·중앙선·치수선)을 흰색으로 지운다.
    글자 줄의 윗선·밑선이 직선으로 잡히지 않게: 길이 ≥ max(40, 3.2·글자 높이), 선 따라 법선 방향 먹 폭 중앙값 ≤ 4px 인 것만."""
    ink = (patch < 205).astype(np.uint8)
    L = int(max(40, 3.2 * h))
    out = patch.copy()
    if min(ink.shape) < 5: return out, 0
    segs = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=int(0.7 * L), minLineLength=L, maxLineGap=2)
    if segs is None: return out, 0
    Hh, Ww = ink.shape
    lm = np.zeros_like(ink); n = 0
    for x0, y0, x1, y1 in segs[:, 0, :]:
        Ls = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / Ls, (y1 - y0) / Ls
        ws = []
        for t in np.linspace(0, Ls, 15):
            px, py = x0 + ux * t, y0 + uy * t; c = 0
            for o in range(-7, 8):
                xx, yy = int(round(px - uy * o)), int(round(py + ux * o))
                if 0 <= yy < Hh and 0 <= xx < Ww and ink[yy, xx]: c += 1
            ws.append(c)
        if float(np.median(ws)) > 4: continue
        one = np.zeros_like(ink); cv2.line(one, (int(x0), int(y0)), (int(x1), int(y1)), 1, 3)
        # 글자 획 보존: 선 띠 바로 바깥(법선 ±3px)에 먹이 있으면 그 자리는 선을 가로지르는 획 → 남김
        nx_, ny_ = -uy, ux
        keep = np.zeros_like(ink)
        for d in (3, 4):
            for sg in (1, -1):
                Mt = np.float32([[1, 0, sg * d * nx_], [0, 1, sg * d * ny_]])
                keep |= cv2.warpAffine(ink, Mt, (Ww, Hh), flags=cv2.INTER_NEAREST)
        lm |= one & (1 - keep); n += 1
    if keep_mask is not None: lm[keep_mask > 0] = 0
    out[lm > 0] = 255
    return out, n


def _prep(patch, k):
    c = cv2.resize(patch, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    _, c = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(c, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _dim_profile(m, g, Z, occ=None, mw=None):
    """Hough 로 찾은 선(g)의 끝 다듬기: 바깥으로 늘림(틈 ≤ 0.8Z, 화살촉·틱을 지나면 멈춤, 선을 덮은 글자·검은 선은 건너뜀) →
    끝의 굵은 구간이 화살촉보다 길면 같은 색 기호로 보고 잘라 냄. occ: 가림 마스크(검은 먹·글자 상자). mw: 약한 색상 마스크(JPEG 로 옅어진 점선 —
    선 바로 위(법선 ±1px)에서만 선 이어짐 근거로 씀, 굵기·장식 판정에는 안 씀).
    → (t0, t1, 선 굵기, 끝마다 기호에 붙었는지, 끝마다 장식(화살촉·틱) 있는지)  (t: g[0:2] 에서 선 따라 px)"""
    L = math.hypot(g[2] - g[0], g[3] - g[1]); ux, uy = (g[2] - g[0]) / L, (g[3] - g[1]) / L; nx, ny = -uy, ux
    H, W = m.shape
    ext = int(1.2 * Z) if occ is None else int(4.5 * Z)
    ts = np.arange(-ext, int(L) + ext + 1)
    core = np.zeros(len(ts), bool); wid = np.zeros(len(ts)); hid = np.zeros(len(ts), bool); so = np.zeros(len(ts)); sn = np.zeros(len(ts)); cst = np.zeros(len(ts), bool)
    for i, t in enumerate(ts):
        px, py = g[0] + ux * t, g[1] + uy * t
        c = w = 0
        for o in range(-8, 9):
            xx, yy = int(round(px + nx * o)), int(round(py + ny * o))
            if 0 <= yy < H and 0 <= xx < W:
                if m[yy, xx]:
                    w += 1
                    if abs(o) <= 2: c = 1; cst[i] = True
                    else: so[i] += o; sn[i] += 1
                elif mw is not None and abs(o) <= 1 and mw[yy, xx]: c = 1
                elif occ is not None and abs(o) <= 1 and occ[yy, xx]: hid[i] = True
        core[i] = c; wid[i] = w
    mid = (ts >= 0.3 * L) & (ts <= 0.7 * L)
    lw = float(np.median(wid[mid & core])) if (mid & core).any() else 3.0
    i0, i1 = ext, ext + int(L)
    lo, hi = i0, min(i1, len(ts) - 1)
    gmax = int(max(6, 0.8 * Z))
    wide_ = wid >= max(5, lw + 3)
    def walk(k, st_):
        """k 에서 st_ 방향으로: 틈 ≤ gmax 뒤 선(3px 이상)이 다시 이어지면 계속(점선·흐린 구간). 화살촉·틱(굵은 곳)을 지난 뒤의 틈에서는 멈춤(옆 치수선으로 넘어가지 않게)"""
        last = k; head = bool(wide_[max(0, k - 6 * st_):k + 1].any()) if st_ > 0 else bool(wide_[k:k + 7].any())
        while 0 <= k + st_ < len(ts) and abs(k + st_ - last) <= (2 if head else gmax):
            k += st_
            cr = cst if (head and mw is not None) else core              # 화살촉·틱을 지난 뒤에는 강한 마스크만(옅은 번짐으로 넘어가지 않게)
            if cr[k]:
                run = 0; j = k
                while 0 <= j < len(ts) and cr[j] and run < 3: run += 1; j += st_
                if run >= 3 or abs(k - last) <= 2:
                    last = k
                    if wide_[k]: head = True
        return last
    for side in (0, 1):
        st_ = -1 if side == 0 else 1
        k = walk(lo if side == 0 else hi, st_)
        for _ in range(2):                                     # 선을 덮은 글자·검은 선(가린 구간) 건너기: 가린 곳만 지나(틈 ≤ 3px, 최대 3Z) 뒤에 선이 0.6Z 이상 이어질 때
            j = k + st_; nh = gap = 0
            while 0 <= j < len(ts) and nh <= 3 * Z and gap <= (6 if nh == 0 else 3) and not core[j]:
                if hid[j]: nh += 1; gap = 0
                else: gap += 1
                j += st_
            if nh < 2 or not (0 <= j < len(ts)): break
            run = 0; jj = j
            while 0 <= jj < len(ts) and core[jj]: run += 1; jj += st_
            if run < 0.6 * Z: break
            k = walk(jj - st_, st_)
        if mw is not None:                                     # 약한 마스크로만 늘어난 끝(번진 가장자리 1~3px)은 되돌림
            for _ in range(3):
                if cst[k] or not (0 <= k - st_ < len(ts)): break
                k -= st_
        if side == 0: lo = k
        else: hi = k
    # 끝 장식(화살촉·틱): 끝에서 안쪽 14px 안에 선보다 확실히 굵은 곳
    deco = [bool(wide_[lo:min(hi, lo + 14) + 1].any()), bool(wide_[max(lo, hi - 14):hi + 1].any())]
    thick = wid > max(7, 2.5 * lw)
    glued = [False, False]
    for side in (0, 1):                                        # 끝의 굵은 구간이 화살촉(≈10px)보다 길면 같은 색 기호
        k = lo if side == 0 else hi
        n = 0
        while lo <= k <= hi and thick[k]: n += 1; k += 1 if side == 0 else -1
        if n > 0.75 * Z and hi - lo - n > 1.2 * Z:
            glued[side] = True
            if side == 0: lo = k
            else: hi = k
    if DIM_TICK:                                               # 틱(빗금) 끝: 치수 끝점은 빗금이 선을 가로지르는 곳 — 빗금 바깥 끝까지 재면 끝마다 ≈0.6·빗금 길이 길어짐.
        mo = np.where(sn > 0, so / np.maximum(sn, 1), 0.0)     # 선 바깥(법선 |o| ≥ 3) 먹의 평균 법선 위치(부호): 빗금은 한쪽 → 다른 쪽으로 넘어감, 화살촉은 양쪽 대칭(≈0)
        for side in (0, 1):
            if glued[side] or not deco[side]: continue
            k0 = lo if side == 0 else hi; st_ = 1 if side == 0 else -1
            ks = [k0 + st_ * j for j in range(0, 16) if lo <= k0 + st_ * j <= hi]
            v = [mo[k] if sn[k] >= 1 else 0.0 for k in ks]
            out_ = [x for x in v[:5] if abs(x) >= 2.5]; in_ = [x for x in v[3:14] if abs(x) >= 2.5]
            if DIMDBG is not None: DIMDBG.append(dict(stage='tick', side=side, mo=[round(x, 1) for x in v]))
            if len(out_) < 2 or len(in_) < 2 or np.sign(np.median(out_)) == np.sign(np.median(in_)): continue
            so_ = np.sign(np.median(out_))
            j_ = next((j for j in range(1, len(v)) if np.sign(v[j]) == -so_ and abs(v[j]) >= 1.5), None)
            if j_ is None: continue
            jo = max(j for j in range(j_) if np.sign(v[j]) == so_ and abs(v[j]) >= 1.5) if any(np.sign(v[j]) == so_ and abs(v[j]) >= 1.5 for j in range(j_)) else j_
            jc = int(round((jo + j_) / 2))                       # 넘어가는 곳 = 빗금이 선을 가로지르는 곳
            if jc >= 2:
                if side == 0: lo = ks[jc]
                else: hi = ks[jc]
    return float(ts[lo]), float(ts[hi]), lw, glued, deco


def _blob_boxes(gray, m, dm, Z):
    """치수선 가운데 둘레를 직접 살펴 글자 덩어리 줄(긴 선·치수선 자체는 지우고) → [(점수, 상자)] — 전체 글자 줄 묶기가 선에 끊긴 경우"""
    (x0, y0), (x1, y1) = dm['p0'], dm['p1']
    L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L; nx, ny = -uy, ux
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = gray.shape
    ra, rc = 0.5 * L + 20, 58
    cs_ = [(mx + ux * a + nx * c, my + uy * a + ny * c) for a in (-ra, ra) for c in (-rc, rc)]
    X0, Y0 = int(max(0, min(p[0] for p in cs_))), int(max(0, min(p[1] for p in cs_)))
    X1, Y1 = int(min(W, max(p[0] for p in cs_))), int(min(H, max(p[1] for p in cs_)))
    if X1 - X0 < 10 or Y1 - Y0 < 10: return []
    P = _flat(gray[Y0:Y1, X0:X1])
    Pc, _ = _clean_lines(P, 14)
    lm = np.zeros(P.shape, np.uint8)                           # 치수선 자체(글자는 남김)
    cv2.line(lm, (int(x0 - X0), int(y0 - Y0)), (int(x1 - X0), int(y1 - Y0)), 1, int(max(3, 2 * dm.get('lw', 3) + 1)))
    lm &= (m[Y0:Y1, X0:X1] > 0).astype(np.uint8)
    Pc[cv2.dilate(lm, np.ones((3, 3), np.uint8)) > 0] = 255
    ink = (Pc < 170).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    cs = [[st[i][0], st[i][1], st[i][0] + st[i][2], st[i][1] + st[i][3]] for i in range(1, n)
          if st[i][4] >= 5 and 5 <= st[i][3] <= 30 and st[i][2] <= 36]
    cs.sort(key=lambda c: c[0])
    rows = []
    for c in cs:
        for r in rows:
            b = r[-1]; ha, hb = c[3] - c[1], b[3] - b[1]
            if min(c[3], b[3]) - max(c[1], b[1]) >= 0.45 * min(ha, hb) and c[0] - max(q[2] for q in r) <= 0.9 * max(ha, hb) and max(ha, hb) / max(1, min(ha, hb)) <= 3: r.append(c); break
        else: rows.append([c])
    out = []
    for r in rows:
        bx = [min(q[0] for q in r) + X0, min(q[1] for q in r) + Y0, max(q[2] for q in r) + X0, max(q[3] for q in r) + Y0]
        if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 7: continue
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        al, ac = (cx - mx) * ux + (cy - my) * uy, (cx - mx) * nx + (cy - my) * ny
        if abs(al) > ra or abs(ac) > rc: continue
        if bx[2] - bx[0] < 0.8 * (bx[3] - bx[1]): continue
        out.append((abs(ac) + 0.3 * abs(al), bx))
    out.sort(key=lambda o: o[0])
    return out[:4]


def _text_boxes(dm, lines, Z):
    """치수선 가운데 근처 글자 줄(같은 줄 이웃은 합침) → 후보 상자 [(점수, 상자, 글자)]"""
    (x0, y0), (x1, y1) = dm['p0'], dm['p1']
    L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L; nx, ny = -uy, ux
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    near = []
    for ln in lines:
        h = ln['y1'] - ln['y0']
        if h < 6 or h > 34: continue
        cx, cy = (ln['x0'] + ln['x1']) / 2, (ln['y0'] + ln['y1']) / 2
        dx, dy = cx - mx, cy - my
        along, across = dx * ux + dy * uy, dx * nx + dy * ny
        if abs(along) > 0.5 * L + 20 or abs(across) > 55: continue
        t = ln.get('text', '') or ''
        if _hangul_label(t): continue                                         # 한글 낱말(소화전 번호 '가상-4'·도로 이름)
        near.append(dict(ln, cx=cx, cy=cy, h=h, _sc=abs(across) + 0.3 * abs(along)))
    # 같은 줄 이웃 합치기('12.5' + 'm')
    near.sort(key=lambda a: a['x0'])
    groups = []
    for a in near:
        for gp in groups:
            b = gp[-1]
            ov = min(a['y1'], b['y1']) - max(a['y0'], b['y0'])
            if ov >= 0.5 * min(a['h'], b['h']) and a['x0'] - max(q['x1'] for q in gp) <= 0.9 * max(a['h'], b['h']): gp.append(a); break
        else: groups.append([a])
    out = []
    for gp in groups:
        bx = [min(q['x0'] for q in gp), min(q['y0'] for q in gp), max(q['x1'] for q in gp), max(q['y1'] for q in gp)]
        out.append((min(q['_sc'] for q in gp), bx, ' '.join((q.get('text_num') or q.get('text') or '') for q in gp)))
    out.sort(key=lambda o: o[0])
    return out[:3]


DIM_STRIP2 = os.environ.get('SITEMAP_DIMSTRIP2', '1' if DIMSET_A else '0') != '0'     # 선 방향 띠: 점선 획 지우기 + 기울어진 선은 첫 판독부터 띠도 읽어 경쟁(기본 끔, 1 = 켬)


def _strip_patches(gray, m, dm, Z, mw=None):
    """선 방향으로 돌린 띠(선 위·아래·선 가운데 끊긴 자리)에서 글자 덩어리 → [(점수, 조각, 뒤집기 필요)] — 기울어진·세로 치수 글자"""
    (x0, y0), (x1, y1) = dm['p0'], dm['p1']
    L = math.hypot(x1 - x0, y1 - y0)
    a = math.degrees(math.atan2(y1 - y0, x1 - x0))
    a = (a + 90) % 180 - 90                                     # (-90, 90]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    Wp, Hp = int(L + 60), 150
    M = cv2.getRotationMatrix2D((mx, my), a, 1.0); M[0, 2] += Wp / 2 - mx; M[1, 2] += Hp / 2 - my
    rp = cv2.warpAffine(gray, M, (Wp, Hp), flags=cv2.INTER_CUBIC, borderValue=255)
    rm = cv2.warpAffine(m, M, (Wp, Hp), flags=cv2.INTER_NEAREST)
    rp = _flat(rp)
    line = cv2.morphologyEx(rm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(15, 0.6 * Z)), 1)))
    yc = Hp // 2
    if DIM_STRIP2:                                              # 점선의 짧은 획: 선 위(가운데 ±2px)에 놓인 납작한 같은 색 조각도 선으로 지움(글자 획은 키가 커서 남음)
        lw_ = dm.get('lw', 3)
        rmw = cv2.warpAffine(mw, M, (Wp, Hp), flags=cv2.INTER_NEAREST) if mw is not None else rm
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(rmw, 8)
        dash = np.zeros_like(line)
        for k_ in range(1, n_):
            x_, y_, w_, h_, a_ = st_[k_]
            if h_ <= lw_ + 4 and w_ >= max(3, 1.5 * h_) and abs(y_ + h_ / 2 - yc) <= 3: dash[lab_ == k_] = 1
        line |= cv2.dilate(dash, np.ones((3, 5), np.uint8))                 # 획의 번진 가장자리까지
    line = cv2.dilate(line, np.ones((3, 3), np.uint8))
    rc, _ = _clean_lines(rp, 12)
    rc[line > 0] = 255
    ink = (rc < 170).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) | (ink & cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), np.ones((3, 3), np.uint8)))
    xa, xb = int(max(0, Wp / 2 - L / 2 - 5)), int(min(Wp, Wp / 2 + L / 2 + 5))
    rows = ink[:, xa:xb].sum(1)
    out = []
    # 글자 띠: 선 위(-)·아래(+)로 나가며 먹이 있는 첫 줄 묶음(1줄 틈 허용), 가운데(0) = 선을 끊고 얹은 글자
    for side in (-1, 1, 0):
        if side == 0:
            if rows[yc] < 2: continue
            y0b = yc; y1b = yc
            while y0b > 0 and rows[y0b - 1] >= 1 and yc - y0b < 16: y0b -= 1
            while y1b < Hp - 1 and rows[y1b + 1] >= 1 and y1b - yc < 16: y1b += 1
        else:
            y = yc + side * 3
            for _try in range(3 if DIM_STRIP2 else 1):              # 선 바로 옆 얇은 찌꺼기 띠(< 6px)는 건너뛰고 다음 띠
                while 0 < y < Hp - 1 and abs(y - yc) < 45 and rows[y] < 2: y += side
                if not (0 < y < Hp - 1) or abs(y - yc) >= 45: break
                y0b = y1b = y; gap = 0
                while 0 < (y1b if side > 0 else y0b) < Hp - 1 and gap <= 1 and (y1b - y0b) < 34:
                    yy = (y1b + 1) if side > 0 else (y0b - 1)
                    if rows[yy] >= 1: gap = 0
                    else: gap += 1
                    if side > 0: y1b = yy
                    else: y0b = yy
                if side > 0: y1b -= gap
                else: y0b += gap
                if side < 0: y0b, y1b = min(y0b, y1b), max(y0b, y1b)
                if y1b - y0b + 1 >= 6: break
                y = (y1b + 1) if side > 0 else (y0b - 1)
            if not (0 < y < Hp - 1) or abs(y - yc) >= 45: continue
        bh = y1b - y0b + 1
        if not (6 <= bh <= 34): continue
        cols = ink[y0b:y1b + 1, :].sum(0) > 0
        c0 = Wp // 2
        cand = np.where(cols[xa:xb])[0] + xa
        if not len(cand): continue
        k = int(cand[np.argmin(np.abs(cand - c0))])
        if abs(k - c0) > 0.5 * L: continue
        lo = hi = k; g_ = int(0.9 * bh) + 1
        while lo > 0:
            nxt = [j for j in range(max(0, lo - g_), lo) if cols[j]]
            if not nxt: break
            lo = min(nxt)
        while hi < Wp - 1:
            nxt = [j for j in range(hi + 1, min(Wp, hi + 1 + g_)) if cols[j]]
            if not nxt: break
            hi = max(nxt)
        if hi - lo < 8: continue
        sc = abs((y0b + y1b) / 2 - yc) + 0.3 * abs((lo + hi) / 2 - c0)
        p = 4
        crop = rc[max(0, y0b - p):y1b + 1 + p, max(0, lo - p):hi + 1 + p]
        raw = rp[max(0, y0b - p):y1b + 1 + p, max(0, lo - p):hi + 1 + p]
        out.append((sc, crop, raw, abs(a) > 45, bh))
    out.sort(key=lambda o: o[0])
    return out[:2]


_DIM_WORDS = re.compile(r'[약악]|[미비마이메]\s*[터타더]|이격|거리|간격|치수')
_KUNIT = re.compile(r'\d\s*[미비마이메]\s*[터타더]')


def _hangul_label(t):
    """한글 이름표(소화전 번호 '가상-4'·도로 이름 '가상대로 12m')인지: 치수 글자에 흔한 낱말(약·미터·이격거리)을 빼고
    첫 숫자 앞에 한글이 2자 이상(숫자가 없으면 전체에서 2자 이상)이면 True. 숫자 뒤 한글('30미터'·오독 '30비터')은 단위로 봄."""
    u = _DIM_WORDS.sub('', t or '')
    m = re.search(r'\d', u)
    return len(re.findall(r'[가-힣]', u[:m.start()] if m else u)) >= 2


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


def _box_patches(gray, pa, h):
    """글자 상자(pa, 여백 포함) → (평탄화 조각, 평탄화 + 지나가는 긴 선 지운 조각). 선은 조금 넓게 잘라 찾은 뒤 다시 자름"""
    H, W = gray.shape; q = 16
    pb = [max(0, pa[0] - q), max(0, pa[1] - q), min(W, pa[2] + q), min(H, pa[3] + q)]
    fw = _flat(gray[pb[1]:pb[3], pb[0]:pb[2]])
    cw, _ = _clean_lines(fw, h)
    sl = (slice(pa[1] - pb[1], pa[3] - pb[1]), slice(pa[0] - pb[0], pa[2] - pb[0]))
    return fw[sl], cw[sl]


def _lead_glyph(v):
    """(이진화·확대한) 치수 글자 조각에서 숫자 앞에 붙은 '약'(한글)·'~' 글자 찾기 → (자를 x, 종류) 또는 None.
    x 범위가 겹치는 먹 조각을 한 글자로 묶고(한글 '약' = ㅇ·ㅑ·ㄱ 여러 조각), 나머지 글자 높이 중앙값 hr 기준으로
    글자 묶음(앞 틈 > 0.8·hr)의 첫 글자가 여러 조각 + 폭 0.75~1.6·hr + 높이 0.85~1.5·hr 이면 한글, 한 조각 + 높이 ≤ 0.5·hr + 폭 ≥ 0.5·hr + 가운데 높이면 '~'
    (바로 뒤 0.8·hr 안에 숫자 높이 글자가 있어야 함). → (자를 x 끝, 종류, 지울 x 시작).
    숫자 글자는 한 조각이라 걸리지 않음(숫자+단위 화이트리스트 판독은 '약'을 숫자('2')로 읽어 '약 13m' → '213m')."""
    ink = (v < 128).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [st[k] for k in range(1, n) if st[k][4] >= 6]
    if len(comps) < 3: return None
    hmax = max(c[3] for c in comps)
    tall = [c for c in comps if c[3] >= 0.6 * hmax and c[2] <= 2 * c[3]]
    if not tall: return None
    bt, bb = float(np.median([c[1] for c in tall])), float(np.median([c[1] + c[3] for c in tall]))       # 글자 줄 띠(위·아래 잡티·선 조각 제외)
    comps = sorted((c for c in comps if c[4] >= 0.004 * hmax * hmax and c[2] <= 2.5 * (bb - bt)
                    and min(bb, c[1] + c[3]) - max(bt, c[1]) >= 0.5 * c[3]), key=lambda c: c[0])
    cl = []
    for c in comps:
        if cl and min(cl[-1][1], c[0] + c[2]) - c[0] >= 0.5 * min(c[2], cl[-1][1] - cl[-1][0]):      # x 가 절반 이상 겹침(기울어진 글자의 이웃 숫자는 조금만 겹침)
            g = cl[-1]; g[1] = max(g[1], c[0] + c[2]); g[2] = min(g[2], c[1]); g[3] = max(g[3], c[1] + c[3]); g[4] += 1
        else: cl.append([c[0], c[0] + c[2], c[1], c[1] + c[3], 1])
    if len(cl) < 3: return None
    hs = [g[3] - g[2] for g in cl]
    tall_ = [g for g in cl if g[3] - g[2] >= 0.6 * max(hs)]
    hr = float(np.median([g[3] - g[2] for g in tall_]))
    yt, yb = float(np.median([g[2] for g in tall_])), float(np.median([g[3] for g in tall_]))
    for i in range(len(cl) - 2):                        # 글자 묶음(앞 틈 > 0.8·hr)의 첫 글자마다 — 왼쪽 잡티(점선 조각·기호) 뒤의 '약' 도 찾게
        g, nx_ = cl[i], cl[i + 1]
        if i > 0 and g[0] - cl[i - 1][1] <= 0.8 * hr: continue
        if nx_[0] - g[1] > 0.8 * hr or nx_[3] - nx_[2] < 0.7 * hr: continue          # 바로 뒤(띄어쓰기 한 칸 안)에 숫자 높이 글자가 와야 함
        w, h = g[1] - g[0], g[3] - g[2]; cy = (g[2] + g[3]) / 2
        if g[4] >= 2 and 0.75 * hr <= w <= 1.6 * hr and 0.85 * hr <= h <= 1.5 * hr: kind = 'hangul'
        elif DIM_LEAD1 and g[4] == 1 and 0.85 * hr <= w <= 1.6 * hr and 1.1 * hr <= h <= 1.5 * hr and w >= 1.4 * (nx_[1] - nx_[0]): kind = 'hangul'   # 번져서 한 조각이 된 '약'(숫자보다 키 크고 넓음)
        elif g[4] == 1 and h <= 0.5 * hr and w >= 0.5 * hr and yt + 0.2 * hr <= cy <= yb - 0.2 * hr: kind = 'tilde'
        else: continue
        return int((g[1] + nx_[0]) / 2), kind, max(0, g[0] - 1)
    return None


DIM_LEAD = os.environ.get('SITEMAP_DIMLEAD', '1') != '0'      # '약'·'~' 앞 글자 지우고 숫자 판독(0 = 끔)
DIM_LEAD1 = os.environ.get('SITEMAP_DIMLEAD1', '1' if DIMSET_A else '0') != '0'    # 번져서 한 조각이 된 '약'도(기본 끔, 1 = 켬)
DIM_LEADOCR = os.environ.get('SITEMAP_DIMLEADOCR', '1' if DIMSET_A else '0') != '0'  # 모양만으로 애매한 첫 글자(깨지거나 번진 '약')는 한 글자 한글 OCR 로 가림(기본 끔, 1 = 켬)


def _lead_cand(v):
    """_lead_glyph 가 못 정한 첫 글자 후보(깨진 '약' = 묶음 둘로 갈라짐, 번진 '약' = 한 조각): 글자 묶음 시작의 첫 묶음(폭 ≥ 0.75·hr) 또는
    좁은 첫 묶음 + 바로 붙은 다음 묶음(합친 폭 0.75~1.6·hr, 높이 ≥ 0.85·hr) 뒤에 숫자 높이 글자가 둘 이상 오면 → (자를 x, 지울 x 시작, 글자 x0, x1).
    숫자 둘이 붙은 것('31')도 후보가 되므로 한 글자 한글 OCR(kor, psm 10)로 한글일 때만 씀."""
    ink = (v < 128).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [st[k] for k in range(1, n) if st[k][4] >= 6]
    if len(comps) < 3: return None
    hmax = max(c[3] for c in comps)
    tall = [c for c in comps if c[3] >= 0.6 * hmax and c[2] <= 2 * c[3]]
    if not tall: return None
    bt, bb = float(np.median([c[1] for c in tall])), float(np.median([c[1] + c[3] for c in tall]))
    comps = sorted((c for c in comps if c[4] >= 0.004 * hmax * hmax and c[2] <= 2.5 * (bb - bt)
                    and min(bb, c[1] + c[3]) - max(bt, c[1]) >= 0.5 * c[3]), key=lambda c: c[0])
    cl = []
    for c in comps:
        if cl and min(cl[-1][1], c[0] + c[2]) - c[0] >= 0.5 * min(c[2], cl[-1][1] - cl[-1][0]):
            g = cl[-1]; g[1] = max(g[1], c[0] + c[2]); g[2] = min(g[2], c[1]); g[3] = max(g[3], c[1] + c[3]); g[4] += 1
        else: cl.append([c[0], c[0] + c[2], c[1], c[1] + c[3], 1])
    if len(cl) < 3: return None
    hs = [g[3] - g[2] for g in cl]
    hr = float(np.median([h for h in hs if h >= 0.6 * max(hs)]))
    for i in range(len(cl) - 2):
        if i > 0 and cl[i][0] - cl[i - 1][1] <= 0.8 * hr: continue
        g = list(cl[i]); k = i + 1
        if g[1] - g[0] < 0.45 * hr: continue                     # 첫 묶음이 '1' 처럼 좁으면 숫자('18' 을 한글로 오독하던 경우)
        if g[1] - g[0] < 0.75 * hr and cl[k][0] - g[1] <= 0.3 * hr and max(g[1], cl[k][1]) - g[0] <= 1.6 * hr:
            g = [g[0], max(g[1], cl[k][1]), min(g[2], cl[k][2]), max(g[3], cl[k][3]), g[4] + cl[k][4]]; k += 1
        if k + 1 >= len(cl): continue
        w, h = g[1] - g[0], g[3] - g[2]; nx_ = cl[k]
        if not (0.75 * hr <= w <= 1.6 * hr and h >= 0.85 * hr): continue
        if nx_[0] - g[1] > 0.8 * hr or nx_[3] - nx_[2] < 0.7 * hr or cl[k + 1][3] - cl[k + 1][2] < 0.5 * hr: continue
        return int((g[1] + nx_[0]) / 2), max(0, g[0] - 1), g[0], g[1]
    return None


def _dim_jobs(i, j, b):
    vs = [('c4', _prep(b['clean'], 4)), ('c3', _prep(b['clean'], 3)), ('r4', _prep(b['raw'], 4)), ('r3', _prep(b['raw'], 3))]
    if b.get('wide') is not None: vs += [('w4', _prep(b['wide'], 4))]
    if b['flip']: vs += [(k + 'f', cv2.rotate(v, cv2.ROTATE_180)) for k, v in vs[:2]]
    out = []
    for k, v in vs:
        lg = _lead_glyph(v) if DIM_LEAD and not k.startswith('w') else None      # 옆으로 넓힌 조각(w)은 상자 밖 이웃 글자가 앞에 올 수 있어 제외
        ke, ve = k, v
        if lg:                                                   # 영문 화이트리스트 판독은 앞 글자 지운 조각으로(판독 앞에 '~' 붙여 '약' 표시), 한글 판독은 원본
            ve = v.copy(); ve[:, lg[2]:lg[0]] = 255; ke = k + '~'
        elif DIM_LEAD and DIM_LEADOCR and not k.startswith('w'):
            lc = _lead_cand(v)
            if lc:                                               # 애매한 첫 글자: 지운 조각도 함께 읽고, 한 글자 한글 OCR 이 한글이라 하면 그쪽을 씀(_run_jobs)
                ve2 = v.copy(); ve2[:, lc[1]:lc[0]] = 255
                gl = cv2.copyMakeBorder(v[:, max(0, lc[2] - 2):lc[3] + 2], 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                out += [(i, j, k + '?', ve2, 'e7'), (i, j, k + '?', gl, 'k10')]
                if k in ('c4', 'c4f'): out += [(i, j, k + '?', ve2, 'e8')]
        out.append((i, j, ke, ve, 'e7'))
        if k in ('c4', 'c4f'): out += [(i, j, ke, ve, 'e8'), (i, j, k, v, 'k7')]
    return out


def _run_jobs(jobs, res, kor, cfs=None):
    # 애매한 첫 글자 후보(k+'?'): 한 글자 한글 OCR 로 한글이면 지운 조각 판독('~' 붙임)을, 아니면 원래 판독을 씀
    kj = [jb for jb in jobs if jb[4] == 'k10']
    hang = set()
    if kj:
        tt, cc = _ocr_batch([jb[3] for jb in kj], 'kor', 10, None, conf=True)
        for jb, t, c in zip(kj, tt, cc):
            if re.search(r'[가-힣]', t or '') and not re.search(r'\d', t or ''): hang.add((jb[0], jb[1], jb[2][:-1]))
    for cfg, (lang, psm, wl) in dict(e7=('eng', 7, DIM_WL), e8=('eng', 8, DIM_WL), k7=('kor+eng', 7, None)).items():
        js = [jb for jb in jobs if jb[4] == cfg]
        if not js: continue
        if DIM_CONF_W: tt, cc = _ocr_batch([jb[3] for jb in js], lang, psm, wl, conf=True)
        else: tt = _ocr_batch([jb[3] for jb in js], lang, psm, wl); cc = [-1.0] * len(tt)
        for jb, t, c in zip(js, tt, cc):
            if jb[2].endswith('?'):
                if (jb[0], jb[1], jb[2][:-1]) not in hang: continue
                if re.search(r'\d', t or ''): t = '~' + t
            elif cfg != 'k7' and (jb[0], jb[1], jb[2]) in hang: continue           # 한글 첫 글자로 확인됨 → 원래(지우지 않은) 영문 판독은 버림
            if jb[2].endswith('~') and re.search(r'\d', t or ''): t = '~' + t
            res.setdefault((jb[0], jb[1]), []).append(t)
            if cfs is not None: cfs.setdefault((jb[0], jb[1]), []).append(c)
            if cfg == 'k7': kor.setdefault((jb[0], jb[1]), []).append(t)


DIM_CONF_W = float(os.environ.get('SITEMAP_DIMCONF', '2.0' if DIMSET_A else '0'))     # 상자 고르기: 맨 위 값을 낸 판독들의 OCR 신뢰도(0~1) 가중(기본 0 = 예전, A 묶음 2.0) — 0 이 아니면 tsv 신뢰도·다른 상자 맨 위 값도 씀


def _top_conf(txts, cfs, v):
    """판독들 중 값 v 를 낸 것들의 tesseract 신뢰도 평균(0~1). 신뢰도 없으면 None"""
    c = [cf for t, cf in zip(txts, cfs) if cf > 1 and any(abs(x[0] - v) < 0.01 for x in parse_dim_vals(t))]    # 화이트리스트 판독은 맞게 읽어도 0 이 자주 나옴 → 0 은 '모름'
    return float(np.percentile(c, 75)) / 100 if c else None                 # 가끔 0 으로 나오는 신뢰도(같은 글자인데)에 덜 흔들리게


def _pick_box(bl, res, kor, i, cfs=None):
    """상자마다(겹치는 상자 판독은 합쳐서) 투표 → (점수, 상자 번호, 표, 판독들, 맨 위 표, 신뢰도).
    점수 = 맨 위 값 표 비율·7 + 단위 0.5 − 0.02·자리 점수 + DIM_CONF_W·(맨 위 값 판독 신뢰도 − 0.5) — 기울어진 글자를 가로 상자로 잘라
    일관되게 잘못 읽은 판독(낮은 신뢰도)이 선 방향 띠의 판독(높은 신뢰도)을 이기지 않게."""
    ok = [j for j in range(len(bl)) if (i, j) in res and not any(_hangul_label(t) for t in kor.get((i, j), []))]
    best = None
    for j in ok:
        txts = list(res[(i, j)]); cf = list((cfs or {}).get((i, j), [-1.0] * len(txts)))
        if bl[j]['box']:
            for j2 in ok:
                if j2 != j and bl[j2]['box'] and _iou(bl[j]['box'], bl[j2]['box']) >= 0.3: txts += res[(i, j2)]; cf += list((cfs or {}).get((i, j2), [-1.0] * len(res[(i, j2)])))
        kt = [t or '' for jj in ([j] + [j2 for j2 in ok if j2 != j and bl[j]['box'] and bl[j2]['box'] and _iou(bl[j]['box'], bl[j2]['box']) >= 0.3]) for t in kor.get((i, jj), [])]
        kh = any(_KUNIT.search(t) for t in kt)
        V = _vote_texts(txts, kh, [k for t in kt for k in _KNUM.findall(t)])
        top = max(e[0] for e in V.values()) if V else 0
        unit = any(e[1] for e in V.values())
        sc = top / max(1, len(txts)) * 7 + 0.5 * unit - 0.02 * bl[j]['sc']
        tc = None
        if V and DIM_CONF_W and cfs is not None:
            tv = max(V, key=lambda k: V[k][0]); tc = _top_conf(txts, cf, tv)
            if tc is not None: sc += DIM_CONF_W * (tc - 0.5)
        if DIM_CONF_W and cfs is not None and not V: sc -= 100                     # 값을 하나도 못 읽은 상자는 값 있는 상자를 이기지 않음
        if DIM_CONF_W and V and not unit and len(f'{max(V, key=lambda k: V[k][0]):g}') == 1: sc -= 3.0   # 단위 없는 한 글자('3') = 돌아간 글자의 한 조각일 가능성 — 여러 글자 판독 상자에 밀리게
        if best is None or sc > best[0]: best = (sc, j, V, txts, top, tc)
    return best, ok


def _slanted(dm, deg=4.0):
    a = abs(math.degrees(math.atan2(dm['p1'][1] - dm['p0'][1], dm['p1'][0] - dm['p0'][0]))) % 180
    return min(a, 180 - a) > deg


def read_dim_texts(img, m, dims, lines, Z, mw=None):
    """치수선마다 글자 후보 상자(글자 줄·가운데 둘레 덩어리) → 여러 번 판독(tesseract 호출은 설정별로 한 번에 모아서)
    → 겹치는 상자끼리 판독을 합쳐 투표 → d['cands'], d['text'], d['tbox'], d['n_read'].
    뚜렷한 값이 없으면 선 방향으로 돌린 띠(기울어진·세로 글자)를 더 읽는다."""
    gray = img.min(axis=2); H, W = gray.shape
    jobs, boxes = [], {}
    for i, dm in enumerate(dims):
        bl = []
        for sc, bx, t0 in _text_boxes(dm, lines, Z):
            h = bx[3] - bx[1]; p = 4
            pa = [max(0, bx[0] - p), max(0, bx[1] - p), min(W, bx[2] + p), min(H, bx[3] + p)]
            raw, cl = _box_patches(gray, pa, h)
            pw = [max(0, bx[0] - int(0.8 * h)), pa[1], min(W, bx[2] + int(0.8 * h)), pa[3]]           # 옆으로 넓힌 조각(끊긴 첫·끝 글자)
            bl.append(dict(sc=sc, box=[int(v) for v in bx], raw=raw, clean=cl, wide=_box_patches(gray, pw, h)[1], flip=False, src='line', h=h))
        for sc, bx in _blob_boxes(gray, m, dm, Z):
            if any(b['box'] and _iou(bx, b['box']) >= 0.5 and bx[0] >= b['box'][0] - 3 and bx[2] <= b['box'][2] + 3 for b in bl): continue      # 이미 있는 상자 안(넓게 잡힌 새 상자는 남김)
            if any(_hangul_label(ln.get('text', '')) and min(bx[2], ln['x1']) - max(bx[0], ln['x0']) > 0.5 * (bx[2] - bx[0]) and min(bx[3], ln['y1']) - max(bx[1], ln['y0']) > 0.5 * (bx[3] - bx[1]) for ln in lines): continue
            h = bx[3] - bx[1]; p = 4
            pa = [max(0, bx[0] - p), max(0, bx[1] - p), min(W, bx[2] + p), min(H, bx[3] + p)]
            raw, cl = _box_patches(gray, pa, h)
            pw = [max(0, bx[0] - int(0.8 * h)), pa[1], min(W, bx[2] + int(0.8 * h)), pa[3]]
            bl.append(dict(sc=sc + 2, box=[int(v) for v in bx], raw=raw, clean=cl, wide=_box_patches(gray, pw, h)[1], flip=False, src='blob', h=h))
        if DIM_STRIP2 and _slanted(dm):                          # 기울어진·세로 선: 선 방향 띠를 처음부터 함께 읽어 경쟁(가로 상자가 기울어진 글자를 잘라 읽는 경우)
            for sc, cl, raw, flip, h in _strip_patches(gray, m, dm, Z, mw):
                bl.append(dict(sc=sc + 5, box=None, raw=raw, clean=cl, flip=flip, src='strip', h=h))
            dm['_strip1'] = True
        boxes[i] = bl
        for j, b in enumerate(bl): jobs += _dim_jobs(i, j, b)
    res, kor, cfs = {}, {}, {}
    _run_jobs(jobs, res, kor, cfs)
    # 2차: 뚜렷한 값(2표 이상)이 없는 치수는 선 방향 띠도 읽음
    jobs = []
    for i, dm in enumerate(dims):
        if dm.pop('_strip1', False): continue
        best, _ = _pick_box(boxes[i], res, kor, i, cfs)
        if best is not None and best[4] >= 2: continue
        for sc, cl, raw, flip, h in _strip_patches(gray, m, dm, Z, mw):
            boxes[i].append(dict(sc=sc + 5, box=None, raw=raw, clean=cl, flip=flip, src='strip', h=h))
            jobs += _dim_jobs(i, len(boxes[i]) - 1, boxes[i][-1])
    if jobs: _run_jobs(jobs, res, kor, cfs)
    if DIMDBG is not None:
        for i, dm in enumerate(dims):
            for j, b in enumerate(boxes[i]): DIMDBG.append(dict(i=i, p0=dm['p0'], p1=dm['p1'], ext=dm.get('ext_px'), src=b['src'], sc=round(b['sc'], 1), box=b['box'], txt=res.get((i, j), []), cf=[round(c) for c in cfs.get((i, j), [])], clean=b['clean'], raw=b['raw']))
    for i, dm in enumerate(dims):
        dm['cands'] = []; dm['n_read'] = 0; dm['text'] = None; dm['tbox'] = None
        best, ok = _pick_box(boxes[i], res, kor, i, cfs)
        if best is None or not best[2]: continue
        _, j, V, txts, _, tconf = best
        dm['tconf'] = None if tconf is None else round(tconf, 2)
        # 다른 상자의 값은 약한 표로(기하 길이와 맞을 때만 쓰임)
        bt = []
        for j2 in ok:
            if j2 == j: continue
            V2 = _vote_texts(res.get((i, j2), []))
            for k, e in V2.items():
                if k not in V: V[k] = [0.0, e[1], e[2], 0.25 * (e[0] + e[3])]
            if V2 and DIM_CONF_W:                              # 다른 상자(선 방향 띠 등)의 맨 위 값: 자기 상자 안에서 뚜렷하고 신뢰도가 높으면 기하 길이와 겨룰 후보(resolve_dim)
                k2 = max(V2, key=lambda k: V2[k][0]); n2 = max(1, len(res[(i, j2)]))
                c2 = _top_conf(res[(i, j2)], cfs.get((i, j2), []), k2)
                if V2[k2][0] / n2 >= 0.3 and c2 is not None: bt.append([k2, round(V2[k2][0], 2), n2, round(c2, 2), V2[k2][1], boxes[i][j2]['src']])
        dm['box_tops'] = bt
        dm['cands'] = sorted([[k, round(e[0], 2), e[1], e[2], round(e[3], 2)] for k, e in V.items()], key=lambda c: (-c[1], -c[4]))
        dm['n_read'] = len(txts)
        top = dm['cands'][0][0]
        dm['text'] = next((t for t in txts if any(abs(v - top) < 0.01 for v, _, _ in parse_dim_vals(t))), txts[0] if txts else None)
        dm['tbox'] = boxes[i][j]['box']
        dm['tsrc'] = boxes[i][j]['src']
        allt = [t for jj in ok for t in res.get((i, jj), [])]
        dm['sep'] = any(re.search(r'\d\s*[.,]\s*\d', t or '') for t in allt)                   # 어느 판독에든 소수 구분자가 보였음
        dm['lead'] = any((t or '').startswith('~') or re.match(r'\s*[약악~]', t or '') for t in allt + [t for jj in ok for t in kor.get((i, jj), [])])   # 숫자 앞 '약'·'~' 보였음
    return dims


def _alt_parses(v, lead, dec):
    """정수 판독 v 의 다른 해석: dec → 마지막 자리 앞 소수점('420'→42.0, '125'→12.5), lead → 첫 자리 뺌('213'→13). 0.5~DIM_MAX_M 안만."""
    s = f'{v:g}'
    if '.' in s or len(s) < 2: return []
    out = []
    if dec: out.append(float(s[:-1] + '.' + s[-1]))
    if lead: out.append(float(s[1:]))
    return [a for a in out if 0.5 <= a <= DIM_MAX_M and a != v]


DIM_RES2 = os.environ.get('SITEMAP_DIMRES2', '0') != '0'      # 값 결정 보강(약한 맨 위 값·반 표 후보·다른 해석 25%·high 기준) — 기본 끔(기본 형식 dev_x 에서 맞음 −1), 1 = 켬


def resolve_dim(dm, s, Z=24.0):
    """후보 값·기하 길이·축척(px/m) → val, val_src(ocr | ocr+geom | geom), val_conf, certainty(high|medium|low|estimated), geom_m, geom_mismatch.
    기하 길이 g = geom_px / s, 허용 = max(1.5 m, 10%) + 끝 기호 보정분. 순서:
      1) 가장 많이 읽힌 값이 g 와 맞음 → ocr
      2) g 와 맞는 다른 직접 판독 값 → ocr+geom  (2' 뚜렷한 판독의 다른 해석, 2'' 맨 위가 뚜렷하지 않을 때 다른 상자의 뚜렷·고신뢰(≥0.6) 판독)
      3) 여러 판독이 한 값을 공통으로 품음('324m'·'424m'·'24m' → 24)인데 g 와 안 맞음 → 그 값 + geom_mismatch(글자 ≠ 실측)
      4) 판독이 뚜렷(절반 이상 같은 값)한데 g 와 안 맞음 → ocr + geom_mismatch  (작전도는 축척대로 그리지 않은 경우가 많아 뚜렷한 글자를 기하로 덮어쓰지 않음)
      5) g 와 맞는 약한 후보(한 글자 뺌·소수점 빠짐) → ocr+geom
      6) 못 읽음 → 기하 길이(geom, 'estimated')
    축척이 없으면(g 없음) 가장 많이 읽힌 값."""
    cands = dm.get('cands') or []
    n = max(1, dm.get('n_read') or 1)
    g = dm['geom_px'] / s if s else None
    tol = (max(DIM_TOL_ABS, DIM_TOL_REL * g) + 0.35 * sum(dm.get('ext_px', [0, 0])) / s) if g else None
    ok = (lambda v: abs(v - g) <= tol) if g else (lambda v: True)
    tol0 = max(DIM_TOL_ABS, DIM_TOL_REL * g) if g else None          # 기호 보정분 뺀 허용(반 표짜리 후보용)
    direct = [c for c in cands if c[1] >= 0.5]
    top = direct[0] if direct else None
    # 포함 지지: 한 글자 더 붙은 다른 판독('324' ⊃ '24')의 표를 더함
    fs = lambda v: f'{v:g}'
    cont = {}
    for c in direct:
        cs = fs(c[0]); sup = [o for o in direct if o is not c and len(fs(o[0])) == len(cs) + 1 and (fs(o[0]).endswith(cs) or fs(o[0]).startswith(cs))]
        cont[c[0]] = (c[1] + sum(o[1] for o in sup), len(sup))
    for c in direct:                                             # 품은 긴 판독보다 자기 표가 너무 적으면('48'·'18' ⊃ '8') 포함 지지 아님
        cs = fs(c[0]); sup = [o for o in direct if o is not c and len(fs(o[0])) == len(cs) + 1 and (fs(o[0]).endswith(cs) or fs(o[0]).startswith(cs))]
        if sup and c[1] < 0.5 * max(o[1] for o in sup): cont[c[0]] = (c[1], 0)
    ctop = max(direct, key=lambda c: (cont[c[0]][0], c[1])) if direct else None
    if ctop is not None and cont[ctop[0]][1] < 2 and top is not None and top[1] > ctop[1]: ctop = None      # 긴 판독 하나만 품는 짧은 값('37' ⊃ '7')은 포함 지지로 보지 않음
    dm.pop('geom_mismatch', None)
    if g is not None: dm['geom_m'] = round(g, 2)
    # 같은 숫자열의 다른 해석(근거 있을 때만): 소수 구분자 빠짐('42,0m'→'420') · 숫자 앞 '약'/'~' 를 숫자로 읽음('약 13m'→'213').
    # 근거 = 300 m 넘는 값(이격거리로 그럴듯하지 않음) 또는 판독에 구분자/앞 글자가 보였음. 해석이 여럿이면 기하 길이에 가까운 것.
    big = bool(top and top[0] > DIM_MAX_M)
    alts = _alt_parses(top[0], big or dm.get('lead'), big or dm.get('sep')) if top else []
    # 뚜렷한 판독: 절반 이상 같은 값(또는 여러 판독이 공통으로 품은 값) — 기하 길이로 덮어쓰지 않음
    dom = bool((top and top[1] >= max(3, 0.5 * n)) or (ctop and cont[ctop[0]][1] >= 1 and cont[ctop[0]][0] >= max(3, 0.5 * n)))
    pick = None
    weak_top = [c for c in cands if ok(c[0]) and (c[1] >= 0.5 or c[4] >= 0.25)] if (DIM_RES2 and g is not None and top and top[1] < 2 and ok(top[0])) else []
    if weak_top and len(weak_top) > 1:                                           # 맨 위도 한 번만 읽힌 약한 값: 기하 길이와 맞는 후보들 중 표·가까움으로
        c = max(weak_top, key=lambda c: c[1] + 0.5 * c[4] - 0.8 * abs(c[0] - g) / tol)
        pick = (c[0], 'ocr' if c is top else 'ocr+geom', False, 'medium')
    elif top and ok(top[0]):
        conf = top[1] >= 2
        hi_ = g is None or not DIM_RES2 or abs(top[0] - g) <= tol0                                       # 'high' 는 기호 보정분 없이도 맞을 때만
        pick = (top[0], 'ocr', conf, ('high' if (g is not None and hi_) else 'medium') if conf else 'medium')
    elif g is not None and any(ok(c[0]) and (not DIM_RES2 or c[1] >= 1 or abs(c[0] - g) <= 0.5 * tol0) for c in direct):     # 반 표(단위 없는 한 번)짜리는 기하 길이에 더 가까워야
        c = max((c for c in direct if ok(c[0]) and (not DIM_RES2 or c[1] >= 1 or abs(c[0] - g) <= 0.5 * tol0)), key=lambda c: c[1] - 0.5 * abs(c[0] - g) / tol)
        pick = (c[0], 'ocr+geom', c[1] >= 1, 'medium')
    elif g is not None and any(ok(a) for a in alts):                            # 뚜렷한 판독의 다른 해석이 기하 길이와 맞음 → 그 해석(글자 숫자열은 그대로)
        pick = (min((a for a in alts if ok(a)), key=lambda a: abs(a - g)), 'ocr+geom', top[1] >= 1, 'medium')
    elif (g is not None and not dom and
          any(ok(b[0]) and b[3] >= 0.6 and (b[1] >= 1 or abs(b[0] - g) <= 0.5 * tol0) for b in dm.get('box_tops') or [])):   # 다른 상자(선 방향 띠 등)의 뚜렷·고신뢰 판독이 기하 길이와 맞음 — 맨 위 판독이 뚜렷하지 않을 때만 겨룸
        b = max((b for b in dm['box_tops'] if ok(b[0]) and b[3] >= 0.6 and (b[1] >= 1 or abs(b[0] - g) <= 0.5 * tol0)), key=lambda b: b[1] / b[2] + b[3])
        pick = (b[0], 'ocr+geom', b[1] >= 1, 'medium')
    elif ctop and cont[ctop[0]][1] >= 1 and cont[ctop[0]][0] >= max(3, 0.5 * n):
        pick = (ctop[0], 'ocr', True, 'low'); dm['geom_mismatch'] = True
    elif top and top[1] >= max(3, 0.5 * n):
        v = top[0]
        if alts and g is not None and not DIM_RES2: v = min(([v] if v <= DIM_MAX_M else []) + alts, key=lambda a: abs(a - g))
        elif alts and g is not None:                                            # 해석이 여럿이면 기하 길이에 가까운 것(불일치는 그대로 표시) —
            near = [a for a in alts if abs(a - g) <= max(tol, 0.25 * g)]       # 글자 그대로가 그럴듯한 값(≤ 300 m)이면 기하 길이에 꽤 가까운(25%) 해석만 글자를 대신함
            if v > DIM_MAX_M: v = min(near or alts, key=lambda a: abs(a - g))
            elif near and abs(v - g) > max(tol, 0.25 * g): v = min(near, key=lambda a: abs(a - g))
        elif big and alts: v = alts[0]
        pick = (v, 'ocr', True, 'low'); dm['geom_mismatch'] = g is not None
    elif g is not None and any(ok(c[0]) and c[4] >= 0.25 for c in cands):
        c = max((c for c in cands if ok(c[0]) and c[4] >= 0.25), key=lambda c: c[1] + c[4] - 0.5 * abs(c[0] - g) / tol)
        pick = (c[0], 'ocr+geom', False, 'medium')
    elif g is not None: pick = (round(g * 2) / 2, 'geom', False, 'estimated')
    elif top: pick = (top[0], 'ocr', False, 'low')
    else: pick = (None, None, False, None)
    val, src, conf, cert = pick
    if dm.get('geom_mismatch') and not all(dm.get('ends', [True, True])):
        dm['geom_mismatch'] = False; cert = 'medium'                            # 선 끝이 확실치 않음(화살촉·틱·기호 없음) → 실측 비교 안 함
    if not dm.get('geom_mismatch'): dm.pop('geom_mismatch', None)
    ap = (next((c[3] for c in cands if c[0] == val), False) or bool(dm.get('lead'))) if src != 'geom' else False
    dm.update(val=val, val_src=src, val_conf=bool(conf), certainty=cert, approx=bool(ap))
    return dm


def scale_consensus(rec, skip=None):
    """recognize 안의 축척(px/m) 합의: 축척 막대(3) · 도로 폭 글자(0.5, 폭 25px 이상) · 판독이 뚜렷한 다른 치수(1).
    7% 안에 모인 가중치가 가장 큰 무리의 가중 중앙값. 치수만으로 된 무리는 둘 이상일 때만."""
    C = []
    if rec.get('scalebar'): C.append((rec['scalebar']['px_per_m'], 3.0, 'scalebar'))
    for r in rec.get('roads', []):
        if r.get('label_m') and r['w_px'] >= 25: C.append((r['w_px'] / r['label_m'], 0.5, 'road'))
    for k, d in enumerate(rec.get('dims', [])):
        if k == skip or not d.get('cands'): continue
        c = d['cands'][0]
        if c[1] >= 3 and c[2] and c[0] > 0.5: C.append((d['geom_px'] / c[0], 1.0, 'dim'))
    if not C: return None, None
    best = None
    for v, _, _ in C:
        grp = [c for c in C if abs(c[0] / v - 1) <= 0.07]
        w = sum(c[1] for c in grp)
        if all(c[2] == 'dim' for c in grp) and len(grp) < 2: continue
        if all(c[2] == 'road' for c in grp) and len(grp) < 2: continue
        if best is None or w > best[0]: best = (w, grp)
    if best is None: return None, None
    grp = sorted(best[1]); ws = np.cumsum([c[1] for c in grp])
    s = grp[int(np.searchsorted(ws, ws[-1] / 2))][0]
    return float(s), '+'.join(sorted(set(c[2] for c in grp)))


DIM_TICK = os.environ.get('SITEMAP_DIMTICK', '0') != '0'      # 틱(빗금) 끝은 빗금 가운데를 끝점으로 — 기본 끔(기본 형식 dev_x·dev_y 에서 틀림 +1씩), 1 = 켬
DIM_LINK = os.environ.get('SITEMAP_DIMLINK', '1' if DIMSET_A else '0') != '0'      # 같은 직선 위 끊긴 조각 잇기·가린 끝을 기호까지 늘리기·버릴 선은 글자 읽기 전에(기본 끔, 1 = 켬)
DIM_WEAK = os.environ.get('SITEMAP_DIMWEAK', '1' if DIMSET_A else '0') != '0'      # 약한 색상 마스크로 옅은 점선 따라가기(기본 끔, 1 = 켬)


def _gap_ev(p, u, t0, t1, mw, occ, road=None, box=None):
    """선(p + u·t) 위 t0..t1 구간(틈)의 근거 비율 → (선 색, 선 색 또는 가림(검은 먹·글자 상자), 회색 도로).
    선 색 = 약한 색상 마스크가 법선 ±2px 안, 가림 = 검은 먹·글자 상자가 법선 ±3px 안.
    box(글자 상자 마스크)를 주면: 틈 대부분(> 50%)을 검은 먹이 선 따라 덮으면 나란히 지나는 다른 선(도로 가장자리 등)으로 보고 검은 먹은 근거에서 뺌."""
    H, W = mw.shape; nx, ny = -u[1], u[0]
    S = []
    for t in np.arange(t0, t1, 1.0):
        x, y = p[0] + u[0] * t, p[1] + u[1] * t
        hl = ho = hb = hr = hc = False
        for o in range(-3, 4):
            xx, yy = int(round(x + nx * o)), int(round(y + ny * o))
            if not (0 <= yy < H and 0 <= xx < W): continue
            if abs(o) <= 2 and mw[yy, xx]: hl = True
            if occ[yy, xx]:
                ho = True
                if abs(o) <= 1: hc = True
            if box is not None and box[yy, xx]: hb = True
            if road is not None and o == 0 and road[yy, xx]: hr = True
        S.append((hl, ho, hb, hr, hc))
    n = max(1, len(S))
    if box is not None and sum(s[4] and not s[2] for s in S) > 0.5 * n:            # 선 따라 나란히 지나는 검은 선
        return sum(s[0] for s in S) / n, sum(s[0] or s[2] for s in S) / n, sum(s[3] for s in S) / n
    return sum(s[0] for s in S) / n, sum(s[0] or s[1] for s in S) / n, sum(s[3] for s in S) / n


def _dedup(out, ang_deg):
    """같은 직선 위에 겹친 선 → 긴 것만(두 끝이 긴 선에서 6px 안, 또는 절반 넘게 긴 선 ±5px 띠 안)"""
    out.sort(key=lambda d: -d['len_px'])
    keep = []
    for d in out:
        a = math.atan2(d['p1'][1] - d['p0'][1], d['p1'][0] - d['p0'][0])
        dup = False
        for k in keep:
            b = math.atan2(k['p1'][1] - k['p0'][1], k['p1'][0] - k['p0'][0])
            if abs(((a - b) + math.pi / 2) % math.pi - math.pi / 2) > math.radians(ang_deg): continue
            Lk = LineString([k['p0'], k['p1']])
            if Lk.distance(Point(d['p0'])) < 6 and Lk.distance(Point(d['p1'])) < 6: dup = True; break
            ov = LineString([d['p0'], d['p1']]).intersection(Lk.buffer(5)).length
            if ov > 0.5 * d['len_px']: dup = True; break
        if not dup: keep.append(d)
    return keep


def _link_dims(out, mw, occ, pts, Z, road=None, box=None):
    """같은 직선 위로 끊긴 치수선 조각 잇기(점선이 도로를 지나며 옅어짐·선을 끊고 얹은 글자 상자·흰 바탕 이름표).
    조건: 방향 차 ≤ 3°, 법선 거리 ≤ max(3px, 0.15Z), 틈 ≤ 4.5Z, 틈 안에 점 기호 없음(소화전→송수구→소화전 일직선은 기호에서 나뉨),
    마주 보는 두 끝이 모두 장식(화살촉·틱)이고 틈 < Z 면 이어진 다른 치수(잇지 않음).
    틈 근거: 선 색 ≥ 25% 또는 (선 색·가림) ≥ 55% 또는 (선 색 ≥ 8% 이고 선 색·가림·회색 도로 ≥ 70%)."""
    def ang(d): return math.atan2(d['p1'][1] - d['p0'][1], d['p1'][0] - d['p0'][0])
    changed = True
    while changed:
        changed = False
        out.sort(key=lambda d: -d['len_px'])
        for i, a in enumerate(out):
            La = a['len_px']; u = ((a['p1'][0] - a['p0'][0]) / La, (a['p1'][1] - a['p0'][1]) / La)
            for j in range(i + 1, len(out)):
                b = out[j]
                if abs(((ang(a) - ang(b)) + math.pi / 2) % math.pi - math.pi / 2) > math.radians(3): continue
                pr = [((q[0] - a['p0'][0]) * u[0] + (q[1] - a['p0'][1]) * u[1], abs(-(q[0] - a['p0'][0]) * u[1] + (q[1] - a['p0'][1]) * u[0])) for q in (b['p0'], b['p1'])]
                if max(r[1] for r in pr) > max(3.0, 0.15 * Z): continue
                tb0, tb1 = min(r[0] for r in pr), max(r[0] for r in pr)
                ovm = min(2 * Z, 0.5 * b['len_px'])                                                         # 조금 겹친 조각(옅은 선을 따라 서로 늘어난 것)도
                if tb0 >= La - ovm and tb1 > La: ga, gb, ea, eb = La, tb0, 1, (0 if pr[0][0] <= pr[1][0] else 1)      # b 가 a 의 p1 쪽
                elif tb1 <= ovm and tb0 < 0: ga, gb, ea, eb = tb1, 0.0, 0, (1 if pr[0][0] <= pr[1][0] else 0)       # b 가 a 의 p0 쪽
                else: continue
                gap = gb - ga
                if gap > 4.5 * Z: continue
                if gap < Z and a['ends'][ea] and b['ends'][eb] and (gap <= 2 or _gap_ev(a['p0'], u, ga, gb, mw, occ, road, box)[1] < 0.5): continue    # 이어진 두 치수(틈에 글자 없음)
                blk = False
                for q in pts:
                    al = (q['px'] - a['p0'][0]) * u[0] + (q['py'] - a['p0'][1]) * u[1]
                    pe = abs(-(q['px'] - a['p0'][0]) * u[1] + (q['py'] - a['p0'][1]) * u[0])
                    if pe <= 0.5 * q.get('size', Z) + 3 and ga - 0.3 * Z <= al <= gb + 0.3 * Z: blk = True; break
                if blk: continue
                fl, fa, fr = _gap_ev(a['p0'], u, ga, gb, mw, occ, road, box) if gap > 2 else (1.0, 1.0, 0.0)
                if not (fl >= 0.25 or fa >= 0.55 or (fl >= 0.08 and fa + fr >= 0.7)): continue
                # 잇기: 먼 두 끝
                pa_, xa_, ea_ = (a['p1'], a['ext_px'][1], a['ends'][1]) if ea == 0 else (a['p0'], a['ext_px'][0], a['ends'][0])
                fb = 1 - eb
                pb_, xb_, eb_ = b['p0' if fb == 0 else 'p1'], b['ext_px'][fb], b['ends'][fb]
                if ea == 0: p0, p1, e0, e1, x0, x1 = pb_, pa_, eb_, ea_, xb_, xa_
                else: p0, p1, e0, e1, x0, x1 = pa_, pb_, ea_, eb_, xa_, xb_
                L2 = math.dist(p0, p1)
                a.update(p0=list(p0), p1=list(p1), len_px=L2, geom_px=L2 + x0 + x1, ext_px=[x0, x1], ends=[e0, e1], lw=max(a['lw'], b['lw']), _linked=True)
                if DIMDBG is not None: DIMDBG.append(dict(stage='link', a=[round(v) for v in a['p0'] + a['p1']], gap=round(gap), ev=(round(fl, 2), round(fa, 2), round(fr, 2))))
                out.pop(j); changed = True; break
            if changed: break
    for d in out: d.pop('_linked', None)
    return out


def _extend_to_symbols(out, mw, occ, pts, Z, box=None):
    """끝이 기호에 닿지 않은 치수선: 선 연장선 위(법선 ≤ 0.5·기호 크기) 끝에서 4Z 안에 소화전·송수구 기호가 있고 그 사이가
    가림(흰 바탕 이름표·글자 상자·검은 먹) 또는 옅은 선이면(≥ 60%) 기호 가장자리까지 늘림(소화전 번호 이름표가 치수선 끝을 덮은 경우)."""
    for d in out:
        L = d['len_px']; u = ((d['p1'][0] - d['p0'][0]) / L, (d['p1'][1] - d['p0'][1]) / L)
        if d.get('lw', 3) > 6: continue                          # 굵은 '선'(글자 획·채운 띠)은 늘리지 않음
        for side in (0, 1):
            if d['ext_px'][side] > 0: continue
            sg = -1 if side == 0 else 1; e = d['p0'] if side == 0 else d['p1']
            best = None
            for q in pts:
                if q['type'] not in ('hyd', 'fdc'): continue
                r = 0.5 * q.get('size', Z)
                al = ((q['px'] - e[0]) * u[0] + (q['py'] - e[1]) * u[1]) * sg
                pe = abs(-(q['px'] - e[0]) * u[1] + (q['py'] - e[1]) * u[0])
                if pe > max(0.5 * r + 3, 0.25 * Z) or not (r + 0.8 * Z < al <= r + 4 * Z): continue
                if best is None or al < best[0]: best = (al, q, r)
            if best is None: continue
            al, q, r = best
            g0, g1 = (0.0, al - r - 2) if sg > 0 else (-(al - r - 2), 0.0)
            fl, fa, _ = _gap_ev(e, u, g0, g1, mw, occ, None, box)
            if fa < 0.6: continue
            ne = [e[0] + u[0] * sg * (al - r - 2), e[1] + u[1] * sg * (al - r - 2)]
            if side == 0: d['p0'] = ne
            else: d['p1'] = ne
            d['ext_px'][side] = round(0.6 * q.get('size', Z), 1); d['ends'][side] = True
            d['len_px'] = math.dist(d['p0'], d['p1']); d['geom_px'] = d['len_px'] + sum(d['ext_px'])
            if DIMDBG is not None: DIMDBG.append(dict(stage='ext_sym', side=side, al=round(al), ev=(round(fl, 2), round(fa, 2))))
    return out


DIM_PAIR = os.environ.get('SITEMAP_DIMPAIR', '1' if DIMSET_A else '0') != '0'      # 글자 상자에 가려 양 끝 조각만 보이는 짧은 치수선(조각 짝짓기, 기본 끔, 1 = 켬)


def _pair_pieces(m0, ink, kept, lines, pts, Z, occ):
    """선을 끊고 얹은 글자 상자(흰 바탕) 때문에 양 끝의 짧은 조각(화살촉·틱 + 몇 px)만 보이는 치수선: 짧은 직선 조각 둘이
    한 직선 위(방향 차 ≤ 6°, 법선 ≤ 3px)에 틈 1~5Z 를 두고 마주 보고, 틈 띠(±8px)에 글자 먹이 절반 이상이고, 두 바깥 끝이 모두
    장식(화살촉·틱) 또는 기호면 하나의 치수선 후보로. 조각은 이미 찾은 선 위·글자 줄 상자 안이 아닌 것만."""
    sp = cv2.HoughLinesP(m0, 1, np.pi / 180, threshold=max(8, int(0.3 * Z)), minLineLength=max(7, int(0.35 * Z)), maxLineGap=2)
    if sp is None: sp = np.zeros((0, 1, 4), np.int32)
    short = [ki for ki, k in enumerate(kept) if math.hypot(k[2] - k[0], k[3] - k[1]) <= 2.5 * Z]     # 짧은 기존 선(한쪽 조각만 선으로 잡힌 것)도 조각으로
    KL = [LineString([(k[0], k[1]), (k[2], k[3])]) for ki, k in enumerate(kept) if ki not in short]
    def on_kept(a):                                            # 이미 찾은 (긴) 선 위·둘레(그 치수의 화살촉·글자 획)
        pm = Point((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        return any(Lk.distance(pm) < max(8, 0.6 * Z) for Lk in KL)
    def in_text(x, y): return any(ln['x0'] - 1 <= x <= ln['x1'] + 1 and ln['y0'] - 1 <= y <= ln['y1'] + 1 for ln in lines if ln.get('col') and (ln['y1'] - ln['y0']) >= 6)
    P = []
    for a in sp[:, 0, :]:
        a = [float(v) for v in a]
        if math.hypot(a[2] - a[0], a[3] - a[1]) > 2.5 * Z or on_kept(a) or in_text((a[0] + a[2]) / 2, (a[1] + a[3]) / 2): continue
        if any(Point(q['px'], q['py']).distance(LineString([(a[0], a[1]), (a[2], a[3])])) < 0.45 * q.get('size', Z) for q in pts): continue     # 기호 자체의 획
        P.append(a)
    P += [list(kept[ki][:4]) + [ki] for ki in short]
    # 같은 직선 위 겹친 짧은 조각 합치기(한 조각이 여러 선분으로 잡힘)
    U = []
    for a in sorted(P, key=lambda a: (len(a) < 5, -math.hypot(a[2] - a[0], a[3] - a[1]))):
        La = math.hypot(a[2] - a[0], a[3] - a[1]); ux, uy = (a[2] - a[0]) / La, (a[3] - a[1]) / La
        for b in U:
            Lb = math.hypot(b[2] - b[0], b[3] - b[1]); vx, vy = (b[2] - b[0]) / Lb, (b[3] - b[1]) / Lb
            if abs(ux * vy - uy * vx) > math.sin(math.radians(8)): continue
            if LineString([(b[0], b[1]), (b[2], b[3])]).distance(LineString([(a[0], a[1]), (a[2], a[3])])) < 3: break
        else: U.append(a)
    H, W = m0.shape
    out, used = [], set()
    cand = []
    if DIMDBG is not None: DIMDBG.append(dict(stage='pieces', P=[[round(v) for v in a] for a in U]))
    for i, a in enumerate(U):
        for j in range(i + 1, len(U)):
            b = U[j]
            # 두 조각의 바깥 끝(서로 먼 끝) 사이 선
            ends = [((a[0], a[1]), (b[0], b[1])), ((a[0], a[1]), (b[2], b[3])), ((a[2], a[3]), (b[0], b[1])), ((a[2], a[3]), (b[2], b[3]))]
            pa, pb = max(ends, key=lambda e: math.dist(e[0], e[1]))
            L = math.dist(pa, pb)
            if L < 2.0 * Z or L > 9 * Z: continue
            ux, uy = (pb[0] - pa[0]) / L, (pb[1] - pa[1]) / L
            okd = True
            for q in (a, b):
                Lq = math.hypot(q[2] - q[0], q[3] - q[1]); qx, qy = (q[2] - q[0]) / Lq, (q[3] - q[1]) / Lq
                if abs(qx * uy - qy * ux) > math.sin(math.radians(6)): okd = False; break
                for px_, py_ in ((q[0], q[1]), (q[2], q[3])):
                    if abs(-(px_ - pa[0]) * uy + (py_ - pa[1]) * ux) > 3: okd = False
            if not okd: continue
            ta = sorted(((a[0] - pa[0]) * ux + (a[1] - pa[1]) * uy, (a[2] - pa[0]) * ux + (a[3] - pa[1]) * uy))
            tb = sorted(((b[0] - pa[0]) * ux + (b[1] - pa[1]) * uy, (b[2] - pa[0]) * ux + (b[3] - pa[1]) * uy))
            g0, g1 = ta[1], tb[0]
            if g1 - g0 < 1.0 * Z or g1 - g0 > 5 * Z: continue
            if any(Point(q['px'], q['py']).distance(LineString([(pa[0] + ux * g0, pa[1] + uy * g0), (pa[0] + ux * g1, pa[1] + uy * g1)])) < 0.5 * q.get('size', Z) for q in pts): continue
            # 틈 띠(±8px)에 글자 먹
            nx, ny = -uy, ux; hit = n = 0
            for t in np.arange(g0, g1, 1.0):
                x, y = pa[0] + ux * t, pa[1] + uy * t; n += 1
                if any(0 <= int(round(y + ny * o)) < H and 0 <= int(round(x + nx * o)) < W and ink[int(round(y + ny * o)), int(round(x + nx * o))] for o in range(-8, 9)): hit += 1
            if hit < 0.5 * max(1, n): continue
            cand.append((g1 - g0, i, j, pa, pb))
    for gap, i, j, pa, pb in sorted(cand):
        if i in used or j in used: continue
        g = [pa[0], pa[1], pb[0], pb[1]]
        if any(LineString([pa, pb]).distance(Lk) < 4 for Lk in KL): continue      # 이미 찾은 선과 겹치거나 가로지름
        am = math.atan2(pb[1] - pa[1], pb[0] - pa[0]); pm = Point((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        if any(Lk.distance(pm) < 1.5 * Z and abs(((am - math.atan2(Lk.coords[1][1] - Lk.coords[0][1], Lk.coords[1][0] - Lk.coords[0][0])) + math.pi / 2) % math.pi - math.pi / 2) < math.radians(20)
               for Lk in KL): continue                                           # 이미 찾은 치수선 옆 거의 나란한 짝(그 치수의 글자 획·화살촉)
        _, _, _, glued, deco = _dim_profile(m0, g, Z)
        sym = [any(math.dist(e, (q['px'], q['py'])) < 0.6 * q.get('size', Z) + 0.8 * Z for q in pts if q['type'] in ('hyd', 'fdc')) for e in (pa, pb)]
        if not all(deco[k] or glued[k] or sym[k] for k in (0, 1)): continue
        used |= {i, j}
        out.append([pa[0], pa[1], pb[0], pb[1], math.atan2(pb[1] - pa[1], pb[0] - pa[0]) % math.pi, 'paired', [q[4] for q in (U[i], U[j]) if len(q) > 4]])
        if DIMDBG is not None: DIMDBG.append(dict(stage='pair', g=[round(v) for v in g], gap=round(gap)))
    return out


def detect_dims(img, LAB, leg, excl, lines, Z, pts=(), keep=None):
    if 'dim' not in leg['icons']: return []
    tp = leg['icons']['dim']
    m0 = _hmask(LAB, tp['lab'])
    m0[excl > 0] = 0
    mw = _hmask(LAB, tp['lab'], 35, 0.12, cmin=6.0, lmax=222) | m0                            # 약한 색상 마스크(JPEG 로 옅어진 가는 점선)
    mw[excl > 0] = 0
    m = m0.copy()
    for ln in lines:                                           # 같은 색 글자(치수 숫자)는 선이 아님 — 점선의 짧은 획(글자 줄로 묶인 '-')은 남김
        if ln['col'] and (ln['y1'] - ln['y0']) >= 6 and (ln.get('n', 2) >= 2 or re.search(r'\d', ln.get('text', '') or '')):
            m[max(0, ln['y0'] - 1):ln['y1'] + 2, max(0, ln['x0'] - 1):ln['x1'] + 2] = 0
    segs = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=int(max(15, 0.8 * Z)), minLineLength=int(1.4 * Z), maxLineGap=int(max(8, 0.6 * Z)))
    # 점선(흐린 획 사이 틈 ≈ 0.7Z)용 두 번째 탐색: 틈을 크게 — 첫 탐색 선이 이미 대부분 덮은 것은 버림(끝이 맞닿은 두 치수를 잇지 않게)
    segs2 = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=int(max(15, 0.8 * Z)), minLineLength=int(2.5 * Z), maxLineGap=int(max(10, 0.9 * Z)))
    if segs is None and segs2 is None: return []
    segs = list(segs[:, 0, :]) if segs is not None else []
    from shapely.ops import unary_union as _uu
    for extra in (segs2,):
        if extra is None: continue
        cov = [LineString([(a[0], a[1]), (a[2], a[3])]).buffer(3) for a in segs]
        U = _uu(cov) if cov else None
        for a in extra[:, 0, :]:
            ln_ = LineString([(a[0], a[1]), (a[2], a[3])])
            if U is None or ln_.intersection(U).length < 0.6 * ln_.length: segs.append(a)
    S = []
    for s in segs:
        x0, y0, x1, y1 = map(float, s)
        S.append([x0, y0, x1, y1, math.atan2(y1 - y0, x1 - x0) % math.pi])
    # 같은 직선 위 조각 합치기
    S.sort(key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
    merged = []
    for s in S:
        for g in merged:
            da = abs(((s[4] - g[4]) + math.pi / 2) % math.pi - math.pi / 2)
            if da > math.radians(4): continue
            ux, uy = math.cos(g[4]), math.sin(g[4])
            def perp(px, py): return abs(-(px - g[0]) * uy + (py - g[1]) * ux)
            if perp(s[0], s[1]) > 4 or perp(s[2], s[3]) > 4: continue
            ts = [(p[0] - g[0]) * ux + (p[1] - g[1]) * uy for p in ((g[0], g[1]), (g[2], g[3]), (s[0], s[1]), (s[2], s[3]))]
            lo, hi = min(ts[:2]), max(ts[:2])
            if min(ts[2:]) > hi + 2.5 * Z or max(ts[2:]) < lo - 2.5 * Z: continue
            a, b = min(ts), max(ts)
            g[:4] = [g[0] + ux * a, g[1] + uy * a, g[0] + ux * b, g[1] + uy * b]
            break
        else: merged.append(list(s))
    merged.sort(key=lambda g: -math.hypot(g[2] - g[0], g[3] - g[1]))
    kept = []
    for g in merged:                                             # 더 긴 선 위에 겹쳐 놓인 짧은 조각 버림
        dup = False
        for k in kept:
            Lk = LineString([(k[0], k[1]), (k[2], k[3])])
            if Lk.distance(Point(g[0], g[1])) < 6 and Lk.distance(Point(g[2], g[3])) < 6: dup = True; break
        if not dup: kept.append(g)
    # 선 가운데에 소화전·송수구 기호가 있으면 거기서 나눔(한 줄로 이어진 두 치수: 소화전 → 송수구 → 소화전)
    spl = []
    for g in kept:
        L = math.hypot(g[2] - g[0], g[3] - g[1]); ux, uy = (g[2] - g[0]) / L, (g[3] - g[1]) / L
        cuts = []
        for q in pts:
            if q['type'] not in ('hyd', 'fdc'): continue
            dx, dy = q['px'] - g[0], q['py'] - g[1]; al, pe = dx * ux + dy * uy, abs(-dx * uy + dy * ux)
            r = 0.5 * q.get('size', Z)
            if pe <= 0.6 * r + 2 and 1.4 * Z + r < al < L - 1.4 * Z - r: cuts.append((al, r))
        t0 = 0.0
        for al, r in sorted(cuts) + [(L + 1e9, 0)]:
            t1 = min(L, al - r - 2)
            if t1 - t0 >= 1.4 * Z: spl.append([g[0] + ux * t0, g[1] + uy * t0, g[0] + ux * t1, g[1] + uy * t1, g[4]])
            t0 = al + r + 2
    kept = spl
    occ = ((img.min(axis=2).astype(np.int16) < 110) & (m0 == 0)).astype(np.uint8)          # 선을 덮은 검은 글자·선
    boxm = np.zeros_like(occ)
    for ln in lines:                                                                          # 선을 덮은 글자 상자(흰 바탕 글자 포함)
        if (ln['y1'] - ln['y0']) >= 6 and (ln.get('n', 2) >= 2): occ[max(0, ln['y0'] - 2):ln['y1'] + 3, max(0, ln['x0'] - 2):ln['x1'] + 3] = 1; boxm[max(0, ln['y0'] - 2):ln['y1'] + 3, max(0, ln['x0'] - 2):ln['x1'] + 3] = 1
    if DIM_PAIR:
        ink_ = ((img.min(axis=2).astype(np.int16) < 150) | (m0 > 0)).astype(np.uint8)
        pr_ = _pair_pieces(m0, ink_, kept, lines, pts, Z, occ)
        gone = {ki for q in pr_ for ki in q[6]}                                                # 짝지은 짧은 기존 선은 새 선으로 대체
        kept = [k for ki, k in enumerate(kept) if ki not in gone] + [q[:6] for q in pr_]
    out = []
    for g in kept:
        L = math.hypot(g[2] - g[0], g[3] - g[1])
        if L < 1.4 * Z: continue
        cxx, cyy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        # 가는 선인지: 선 따라 법선 방향 마스크 폭 중앙값(채운 기호·글자 덩어리 제외)
        ux_, uy_ = (g[2] - g[0]) / L, (g[3] - g[1]) / L
        ws = []
        for tt in np.arange(0.15, 0.86, 0.05):
            px_, py_ = g[0] + (g[2] - g[0]) * tt, g[1] + (g[3] - g[1]) * tt
            cnt = 0
            for o in range(-8, 9):
                xx, yy = int(round(px_ - uy_ * o)), int(round(py_ + ux_ * o))
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx]: cnt += 1
            ws.append(cnt)
        if float(np.median(ws)) > 6: continue
        if len(g) < 6 and any(ln['x0'] - 2 <= cxx <= ln['x1'] + 2 and ln['y0'] - 2 <= cyy <= ln['y1'] + 2 and L < 1.5 * (ln['x1'] - ln['x0']) for ln in lines if (ln['y1'] - ln['y0']) >= 6): continue
        # 끝점 다듬기(화살촉·틱까지, 같은 색 기호에 붙은 끝은 잘라 냄)
        t0, t1, lw, glued, deco = _dim_profile(m0, g, Z, occ, mw if DIM_WEAK else None)
        if DIMDBG is not None: DIMDBG.append(dict(stage='profile', g=[round(v) for v in g[:4]], t=(t0, t1), lw=lw, glued=glued, deco=deco))
        if t1 - t0 >= 1.2 * Z:
            p0 = [g[0] + ux_ * t0, g[1] + uy_ * t0]; p1 = [g[0] + ux_ * t1, g[1] + uy_ * t1]
        else: p0, p1 = [g[0], g[1]], [g[2], g[3]]
        # 같은 색 기호를 뚫고 들어간 끝: 인식한 점 기호(중심이 선 위, 끝에서 1.5Z 안)의 가장자리에서 자름
        for side in (0, 1):
            L2 = math.dist(p0, p1)
            if L2 < 2.5 * Z: break
            for q in pts:
                dx, dy = q['px'] - p0[0], q['py'] - p0[1]
                al, pe = dx * ux_ + dy * uy_, abs(-dx * uy_ + dy * ux_)
                r = 0.5 * q.get('size', Z)
                if pe > 0.6 * r + 2: continue
                if side == 0 and -0.2 * Z <= al <= 1.5 * Z and al + r + 2 < L2 - 1.2 * Z: p0 = [p0[0] + ux_ * (al + r + 2), p0[1] + uy_ * (al + r + 2)]; glued[0] = True; break
                if side == 1 and L2 - 1.5 * Z <= al <= L2 + 0.2 * Z and al - r - 2 > 1.2 * Z: p1 = [p0[0] + ux_ * (al - r - 2), p0[1] + uy_ * (al - r - 2)]; glued[1] = True; break
        L2 = math.dist(p0, p1)
        # 끝이 소화전·송수구 기호에 붙어 있으면(선 끝 앞뒤 1.0Z 안) 기호 중심까지 — 기호 사이 치수는 기호 중심 거리로 적는다
        ext = [0.0, 0.0]
        for side, q0, sg in ((0, p0, -1), (1, p1, 1)):
            best = None
            for q in pts:
                if q['type'] not in ('hyd', 'fdc'): continue
                dx, dy = q['px'] - q0[0], q['py'] - q0[1]
                al, pe = (dx * ux_ + dy * uy_) * sg, abs(-dx * uy_ + dy * ux_)
                if -0.5 * Z <= al <= 1.2 * Z and pe <= 1.2 * Z and math.hypot(al, pe) <= 1.3 * Z and (best is None or math.hypot(al, pe) < best[0]): best = (math.hypot(al, pe), q)
            if best: ext[side] = 0.6 * best[1].get('size', Z)
            elif glued[side]: ext[side] = 0.6 * Z
        out.append(dict(p0=p0, p1=p1, len_px=L2, geom_px=L2 + sum(ext), ext_px=[round(e, 1) for e in ext], lw=round(lw, 1),
                        ends=[bool(deco[k] or glued[k] or ext[k] > 0) for k in (0, 1)]))
    # 다듬은 뒤 같은 직선 위에 겹친 선(조각이 따로 늘어난 것) → 긴 것만
    out = _dedup(out, 5)
    if DIM_LINK:
        lum_ = img.mean(axis=2); bg_ = float(np.percentile(lum_, 70)); sat_ = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).astype(np.int16)
        road = ((lum_ < bg_ - 14) & (lum_ > bg_ - 85) & (sat_ < 28)).astype(np.uint8)
        _link_dims(out, mw, occ, pts, Z, road, boxm)
        _extend_to_symbols(out, mw, occ, pts, Z, boxm)
        out = _dedup(out, 10)                                   # 이은 긴 선에 겹친 조각(같은 조각이 다른 기울기로 또 잡힌 것)
    if keep is not None and DIM_LINK: out = [d for d in out if keep(d)]      # 건물 안·도로 중앙선 등 버릴 선은 글자 읽기 전에(같은 글자 상자를 나눠 가진 뒤 버려져 진짜 치수선이 글자를 잃지 않게)
    read_dim_texts(img, m0, out, lines, Z, mw)
    # 한 글자 상자를 두 선이 나눠 가지면: 글자 가운데가 선 가운데에 더 가까운 쪽만(다른 선은 글자 없음)
    for i, a in enumerate(out):
        if not a.get('tbox'): continue
        for b in out[i + 1:]:
            if not b.get('tbox') or _iou(a['tbox'], b['tbox']) < 0.5: continue
            def off(d):
                cx, cy = (d['tbox'][0] + d['tbox'][2]) / 2, (d['tbox'][1] + d['tbox'][3]) / 2
                mx, my = (d['p0'][0] + d['p1'][0]) / 2, (d['p0'][1] + d['p1'][1]) / 2
                return math.hypot(cx - mx, cy - my) / max(1.0, d['len_px'])
            lose = b if off(a) <= off(b) else a
            lose.update(cands=[], n_read=0, text=None, tbox=None, shared_text=True)
            if lose is a: break
    # 글자도 없고 양 끝 장식(화살촉·틱·기호)도 없는 선은 치수선이 아님
    out = [d for d in out if d.get('cands') or all(d['ends'])]
    return out


# ─────────────────────────────── 5) 방위·축척·건물·도로
def detect_north(img, lines, lum, bg):
    dk = (lum < bg - 100).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dk, 8)
    best = None
    for ln in lines:
        t = ln.get('text', '').strip()
        if t not in ('N', 'n', 'N.', 'W', 'H') or ln['n'] > 2: continue
        for i in range(1, n):
            x, y, w, h, a = st[i]
            s = max(w, h)
            if not (22 <= s <= 110) or a < 40: continue
            c = cen[i]
            dd = math.hypot(c[0] - ln['cx'], c[1] - ln['cy'])
            if dd > 3.2 * max(ln['h'], 12) + s / 2: continue
            # 화살표 끝이 글자 쪽: 성분 안에서 글자에 가장 가까운 점까지 거리가 작아야
            ys, xs = np.where(lab[y:y + h, x:x + w] == i)
            near = np.min(np.hypot(xs + x - ln['cx'], ys + y - ln['cy']))
            if near > 2.2 * max(ln['h'], 12): continue
            sc = near + 0.1 * dd
            if best is None or sc < best[0]: best = (sc, c, ln)
    if best is None: return None
    _, c, ln = best
    ang = math.degrees(math.atan2(ln['cx'] - c[0], -(ln['cy'] - c[1])))
    return dict(deg=ang, px=[float(c[0]), float(c[1])], n=[ln['cx'], ln['cy']])


def detect_scalebar(img, lines, lum, bg):
    dk = (lum < bg - 100).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dk, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 50 or not (5 <= h <= 18) or w / h < 5: continue
        fill = a / (w * h)
        if not (0.35 <= fill <= 0.85): continue
        # 칸 무늬: 가운데 줄에서 흑백 전환이 3번 이상
        row = (lab[y + h // 2, x:x + w] == i).astype(np.int8)
        if np.abs(np.diff(row)).sum() < 3: continue
        nums = []
        for ln in lines:
            if ln['y1'] < y - 30 or ln['y0'] > y + h + 30 or ln['x1'] < x - 20 or ln['x0'] > x + w + 30: continue
            for mm in re.finditer(r'(\d+(?:\.\d+)?)\s*(k?m)?', (ln.get('text_num') or ln.get('text', '')).replace(' ', '')):
                v = float(mm.group(1)) * (1000 if (mm.group(2) or '').lower() == 'km' else 1)
                if v > 0: nums.append((v, ln['cx']))
        if not nums: continue
        v, xc = max(nums)
        if abs(xc - (x + w)) > 0.25 * w + 25: continue
        sc = w
        if best is None or sc > best[0]: best = (sc, w / v, [int(x), int(y), int(x + w), int(y + h)], v)
    if best is None: return None
    return dict(px_per_m=best[1], box=best[2], m=best[3])


def detect_buildings(img, lum, sat, bg, excl, pts, Z):
    """닫힌 어두운 윤곽 안쪽 영역 → 건물 후보. 대상 = 둘레에 주출입구·비상구·송수구 기호가 가장 많이 붙은 것.
    외곽선이 글자 바탕·치수선에 조금 끊겨도 닫히도록 메움 크기를 3→5→9 px 로 늘려 가며 찾는다."""
    dk0 = ((lum < bg - 120) & (sat < 60)).astype(np.uint8)                        # 진한 외곽선 — 빗금·연한 선 제외
    colr = ((sat >= 60) & (lum < bg - 40)).astype(np.uint8)
    dk0 |= colr & cv2.dilate(dk0, np.ones((7, 7), np.uint8))                          # 외곽선 위에 덧그린 색(치수선 끝·기호)은 선의 일부로
    n_ent = sum(1 for q in pts if q['type'] in ('main', 'exit', 'fdc'))
    best = None
    for k in (3, 5, 9):
        r = _buildings_k(img, lum, sat, cv2.morphologyEx(dk0, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), excl, pts, Z)
        if r[0] is not None and (best is None or r[0]['near'] > best[0]['near']): best = r
        if best is not None and best[0]['near'] >= max(1, 0.6 * n_ent): break
    return best if best else (None, [])


def _buildings_k(img, lum, sat, dk, excl, pts, Z):
    cnts, hier = cv2.findContours(dk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = lum.shape
    cands = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0: continue                     # 구멍(안쪽 윤곽)만
        A = cv2.contourArea(c)
        if A < (1.5 * Z) ** 2 or A > 0.25 * H * W: continue
        (rx, ry), (rw, rh), ra = cv2.minAreaRect(c)
        if A / max(1, rw * rh) < 0.6 or max(rw, rh) / max(1, min(rw, rh)) > 9: continue
        x, y, w, h = cv2.boundingRect(c)
        if excl[int(ry), int(rx)]: continue
        ap = cv2.approxPolyDP(c, max(2.0, 0.01 * cv2.arcLength(c, True)), True)[:, 0, :].astype(float)
        if len(ap) < 3: continue
        P = Polygon(ap).buffer(0)
        if P.is_empty: continue
        P = P.buffer(1.5, join_style=2)                    # 윤곽선 굵기 절반 되돌림
        ins = img[y:y + h, x:x + w]
        mk = np.zeros((h, w), np.uint8); cv2.drawContours(mk, [c - [x, y]], -1, 1, -1)
        fs = float(sat[y:y + h, x:x + w][mk > 0].mean()) if mk.any() else 0
        near = sum(1 for q in pts if q['type'] in ('main', 'exit', 'fdc') and P.exterior.distance(Point(q['px'], q['py'])) < 1.6 * Z and not P.contains(Point(q['px'], q['py'])))
        cands.append(dict(poly=P, area=P.area, near=near, fill_sat=fs, rect=((rx, ry), (rw, rh), ra)))
    if not cands: return None, []
    tgt = max(cands, key=lambda c: (c['near'], c['fill_sat'] > 12, c['area']))
    return tgt, [c for c in cands if c is not tgt]


def detect_roads(img, lum, sat, bg, excl, theta_axis, lines, polys=()):
    """회색 띠 → 건물 축으로 돌려 가로·세로 긴 띠 → 중심선·폭(px)"""
    H, W = lum.shape
    lm = cv2.medianBlur(lum.astype(np.uint8), 5).astype(np.float32)
    sm = cv2.medianBlur(sat.astype(np.uint8), 5)
    g = ((lm < bg - 14) & (lm > bg - 85) & (sm < 28)).astype(np.uint8)
    g[excl > 0] = 0
    for P_ in polys:                                   # 건물(대상·인접) 안은 도로 아님
        cv2.fillPoly(g, [np.array(P_.buffer(3).exterior.coords, np.int32)], 0)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), theta_axis, 1.0)        # 반시계 theta_axis → 건물 축이 화면 축
    D = int(math.hypot(W, H))
    M[0, 2] += (D - W) / 2; M[1, 2] += (D - H) / 2
    R = cv2.warpAffine(g, M, (D, D), flags=cv2.INTER_NEAREST)
    Mi = cv2.invertAffineTransform(M)
    back = lambda x, y: (Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2], Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2])
    def bands(Lk):
        out = []
        for ori in ('h', 'v'):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (Lk, 1) if ori == 'h' else (1, Lk))
            B = cv2.morphologyEx(R, cv2.MORPH_OPEN, k)
            n, lab, st, cen = cv2.connectedComponentsWithStats(B, 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                sub = (lab[y:y + h, x:x + w] == i)
                prof = sub.sum(0) if ori == 'h' else sub.sum(1)
                prof = prof[prof > 0]
                out.append((ori, x, y, w, h, a, sub, float(np.median(prof)) if len(prof) else 0))
        return out
    Lk = 140
    B0 = bands(Lk)
    T = max([b[7] for b in B0 if (b[3] if b[0] == 'h' else b[4]) >= 2.5 * b[7]] + [0])
    if T > 0.5 * Lk: Lk = int(2.2 * T); B0 = bands(Lk)
    roads = []
    for ori, x, y, w, h, a, sub, med in B0:
            L, T = (w, h) if ori == 'h' else (h, w)
            if L < Lk or L < 2.5 * med: continue
            th = med + 2.0                                                 # 가장자리 선 굵기 보정
            if med < 7: continue
            if ori == 'h':
                ys = np.where(sub.any(1))[0]; cyy = y + float(np.average(np.arange(h), weights=sub.sum(1)))
                p0, p1 = back(x, cyy), back(x + w, cyy)
            else:
                cxx = x + float(np.average(np.arange(w), weights=sub.sum(0)))
                p0, p1 = back(cxx, y), back(cxx, y + h)
            roads.append(dict(p0=p0, p1=p1, w_px=th, ori=ori))
    return roads


# ─────────────────────────────── 전체
def recognize(img, legend=None):
    """img: BGR(np) 또는 PIL. legend: 정보카드 범례 표 낱말(참고, 없어도 됨) → 화소 좌표 결과 dict"""
    if not isinstance(img, np.ndarray): img = cv2.cvtColor(np.asarray(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    lum, sat = _lum_sat(img)
    bg = float(np.percentile(lum, 70))
    gray = img.min(axis=2)                                    # 채널 최솟값: 검정·색 글자 모두 어둡게
    LAB = _lab(cv2.GaussianBlur(img, (3, 3), 0))
    lines = ocr_lines(gray, text_lines(img, lum, sat, bg))
    leg = find_legend(img, lines, lum, sat, bg, LAB)
    rec = dict(size=[W, H], legend=leg, lines=[{k: ln[k] for k in ('x0', 'y0', 'x1', 'y1', 'text', 'text_num', 'col') if k in ln} for ln in lines])
    rec['_img'] = img                                                     # to_frame 의 재확인 OCR 용(저장 안 함)
    if leg is None: rec['error'] = 'legend_not_found'; return rec
    excl = np.zeros((H, W), np.uint8)
    b = leg['box']; excl[max(0, b[1] - 3):b[3] + 3, max(0, b[0] - 3):b[2] + 3] = 1
    pts, Z = detect_points(img, LAB, leg, excl)
    rec['Z'] = Z
    north = detect_north(img, lines, lum, bg)
    sbar = detect_scalebar(img, lines, lum, bg)
    tgt, others = detect_buildings(img, lum, sat, bg, excl, pts, Z)
    rec.update(points=pts, dims=[], north=north, scalebar=sbar)
    if tgt is None:
        rec['dims'] = detect_dims(img, LAB, leg, excl, lines, Z, pts); rec['error'] = 'footprint_not_found'
        for k, d in enumerate(rec['dims']): resolve_dim(d, scale_consensus(rec, skip=k)[0], Z)
        return rec
    rec['footprint_px'] = [[round(x, 1), round(y, 1)] for x, y in list(tgt['poly'].exterior.coords)[:-1]]
    rec['others_px'] = [[[round(x, 1), round(y, 1)] for x, y in list(o['poly'].exterior.coords)[:-1]] for o in others]
    # 건물 축 각도(화면, 시계 +): 외곽 최소 사각형 긴 변 방향 → 방위표 각도에 가장 가까운 90° 배수로
    (rx, ry), (rw, rh), ra = tgt['rect']
    rec['axis_deg'] = float(ra % 90)
    th = rec['axis_deg']
    if north is not None:
        k = round((north['deg'] - th) / 90.0); rec['north_deg'] = th + 90 * k
        if abs(((north['deg'] - rec['north_deg']) + 180) % 360 - 180) > 30: rec['north_deg'] = north['deg']
        rec['north_src'] = 'north_arrow+footprint_axis'
    else:
        rec['north_deg'] = th if th <= 45 else th - 90; rec['north_src'] = 'footprint_axis(방위표 못 찾음)'
        rec['north_ambiguous'] = True                                  # to_frame 에서 평면도 외곽 가로세로비로 90° 모호성 해소
    # 도로: 건물 축(= 방위 각)으로 돌려 찾기. cv2 는 반시계 + → 화면 시계 각의 부호 반대
    rds = detect_roads(img, lum, sat, bg, excl, rec['north_deg'], lines, [c['poly'] for c in [tgt] + others])
    for r in rds:                                                         # 폭 글자(띠 안 또는 옆)
        L = LineString([r['p0'], r['p1']])
        best = None
        for ln in lines:
            v = _num(ln)
            if v is None or v > 60 or ln['col']: continue
            dd = L.distance(Point(ln['cx'], ln['cy']))
            if dd > r['w_px'] / 2 + 30: continue
            if re.search(r'폭|B=|도로|로|길', ln.get('text', '')): dd -= 40
            if best is None or dd < best[0]: best = (dd, v, ln['text'])
        if best: r['label_m'], r['label'] = best[1], best[2]
    rec['roads'] = rds
    inner = [c['poly'].buffer(-4) for c in [tgt] + others]                 # 건물 안(빗금·채움)을 지나는 '선'은 치수선이 아님
    def on_center(d):                                                      # 도로 중앙선(도로 띠 가운데·평행) 제외
        a = math.atan2(d['p1'][1] - d['p0'][1], d['p1'][0] - d['p0'][0])
        for r in rds:
            b = math.atan2(r['p1'][1] - r['p0'][1], r['p1'][0] - r['p0'][0])
            if abs(((a - b) + math.pi / 2) % math.pi - math.pi / 2) > math.radians(5): continue
            Lr = LineString([r['p0'], r['p1']])
            if Lr.distance(Point(d['p0'])) < 0.25 * r['w_px'] and Lr.distance(Point(d['p1'])) < 0.25 * r['w_px']: return True
        return False
    def keep_dim(d):
        return (not any(q.contains(Point((d['p0'][0] + d['p1'][0]) / 2, (d['p0'][1] + d['p1'][1]) / 2)) for q in inner if not q.is_empty)
                and d['len_px'] < 0.6 * max(W, H) and not on_center(d))      # 도면 전체를 가로지르는 선도 아님
    dims = detect_dims(img, LAB, leg, excl, lines, Z, pts, keep_dim)
    rec['dims'] = [d for d in dims if keep_dim(d)]
    # 치수 값 결정: 축척 합의(자기 자신 빼고)로 기하 길이와 맞춰 봄 — to_frame 에서 최종 축척으로 다시
    for k, d in enumerate(rec['dims']):
        s_, src_ = scale_consensus(rec, skip=k)
        resolve_dim(d, s_, Z); d['scale_src'] = src_
    return rec


DIM_JOIN = os.environ.get('SITEMAP_DIMJOIN', '1') != '0'


def _join_fragments(D, Z):
    """글자는 뚜렷한데 선 길이와 안 맞는 치수(geom_mismatch) d: 같은 직선 위에 이어지는 글자 없는 조각 q(점선이 도로·건물선을 지나며 끊긴 것)가 있고
    둘을 이은 길이가 글자와 맞으면 하나로 잇는다(글자가 이음 근거 — 이어도 안 맞으면 그대로 두고 불일치 표시 유지).
    D: to_frame 에서 최종 축척으로 다시 맞춘 치수들(목록 순서는 rec['dims'] 와 같게 둠 — 이어 붙인 조각은 값 없이 joined_into 표시)."""
    if not DIM_JOIN: return
    drop = set()
    for k, d in enumerate(D):
        if k in drop or not d.get('geom_mismatch') or d.get('val') is None or not d.get('geom_m'): continue
        s_ = d['geom_px'] / d['geom_m']
        (x0, y0), (x1, y1) = d['p0'], d['p1']; L = math.hypot(x1 - x0, y1 - y0) or 1; ux, uy = (x1 - x0) / L, (y1 - y0) / L
        best = None
        for k2, q in enumerate(D):
            if k2 == k or k2 in drop or max((c[1] for c in q.get('cands') or []), default=0) >= 1: continue     # 조각에 자기 글자가 있으면 다른 치수
            ts = [((px - x0) * ux + (py - y0) * uy, abs(-(px - x0) * uy + (py - y0) * ux)) for px, py in (q['p0'], q['p1'])]
            if max(t[1] for t in ts) > max(4.0, 0.25 * Z): continue
            ta, tb = min(t[0] for t in ts), max(t[0] for t in ts)
            if ta < L - 2 and tb > 2: continue                                                   # d 와 겹침
            gap = ta - L if ta >= L - 2 else -tb
            if gap > max(3 * Z, 0.5 * L): continue
            lo, hi = min(0.0, ta), max(L, tb)
            ext = (d['ext_px'][0] if lo == 0 else q['ext_px'][0 if ts[0][0] < ts[1][0] else 1]) + (d['ext_px'][1] if hi == L else q['ext_px'][1 if ts[0][0] < ts[1][0] else 0])
            g2 = (hi - lo + ext) / s_
            if abs(d['val'] - g2) <= max(DIM_TOL_ABS, DIM_TOL_REL * g2) and (best is None or gap < best[0]): best = (gap, k2, lo, hi, ext, ts)
        if best is None: continue
        _, k2, lo, hi, ext, ts = best; q = D[k2]
        qa, qb = (q['p0'], q['p1']) if ts[0][0] < ts[1][0] else (q['p1'], q['p0'])
        qend = q['ends'][0 if ts[0][0] < ts[1][0] else 1], q['ends'][1 if ts[0][0] < ts[1][0] else 0]
        p0, e0 = (d['p0'], d['ends'][0]) if lo == 0 else (qa, qend[0])
        p1, e1 = (d['p1'], d['ends'][1]) if hi == L else (qb, qend[1])
        d.update(p0=p0, p1=p1, len_px=hi - lo, geom_px=hi - lo + ext, ends=[e0, e1], joined=True)
        drop.add(k2)
        resolve_dim(d, s_, Z)
        q.update(val=None, val_src=None, certainty=None, joined_into=k, geom_mismatch=False)


# ─────────────────────────────── 좌표 변환·정합
def _facade_name(n):
    ang = math.degrees(math.atan2(n[0], n[1])) % 360
    return min([(0, '북측면'), (90, '동측면'), (180, '남측면'), (270, '서측면')], key=lambda d: abs(((d[0] - ang + 180) % 360) - 180))[1]


def _open_end(rec, k, pp):
    """도로 k 의 끝점 pp(화소)가 다른 도로와 만나는 교차점이 아니면 True(= 도면 틀에서 끊긴 끝 → 도로가 이어짐)"""
    for j, r in enumerate(rec.get('roads', [])):
        if j == k: continue
        if LineString([r['p0'], r['p1']]).distance(Point(pp)) < r['w_px'] / 2 + 15: return False
    return True


def to_frame(rec, ref_outline=None, card_area=None, cfg=None):
    """화소 결과 → 건물 좌표(m, 동 +x · 북 +y). ref_outline: 평면도 1층 외곽(엔진 좌표, shapely) — 있으면 축척·위치를 여기에 맞춘다.
    반환: dict(scale, north_deg, reg, footprint, entrances, hydrants, fdc, dims, roads, checks)"""
    if 'footprint_px' not in rec: return None
    if rec.get('north_ambiguous') and ref_outline is not None:
        rx0, ry0, rx1, ry1 = ref_outline.bounds; ar_ref = math.log((rx1 - rx0) / max(1e-6, ry1 - ry0))
        best = None
        for k in (0, 90):
            t_ = math.radians(rec['north_deg'] + k)
            q = [((u) * math.cos(t_) + (v) * math.sin(t_), (u) * math.sin(t_) - (v) * math.cos(t_)) for u, v in rec['footprint_px']]
            xs, ys = [a for a, _ in q], [b for _, b in q]
            e = abs(math.log((max(xs) - min(xs)) / max(1e-6, max(ys) - min(ys))) - ar_ref)
            if best is None or e < best[0]: best = (e, k)
        rec = dict(rec, north_deg=rec['north_deg'] + best[1], north_src=rec['north_src'] + '+평면도 가로세로비')
    th = math.radians(rec['north_deg'])
    fpx = Polygon(rec['footprint_px'])
    c0 = fpx.centroid
    def rot(u, v):                                   # 화소 → 북쪽 위 화소 단위(동 +a, 북 +b)
        a, c = u - c0.x, v - c0.y
        return (a * math.cos(th) + c * math.sin(th), a * math.sin(th) - c * math.cos(th))
    fr = Polygon([rot(*p) for p in fpx.exterior.coords])
    bx0, by0, bx1, by1 = fr.bounds; fw, fh = bx1 - bx0, by1 - by0
    srcs = {}
    if rec.get('scalebar'): srcs['scalebar'] = rec['scalebar']['px_per_m']
    if ref_outline is not None:
        rx0, ry0, rx1, ry1 = ref_outline.bounds
        srcs['plan_outline'] = math.sqrt((fw / max(1e-6, rx1 - rx0)) * (fh / max(1e-6, ry1 - ry0)))
    if card_area: srcs['card_area'] = math.sqrt(fr.area / card_area)
    dvals = [d.get('geom_px', d['len_px']) / d['val'] for d in rec.get('dims', []) if d.get('val') and d.get('val_src', 'ocr') in ('ocr', 'ocr+geom')]
    checks = []
    if 'plan_outline' in srcs and 'scalebar' in srcs:
        r = srcs['scalebar'] / srcs['plan_outline']
        s, ssrc = (srcs['scalebar'], 'scalebar') if abs(r - 1) <= 0.1 else (srcs['plan_outline'], 'plan_outline')
        if abs(r - 1) > 0.1: checks.append(dict(kind='sitemap_scale', scalebar=round(srcs['scalebar'], 3), plan=round(srcs['plan_outline'], 3)))
    elif 'plan_outline' in srcs: s, ssrc = srcs['plan_outline'], 'plan_outline'
    elif 'scalebar' in srcs: s, ssrc = srcs['scalebar'], 'scalebar'
    elif 'card_area' in srcs: s, ssrc = srcs['card_area'], 'card_area'
    elif dvals: s, ssrc = float(np.median(dvals)), 'dims'
    else: return None
    # 평행이동: 외곽 bbox 중심 → 평면도 외곽 bbox 중심(없으면 원점 = 작전도 외곽 중심)
    if ref_outline is not None:
        rx0, ry0, rx1, ry1 = ref_outline.bounds; tx, ty = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    else: tx, ty = 0.0, 0.0
    ox, oy = (bx0 + bx1) / 2, (by0 + by1) / 2
    def M(u, v):
        a, b = rot(u, v); return ((a - ox) / s + tx, (b - oy) / s + ty)
    FP = Polygon([M(*p) for p in rec['footprint_px']]).buffer(0)
    if ref_outline is not None:
        # 외곽이 거의 직사각형이면 평면도 외곽 쪽 크기로(축척 차이는 checks 에)
        pass
    fpo = FP
    ents, hyds, fdcs = [], [], []
    for q in rec.get('points', []):
        x, y = M(q['px'], q['py'])
        if q['type'] in ('main', 'exit', 'fdc'):
            p = nearest_points(fpo.exterior, Point(x, y))[0]
            n = (x - p.x, y - p.y); L = math.hypot(*n) or 1
            e = dict(kind=q['type'], x=round(p.x, 2), y=round(p.y, 2), icon=[round(x, 2), round(y, 2)], facade=_facade_name((n[0] / L, n[1] / L)), px=[q['px'], q['py']])
            (fdcs if q['type'] == 'fdc' else ents).append(e)
        else:
            hyds.append(dict(x=round(x, 2), y=round(y, 2), px=[q['px'], q['py']]))
    # 소화전 글자(번호·종류·거리)
    lines = rec.get('lines', [])
    Z = rec.get('Z', 20)
    for h in hyds:
        best = None
        for ln in lines:
            t = ln.get('text', '')
            if not re.search(r'[가-힣]+\s*-\s*\d|\d+\s*k?m', t.replace(' ', '')): continue
            cx, cy = (ln['x0'] + ln['x1']) / 2, (ln['y0'] + ln['y1']) / 2
            dd = math.hypot(cx - h['px'][0], cy - h['px'][1]) - (ln['x1'] - ln['x0']) / 2
            if dd > 3 * Z: continue
            if best is None or dd < best[0]: best = (dd, t, ln.get('text_num'), [ln['x0'], ln['y0'], ln['x1'], ln['y1']])
        if best:
            fix = lambda z: re.sub(r'(-\s*\d)(0\.\d+\s*[kK][mM])', r'\1 \2', z or '')      # '가상-10.150km' → '가상-1 0.150km'
            best = (best[0], fix(best[1]), fix(best[2]) if best[2] else best[2], best[3])
            t = best[1]
            m = re.search(r'([가-힣]{1,3})\s*-?\s*(\d{1,3})(?!\d)', t)
            if m: h['id'] = f"{m.group(1)}-{m.group(2)}"
            if '지하' in t: h['type'] = '지하식'
            elif '지상' in t: h['type'] = '지상식'
            tn = best[2] or t
            dv = parse_dist(re.sub(r'^.*?-\s*\d+\s*(\([^)]*\))?', '', tn) if '-' in tn else tn)
            if dv is None: dv = parse_dist(re.sub(r'^.*?[가-힣]\s*-\s*\d+\)?', '', t) if m else t)
            if dv is not None and 5 < dv <= 5000: h['dist_label'] = dv
            h['label'] = t; h['label_box'] = best[3]
    roads = []
    for k, r in enumerate(rec.get('roads', [])):
        p0, p1 = M(*r['p0']), M(*r['p1'])
        w = r['w_px'] / s
        if r.get('label_m') and abs(r['label_m'] / w - 1) < 0.3: w, wsrc = r['label_m'], 'label'
        else: w, wsrc = round(w * 2) / 2, 'measured'
        # 도면 틀에 닿은 끝은 도로가 이어진다고 보고 길게
        L = math.dist(p0, p1); ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
        def edge(pp): return _open_end(rec, k, pp)
        if edge(r['p0']): p0 = (p0[0] - ux * 500, p0[1] - uy * 500)
        if edge(r['p1']): p1 = (p1[0] + ux * 500, p1[1] + uy * 500)
        roads.append(dict(id=f'D{k + 1}', name=f"작전도 도로{k + 1}", width=w, width_src=wsrc, line=[[round(p0[0], 2), round(p0[1], 2)], [round(p1[0], 2), round(p1[1], 2)]]))
    # 도면 밖 소화전(거리 글자): 가까운 도로를 따라 건물에서 멀어지는 쪽으로 글자 거리만큼.
    # 거리 글자는 다른 배율로 한 번 더 읽어 같을 때만 믿는다(아니면 위치 미상 → 부지 계산에서 뺌)
    img_ = rec.get('_img')
    for h in hyds:
        if not h.get('dist_label') or img_ is None: continue
        b_ = h['label_box']; H_, W_ = img_.shape[:2]
        c3 = cv2.resize(img_.min(axis=2)[max(0, b_[1] - 4):min(H_, b_[3] + 4), max(0, b_[0] - 4):min(W_, b_[2] + 4)], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, c3 = cv2.threshold(c3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t3 = _ocr(cv2.copyMakeBorder(c3, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255), 'eng', 7, '0123456789.kmM-=() L')
        v3 = parse_dist(re.sub(r'^.*?-\s*\d+\s*(\([^)]*\))?', '', t3) if '-' in t3 else t3)
        h['dist_conf'] = v3 is not None and abs(v3 - h['dist_label']) < 0.5
    def near_open_end(px):
        """화소 점이 어떤 도로의 '열린 끝'(도면 틀에서 끊긴 쪽) 가까이(도로 폭 안쪽·끝에서 4Z 이내)에 있으면 그 도로 번호"""
        best = None
        for j, r in enumerate(rec.get('roads', [])):
            L_ = math.dist(r['p0'], r['p1']); ux_, uy_ = (r['p1'][0] - r['p0'][0]) / L_, (r['p1'][1] - r['p0'][1]) / L_
            dx_, dy_ = px[0] - r['p0'][0], px[1] - r['p0'][1]
            al, pe = dx_ * ux_ + dy_ * uy_, abs(-dx_ * uy_ + dy_ * ux_)
            if pe > r["w_px"] / 2 + 5 * Z: continue
            for q, t_ in ((r['p0'], 0.0), (r['p1'], L_)):
                if abs(al - t_) <= 6 * Z and _open_end(rec, j, q) and (best is None or abs(al - t_) < best[0]): best = (abs(al - t_), j)
        return best[1] if best else None
    for h in hyds:
        if not h.get('dist_label'): continue
        # 도면 밖 소화전 표시는 도로가 도면 틀에서 끊기는 끝 근처에 있다(도면 안 소화전 글자를 거리로 잘못 읽은 경우 제외)
        h['_road'] = near_open_end(h['px'])
        if rec.get('roads') and h['_road'] is None: h['dist_label_ignored'] = h.pop('dist_label'); continue
        if fpo.distance(Point(h['x'], h['y'])) >= h['dist_label'] - 5: continue
        rl = [LineString(r['line']) for r in roads]
        if not rl: continue
        rp = rec.get('roads', [])
        if len(rp) == len(roads) and h.get('_road') is not None: j = h['_road']      # 아이콘이 붙은 '열린 끝' 도로
        else: j = min(range(len(rl)), key=lambda j: rl[j].distance(Point(h['x'], h['y'])))
        a, b = roads[j]['line']; L = math.dist(a, b); u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        pt = (h['x'], h['y'])
        if fpo.distance(Point(pt[0] + u[0], pt[1] + u[1])) < fpo.distance(Point(pt[0] - u[0], pt[1] - u[1])): u = (-u[0], -u[1])
        lo, hi = 0.0, 3000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if fpo.distance(Point(pt[0] + u[0] * mid, pt[1] + u[1] * mid)) < h['dist_label']: lo = mid
            else: hi = mid
        h['icon_m'] = [h['x'], h['y']]; h['x'], h['y'] = round(pt[0] + u[0] * lo, 2), round(pt[1] + u[1] * lo, 2); h['offmap'] = True
    dims, dds = [], []
    for dm in rec.get('dims', []):
        dd = dict(dm)
        if 'geom_px' in dd and ssrc != 'dims': resolve_dim(dd, s, Z)          # 최종 축척으로 글자 ↔ 기하 길이 다시 맞춤
        dds.append(dd)
    _join_fragments(dds, Z)                                                      # 끊긴 점선 조각 잇기(글자와 이은 길이가 맞을 때만)
    for dd in dds:
        a, b = M(*dd['p0']), M(*dd['p1'])
        dims.append(dict(a=[round(v, 2) for v in a], b=[round(v, 2) for v in b], len_m=round(math.dist(a, b), 2), val=dd.get('val'), val_conf=dd.get('val_conf'), text=dd.get('text'),
                         val_src=dd.get('val_src'), certainty=dd.get('certainty'), geom_m=dd.get('geom_m'), approx=bool(dd.get('approx')), geom_mismatch=bool(dd.get('geom_mismatch')),
                         ends=dd.get('ends'), cands=[c[:2] for c in (dd.get('cands') or [])[:4] if c[1] > 0], tconf=dd.get('tconf'), n_read=dd.get('n_read'), **({'joined': True} if dd.get('joined') else {}),
                         **({'joined_into': dd['joined_into']} if 'joined_into' in dd else {})))
    reg = dict(scale_px_per_m=round(s, 4), scale_src=ssrc, scale_cands={k: round(v, 4) for k, v in srcs.items()}, north_deg=round(rec['north_deg'], 2),
               north_src=rec.get('north_src'), t=[round(tx - ox / s, 3), round(ty - oy / s, 3)])
    return dict(reg=reg, footprint=[[round(x, 2), round(y, 2)] for x, y in list(FP.exterior.coords)[:-1]], entrances=ents, hydrants=hyds, fdc=fdcs, dims=dims,
                roads=roads, checks=checks, M=M)


# ─────────────────────────────── PDF 에서 진압작전도 이미지 찾기(캡션)
def find_in_pdf(pdf):
    """정보카드 PDF → '진압작전도' 캡션 바로 아래 이미지(BGR) 또는 None. (정식 해석기가 info['sitemap'] 을 주면 그것을 먼저 쓴다)"""
    import pymupdf
    doc = pymupdf.open(pdf) if isinstance(pdf, str) else pdf
    for page in doc:
        blocks = [b for b in page.get_text('blocks') if b[6] == 0]
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info['bbox']
            cands = [b for b in blocks if b[3] <= y0 + 2 and y0 - b[3] < 80]
            cap = max(cands, key=lambda b: b[3])[4].strip() if cands else ''
            if '진압작전도' not in cap.replace(' ', ''): continue
            pix = pymupdf.Pixmap(doc, info['xref'])
            if pix.n - pix.alpha >= 4: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
            return arr[..., ::-1].copy()
    return None


def overlay(img, rec, out):
    """검수용 겹침 그림"""
    im = img.copy()
    col = dict(main=(0, 0, 255), exit=(0, 160, 0), hyd=(255, 0, 0), fdc=(200, 0, 200))
    if rec.get('legend'): b = rec['legend']['box']; cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
    for q in rec.get('points', []):
        cv2.circle(im, (int(q['px']), int(q['py'])), int(rec.get('Z', 20) * 0.8), col[q['type']], 2)
        cv2.putText(im, q['type'], (int(q['px']) + 12, int(q['py']) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[q['type']], 1)
    for d in rec.get('dims', []):
        cv2.line(im, tuple(map(int, d['p0'])), tuple(map(int, d['p1'])), (0, 120, 255), 2)
        m = ((d['p0'][0] + d['p1'][0]) / 2, (d['p0'][1] + d['p1'][1]) / 2)
        cv2.putText(im, str(d.get('val')), (int(m[0]), int(m[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
    if rec.get('footprint_px'): cv2.polylines(im, [np.array(rec['footprint_px'], np.int32)], True, (255, 0, 255), 2)
    for r in rec.get('roads', []):
        cv2.line(im, tuple(map(int, r['p0'])), tuple(map(int, r['p1'])), (255, 160, 0), 2)
    if rec.get('north'):
        n = rec['north']; cv2.arrowedLine(im, tuple(map(int, n['px'])), tuple(map(int, n['n'])), (0, 0, 255), 2)
    if rec.get('scalebar'): b = rec['scalebar']['box']; cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cv2.imwrite(out, im)
