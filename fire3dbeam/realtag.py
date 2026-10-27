# -*- coding: utf-8 -*-
"""실제 CAD 평면도의 원형 표기 판독(2026-10-03 2차 강건화) — 창호 태그(위 칸 '폭X높이' · 아래 칸 기호 AW·ASD·SD·FD …)와 그리드 버블(X1·Y2·1·A).

창호 태그 기호(아래 칸)는 글자가 4~8 px 로 작고, 기둥 빗금·문 선·원 둘레가 겹쳐 OCR 이 자주 틀린다(ASD → ES·ASB·빈칸). 그래서
  1) 칸 정리: 원 둘레(고리) 지우기, 진한 글자 문턱(빗금은 연한 회색 150~230 → 고정 문턱 ≤135), 칸을 가로지르는 긴 선·가장자리 덩어리 제거.
  2) 기호 칸 OCR = 대문자 화이트리스트 모자이크(tesseract 1회) + 같은 칸을 굵게 한 판(1회) → 두 판독 중 어휘에 더 가까운 것.
  3) 어휘 맞추기: 그 도면(같은 건물 모든 층) 글자층·OCR 에서 찾은 기호 + 표준 창호 기호 어휘에 OCR 혼동 비용(D↔O·B·0, S↔5·E …)을 준 편집거리로 맞춤.
  4) 본보기 투표: 같은 건물 안에서 확실히 읽힌 태그(어휘와 정확히 일치)의 기호 칸 그림을 본보기로 삼아, 못 읽은 태그는 그림이 가장 닮은 본보기(정규화 상관 ≥0.6)의
     기호를 받음(위 칸 '폭X높이' 가 같으면 가산) — ASD 태그 3개 중 하나만 바로 읽혀도 나머지가 따라옴.
그리드 버블: 축(세로선 버블 = 가로 방향 이름 줄, 가로선 버블 = 세로 방향 줄)마다 1차 판독 다수결로 이름 꼴(숫자 · 영문 · X숫자 · Y숫자)을 정하고
  그 꼴의 화이트리스트로 다시 읽은 뒤, 위치 순서대로 연속 이름(A,B,C… · 1,2,3…, I·O 건너뜀)에 맞춰 빈칸·오독(D·C→0)을 채움(동적 계획, 간격 일관성 검사)."""
import os, re, math
import numpy as np, cv2

STD_CODES = ['AW', 'PW', 'SW', 'WW', 'SSW', 'GW', 'FW', 'CW', 'AFW', 'AD', 'SD', 'WD', 'PD', 'GD', 'SSD', 'ASD', 'AAD', 'SAD', 'FSD', 'FD', 'AFD', 'ARD',
             'SFD', 'SS', 'ST', 'AG']
DOOR_RE = re.compile(r'D$')                 # …D = 문(ASD 자동문·SD 강재문·FD 접이·방화문·WD 목재문), …W = 창
CONF = {frozenset(p): c for p, c in [(('D', 'O'), .35), (('D', '0'), .35), (('D', 'B'), .4), (('D', 'Q'), .45), (('B', '8'), .35), (('S', '5'), .3),
                                       (('S', 'E'), .55), (('S', 'B'), .55), (('A', '4'), .4), (('W', 'V'), .45), (('F', 'E'), .4), (('F', 'P'), .5),
                                       (('S', 'G'), .5), (('D', 'P'), .55), (('A', 'R'), .6), (('I', '1'), .3), (('I', 'L'), .5), (('O', '0'), .2),
                                       (('C', 'G'), .5), (('W', 'M'), .5), (('A', 'H'), .55)]}


def sub_cost(a, b):
    return 0.0 if a == b else CONF.get(frozenset((a, b)), 1.0)


def edit(a, b, ins=0.8):
    """OCR 혼동 비용 편집거리"""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * ins; D[0, :] = np.arange(m + 1) * ins
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + ins, D[i, j - 1] + ins, D[i - 1, j - 1] + sub_cost(a[i - 1], b[j - 1]))
    return float(D[n, m])


def match_code(raw, vocab, sheet=()):
    """OCR 기호 → (어휘 기호, 닮음 0~1). sheet(그 도면에서 본 기호)는 가산. 못 맞추면 (None, 0)"""
    raw = re.sub(r'[^A-Z0-9]', '', (raw or '').upper())
    if not raw: return None, 0.0
    best = (None, 0.0)
    for c in vocab:
        s = 1.0 - edit(raw, c) / max(len(raw), len(c))
        if c in sheet: s += 0.08
        if s > best[1]: best = (c, s)
    return (best[0], round(min(1.0, best[1]), 3)) if best[1] >= 0.5 else (None, 0.0)


# ───────────────────────── 원 안 칸 정리 ─────────────────────────
def part_tile(g, x, y, r, part, up=4):
    """원 안 칸(part='top'|'bottom'|'all') → 깨끗한 이진 타일(검은 글자/흰 바탕, up 배) 과 본보기용 작은 그림(16×40, 0~1) | (None, None)"""
    R = int(math.ceil(r + 3)); x0, y0 = int(round(x)) - R, int(round(y)) - R
    H, Wd = g.shape
    c = np.full((2 * R + 1, 2 * R + 1), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(Wd, x0 + 2 * R + 1), min(H, y0 + 2 * R + 1)
    if sx1 <= sx0 or sy1 <= sy0: return None, None
    c[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = g[sy0:sy1, sx0:sx1]
    yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    d = np.hypot(xx - (x - x0), yy - (y - y0))
    inside = d < r - max(2.5, 0.09 * r)
    vals = c[inside]
    if vals.size < 20: return None, None
    otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T = min(float(otsu), 135.0) if part == 'bottom' else float(otsu)  # 기호 칸: 빗금·회색 칠(150~230)은 글자가 아님 / 치수 칸: 소수점·가는 획 보존
    ink = ((c < T) & inside).astype(np.uint8)
    cy = int(round(y - y0))
    gap = max(1, int(round(0.10 * r)))
    if part == 'top': ink[cy - gap:, :] = 0
    elif part == 'bottom': ink[:cy + gap + 1, :] = 0
    # 칸을 가로지르는 긴 선(문 선·벽·가름선)·원 둘레 잔여 지우기: 칸 폭 85% 넘거나 칸 높이 90% 넘는 덩어리, 면적 2 이하 점
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    ys = np.nonzero(ink.any(1))[0]
    if not len(ys): return None, None
    ph = (cy - gap - (y - y0 - r)) if part == 'top' else ((y - y0 + r) - (cy + gap)) if part == 'bottom' else 2 * r
    keep = np.zeros(n, bool)
    for i in range(1, n):
        bx, by, bw, bh, ar = st[i]
        if ar <= (2 if part == 'bottom' else 0) or bw > 0.85 * 2 * r or bh > 0.9 * ph: continue
        if bh <= 2 and bw > 0.5 * r: continue                         # 가는 가로선 조각
        if bw <= 2 and bh > 0.6 * ph: continue                         # 가는 세로선 조각
        keep[i] = True
    ink = keep[lab].astype(np.uint8)
    ys, xs = np.nonzero(ink)
    if len(ys) < 6: return None, None
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    b = ink[by0:by1, bx0:bx1]
    small = cv2.resize(b.astype(np.float32), (40, 16), interpolation=cv2.INTER_AREA)
    pad = 3
    t = np.full((b.shape[0] + 2 * pad, b.shape[1] + 2 * pad), 255, np.uint8)
    t[pad:-pad, pad:-pad][b > 0] = 0
    t = cv2.resize(t, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    _, t = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)
    return t, small


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


# ───────────────────────── 창호 태그 ─────────────────────────
TAG_M = re.compile(r'(?<![.\d])(\d)\s*[.,]\s*(\d{1,2})\s*[Xx×*]+\s*\d')
TAG_MM = re.compile(r'(?<!\d)([1-9]\d{2,3})\s*[Xx×*]+\s*([1-9]\d{2,3})(?!\d)')


def parse_tag(t):
    t = (t or '').replace(' ', '')
    m = TAG_M.search(t)
    if m:
        w = float(m.group(1) + '.' + m.group(2))
        return w if 0.4 <= w <= 9.0 else None
    m = TAG_MM.search(t)
    if m and int(m.group(2)) >= 500:
        w = float(m.group(1)) / 1000.0
        return w if 0.4 <= w <= 9.0 else None
    return None


def has_divider(g, x, y, r):
    """원 가운데를 가로지르는 가름선(창호 태그 위·아래 칸) 여부: 중심 ±0.12r 줄 중 하나에 지름의 70% 넘는 잉크"""
    H, Wd = g.shape
    x0, x1 = int(round(x - 0.8 * r)), int(round(x + 0.8 * r))
    if x0 < 0 or x1 >= Wd: return False
    for dy in range(-max(1, int(0.12 * r)), max(1, int(0.12 * r)) + 1):
        yy = int(round(y)) + dy
        if 0 <= yy < H and (g[max(0, yy - 1):yy + 2, x0:x1 + 1] < 170).any(0).mean() >= 0.75: return True
    return False


def text_in_circle(text, x, y, r, frac=0.92):
    """글자층 줄 중 가운데가 원 안인 것 → [(줄, 위치 'top'|'bottom'|'mid')]"""
    out = []
    for l in text or ():
        b = l['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if math.hypot(cx - x, cy - y) > frac * r: continue
        if max(b[2] - b[0], b[3] - b[1]) > 2.2 * r: continue                # 원보다 긴 글자(지나가는 주석)
        out.append((l, 'top' if cy < y - 0.12 * r else 'bottom' if cy > y + 0.12 * r else 'mid'))
    return out


def read_tags_text(g, circles, text):
    """글자층이 있는 벡터 도면: 원 안 글자로 창호 태그 판독(OCR 없음) — 위 칸 '폭X높이', 아래 칸 기호(ASD·AW …)"""
    out = []
    for x, y, r in circles:
        T = text_in_circle(text, x, y, r)
        if not T: continue
        top = ' '.join(l['t'] for l, p in sorted(T, key=lambda q: q[0]['box'][0]) if p == 'top')
        bot = ' '.join(l['t'] for l, p in sorted(T, key=lambda q: q[0]['box'][0]) if p == 'bottom')
        mid = ' '.join(l['t'] for l, p in T if p == 'mid')
        w = parse_tag(top) or (parse_tag(mid) if not top else None)
        c = re.sub(r'[^A-Z]', '', bot.upper())
        if not c and mid and w is None:
            c = re.sub(r'[^A-Z]', '', mid.upper()) if CODE_WORD.match(mid.replace(' ', '').upper()) else ''
        if w is None and not c: continue
        split = bool(top and bot) or has_divider(g, x, y, r)
        code0, sim = match_code(c, STD_CODES) if c else (None, 0.0)
        if c and (code0 != c): code0, sim = c, 1.0                      # 글자층 기호는 그대로(표준 어휘 밖 기호 포함)
        if not split and not (w or (len(c) >= 2)): continue
        out.append(dict(x=x, y=y, r=r, W=w, text=top or mid, raw=c, raws=[c], code0=code0, sim0=sim, small=None, split=split, src='pdf'))
    return out


def read_tags(g, circles, text=None):
    """창호 태그 원 → [{x,y,r,W,text,raw,code0,small}] (code0 = 1차 어휘 맞춤, 건물 단위 resolve_codes 로 확정). tesseract 3회(모자이크).
    text(벡터 도면 글자층 줄, fire3dbeam/pdftext.py)가 있으면 원 안 글자로 읽음(OCR 없음)"""
    if text: return read_tags_text(g, circles, text)
    from fire3dbeam.realsym import _ocr_mosaic
    tops, bots, bold, meta = [], [], [], []
    for x, y, r in circles:
        t, _ = part_tile(g, x, y, r, 'top'); b, sm = part_tile(g, x, y, r, 'bottom')
        if t is None and b is None: continue
        tops.append(t if t is not None else np.full((20, 20), 255, np.uint8))
        bots.append(b if b is not None else np.full((20, 20), 255, np.uint8))
        bold.append(cv2.erode(bots[-1], np.ones((3, 3), np.uint8)) if b is not None else bots[-1])
        meta.append((x, y, r, sm))
    if not meta: return []
    txt = _ocr_mosaic(tops, lang='eng', whitelist='0123456789.Xx')
    cod = _ocr_mosaic(bots, lang='eng', whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    cod2 = _ocr_mosaic(bold, lang='eng', whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    out = []
    for (t, _), (c, cf), (c2, cf2), (x, y, r, sm) in zip(txt, cod, cod2, meta):
        w = parse_tag(t)
        c = re.sub(r'[^A-Z]', '', (c or '').upper()); c2 = re.sub(r'[^A-Z]', '', (c2 or '').upper())
        m1, s1 = match_code(c, STD_CODES); m2, s2 = match_code(c2, STD_CODES)
        raw, code0, sim = (c, m1, s1) if s1 >= s2 else (c2, m2, s2)
        if w is None and not raw: continue
        split = has_divider(g, x, y, r)
        if not split and not (w or (len(raw) >= 2 and sim >= 0.99)): continue      # 가름선 없는 원 = 그리드 버블·기호(폭·기호가 확실히 읽힌 홑칸 태그만 예외)
        out.append(dict(x=x, y=y, r=r, W=w, text=t, raw=raw, raws=[c, c2], code0=code0, sim0=sim, small=sm, split=split))
    return out


def resolve_codes(tags, sheet_codes=()):
    """건물 모든 층 태그(같은 dict 를 고침) → code·door·code_src. sheet_codes: 그 도면 글자(글자층·OCR)에서 본 기호(어휘 가산)."""
    vocab = sorted(set(STD_CODES) | set(sheet_codes))
    raws = [t['raw'] for t in tags if t.get('raw')]
    for t in tags:
        best = (None, 0.0)
        for r in t.get('raws') or [t.get('raw')]:
            m, s = match_code(r, vocab, sheet_codes)
            if s > best[1]: best = (m, s)
        t['code1'], t['sim1'] = best
    # 확실한 본보기: 판독 그대로가 어휘 기호 + (도면에서 본 기호이거나 같은 판독이 2번 이상)
    ex = [t for t in tags if t.get('small') is not None and t.get('raw') and t['raw'] in vocab and (t['raw'] in sheet_codes or raws.count(t['raw']) >= 2)]
    for t in tags:
        if t in ex: t['code'], t['code_src'] = t['raw'], 'read'; continue
        cand = None
        if t.get('small') is not None and ex:
            sc = []
            for e in ex:
                v = _ncc(t['small'], e['small'])
                if t.get('text') and e.get('text') and t['W'] and t['W'] == e['W']: v += 0.1
                sc.append((v, e['raw']))
            v, c = max(sc)
            # 본보기 기호들끼리 투표(닮음 0.6 이상 중 최고와 0.1 안의 것들)
            near = [cc for vv, cc in sc if vv >= max(0.6, v - 0.1)]
            if v >= 0.6: cand = (max(set(near), key=near.count), 'exemplar', round(v, 2))
        if cand and (t['sim1'] < 0.99 or cand[0] == t['code1']):
            t['code'], t['code_src'], t['ex_sim'] = cand
        elif t['code1'] and t['sim1'] >= 0.6:
            t['code'], t['code_src'] = t['code1'], 'fuzzy'
        else:
            t['code'], t['code_src'] = None, None
    for t in tags:
        c = t.get('code')
        t['door'] = bool(c) and bool(DOOR_RE.search(c)) and 'W' not in c
        t['window'] = bool(c) and c.endswith('W')
    return tags


CODE_WORD = re.compile(r'^\(?([A-Z]{1,3}[DW])\)?[-_]?\d{0,2}$')


def sheet_codes_from_words(words):
    """도면 글자(글자층·OCR 단어) 중 창호 기호 꼴(…D·…W, 2~4자)인 것 → 집합"""
    out = set()
    for w in words or ():
        t = (w.get('text') if isinstance(w, dict) else w) or ''
        m = CODE_WORD.match(t.replace(' ', '').upper())
        if m and len(m.group(1)) >= 2: out.add(m.group(1))
    return out


# ───────────────────────── 그리드 버블 ─────────────────────────
LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'              # I·O 는 그리드 이름에 안 씀


def _kind(t):
    if re.fullmatch(r'X\d{1,2}', t): return 'X'
    if re.fullmatch(r'Y\d{1,2}', t): return 'Y'
    if re.fullmatch(r'\d{1,2}', t): return 'N'
    if re.fullmatch(r'[A-Z]', t): return 'L'
    return None


def _seq(kind, n=30):
    if kind == 'N': return [str(i) for i in range(1, n + 1)]
    if kind == 'L': return list(LETTERS)
    return [f'{kind}{i}' for i in range(1, n + 1)]


def _assign(reads, kind, pos):
    """위치 순서 reads(문자열|None) → 연속 이름 배정(동적 계획). 각 버블은 다음 이름 또는 '건너뜀'(그리드 버블이 아닌 원·같은 선 중복 — 판독이 없거나 꼴이 다를 때 싸게).
    정방향·역방향 중 점수 큰 쪽. 이름 건너뛰기(빠진 버블)는 간격이 넓을수록 싸게."""
    S = _seq(kind); n = len(reads); m = len(S)
    if n == 0: return None, 0
    gaps = np.diff(pos) if n > 1 else np.array([])
    med = float(np.median(gaps)) if len(gaps) else 1.0
    def sc(r, j):
        if not r: return 0.0
        if r == S[j]: return 1.0
        return -0.6 if _kind(r) == kind else -0.1
    def skip_cost(r):
        if not r: return -0.05
        return -0.15 if _kind(r) != kind else -0.5
    best = None
    for direction in (1, -1):
        R = reads if direction == 1 else reads[::-1]
        # 상태: (마지막 배정 이름 j 또는 -1) — 점수 최대
        NEG = -1e9
        dp = {-1: (0.0, [])}
        G = list(gaps) if direction == 1 else list(gaps[::-1])
        for i in range(n):
            wide = i > 0 and G[i - 1] > 1.5 * med
            nd = {}
            for j0, (v, lab) in dp.items():
                # 건너뜀
                c = (v + skip_cost(R[i]), lab + [None])
                if c[0] > nd.get(j0, (NEG,))[0]: nd[j0] = c
                for j in range(j0 + 1, min(m, j0 + 4)):
                    pen = 0.0 if (j == j0 + 1 or j0 == -1) else (0.25 if wide else 0.8) * (j - j0 - 1)
                    if j0 == -1: pen = 0.05 * j
                    c = (v - pen + sc(R[i], j), lab + [S[j]])
                    if c[0] > nd.get(j, (NEG,))[0]: nd[j] = c
            dp = nd
        v, lab = max(dp.values(), key=lambda q: q[0])
        if direction == -1: lab = lab[::-1]
        hits = sum(1 for a_, b_ in zip(reads, lab) if a_ and a_ == b_)
        if best is None or v > best[0]: best = (v, lab, hits)
    return best[1], best[2]


def read_bubbles(g, circles, vl, hl, return_debug=False, text=None):
    """그리드 버블 → {('v', x): 이름, ('h', y): 이름}. 축마다 이름 꼴 다수결 → 꼴 화이트리스트 재판독 → 순서 맞춤.
    text(벡터 도면 글자층 줄)가 있으면 원 안 글자를 판독값으로(OCR 없음) — 같은 순서 맞춤·중복 검사는 그대로"""
    from fire3dbeam.realsym import _ocr_mosaic
    tiles, meta, treads = [], [], []
    for x, y, r in circles:
        on = [('v', v) for v in vl if abs(x - v) <= max(2.0, 0.25 * r)] + [('h', h) for h in hl if abs(y - h) <= max(2.0, 0.25 * r)]
        if len(on) != 1: continue
        if text:
            T = text_in_circle(text, x, y, r)
            if not T: continue
            tiles.append(None); meta.append((on[0], x, y, r)); treads.append(re.sub(r'[^A-Z0-9]', '', ''.join(l['t'] for l, _ in T).upper()))
            continue
        t, _ = part_tile(g, x, y, r, 'all')
        if t is not None: tiles.append(t); meta.append((on[0], x, y, r))
    if not tiles: return ({}, {}) if return_debug else {}
    if text: reads = treads
    else:
        txt = _ocr_mosaic(tiles, lang='eng', whitelist='ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
        reads = [re.sub(r'[^A-Z0-9]', '', (t or '').upper()) for t, _ in txt]
    out, dbg = {}, {}
    for o in ('v', 'h'):
        idx = [i for i, m in enumerate(meta) if m[0][0] == o]
        if not idx: continue
        # 같은 선 위 버블이 위·아래(양 끝) 둘이면 하나로(이름 같음): 선 좌표로 묶음
        kinds = [_kind(reads[i]) for i in idx if _kind(reads[i])]
        if not kinds: continue
        kind = max(set(kinds), key=kinds.count)
        rr = [meta[i][3] for i in idx if _kind(reads[i]) == kind]
        if len(rr) >= 2:                                         # 같은 축 버블은 같은 크기(±25%) — 작은 기호·큰 원은 버블 아님
            rm = float(np.median(rr)); idx = [i for i in idx if abs(meta[i][3] / rm - 1) <= 0.25]
        lines = {}
        for i in idx: lines.setdefault(meta[i][0], []).append(i)
        wl = {'N': '0123456789', 'L': LETTERS, 'X': 'X0123456789', 'Y': 'Y0123456789'}[kind]
        sub = [tiles[i] for i in idx]
        if not sub: continue
        if text: r2 = {i: reads[i] for i in idx}
        else:
            txt2 = _ocr_mosaic(sub, lang='eng', whitelist=wl, psm=8)
            r2 = {i: re.sub(r'[^A-Z0-9]', '', (t or '').upper()) for i, (t, _) in zip(idx, txt2)}
        keys = sorted(lines, key=lambda k: k[1])
        lr = []
        for k in keys:
            vs = [r2[i] for i in lines[k] if _kind(r2[i]) == kind] + [reads[i] for i in lines[k] if _kind(reads[i]) == kind]
            lr.append(max(set(vs), key=vs.count) if vs else None)
        lab, hits = _assign(lr, kind, [k[1] for k in keys])
        ok = lab is not None and hits >= 2 and hits >= 0.5 * sum(1 for v in lr if v)
        dbg[o] = dict(kind=kind, reads=lr, assigned=lab, hits=hits, ok=ok)
        if ok:
            for k, nm in zip(keys, lab):
                if nm: out[k] = nm
        else:                                    # 순서 맞춤 실패: 예전처럼 꼴에 맞는 판독만, 같은 이름 둘이면 버림
            seen = {}
            for k, v in zip(keys, lr):
                if v: seen[v] = seen.get(v, 0) + 1
            for k, v in zip(keys, lr):
                if v and seen[v] == 1: out[k] = v
    return (out, dbg) if return_debug else out


AXIS_TOK = re.compile(r'^([A-Z]|\d{1,2})$')


def text_axes(lines, min_hits=4):
    """글자층의 원 없는 그리드 이름 줄 → {('v', x): 이름, ('h', y): 이름}.
    같은 줄(가로로 늘어선 숫자·영문 = 세로 그리드선 이름, 세로로 늘어선 것 = 가로선 이름)에서 크기가 비슷한(줄 중앙값의 0.8배 이상 — 위 첨자 제외) 한두 글자 토막을
    위치 순서대로 연속 이름(1,2,3… / A,B,C… I·O 제외)에 동적 계획으로 맞춰 맞은 것이 min_hits 이상인 줄만. 같은 이름이 두 번(위·아래 줄) 나오면 첫 줄만."""
    T = []
    for l in lines or ():
        t = l['t'].replace(' ', '').upper()
        if AXIS_TOK.match(t):
            b = l['box']; T.append((t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(1.0, min(b[2] - b[0], b[3] - b[1]) if len(t) > 1 else max(b[2] - b[0], b[3] - b[1]))))
    out = {}
    for o in ('v', 'h'):                                           # v: 가로 줄(같은 y) → x 위치
        key = (lambda q: q[2]) if o == 'v' else (lambda q: q[1])
        pos = (lambda q: q[1]) if o == 'v' else (lambda q: q[2])
        S = sorted(T, key=key); rows, cur = [], []
        for q in S:
            if cur and abs(key(q) - key(cur[0])) > 0.6 * float(np.median([c[3] for c in cur])): rows.append(cur); cur = []
            cur.append(q)
        if cur: rows.append(cur)
        seen = set()
        for row in rows:
            if len(row) < min_hits: continue
            mh = float(np.median([c[3] for c in row]))
            row = sorted([c for c in row if c[3] >= 0.8 * mh], key=pos)
            kinds = [_kind(c[0]) for c in row]
            for kd in ('N', 'L'):
                sub = [c for c, k_ in zip(row, kinds) if k_ == kd]
                if len(sub) < min_hits: continue
                lab, hits = _assign([c[0] for c in sub], kd, [pos(c) for c in sub])
                if lab is None or hits < min_hits or hits < 0.6 * len(sub): continue
                for c, nm in zip(sub, lab):
                    if nm and nm == c[0] and (o, nm) not in seen:
                        seen.add((o, nm)); out[(o, round(pos(c)))] = nm
    return out

