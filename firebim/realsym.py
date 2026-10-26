# -*- coding: utf-8 -*-
"""실제 평면도(AI허브 건축 도면 등, 회색조 스캔·CAD 출력) 전용 3종 검출기 — 가상 파이프라인과 독립.
  1) 실명 OCR → 실 유형(인명검색 우선순위·대피공간)       detect_rooms()
  2) 여닫이문 호(1/4원 + 문짝선) → 문 폭·경첩·열림 방향       detect_doors()
  3) 계단(등간격 평행 디딤판) · 승강기(X 표시 상자 + EV 글자)  detect_stairs(), detect_elevators()
모든 좌표는 입력 영상 픽셀. 축척(px/m)은 문 호 반경 중앙값(문 폭≈0.9m)으로 자체 추정(scale_from_doors).
"""
import os, re, subprocess, tempfile, difflib
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np, cv2

# ─────────────────────────── 공통 ───────────────────────────
def plan_regions(g, f=0.125, min_frac=0.02):
    """시트(도면 한 장)에서 평면 영역 상자들. 긴 테두리선 제거 후 잉크 덩어리. (간이 — real1 의 시트 배치 해석이 있으면 그것을 쓸 것)"""
    s = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ink = (s < 200).astype(np.uint8)
    H, W = ink.shape
    for k in [(int(W * 0.3), 1), (1, int(H * 0.3))]:
        ink[cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k)) > 0] = 0
    d = cv2.dilate(ink, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w * h > min_frac * W * H and w < 0.95 * W and h < 0.95 * H:
            out.append([int(x / f), int(y / f), int((x + w) / f), int((y + h) / f)])
    return out


def line_mask(g, k=7, thr=40):
    """배경(흰 바탕·회색 칠) 위의 가는 어두운 선만: black-hat. 두꺼운 벽 속은 0."""
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return (bh > thr).astype(np.uint8)


# ─────────────────────────── 1) 실명 OCR ───────────────────────────
# 소방 관점 범주: 침실(인명검색 1순위) / 거실 / 주방 / 욕실·화장실 / 발코니·대피공간 / 현관 / 계단실 / 승강기 / 복도 / 기타
VOCAB = [('침실', '침실'), ('안방', '침실'), ('방', '침실'), ('자녀방', '침실'), ('부부침실', '침실'), ('게스트룸', '침실'), ('작은방', '침실'), ('큰방', '침실'),
         ('거실', '거실'), ('가족실', '거실'), ('라운지', '거실'),
         ('주방', '주방'), ('식당', '주방'), ('부엌', '주방'), ('주방식당', '주방'), ('주방및식당', '주방'), ('주방및거실', '주방'),
         ('욕실', '욕실·화장실'), ('화장실', '욕실·화장실'), ('파우더룸', '욕실·화장실'), ('샤워실', '욕실·화장실'),
         ('발코니', '발코니·대피공간'), ('베란다', '발코니·대피공간'), ('테라스', '발코니·대피공간'), ('데크', '발코니·대피공간'),
         ('대피공간', '발코니·대피공간'), ('대피실', '발코니·대피공간'),
         ('현관', '현관'), ('전실', '현관'),
         ('계단실', '계단실'), ('계단', '계단실'),
         ('승강기', '승강기'), ('엘리베이터', '승강기'), ('승강장', '승강기'),
         ('복도', '복도'), ('홀', '복도'), ('통로', '복도'),
         ('다용도실', '기타'), ('드레스룸', '기타'), ('실외기실', '기타'), ('창고', '기타'), ('팬트리', '기타'), ('서재', '기타'), ('알파룸', '기타'),
         ('보일러실', '기타'), ('세탁실', '기타'), ('수납', '기타'), ('연습실', '기타'), ('포치', '기타'), ('주차장', '기타'), ('작업실', '기타'),
         ('기계실', '기타'), ('전기실', '기타'), ('관리실', '기타'), ('다락', '기타'), ('서비스', '기타')]
# 비주거(복지·업무·교육·의료) 실명(2026-10-03, 공개 실제 도면 시험 후속) — 범주는 기존 10개 안에서(로비→복도, 조리실→주방, 나머지 '기타').
# 소방 용도 범주(인명검색 우선순위)는 실명 글자로 firebim/search.py NAME_VOCAB 이 정함(사무·교육·치료 → 주간 재실, 병실·요양실 → 취침, 기계·전기 → 설비)
VOCAB_NONRES = [('로비', '복도'), ('조리실', '주방'), ('사무실', '기타'), ('회의실', '기타'), ('강당', '기타'), ('프로그램실', '기타'), ('상담실', '기타'),
                ('교실', '기타'), ('강의실', '기타'), ('치료실', '기타'), ('물리치료실', '기타'), ('작업치료실', '기타'), ('재활실', '기타'), ('진료실', '기타'),
                ('병실', '기타'), ('요양실', '기타'), ('간호사실', '기타'), ('숙직실', '기타'), ('휴게실', '기타'), ('대기실', '기타'), ('다목적실', '기타'),
                ('도서실', '기타'), ('열람실', '기타'), ('체력단련실', '기타'), ('수유실', '기타'), ('탈의실', '기타'), ('방재실', '기타'), ('경비실', '기타'),
                ('전산실', '기타'), ('저수조', '기타'), ('펌프실', '기타'), ('발전기실', '기타'), ('물탱크실', '기타'), ('식당', '주방')]
if os.environ.get('REALSYM_NONRES', '1') != '0':
    VOCAB += [v for v in VOCAB_NONRES if v[0] not in {n for n, _ in VOCAB}]
REFUGE = ('대피공간', '대피실')
SEARCH_PRIORITY = {'침실': 1, '거실': 2, '주방': 3, '욕실·화장실': 3, '발코니·대피공간': 2, '현관': 4, '복도': 4, '계단실': 5, '승강기': 5, '기타': 4}
AREA_RE = re.compile(r'(\d{1,3}[.,]\d{1,2})\s*(?:㎡|m2|M2|m²|M²|㎥)')
PYEONG_RE = re.compile(r'(\d{1,3}[.,]\d{1,2})\s*평')


def _norm(t):
    t = re.sub(r'[\s\-_·.,:;\'"|()\[\]{}<>~!?@#$%^&*+=/\\0-9A-Za-z\u3131-\u318e\u2018-\u201f\u2026」『』「]', '', t)
    return t


def _jamo(s):
    """한글 음절 → 초·중·종성 자모열 (유사도 비교용: '첨실'≈'침실')"""
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out += [chr(0x1100 + o // 588), chr(0x1161 + (o % 588) // 28)]
            if o % 28: out.append(chr(0x11A7 + o % 28))
        else: out.append(ch)
    return out


SYL_MATCH = True
STOP_WORDS = {'도로', '차로', '보도', '대지', '도면', '면도', '평면도', '축척'}     # 음절 유사도로 실명에 붙기 쉬운 비실명(도로≈통로)
UNNUMBERED = ('거실', '주방', '식당', '현관', '부엌', '전실', '데크')     # 보통 번호가 붙지 않는 실
# 저해상도 OCR 에서 서로 헷갈리는 자모 묶음(초성·종성 / 중성) — 같은 묶음이면 부분 일치 0.6
_CG = [set('ㄱㅋㄲ'), set('ㄷㅌㄹㄸ'), set('ㅂㅍㅁㅃ'), set('ㅅㅈㅊㅆㅉ'), set('ㅇㅎ')]
_VG = [set('ㅓㅕㅔㅖ'), set('ㅏㅑㅐㅒ'), set('ㅗㅛ'), set('ㅜㅠㅡ'), set('ㅣㅢㅟ'), set('ㅘㅙㅚ'), set('ㅝㅞ')]
_CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
_JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
_JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def _dec(ch):
    o = ord(ch) - 0xAC00
    if not 0 <= o < 11172: return None
    return _CHO[o // 588], _JUNG[(o % 588) // 28], _JONG[o % 28]


def _jsim(a, b, groups):
    if a == b: return 1.0
    return 0.6 if any(a in g and b in g for g in groups) else 0.0


def _syl_sim(a, b):
    da, db = _dec(a), _dec(b)
    if da is None or db is None: return 1.0 if a == b else 0.0
    return 0.35 * _jsim(da[0], db[0], _CG) + 0.35 * _jsim(da[1], db[1], _VG) + 0.3 * _jsim(da[2], db[2], _CG)


def _word_sim(t, n):
    """음절 정렬(DP, 빠짐·덧붙음 허용) 후 음절 유사도 합 / 긴 쪽 길이 — '겨일'≈'거실', '욱실'≈'욕실'"""
    if abs(len(t) - len(n)) > 1: return 0.0
    D = np.zeros((len(t) + 1, len(n) + 1))
    for i in range(1, len(t) + 1):
        for j in range(1, len(n) + 1):
            D[i, j] = max(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1] + _syl_sim(t[i - 1], n[j - 1]))
    return D[-1, -1] / max(len(t), len(n))


def match_room(text):
    """OCR 문자열 → (표준명, 범주, 점수) 또는 None. 포함 일치 우선, 이후 자모 단위 difflib 유사도."""
    t = _norm(text)
    if not t or len(t) > 9 or t in STOP_WORDS: return None
    best = None
    jt = _jamo(t)
    for name, cat in VOCAB:
        if len(name) == 1:                    # '방','홀' 은 정확히 한 글자일 때만
            sc = 1.0 if t == name else 0.0
        elif name in t and len(t) <= len(name) + 2:
            sc = 0.95
        elif len(name) >= 4 and len(t) >= 2 and (name.startswith(t) or name.endswith(t)):
            sc = 0.78                         # 두 줄 표기의 한 줄('드레'/'스룸', '다용'/'도실')
        elif len(t) < max(2, len(name) - 1):
            sc = 0.0
        else:
            jn = _jamo(name)
            sc = difflib.SequenceMatcher(None, jt, jn).ratio()
            if len(t) > len(name) + 1: sc *= 0.85
            if SYL_MATCH and len(name) >= 2:
                ws = _word_sim(t, name)
                if len(name) == 2 and not (set(t) & set(name)) and ws < 0.85: ws = 0   # 2음절 이름은 한 음절 이상 정확 일치 필요('더그'≠'데크')
                sc = max(sc, ws)
        if best is None or sc > best[2]: best = (name, cat, sc)
    thr = 0.72
    if SYL_MATCH and best[2] < 0.95 and re.search(r'\d', text) and best[0] in UNNUMBERED:
        best = (best[0], best[1], best[2] * 0.9)        # 번호 붙은 실(침실3·욕실2)을 거실·현관으로 잘못 읽는 것 방지
    return best if best[2] >= thr else None


def text_blobs(g, passes=((15, 25, 6, 26), (31, 61, 14, 64))):
    """글자 덩어리 찾기(다중 크기). pass=(blackhat 커널, 직선제거 길이, 글자 최소·최대 높이).
    black-hat(국소 대비)로 회색 칠 바탕 위 글자도 잡고 두꺼운 벽은 뺀 뒤, 긴 수평·수직선(테두리·격자·치수선)을 지우고
    글자 크기 성분을 같은 줄·비슷한 높이·가까운 간격으로 묶는다. 반환: [{box, n, h, mask(상자 크기 이진)}]"""
    out = []
    for k, L, hmin, hmax in passes:
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        ink = (bh > 60).astype(np.uint8)
        for kk in [(L, 1), (1, L)]:
            ink[cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kk)) > 0] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        X, Y, Wc, Hc, Ac = [st[1:, j].astype(np.int64) for j in range(5)]
        mx, mn_ = np.maximum(Wc, Hc), np.minimum(Wc, Hc)
        sel = (mx <= hmax) & (mx >= 3) & (Ac >= 4) & ~((mn_ <= 2) & (mx > hmax * 0.9))
        ids = list(np.nonzero(sel)[0] + 1)
        if not ids: continue
        cs = np.c_[X[sel], Y[sel], X[sel] + Wc[sel], Y[sel] + Hc[sel]]
        ww, hh = cs[:, 2] - cs[:, 0], cs[:, 3] - cs[:, 1]
        thin = (np.minimum(ww, hh) <= np.maximum(2, 0.2 * np.maximum(ww, hh))) & (np.maximum(ww, hh) >= 5) | (np.maximum(ww, hh) <= 3)
        par = list(range(len(cs)))
        def f(i):
            while par[i] != i: par[i] = par[par[i]]; i = par[i]
            return i
        solid = np.where(~thin)[0]
        order = solid[np.argsort(cs[solid, 0])]
        S = cs[order].astype(np.int64)                            # x0 정렬
        x0s = S[:, 0]
        for ii in range(len(order)):                              # (벡터화: 같은 결과, 큰 도면에서 수 배 빠름)
            a = S[ii]; ha = a[3] - a[1]
            je = int(np.searchsorted(x0s, a[2] + 1.4 * hmax, side='right'))
            if je <= ii + 1: continue
            B = S[ii + 1:je]
            B = B[(B[:, 1] < a[3] + 2 * hmax) & (B[:, 3] > a[1] - 2 * hmax)]
            if not len(B): continue
            jidx = np.nonzero((S[ii + 1:je, 1] < a[3] + 2 * hmax) & (S[ii + 1:je, 3] > a[1] - 2 * hmax))[0] + ii + 1
            hb = B[:, 3] - B[:, 1]
            ov = np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1])
            oxx = np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])
            hm = np.maximum(np.maximum(ha, hb), hmin)
            mn = np.minimum(ha, hb)
            c1 = (ov > 0.2 * mn) & (B[:, 0] - a[2] < 1.2 * hm) & (hm < 4 * np.maximum(mn, 3))          # 같은 줄 이웃 글자·자모
            c2 = (-0.45 * hm < ov) & (oxx > -0.15 * hm) & (np.maximum(a[2] - a[0], B[:, 2] - B[:, 0]) < 2.5 * hm)   # 한 글자 안 위·아래 자모(받침)
            for jj in jidx[c1 | c2]:
                par[f(order[ii])] = f(order[jj])
        # 가는 획(ㅣ·ㅡ 등)은 글자끼리 잇는 데 쓰지 않고(점선·테두리 사슬 방지) 이미 만들어진 묶음 상자 안쪽이면 붙인다
        gb = {}
        for i in solid: gb.setdefault(f(i), []).append(i)
        boxes = {r: (cs[v, 0].min(), cs[v, 1].min(), cs[v, 2].max(), cs[v, 3].max()) for r, v in gb.items()}
        if boxes:
            rk = list(boxes.keys()); BB = np.array([boxes[r] for r in rk], float)
            Mg = 0.5 * np.maximum(BB[:, 3] - BB[:, 1], 1.5 * hmin)
            for i in np.where(thin)[0]:
                x0, y0, x1, y1 = cs[i]
                hit = np.nonzero((x0 >= BB[:, 0] - Mg) & (x1 <= BB[:, 2] + Mg) & (y0 >= BB[:, 1] - 0.25 * Mg) & (y1 <= BB[:, 3] + 0.25 * Mg))[0]
                if len(hit): par[i] = rk[hit[0]]
        groups = {}
        for i in range(len(cs)):
            if thin[i] and par[i] == i: continue
            groups.setdefault(f(i), []).append(i)
        for idx in groups.values():
            b = cs[idx]
            x0, y0, x1, y1 = int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())
            w, h = x1 - x0, y1 - y0
            if h < hmin or h > 1.6 * hmax or w < hmin or w > 14 * hmax: continue
            if len(idx) < 2 and w < 1.5 * h: continue
            m = np.isin(lab[y0:y1, x0:x1], [ids[i] for i in idx]).astype(np.uint8)
            if m.sum() < 0.06 * w * h: continue
            out.append(dict(box=[x0, y0, x1, y1], n=len(idx), h=h, mask=m, pas=k))
    # 두 크기 통과에서 겹치는 상자: 큰 통과 쪽이 작은 걸 완전히 포함하면 둘 다 유지(OCR 후 점수로 선택)
    return out


def _vertical_merge(blobs):
    """세로쓰기(글자 위아래 나열: '발/코/니') 후보: 가로폭 비슷·x 겹침·세로로 가까운 단일 글자 상자 묶음."""
    B = sorted([b for b in blobs if 0.6 < (b['box'][2] - b['box'][0]) / max(1, b['box'][3] - b['box'][1]) < 1.7], key=lambda b: b['box'][1])
    out, used = [], set()
    for i, a in enumerate(B):
        if i in used: continue
        chain = [a]; cur = a
        for j in range(i + 1, len(B)):
            if j in used: continue
            b = B[j]; ca, cb = cur['box'], b['box']
            h = ca[3] - ca[1]
            if cb[1] - ca[3] > 0.7 * h: continue
            if abs((ca[0] + ca[2]) - (cb[0] + cb[2])) < 0.5 * (ca[2] - ca[0]) and abs((cb[3] - cb[1]) - h) < 0.5 * h and cb[1] >= ca[3] - 0.2 * h:
                chain.append(b); used.add(j); cur = b
        if len(chain) >= 2:
            bx = [min(c['box'][0] for c in chain), chain[0]['box'][1], max(c['box'][2] for c in chain), chain[-1]['box'][3]]
            out.append(dict(box=bx, n=len(chain), vert=True, chars=chain))
    return out


OCR_LANG = os.environ.get('REALSYM_OCR_LANG', 'kor')
GRAY_TILE = os.environ.get('REALSYM_GRAY_TILE', '0') == '1'   # 회색 타일 확대 후 이진화(SPA 개발 표본에서 이득 없음 → 기본 끔)
ROOM_THR = 0.55


def _ocr_mosaic(tiles, lang=None, psm=7, chunk=160, whitelist=None):
    """여러 글자 상자를 한 장(세로로 쌓기)에 붙여 tesseract 1회 호출 → 타일별 문자열. (타일마다 호출하면 느림)
    tesseract 는 높이 32767 px 넘는 영상을 처리하지 못함 → chunk 개씩 나눠 호출."""
    if not tiles: return []
    lang = lang or OCR_LANG
    if len(tiles) > chunk:
        out = []
        for i in range(0, len(tiles), chunk): out += _ocr_mosaic(tiles[i:i + chunk], lang, psm, chunk, whitelist)
        return out
    TH, PAD = 56, 28
    rows, ys = [], []
    y = PAD
    Wm = 0
    for t in tiles:
        h, w = t.shape
        s = TH / max(h, 1)
        r = cv2.resize(t, (max(1, int(w * s)), TH), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if GRAY_TILE and r.size and (t[(t > 0) & (t < 255)].size > 0.05 * t.size):     # 회색 타일: 확대 후 이진화(계단 모양 글자 방지)
            _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rows.append(r); ys.append(y); y += TH + PAD; Wm = max(Wm, r.shape[1])
    M = np.full((y + PAD, Wm + 2 * PAD), 255, np.uint8)
    for r, yy in zip(rows, ys): M[yy:yy + TH, PAD:PAD + r.shape[1]] = r
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, M); fn = tf.name
    try:
        wl = ['-c', 'tessedit_char_whitelist=' + whitelist] if whitelist else []
        out = subprocess.run(['tesseract', fn, '-', '-l', lang, '--psm', '6', '-c', 'preserve_interword_spaces=1'] + wl + ['tsv'],
                             capture_output=True, text=True, timeout=180).stdout
    finally:
        os.unlink(fn)
    res = [[] for _ in tiles]
    for ln in out.splitlines()[1:]:
        p = ln.split('\t')
        if len(p) == 12 and p[11].strip():
            cy = int(p[7]) + int(p[9]) / 2
            k = int(np.argmin([abs(cy - (yy + TH / 2)) for yy in ys]))
            if abs(cy - (ys[k] + TH / 2)) < TH: res[k].append((int(p[6]), p[11].strip(), float(p[10])))
    return [(' '.join(t for _, t, _ in sorted(r)), float(np.mean([c for _, _, c in r])) if r else -1) for r in res]


def _clean_tile(g, box, pad=4):
    x0, y0, x1, y1 = box
    H, W = g.shape
    t = g[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)].copy()
    if GRAY_TILE: return t                # 이진화는 모자이크에서 확대 후
    # 글자 상자 안 배경 회색 칠(APT 분양도) → 흰색으로: 오츠 이진화
    _, b = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def detect_rooms(g, box=None, want_area=True, up=1.0):
    """실명 라벨 → [{text,name,cat,refuge,priority,xy,box,score,area_m2?,pyeong?}]  (xy=원 영상 좌표)
    up>1: 글자가 작은 저해상도 도면(글자 높이 < 12 px, 예: A3 2048 px 축소본)은 up 배 확대 후 검출(좌표는 원 영상으로 환산)."""
    if up and up != 1.0:
        sub = g[box[1]:box[3], box[0]:box[2]] if box else g
        u = cv2.resize(sub, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        keep, cands = detect_rooms(u, None, want_area, 1.0)
        ox, oy = (box[0], box[1]) if box else (0, 0)
        for c in cands:
            c['box'] = [c['box'][0] / up + ox, c['box'][1] / up + oy, c['box'][2] / up + ox, c['box'][3] / up + oy]
            c['xy'] = [c['xy'][0] / up + ox, c['xy'][1] / up + oy]
        return keep, cands
    ox, oy = (box[0], box[1]) if box else (0, 0)
    sub = g[box[1]:box[3], box[0]:box[2]] if box else g
    blobs = text_blobs(sub)
    vb = _vertical_merge([b for b in blobs if b['n'] <= 4])
    tiles, meta = [], []
    def mtile(m, pad=6):                  # 묶인 성분만 그린 깨끗한 타일(검은 글자/흰 바탕)
        t = np.full((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), 255, np.uint8)
        t[pad:-pad, pad:-pad][m > 0] = 0
        return t
    for b in blobs:
        x0, y0, x1, y1 = b['box']
        w, h = x1 - x0, y1 - y0
        if h > 1.8 * w and b['n'] >= 2:   # 90° 회전 글자(세로 방향 가로쓰기)
            t = mtile(b['mask'])
            tiles.append(cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)); meta.append((b, 'r1'))
            tiles.append(cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)); meta.append((b, 'r2'))
            continue
        tiles.append(mtile(b['mask'])); meta.append((b, 'h'))
        tiles.append(_clean_tile(sub, b['box'])); meta.append((b, 'g'))
    for b in vb:                          # 세로쓰기 → 글자를 가로로 이어붙임
        cs = [mtile(c['mask'], 3) for c in b['chars']]
        h = max(c.shape[0] for c in cs)
        cs = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=255) for c in cs]
        tiles.append(np.hstack(cs)); meta.append((b, 'v'))
    txt = _ocr_mosaic(tiles)
    items = []
    for (b, how), (t, conf) in zip(meta, txt):
        x0, y0, x1, y1 = b['box']
        items.append(dict(text=t, box=[x0 + ox, y0 + oy, x1 + ox, y1 + oy], xy=[(x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy], how=how, conf=conf))
    return select_rooms(items, want_area)


def _room_score(m, conf):
    return round(m[2] * (0.6 + 0.4 * max(0, min(conf, 90)) / 90), 3)


def select_rooms(items, want_area=True):
    """OCR 결과 [{text,box,xy,how,conf}] → (실명 keep, 전체 후보 cands). 어휘 매칭·두 줄 결합·겹침 정리·면적 파싱.
    (OCR 과 분리 → 캐시된 OCR 결과로 어휘·임계 재조정 가능: evalr/spa_eval.py --rematch)"""
    cands = []
    for it in items:
        item = dict(it); t, conf = item['text'], item['conf']
        m = match_room(t)
        if m:
            item.update(name=m[0], cat=m[1], score=_room_score(m, conf), refuge=m[0] in REFUGE, priority=SEARCH_PRIORITY[m[1]])
        cands.append(item)
    # 두 줄 표기(주방/식당, 드레/스룸, 다용/도실): 위아래로 붙은 상자 쌍의 이어붙인 글자도 후보로
    hs = [c for c in cands if c['how'] in ('h', 'g') and c['text'].strip()]
    for i, a_ in enumerate(hs):
        for b_ in hs:
            if b_ is a_: continue
            A, B = a_['box'], b_['box']
            h = A[3] - A[1]
            if 0 <= B[1] - A[3] < 0.9 * h and abs((A[0] + A[2]) - (B[0] + B[2])) < 0.6 * max(A[2] - A[0], B[2] - B[0]) and abs((B[3] - B[1]) - h) < 0.5 * h:
                t = a_['text'] + b_['text']
                m = match_room(t)
                if m and (m[2] > max(a_.get('score', 0), b_.get('score', 0)) + 0.05 or len(m[0]) > max(len(a_.get('name', '')), len(b_.get('name', '')))):
                    bx = [min(A[0], B[0]), A[1], max(A[2], B[2]), B[3]]
                    conf = min(a_['conf'], b_['conf'])
                    cands.append(dict(text=t, box=bx, xy=[(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2], how='2line', conf=conf, name=m[0], cat=m[1],
                                      score=_room_score(m, conf) + 0.02, refuge=m[0] in REFUGE, priority=SEARCH_PRIORITY[m[1]]))
    rooms = [c for c in cands if 'name' in c and c['score'] >= ROOM_THR]
    # 겹치는 후보(가로/회전/세로) 중 최고 점수만
    rooms.sort(key=lambda c: -c['score'])
    keep = []
    for c in rooms:
        if all(not _ovl(c['box'], k['box']) for k in keep): keep.append(c)
    if want_area:
        for c in cands:
            a = AREA_RE.search(c['text'].replace(' ', '')); p = PYEONG_RE.search(c['text'].replace(' ', ''))
            if not (a or p): continue
            best = min(keep, key=lambda k: np.hypot(k['xy'][0] - c['xy'][0], k['xy'][1] - c['xy'][1]), default=None)
            if best is not None and np.hypot(best['xy'][0] - c['xy'][0], best['xy'][1] - c['xy'][1]) < 6 * (c['box'][3] - c['box'][1]):
                if a: best['area_m2'] = float(a.group(1).replace(',', '.'))
                if p: best['pyeong'] = float(p.group(1).replace(',', '.'))
    return keep, cands


def _ovl(a, b, m=0):
    return a[0] < b[2] + m and b[0] < a[2] + m and a[1] < b[3] + m and b[1] < a[3] + m


# ─────────────────────────── 2) 여닫이문 호 ───────────────────────────
def _fit_circle(p):
    """대수적 최소제곱 원 적합 (Kasa). p: (N,2) float → (cx, cy, r)"""
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(p))]
    b = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    return (c[0], c[1], np.sqrt(r2)) if r2 > 0 else None


def _fit_ellipse_aa(p):
    """축 정렬 타원 A x² + C y² + D x + E y = 1 최소제곱 → (cx, cy, rx, ry). (원 포함; 분양 도면의 찌그러진 문 호 대응)"""
    x, y = p[:, 0], p[:, 1]
    mx, my = x.mean(), y.mean(); s = max(x.std() + y.std(), 1.0)
    xn, yn = (x - mx) / s, (y - my) / s
    M = np.c_[xn * xn, yn * yn, xn, yn]
    try:
        c, *_ = np.linalg.lstsq(M, np.ones(len(p)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, C, D, E = c
    if A <= 1e-9 or C <= 1e-9: return None
    cx, cy = -D / (2 * A), -E / (2 * C)
    F = 1 + D * D / (4 * A) + E * E / (4 * C)
    if F <= 0: return None
    rx, ry = np.sqrt(F / A) * s, np.sqrt(F / C) * s
    return (cx * s + mx, cy * s + my, rx, ry)


def _edist(p, c):
    cx, cy, rx, ry = c
    return np.abs(np.hypot((p[:, 0] - cx) / rx, (p[:, 1] - cy) / ry) - 1) * min(rx, ry)


def _ransac_circles(p, rmin, rmax, rng, iters=90, max_circ=3):
    """한 성분 안의 원·타원호(여럿 가능) — RANSAC(4점 축정렬 타원) 후 인라이어 재적합."""
    out = []
    p = p.astype(np.float64)
    ok = lambda c: c is not None and rmin <= min(c[2], c[3]) and max(c[2], c[3]) <= rmax and 0.62 <= c[2] / c[3] <= 1.6
    for _ in range(max_circ):
        if len(p) < 12: break
        best, bi = None, None
        for _ in range(iters):
            c = _fit_ellipse_aa(p[rng.choice(len(p), 4, replace=False)])
            if not ok(c): continue
            inl = _edist(p, c) < max(1.8, 0.03 * min(c[2], c[3]))
            if best is None or inl.sum() > bi.sum(): best, bi = c, inl
        if best is None or bi.sum() < 12: break
        c = _fit_ellipse_aa(p[bi])
        if not ok(c): break
        ang = np.degrees(np.arctan2((p[bi][:, 1] - c[1]) / c[3], (p[bi][:, 0] - c[0]) / c[2])) % 360
        h = np.histogram(ang, bins=36, range=(0, 360))[0] > 0
        if h.sum() >= 4: out.append((c, int(bi.sum()), int(h.sum()) * 10))
        p = p[~bi]
    return out


def _cover(m, pts):
    H, W = m.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    ins = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    return (m[y, x] > 0) & ins


def _verify_door(lmd, ink, cx, cy, rx, ry):
    """타원 (cx,cy,rx,ry) 이 여닫이문인지: 90° 호(4분면 중 하나)만 칠해지고, 경계 반지름 한쪽에 문짝선, 1/4원 안은 비어 있음."""
    th = np.radians(np.arange(0, 360, 3))
    ring = np.c_[cx + rx * np.cos(th), cy + ry * np.sin(th)]
    cov = _cover(lmd, ring).astype(float)                       # 120 bins
    best = None
    for a0 in (0, 90, 180, 270):
        idx_in = [(a0 + d) // 3 % 120 for d in range(9, 82, 3)]
        idx_out = [(a0 + d) // 3 % 120 for d in range(111, 340, 3)]
        cin, cout = cov[idx_in].mean(), cov[idx_out].mean()
        if cin < 0.62 or cout > 0.42: continue
        leaf = []
        for ad in (a0, a0 + 90):
            t = np.linspace(0.12, 0.92, 24) * (rx if ad % 180 == 0 else ry)
            u = np.radians(ad)
            leaf.append(_cover(lmd, np.c_[cx + t * np.cos(u), cy + t * np.sin(u)]).mean())
        k = int(np.argmax(leaf))
        if leaf[k] < 0.6: continue
        # 1/4원 내부 비어 있음(해치·글자·가구 아님)
        rr, aa = np.meshgrid(np.linspace(0.25, 0.8, 6), np.radians(a0 + np.linspace(18, 72, 7)))
        inside = _cover(ink, np.c_[cx + (rx * rr * np.cos(aa)).ravel(), cy + (ry * rr * np.sin(aa)).ravel()]).mean()
        if inside > 0.3: continue
        sc = cin - 0.5 * cout + 0.3 * leaf[k] - 0.5 * inside
        if best is None or sc > best['score']:
            best = dict(a0=a0, cin=round(cin, 2), cout=round(cout, 2), leaf=round(leaf[k], 2), inside=round(inside, 2), score=round(sc, 3),
                        leaf_dir=(a0 + 90 * k) % 360, closed_dir=(a0 + 90 * (1 - k)) % 360)
    return best


def detect_doors(g, box=None, rmin=16, rmax=320, seed=0):
    """여닫이문: 곡선 성분 → RANSAC 원 적합 → 4분면 호·문짝선·빈 내부 검증 → 반경 일관성(중앙값 대비) 필터.
    반환 [{hinge, r(px)=문 폭, a0(호 시작각°, 화면좌표 시계방향), leaf_dir, closed_dir, xy(호 상자 중심), swing}]"""
    ox, oy = (box[0], box[1]) if box else (0, 0)
    sub = g[box[1]:box[3], box[0]:box[2]] if box else g
    lm = line_mask(sub, k=9, thr=35)
    lmd = cv2.dilate(lm, np.ones((3, 3), np.uint8))
    ink = (cv2.morphologyEx(sub, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))) > 35).astype(np.uint8)
    L = 21
    straight = cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) | \
               cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    curve = lm & (1 - cv2.dilate(straight, np.ones((3, 3), np.uint8)))
    curve = cv2.morphologyEx(curve, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(curve, 8)
    rng = np.random.default_rng(seed)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 14 or max(w, h) < 0.45 * rmin or max(w, h) > 2.2 * rmax: continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        p = np.c_[xs + x, ys + y]
        if len(p) > 600: p = p[rng.choice(len(p), 600, replace=False)]
        for (cx, cy, rx, ry), ninl, span in _ransac_circles(p, rmin, rmax, rng):
            if span < 40 or span > 130: continue
            v = _verify_door(lmd, ink, cx, cy, rx, ry)
            if not v:                     # 재적합: 곡선만 남기며 잘린 접선 부분까지 선 마스크의 고리 영역 점으로 다시 맞춤
                for _ in range(2):
                    R = 1.25 * max(rx, ry)
                    xa, ya = int(max(0, cx - R)), int(max(0, cy - R))
                    yy, xx = np.nonzero(lm[ya:int(cy + R) + 1, xa:int(cx + R) + 1])
                    q = np.c_[xx + xa, yy + ya].astype(float)
                    if len(q) < 12: break
                    q = q[np.abs(np.hypot((q[:, 0] - cx) / rx, (q[:, 1] - cy) / ry) - 1) < 0.22]
                    if len(q) > 800: q = q[rng.choice(len(q), 800, replace=False)]
                    rf = _ransac_circles(q, rmin, rmax, rng, iters=60, max_circ=1)
                    if not rf: break
                    (cx, cy, rx, ry), _, span2 = rf[0]
                    v = _verify_door(lmd, ink, cx, cy, rx, ry)
                    if v: break
            if v:
                r = rx if v['closed_dir'] % 180 == 0 else ry          # 문 폭 = 닫힌 방향(문틀 쪽) 반축
                cands.append(dict(hinge=[cx, cy], r=r, rx=rx, ry=ry, **v))
    # 같은 문 중복 제거
    cands.sort(key=lambda d: -d['score'])
    keep = []
    for d in cands:
        if all(np.hypot(d['hinge'][0] - k['hinge'][0], d['hinge'][1] - k['hinge'][1]) > 0.35 * k['r'] or abs(d['r'] - k['r']) > 0.3 * k['r'] for k in keep):
            keep.append(d)
    if len(keep) >= 3:                                   # 도면 안 문 폭 일관성(축척 무관 자기보정)
        med = float(np.median([d['r'] for d in keep]))
        keep = [d for d in keep if 0.5 * med <= d['r'] <= 1.9 * med]
    for d in keep:
        u0, u1 = np.radians(d['a0']), np.radians(d['a0'] + 90)
        cx, cy = d['hinge']
        d['xy'] = [cx + 0.5 * d['rx'] * (np.cos(u0) + np.cos(u1)) + ox, cy + 0.5 * d['ry'] * (np.sin(u0) + np.sin(u1)) + oy]
        d['hinge'] = [cx + ox, cy + oy]
        d['swing'] = {0: 'SE', 90: 'SW', 180: 'NW', 270: 'NE'}[d['a0']]      # 열리는 1/4면(화면 기준)
    return keep


def scale_from_doors(doors, door_m=0.9):
    """문 호 반경 중앙값 ≈ 0.9 m 가정 → px/m (문 3개 이상일 때만)"""
    if len(doors) < 3: return None
    return float(np.median([d['r'] for d in doors])) / door_m


# ─────────────────────────── 3) 계단 · 승강기 ───────────────────────────
STAIR_WORDS = re.compile(r'(UP|DN|D\.?N|U\.?P|계단|올라|내려)', re.I)
EV_WORDS = re.compile(r'(E\s*/\s*V|E\.V|EV|ELEV|엘리베|승강기|승강로|인승|비상용|장애인)', re.I)
EV_P = dict(minlen=30, gap=10, merge=6, tol=0.12)    # 승강기 X 대각선: Hough 최소길이·끊김 허용, 조각 병합 여유, 대각선-상자 일치 허용비
NOT_EV = re.compile(r'(OPEN|VOID|DUCT|PIT|\bPS\b|\bAD\b|\bDA\b|정화조|덕트|샤프트|오수|배기|급기|환기|보일러|수납|창고|욕조)', re.I)


def _segments(lm, o, Lmin):
    """축 방향 선분 목록 [(a0,a1,c)] — o='h': x구간(a0,a1), y=c"""
    k = (Lmin, 1) if o == 'h' else (1, Lmin)
    m = cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if o == 'h' and h <= 6: out.append((x, x + w, cen[i][1]))
        if o == 'v' and w <= 6: out.append((y, y + h, cen[i][0]))
    return out


def _runs(segs, min_n):
    """비슷한 구간(a0,a1)의 평행선들 중 등간격으로 min_n 개 이상 이어지는 묶음 → [(a0,a1,c0,c1,n,gap)]"""
    out = []
    segs = sorted(segs, key=lambda s: s[2])
    used = set()
    for i, s in enumerate(segs):
        if i in used: continue
        L = s[1] - s[0]
        if L < 12: continue
        tol = 0.15 * L + 3
        sim = [j for j, t in enumerate(segs) if abs(t[0] - s[0]) <= tol and abs(t[1] - s[1]) <= tol and t[2] >= s[2] - 0.5]
        cs = [segs[j][2] for j in sim]
        # 같은 c(두 줄 겹침) 제거
        uc, uj = [], []
        for c, j in zip(cs, sim):
            if not uc or c - uc[-1] > 2.5: uc.append(c); uj.append(j)
        if len(uc) < min_n: continue
        gaps = np.diff(uc)
        # 첫 선에서 시작하는 등간격 연쇄
        best = None
        for k0 in range(0, len(gaps)):
            g0 = gaps[k0]
            if not (max(6, 0.09 * L) <= g0 <= 0.65 * L): continue
            chain = [k0]
            for k in range(k0 + 1, len(gaps)):
                if 0.72 * g0 <= gaps[k] <= 1.33 * g0: chain.append(k)
                else: break
            if best is None or len(chain) > len(best): best = chain
            if k0 > 0: break                     # 시작점 s 에서만(나머지 시작점은 다른 i 가 담당)
        if best and len(best) + 1 >= min_n:
            idx = [uj[best[0]]] + [uj[k + 1] for k in best]
            used.update(idx)
            a0 = min(segs[j][0] for j in idx); a1 = max(segs[j][1] for j in idx)
            out.append((a0, a1, uc[best[0]], uc[best[-1] + 1], len(idx), float(np.median(gaps[best]))))
    return out


def detect_stairs(g, box=None, words=(), ppm=None, min_n=5, min_area_m2=0.0):
    """계단: 길이가 같고 등간격인 짧은 평행선(디딤판) min_n 개 이상. 타일·격자(직교선 다수) 제외, UP/DN/계단 글자 근처면 4개도 인정."""
    ox, oy = (box[0], box[1]) if box else (0, 0)
    sub = g[box[1]:box[3], box[0]:box[2]] if box else g
    lm = line_mask(sub, k=9, thr=35)
    tw = [w for w in words if STAIR_WORDS.search(w.get('text', '').replace(' ', ''))]
    cands = []
    segH, segV = _segments(lm, 'h', 12), _segments(lm, 'v', 12)
    for o, segs, perp in (('h', segH, segV), ('v', segV, segH)):
        for a0, a1, c0, c1, n, gap in _runs(segs, min(4, min_n)):
            bx = [a0, c0, a1, c1] if o == 'h' else [c0, a0, c1, a1]
            L = a1 - a0; D = c1 - c0
            # 격자 판정: 디딤판과 직교하는 선이 상자 안을 D*0.6 이상 가로지르는 개수
            cross = sum(1 for p0, p1, pc in perp if a0 + 0.1 * L < pc < a1 - 0.1 * L and min(p1, c1) - max(p0, c0) > 0.6 * D)
            if cross >= 3: continue
            # 벽돌·타일 줄눈: 상자 안 짧은 직교 선분이 디딤판 수보다 많으면 무늬
            joints = sum(1 for p0, p1, pc in perp if a0 + 0.05 * L < pc < a1 - 0.05 * L and c0 - 2 <= p0 and p1 <= c1 + 2 and p1 - p0 < 0.6 * D)
            if joints > 1.2 * n: continue
            near = any(_ovl([w['box'][0] - ox, w['box'][1] - oy, w['box'][2] - ox, w['box'][3] - oy], bx, m=0.4 * max(L, D)) for w in tw)
            walk = 1 <= cross <= 2                   # 디딤판을 가로지르는 보행선·화살표
            if ppm:                                  # 축척(문 호) 있으면 실치수: 디딤판 깊이 0.17~0.45 m, 폭 0.6~3.2 m
                if not (0.17 <= gap / ppm <= 0.45 and 0.6 <= L / ppm <= 3.2): continue
                if not (near or walk) and not (0.21 <= gap / ppm <= 0.4 and n <= 18): continue   # 난간살·데크(0.1~0.18 m 간격) 배제
            elif not (near or walk) and n > 18: continue
            if n < min_n and not (near or walk): continue
            cands.append(dict(box=bx, n=n, o=o, gap=round(gap, 1), tread_len=int(L), label=near, walk=walk))
    # 같은 계단의 여러 흐름(꺾인 계단) 병합
    merged = True
    while merged:
        merged = False
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                A, B = cands[i]['box'], cands[j]['box']
                m = 0.6 * max(cands[i]['gap'], cands[j]['gap']) + 4
                if _ovl(A, B, m=m):
                    cands[i]['box'] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                    cands[i]['n'] += cands[j]['n']; cands[i]['label'] |= cands[j]['label']
                    cands.pop(j); merged = True; break
            if merged: break
    if ppm and min_area_m2:               # 병합 뒤 계단 전체 면적(실치수) 하한: 난간살·줄눈 조각 등 작은 오검출 제거(글자 확인된 것은 유지)
        cands = [c for c in cands if c['label'] or (c['box'][2] - c['box'][0]) * (c['box'][3] - c['box'][1]) / ppm ** 2 >= min_area_m2]
    for c in cands:
        c['box'] = [int(c['box'][0] + ox), int(c['box'][1] + oy), int(c['box'][2] + ox), int(c['box'][3] + oy)]
    return cands


def detect_elevators(g, box=None, words=(), ppm=None):
    """승강기: 대각선 X 가 꽉 찬 직사각형(네 변 존재). 크기(문 호로 추정한 축척 1.1~3.2 m, 가로세로비≤1.7) 또는 EV 글자로 확정.
    덕트(PS·AD, 작음)·보이드(OPEN, 큼)는 크기로 제외."""
    ox, oy = (box[0], box[1]) if box else (0, 0)
    sub = g[box[1]:box[3], box[0]:box[2]] if box else g
    lm = line_mask(sub, k=9, thr=18)                 # X 표시는 흐린 회색선인 경우가 많음
    lmd = cv2.dilate(lm, np.ones((3, 3), np.uint8))
    L = 25
    straight = cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) | \
               cv2.morphologyEx(lm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    diag = lm & (1 - straight)
    ls = cv2.HoughLinesP(diag * 255, 1, np.pi / 180, 30, minLineLength=EV_P['minlen'], maxLineGap=EV_P['gap'])
    if ls is None: return []
    P, N = [], []
    for x1, y1, x2, y2 in ls[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx)) % 180
        if 15 < ang < 75: P.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))       # ↘
        elif 105 < ang < 165: N.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))   # ↙
    ew = [w for w in words if EV_WORDS.search(w.get('text', '').replace(' ', ''))]
    out = []
    # 같은 대각선 조각 합치기(끊긴 X) — 상자 겹침 기준
    def merge(S):
        S = [list(s) for s in S]
        ch = True
        while ch:
            ch = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    if _ovl(a, b, m=EV_P['merge']) and iou_like(a, b):
                        S[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]; S.pop(j); ch = True; break
                if ch: break
        return S
    def iou_like(a, b):             # 같은 직선 위(기울기 비슷)
        sa = (a[3] - a[1]) / max(1, a[2] - a[0]); sb = (b[3] - b[1]) / max(1, b[2] - b[0])
        return 0.7 < sa / max(sb, 1e-3) < 1.4
    P, N = merge(P), merge(N)
    for p in P:
        for q in N:
            bx = [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3])]
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            if w < 25 or h < 25: continue
            # 두 대각선이 같은 상자를 대각으로 채워야
            if not all(abs(u - v) < EV_P['tol'] * max(w, h) + 4 for u, v in zip(p, bx)) or not all(abs(u - v) < EV_P['tol'] * max(w, h) + 4 for u, v in zip(q, bx)): continue
            # 네 변
            sides = []                                   # X 교차점이 상자 중심 근처(마름모 탁자 등 대각 변 배제)
            (px1, py1), (px2, py2) = (p[0], p[1]), (p[2], p[3])
            (qx1, qy1), (qx2, qy2) = (q[0], q[3]), (q[2], q[1])
            den = (px2 - px1) * (qy2 - qy1) - (py2 - py1) * (qx2 - qx1)
            if abs(den) < 1e-6: continue
            tt = ((qx1 - px1) * (qy2 - qy1) - (qy1 - py1) * (qx2 - qx1)) / den
            ix, iy = px1 + tt * (px2 - px1), py1 + tt * (py2 - py1)
            if abs(ix - (bx[0] + bx[2]) / 2) > 0.15 * w + 3 or abs(iy - (bx[1] + bx[3]) / 2) > 0.15 * h + 3: continue
            # 상자 안에 승강기 아닌 글자(정화조·PS·AD·OPEN 등)가 있으면 제외
            # (저해상도 2배 OCR 에선 한글 잡음이 많아 '한글 2자 이상' 대신 명시 단어·실명 일치로만 제외)
            if any(_ovl([t['box'][0] - ox, t['box'][1] - oy, t['box'][2] - ox, t['box'][3] - oy], bx) and not EV_WORDS.search(t['text'].replace(' ', ''))
                   and (NOT_EV.search(t['text'].replace(' ', '')) or (t.get('name') and t.get('cat') != '승강기' and t.get('score', 0) >= 0.7))
                   and t.get('conf', 0) > 50 for t in words): continue
            # 네 변: 카(안쪽 상자) 또는 승강로 벽(바깥 0~15%) 중 가장 잘 칠해진 선
            t = np.linspace(0.05, 0.95, 30)
            for sd in range(4):
                best_c = 0
                for off in np.linspace(0, 0.15 * max(w, h), 6):
                    xa, ya, xb, yb = ((bx[0], bx[1] - off, bx[2], bx[1] - off), (bx[0], bx[3] + off, bx[2], bx[3] + off),
                                      (bx[0] - off, bx[1], bx[0] - off, bx[3]), (bx[2] + off, bx[1], bx[2] + off, bx[3]))[sd]
                    best_c = max(best_c, _cover(lmd, np.c_[xa + t * (xb - xa), ya + t * (yb - ya)]).mean())
                sides.append(best_c)
            if min(sides) < 0.45: continue
            asp = max(w, h) / min(w, h)
            near = any(_ovl([t['box'][0] - ox, t['box'][1] - oy, t['box'][2] - ox, t['box'][3] - oy], bx, m=0.5 * max(w, h)) for t in ew)
            wm, hm = (w / ppm, h / ppm) if ppm else (None, None)
            size_ok = ppm is not None and 1.1 <= min(wm, hm) and max(wm, hm) <= 3.2 and asp <= 1.7
            if not (size_ok or (near and asp <= 2.2 and (ppm is None or min(wm, hm) >= 0.8))): continue
            out.append(dict(box=[int(bx[0] + ox), int(bx[1] + oy), int(bx[2] + ox), int(bx[3] + oy)], size_m=[round(wm, 2), round(hm, 2)] if ppm else None,
                            label=near, sides=round(min(sides), 2)))
    # 중첩(승강로 안 카) 제거: 큰 것 유지
    out.sort(key=lambda e: -(e['box'][2] - e['box'][0]) * (e['box'][3] - e['box'][1]))
    keep = []
    for e in out:
        if all(not _ovl(e['box'], k['box']) for k in keep): keep.append(e)
    return keep


# ─────────────────────────── 통합 ───────────────────────────
SPA2CAT = {'침실': '침실', '거실': '거실', '주방': '주방', '화장실': '욕실·화장실', '발코니': '발코니·대피공간', '현관': '현관', '계단실': '계단실',
           '엘리베이터': '승강기', '엘리베이터홀': '복도', '드레스룸': '기타', '실외기실': '기타', '다목적공간': '기타', '기타': '기타'}


FUSE_RULE_ALL = False     # True: 모델과 안 겹치는 규칙 검출 전부 추가(재현율↑ 정밀도↓)


# 실 합치기 방식(2026-10-02, SPA devval124 + 손 정답 개발 16평면에서만 고름):
#  'v2'(기본) = OCR 실마다 그 글자를 품은 모델 실(가장 작은 성분 상자)과 대응 → 같은 실 종류면 글자 쪽을 남기고 가장 가까운 모델 점 + 글자에서
#   FUSE_DUP_M(m) 안의 같은 종류 모델 점(같은 실의 중복 봉우리)만 지움, 소방 범주가 다르면 FUSE_CONFLICT 로 결정
#   ('auto' = 고해상도 시트(OCR 확대 안 함)는 OCR, 저해상도 축소본(2배 확대 OCR — 오독 많음)은 모델 범주가 이김 · 'model' · 'ocr' · 'both'),
#   모델 실 밖 글자는 그대로 추가. '기타'·'복도' 범주는 SPA 부류 수준(드레스룸·실외기실·승강기홀…)으로 같은 종류를 판정.
#  'old' = 이전 방식(같은 범주 글자가 성분 상자 안에 있으면 그 성분의 모델 점 전부 지움 — 문으로 이어진 침실1·2 중 하나만 읽혀도 둘 다 지워짐,
#   다른 범주 글자는 모델 점과 함께 남김) — devval124 합침 소방6종 .818/정밀 .954 → v2 .826/.970 (모델 단독 .824/.972)
FUSE_ROOMS = os.environ.get('REALSYM_FUSE', 'v2')
FUSE_CONFLICT = os.environ.get('REALSYM_FUSE_CONFLICT', 'auto')
FUSE_DUP_M = float(os.environ.get('REALSYM_FUSE_DUP_M', 2.0))   # 같은 실 중복 봉우리 반경(m, 모델 학습 축척 17.5 px/m 기준)


def spa_seg_ref_ppm():
    from firebim import spa_seg
    return spa_seg.REF_PPM


_FINE_OCR = {'드레스룸': '드레스룸', '실외기실': '실외기실', '알파룸': '다목적공간', '홀': '엘리베이터홀'}


def _fine(r):
    """실 세부 종류 열쇠: 소방 범주가 '기타'·'복도'(여러 종류를 묶은 범주)면 SPA 모델 부류 수준으로 나눔"""
    if r['cat'] not in ('기타', '복도'): return r['cat']
    if r.get('src') == 'model': return r['name']
    return _FINE_OCR.get(r.get('name'), '기타' if r['cat'] == '기타' else '복도')


def _fuse_rooms_v2(rooms, M, conflict, dup_px=0.0):
    """rooms: 규칙(OCR) 실, M: 모델 실(좌표 이동 끝). 반환 합친 목록.
    dup_px: 같은 성분·같은 종류 모델 점 중 OCR 글자에서 이 거리(px) 안의 것도 같은 실의 중복 봉우리로 보고 지움(0 = 가장 가까운 점 하나만)"""
    alive = [True] * len(M)
    keep_r = []
    for o in rooms:
        if o.get('src') == 'model': keep_r.append(o); continue
        x, y = o['xy']
        C = [i for i, m in enumerate(M) if m['box'][0] <= x <= m['box'][2] and m['box'][1] <= y <= m['box'][3]]
        if not C: keep_r.append(o); continue
        area = lambda i: (M[i]['box'][2] - M[i]['box'][0]) * (M[i]['box'][3] - M[i]['box'][1])
        a0 = min(area(i) for i in C)
        comp = [i for i in C if area(i) <= a0 * 1.0001 and M[i]['box'] == M[min(C, key=area)]['box']]
        same = [i for i in comp if _fine(M[i]) == _fine(o)]
        if same:                                             # 같은 실 종류: 글자 쪽을 남기고 가장 가까운 모델 점 하나 지움
            live = [i for i in same if alive[i]]
            if live:
                j = min(live, key=lambda i: (M[i]['xy'][0] - x) ** 2 + (M[i]['xy'][1] - y) ** 2)
                alive[j] = False
                for i in live:
                    if np.hypot(M[i]['xy'][0] - x, M[i]['xy'][1] - y) <= dup_px: alive[i] = False
            keep_r.append(o); continue
        if any(M[i]['cat'] == o['cat'] for i in comp):       # 같은 소방 범주(기타·복도)의 다른 세부 종류(드레스룸↔실외기실, 복도↔승강기홀): 둘 다 둠
            keep_r.append(o); continue
        if conflict == 'model': continue                     # 범주가 다름: 모델 범주가 이김(OCR 오독·어휘 오매칭 대비)
        if conflict == 'ocr':
            for i in comp: alive[i] = False
        keep_r.append(o)
    return keep_r + [m for m, a in zip(M, alive) if a]


def fuse_model(out, seg, box=None, mode=None, conflict=None):
    """학습 분할망(spa_seg) 결과를 규칙 결과에 합침.
    실: mode(기본 FUSE_ROOMS) — 위 설명. 모델 실은 src='model' 로 추가(OCR 글자와 겹친 같은 종류 모델 점은 빠짐).
    계단: 모델 검출 + (모델과 겹치지 않는) 규칙 검출 중 UP/DN·계단 글자가 있는 것. 승강기: 모델 + EV 글자 있는 규칙 검출."""
    ox, oy = (box[0], box[1]) if box else (0, 0)
    sh = lambda b: [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
    mode = mode or FUSE_ROOMS; conflict = conflict or FUSE_CONFLICT
    rooms = list(out.get('rooms', []))
    M = []
    for r in seg['rooms']:
        cat = SPA2CAT.get(r['name'], '기타')
        M.append(dict(text='', name=r['name'], spa=r.get('spa'), cat=cat, xy=[r['xy'][0] + ox, r['xy'][1] + oy], box=sh(r['box']), score=r['score'], refuge=False,
                      priority=SEARCH_PRIORITY[cat], src='model'))
    if mode == 'v2':
        if conflict == 'auto':                                 # 글자 판독 신뢰: 고해상도 시트(OCR 확대 안 함)면 OCR, 저해상도 축소본이면 모델 범주가 이김
            conflict = 'ocr' if (seg.get('ocr_up') or out.get('ocr_up') or 1.0) <= 1.0 else 'model'
        sc = seg.get('scale') or 0
        dup_px = FUSE_DUP_M * spa_seg_ref_ppm() / sc if (sc and FUSE_DUP_M > 0) else 0.0
        rooms = _fuse_rooms_v2(rooms, M, conflict, dup_px)
    else:
        R0 = [o for o in rooms if o.get('src') != 'model']
        for m in M:
            bx = m['box']
            if any(bx[0] <= o['xy'][0] <= bx[2] and bx[1] <= o['xy'][1] <= bx[3] and m['cat'] == o['cat'] for o in R0): continue
            rooms.append(m)
    out['rooms'] = rooms
    for key, keep_rule in (('stairs', lambda d: d.get('label')), ('elevators', lambda d: d.get('label'))):   # 규칙은 글자(UP·DN·계단·EV)로 확인된 것만(SPA 개발: 보행선만으로는 정밀도↓)
        if key not in out: continue
        M = [dict(box=[int(v) for v in sh(d['box'])], score=d['score'], src='model') for d in seg[key]]
        other = [sh(d['box']) for d in seg['elevators' if key == 'stairs' else 'stairs']]     # 모델이 다른 종류로 본 자리(승강기 속 '계단' 등)는 제외
        R = [dict(d, src='rule') for d in out[key] if (FUSE_RULE_ALL or keep_rule(d)) and not any(_ovl(d['box'], m['box']) for m in M)
             and not any(_ovl(d['box'], o) for o in other)]
        out[key] = M + R
    return out


def analyze(g, box=None, dets=('rooms', 'doors', 'stairs', 'elev'), ocr_up='auto', model=False, model_scale=None):
    """ocr_up: 'auto' = 시트(g) 긴 변 ≤ 2600 px(저해상도 축소본, 실명 글자 높이 ≈ 8~11 px)이면 2배 확대 OCR. model: True/경로 → firebim/spa_seg ONNX 분할망 결과 합침
    (out['seg'] 에 모델 원 결과, rooms/stairs/elevators 는 합친 결과. 모델 파일 없으면 규칙만).
    model_scale: 모델 입력 배율(기본 1024/시트 긴 변 — 학습 조건: A3 시트 전체를 긴 변 1024 px 로). 시트가 아닌 잘린 평면 영상이면 직접 지정."""
    out = dict(box=box)
    words = []
    sub = g[box[1]:box[3], box[0]:box[2]] if box else g
    if ocr_up == 'auto': ocr_up = 2.0 if max(g.shape[:2]) <= 2600 else 1.0     # 시트 전체 긴 변 기준(A3 원본 ≈4960 px 는 그대로)
    out['ocr_up'] = ocr_up
    if 'rooms' in dets or 'stairs' in dets or 'elev' in dets:
        rooms, cands = detect_rooms(g, box, up=ocr_up)
        out['rooms'] = rooms if 'rooms' in dets else []
        out['words'] = words = cands
    if 'doors' in dets or 'elev' in dets or 'stairs' in dets:
        out['doors'] = detect_doors(g, box)
        out['px_per_m'] = scale_from_doors(out['doors'])
    if 'stairs' in dets:
        out['stairs'] = detect_stairs(g, box, words, out.get('px_per_m'))
    if 'elev' in dets:
        out['elevators'] = detect_elevators(g, box, words, out.get('px_per_m'))
    if model:
        from firebim import spa_seg
        path = model if isinstance(model, str) else spa_seg.MODEL
        if spa_seg.available(path):
            # 학습 축척 = 시트 전체 긴 변 1024 px. 평면 상자(box)만 넘겨도 축척은 시트 전체(g) 기준으로
            out['seg'] = spa_seg.detect(sub, scale=model_scale or spa_seg.sheet_scale(max(g.shape[:2]), path), path=path)
            fuse_model(out, out['seg'], box)
    return out
