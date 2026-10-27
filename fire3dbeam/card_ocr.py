# -*- coding: utf-8 -*-
"""스캔본(글자층 없는 이미지) 표준양식 정보카드 OCR — card.py 가 글자층 없는 쪽을 만나면 이 모듈로 자동 전환(source='ocr').

쪽마다: 쪽 그림 → 바탕 평탄화 → 기울기 보정(deskew, 투영 분산 최대) → 표 선(OpenCV 형태학) → 칸 격자(병합 칸 = 선이 없는 경계 합치기)
→ 칸 글자 OCR(tesseract kor+eng, 여러 칸을 한 장에 쌓아 1회 호출) → 칸 이름(양식 항목명) 자모 유사도 대응 → 숫자 칸 재독(eng, 숫자 화이트리스트)
→ 영역 어휘 보정(소방시설명 fac_catalog, 주용도·구조·지붕·층 표기·위험물) → pymupdf find_tables 와 같은 모양의 행 [(쪽, y, [칸 글자|None])]
→ card.parse_std 가 그대로 해석(하류 불변). 표 밖 글줄(절 제목·⑪ 캡션)도 OCR → 그림 영역(표 선 틀 + 색·잉크 덩어리) 잘라
(종이색→흰색, 틀 선 조각 제거, 무채색 감마 1.5) extract_figures 에 공급. 평면도 캡션 층은 ⑩ 층 목록으로 검증·보정.
항목별 신뢰도(field_confs) → card.ocr_meta → 낮은 항목은 pipeline 검수 플래그 kind='card_ocr'.

개인정보: ⑧ 관계인 칸 글자는 로그·디버그 출력에 남기지 않는다(tesseract 입력은 표준입력으로 넘겨 임시 파일도 만들지 않음; 실패 시에만 임시 파일 즉시 삭제).
결과는 문서 내용 해시로 프로세스 안에만 캐시(같은 PDF 를 parse_card·extract_plans·extract_figures 가 여러 번 열어도 OCR 1회)."""
import hashlib, os, re, subprocess, tempfile, time
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np
import cv2

LOG = os.environ.get('FIRE3DBEAM_OCR_LOG', '1') != '0'
TESS = os.environ.get('FIRE3DBEAM_TESSERACT', 'tesseract')
_STDIN_OK = [None]
_CACHE = {}                    # 쪽 해시 → 쪽 분석 결과(최근 64쪽)


def log(*a):
    if LOG: print('[card_ocr]', *a, flush=True)


# ───────────────────────────── tesseract 호출(표준입력 → 실패 시 임시 파일)
def tesseract_tsv(img, lang='kor+eng', psm=6, wl=None, extra=()):
    """img: gray uint8 → tsv 행 [dict(x,y,w,h,conf,text,line_key)]"""
    ok, png = cv2.imencode('.png', img)
    cmd = ['-l', lang, '--psm', str(psm), '-c', 'preserve_interword_spaces=1'] + (['-c', 'tessedit_char_whitelist=' + wl] if wl else []) + list(extra) + ['tsv']
    out = None
    if _STDIN_OK[0] is not False:
        try:
            r = subprocess.run([TESS, 'stdin', 'stdout'] + cmd, input=png.tobytes(), capture_output=True, timeout=300)
            if r.returncode == 0 and r.stdout: out = r.stdout.decode('utf-8', 'replace'); _STDIN_OK[0] = True
            elif _STDIN_OK[0] is None: _STDIN_OK[0] = False
        except (OSError, subprocess.SubprocessError):
            _STDIN_OK[0] = False
    if out is None:
        fd, fn = tempfile.mkstemp(suffix='.png')
        try:
            os.write(fd, png.tobytes()); os.close(fd)
            out = subprocess.run([TESS, fn, 'stdout'] + cmd, capture_output=True, timeout=300).stdout.decode('utf-8', 'replace')
        finally:
            try: os.unlink(fn)
            except OSError: pass
    rows = []
    for ln in out.splitlines()[1:]:
        p = ln.split('\t')
        if len(p) == 12 and p[11].strip():
            rows.append(dict(x=int(p[6]), y=int(p[7]), w=int(p[8]), h=int(p[9]), conf=float(p[10]), text=p[11].strip(),
                             line=(int(p[2]), int(p[3]), int(p[4]))))
    return rows


# ───────────────────────────── 쪽 판별·그림
def page_text_len(page):
    return len(re.sub(r'\s+', '', page.get_text() or ''))


def is_image_page(page, min_cover=0.5):
    """글자층이 (거의) 없고 그림(들)이 쪽 대부분을 덮는 쪽 = 스캔본 쪽. 글자를 곡선으로 바꿔 내보낸 쪽(글자층 없음 + 벡터 그림 많음)도 포함"""
    if page_text_len(page) > 8: return False
    pa = page.rect.width * page.rect.height
    cov = 0.0
    for i in page.get_image_info():                      # 띠로 나뉜 스캔(여러 그림)도 합산
        x0, y0, x1, y1 = i['bbox']
        cov += max(0, min(x1, page.rect.width) - max(0, x0)) * max(0, min(y1, page.rect.height) - max(0, y0)) / pa
    if cov >= min_cover: return True
    try: return len(page.get_drawings()) > 150            # 글자 곡선화 PDF
    except Exception: return False


MAX_PPT = 4.2                                            # 300 dpi 넘는 스캔은 줄여서(속도) 처리


def page_image(page, dpi=200):
    """쪽 → BGR 영상 + 화소/pt. 쪽을 덮는 그림 한 장이면 원본 화소(스캔 해상도), 아니면 dpi 로 그리기"""
    import pymupdf
    infos = [i for i in page.get_image_info(xrefs=True) if i.get('xref')]
    pa = page.rect.width * page.rect.height
    big = [i for i in infos if (i['bbox'][2] - i['bbox'][0]) * (i['bbox'][3] - i['bbox'][1]) > 0.9 * pa]
    pix = None
    if len(big) == 1 and len(infos) == 1 and page.rotation == 0:
        try:
            pix = pymupdf.Pixmap(page.parent, big[0]['xref'])
            if pix.w < 1.2 * page.rect.width or pix.h < 1.2 * page.rect.height: pix = None      # 90 dpi 미만 → 그리기(확대)
        except Exception:
            pix = None
    if pix is None: pix = page.get_pixmap(dpi=dpi)
    if pix.n - pix.alpha >= 4 or pix.n - pix.alpha == 1: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3][..., ::-1].copy()
    ppt = pix.w / page.rect.width
    if ppt > MAX_PPT:
        f = MAX_PPT / ppt
        arr = cv2.resize(arr, (int(arr.shape[1] * f), int(arr.shape[0] * f)), interpolation=cv2.INTER_AREA); ppt = MAX_PPT
    return arr, ppt


def page_key(page):
    """쪽 내용 해시(그림 스트림 원본 바이트 + 쪽 내용 + 쪽 크기) — OCR 캐시 키"""
    h = hashlib.sha1(f'{page.rect.width:.1f}x{page.rect.height:.1f}r{page.rotation}'.encode())
    for i in page.get_images(full=True):
        try: h.update(page.parent.xref_stream_raw(i[0]) or b'')
        except Exception: h.update(str(i).encode())
    try: h.update(page.read_contents() or b'')
    except Exception: pass
    return h.hexdigest()


# ───────────────────────────── 전처리·기울기
def flatten(gray):
    """바탕(종이색·명암 기울기) 평탄화 → 흰 바탕 회색조"""
    k = max(31, (min(gray.shape) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    out = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255)
    # 잉크 대비 늘리기(가장 진한 1% → 0)
    lo = float(np.percentile(out, 0.5))
    if lo > 5 and (out < 215).mean() > 0.003: out = np.clip((out - lo) / (255 - lo) * 255, 0, 255)      # 잉크가 있는 쪽만(빈 쪽의 잡음을 키우지 않게)
    return out.astype(np.uint8)


def ink_mask(g):
    """글자·선(잉크) = 1"""
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = min(200, max(110, int(_ * 1.0)))
    return (g < t).astype(np.uint8)


def skew_angle(ink, lim=2.5):
    """투영 분산 최대 각도(도). 작게 줄인 잉크 영상을 돌려가며 행 합의 분산 비교: 거친 0.1° → 고운 0.02°"""
    H, W = ink.shape
    f = min(1.0, 900.0 / W)
    small = cv2.resize(ink.astype(np.float32), (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR)
        p = r.sum(1); return float(np.var(p))
    best = max(np.arange(-lim, lim + 1e-6, 0.1), key=score)
    best = max(np.arange(best - 0.1, best + 0.1 + 1e-6, 0.02), key=score)
    return float(best)


def rotate(img, ang, border=255):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    bv = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=bv)


# ───────────────────────────── 표 선·칸 격자
def line_masks(ink, ppt):
    """가로선·세로선 마스크(형태학 열림). ppt = 화소/pt"""
    L = max(15, int(18 * ppt))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(10 * ppt)))))
    # 짧은 틈 메우기(흐림·JPEG 으로 끊긴 선)
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * ppt) + 1, 1)))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(3 * ppt) + 1)))
    return h, v


def _peaks(prof, thr, gap):
    idx = np.where(prof >= thr)[0]
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap: out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    return [(c[0], c[-1]) for c in out]


def find_tables(hm, vm, ppt, min_w_pt=120, min_h_pt=12):
    """선 마스크 → 표 후보 [(x0,y0,x1,y1)] (가로선·세로선이 함께 있는 연결 덩어리)"""
    grid = cv2.dilate(((hm | vm) > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid, 8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if w < min_w_pt * ppt or h < min_h_pt * ppt: continue
        sub_h = hm[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == k); sub_v = vm[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == k)
        if sub_h.sum() < w or sub_v.sum() < h * 0.5: continue
        out.append((x, y, x + w, y + h))
    return sorted(out, key=lambda b: b[1])


def cell_grid(hm, vm, box, ppt):
    """표 한 개 → (ys, xs, cells) — cells: [dict(r0,c0,r1,c1,box)] (병합 = 경계선 없음). 경계 판정은 띠 길이의 55% 이상 선"""
    x0, y0, x1, y1 = box
    H = hm[y0:y1, x0:x1]; V = vm[y0:y1, x0:x1]
    h, w = H.shape
    gap = max(2, int(2.5 * ppt))
    ry = _peaks(H.sum(1), max(6 * ppt, 0.04 * w), gap)
    cx = _peaks(V.sum(0), max(5 * ppt, 0.25 * min(h, 30 * ppt)), gap)
    ys = [int((a + b) / 2) for a, b in ry]; xs = [int((a + b) / 2) for a, b in cx]
    if not ys or ys[0] > 3 * ppt: ys = [0] + ys
    if ys[-1] < h - 3 * ppt: ys = ys + [h - 1]
    if not xs or xs[0] > 3 * ppt: xs = [0] + xs
    if xs[-1] < w - 3 * ppt: xs = xs + [w - 1]
    nr, nc = len(ys) - 1, len(xs) - 1
    if nr < 1 or nc < 1: return ys, xs, []
    t = max(2, int(1.5 * ppt))
    def vline(c, r):        # 세로 경계 xs[c] 가 행 r 에 있나
        a, b = ys[r] + t, ys[r + 1] - t
        if b <= a: return True
        seg = V[a:b, max(0, xs[c] - t):xs[c] + t + 1].max(1) if b > a else np.zeros(0)
        return seg.mean() > 0.55
    def hline(r, c):        # 가로 경계 ys[r] 가 열 c 에 있나
        a, b = xs[c] + t, xs[c + 1] - t
        if b <= a: return True
        seg = H[max(0, ys[r] - t):ys[r] + t + 1, a:b].max(0)
        return seg.mean() > 0.55
    par = list(range(nr * nc))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for r in range(nr):
        for c in range(nc):
            if c + 1 < nc and not vline(c + 1, r): par[f(r * nc + c)] = f(r * nc + c + 1)
            if r + 1 < nr and not hline(r + 1, c): par[f(r * nc + c)] = f((r + 1) * nc + c)
    groups = {}
    for i in range(nr * nc): groups.setdefault(f(i), []).append(i)
    cells = []
    for g in groups.values():
        rr = [i // nc for i in g]; cc = [i % nc for i in g]
        r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        if len(g) != (r1 - r0) * (c1 - c0):        # 직사각형 아닌 병합(선 끊김) → 칸마다 따로
            for i in g: cells.append(dict(r0=i // nc, r1=i // nc + 1, c0=i % nc, c1=i % nc + 1))
            continue
        cells.append(dict(r0=r0, r1=r1, c0=c0, c1=c1))
    for c in cells: c['box'] = (x0 + xs[c['c0']], y0 + ys[c['r0']], x0 + xs[c['c1']], y0 + ys[c['r1']])
    return [y0 + y for y in ys], [x0 + x for x in xs], cells


# ───────────────────────────── 칸·글줄 OCR(모자이크 1회 호출)
OCR_PPT = float(os.environ.get('FIRE3DBEAM_OCR_PPT', '3.2'))     # 표 칸 OCR 입력 해상도(화소/pt ≈ 230 dpi)
LINE_PPT = 4.5                                                 # 표 밖 글줄(굵은 밑줄 캡션) OCR 해상도


def ink_bbox(g, thr=160, min_px=6):
    """회색 칸 영상 안 잉크 외접 상자(없으면 None)"""
    m = (g < thr).astype(np.uint8)
    if m.sum() < min_px: return None
    ys, xs = np.where(m > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clean_cell(g, box, ppt):
    """칸 상자(쪽 화소) → 선을 뺀 칸 안쪽 회색 영상(잉크 없으면 None). 칸 테두리 선 두께만큼 안쪽으로 줄이고, 가장자리에 붙은 선 조각 제거"""
    x0, y0, x1, y1 = box
    t = max(2, int(1.6 * ppt))
    sub = g[y0 + t:y1 - t, x0 + t:x1 - t]
    if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4: return None
    sub = sub.copy()
    m = (sub < 150).astype(np.uint8)
    # 가장자리 긴 선(칸 경계 잔여) 지우기
    for axis in (0, 1):
        prof = m.mean(axis=axis)
        for i in list(range(0, min(4, len(prof)))) + list(range(max(0, len(prof) - 4), len(prof))):
            if prof[i] > 0.6:
                if axis == 0: sub[:, i] = 255
                else: sub[i, :] = 255
    bb = ink_bbox(sub)
    if bb is None: return None
    # 잡음 점만 있는 칸(스캔 먼지) → 빈 칸
    n, lab, st, _ = cv2.connectedComponentsWithStats((sub < 150).astype(np.uint8), 8)
    big = [k for k in range(1, n) if st[k, 4] >= max(4, 0.6 * ppt * ppt)]
    if not big: return None
    xs0 = min(st[k, 0] for k in big); ys0 = min(st[k, 1] for k in big)
    xs1 = max(st[k, 0] + st[k, 2] for k in big); ys1 = max(st[k, 1] + st[k, 3] for k in big)
    if (ys1 - ys0) < 2.5 * ppt and (xs1 - xs0) < 2.5 * ppt: return None
    p = int(1.5 * ppt)
    return sub[max(0, ys0 - p):ys1 + p, max(0, xs0 - p):xs1 + p]


def ocr_tiles(tiles, ppt, lang='kor+eng', wl=None, psm=6, target=None):
    """회색 타일 여러 개 → 타일마다 [(줄 y, [(x, 글자, conf)])] (한 장에 세로로 쌓아 tesseract 1회). target = OCR 해상도(화소/pt)"""
    if not tiles: return []
    s = (target or OCR_PPT) / ppt
    PAD = int(OCR_PPT * 14)
    rs, ys = [], []
    y = PAD; Wm = 0
    for t in tiles:
        r = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if abs(s - 1) > 0.05 else t
        rs.append(r); ys.append(y); y += r.shape[0] + PAD; Wm = max(Wm, r.shape[1])
    M = np.full((y + PAD, Wm + 2 * PAD), 255, np.uint8)
    for r, yy in zip(rs, ys): M[yy:yy + r.shape[0], PAD:PAD + r.shape[1]] = r
    if M.shape[0] > 30000:                                  # tesseract 높이 한도 → 나눠 호출
        k = len(tiles) // 2
        return ocr_tiles(tiles[:k], ppt, lang, wl, psm, target) + ocr_tiles(tiles[k:], ppt, lang, wl, psm, target)
    rows = tesseract_tsv(M, lang=lang, psm=psm, wl=wl)
    res = [dict() for _ in tiles]
    for w in rows:
        cy = w['y'] + w['h'] / 2
        k = int(np.searchsorted(np.array(ys), cy, side='right')) - 1
        if k < 0 or cy > ys[k] + rs[k].shape[0] + PAD / 2: continue
        res[k].setdefault(w['line'], []).append((w['x'], w['y'], w['w'], w['h'], w['text'], w['conf']))
    out = []
    for d in res:
        lines = []
        for key, ws in d.items():
            ws.sort()
            lines.append((min(v[1] for v in ws), [(v[0], v[2], v[3], v[4], v[5]) for v in ws]))
        lines.sort()
        out.append(lines)
    return out


def _join_words(ws):
    """단어 상자 간격으로 띄어쓰기 복원(tesseract kor 는 음절마다 끊음): 간격 > 글자 높이 0.33 → 공백"""
    if not ws: return ''
    hs = sorted(w[2] for w in ws); hm = hs[len(hs) // 2]
    out = ws[0][3]
    for a, b in zip(ws, ws[1:]):
        gap = b[0] - (a[0] + a[1])
        out += (' ' if gap > 0.33 * hm else '') + b[3]
    return out


def lines_text(lines, joiner='\n'):
    """ocr_tiles 한 타일 결과 → (글자, 평균 conf 0~100, 단어 수)"""
    if not lines: return '', -1.0, 0
    txt = joiner.join(_join_words(ws) for _, ws in lines)
    cf = [w[4] for _, ws in lines for w in ws if w[4] >= 0]
    return txt, (float(np.mean(cf)) if cf else -1.0), len(cf)


# ───────────────────────────── 자모 유사도·어휘 보정
def _jamo_seq(s):
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out += [chr(0x1100 + o // 588), chr(0x1161 + (o % 588) // 28)] + ([chr(0x11A7 + o % 28)] if o % 28 else [])
        else: out.append(ch)
    return out


def _lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def jsim(a, b):
    """자모 편집거리 유사도 0~1 (OCR 오독 '충수'↔'층수', '제단'↔'계단' 허용)"""
    A, B = _jamo_seq(a), _jamo_seq(b)
    if not A or not B: return 0.0
    return 1.0 - _lev(A, B) / max(len(A), len(B))


def best_match(s, vocab, key=lambda v: v):
    """→ (항목, 유사도, 2등 유사도)"""
    sc = sorted(((jsim(s, key(v)), i) for i, v in enumerate(vocab)), reverse=True)
    if not sc: return None, 0.0, 0.0
    return vocab[sc[0][1]], sc[0][0], (sc[1][0] if len(sc) > 1 else 0.0)


def hangul(s): return re.sub(r'[^가-힣]', '', s or '')


# 양식 칸 이름: (비교 글자(한글만), 출력 글자 — pymupdf 글자층과 같은 모양, 무리)
LABELS = [('관리번호', '① 관리번호', 'top'), ('주용도', '② 주 용 도', 'top'), ('상호', '③ 상 호', 'top'), ('소재지', '④ 소 재 지', 'top'),
          ('소방대상물과의거리', '⑤ 소방대상물과의거리', 'top'), ('소방용수시설', '⑥ 소방용수시설', 'top'), ('건물구조', '⑦ 건물구조', 'sec'),
          ('양식', '양 식', 's0'), ('구조', '구 조', 's0'), ('지붕', '지 붕', 's0'), ('동명', '동 명', 's0'),
          ('지하층수', '지하 층수', 's1'), ('지상층수', '지상 층수', 's1'), ('연면적', '연면적(㎡)', 's1'), ('바닥면적', '바닥면적(㎡)', 's1'),
          ('특별피난계단수', '특별피난계단수', 's2'), ('피난계단수', '피난계단수', 's2'), ('일반계단수', '일반계단수', 's2'), ('옥외계단수', '옥외계단수', 's2'),
          ('승강기비상', '승강기(비상)', 's3'), ('승강기일반', '승강기(일반)', 's3'),
          ('관계인등', '⑧ 관계인등', 'sec'), ('소유자', '소유자', 'role'), ('관리자', '관리자', 'role'), ('소방안전관리자', '소방안전관리자', 'role'),
          ('위험물안전관리자', '위험물안전관리자', 'role'), ('성명', '성 명', 'rel'), ('연락처', '연 락 처', 'rel'),
          ('위험물시설등', '⑨ 위험물시설등', 'sec'), ('구분', '구 분', 'hh'), ('시설명제조소구분류별등', '시 설 명(제조소 구분, 류별 등)', 'hh'), ('품명', '품 명', 'hh'),
          ('수량', '수 량', 'hh'), ('위험물시설', '위험물 시설', 'haz'), ('가스취급현황', '가스취급 현황', 'haz'), ('특수가연물', '특수가연물', 'haz'), ('기타', '기 타', 'haz'),
          ('층별현황', '⑩ 층별현황', 'sec'), ('층별', '층 별', 'fh'), ('용도', '용 도', 'fh'), ('면적', '면 적(㎡)', 'fh'), ('소방시설설치현황', '소방시설 설치현황', 'fh')]
TEMPLATE_ROWS = {g: [l[1] for l in LABELS if l[2] == g] for g in ('s0', 's1', 's2', 's3', 'top', 'fh', 'hh')}
NUM_LABELS = {'지하 층수': 'int', '지상 층수': 'int', '연면적(㎡)': 'area', '바닥면적(㎡)': 'area', '특별피난계단수': 'int', '피난계단수': 'int', '일반계단수': 'int',
              '옥외계단수': 'int', '승강기(비상)': 'int', '승강기(일반)': 'int'}

USES = ['단독주택', '공동주택', '제1종근린생활시설', '제2종근린생활시설', '문화및집회시설', '종교시설', '판매시설', '운수시설', '의료시설', '교육연구시설', '노유자시설',
        '수련시설', '운동시설', '업무시설', '숙박시설', '위락시설', '공장', '창고시설', '위험물저장및처리시설', '자동차관련시설', '동물및식물관련시설', '자원순환관련시설',
        '교정시설', '국방·군사시설', '방송통신시설', '발전시설', '묘지관련시설', '관광휴게시설', '장례시설', '야영장시설', '복합건축물']        # 건축법 시행령 별표 1
STRUCTS = ['철근콘크리트조', '철골철근콘크리트조', '철골콘크리트조', '철골조', '일반철골구조', '경량철골구조', '조적조', '벽돌조', '블록조', '석조', '목구조', '목조',
           '프리캐스트콘크리트조', '강파이프구조', '기타조', '철근콘크리트구조', '철골구조']
ROOFS = ['슬라브', '평슬라브', '경사슬라브', '(철근)콘크리트', '철근콘크리트', '샌드위치패널', '칼라강판', '강판', '기와', '슬레이트', '아스팔트싱글', '패널', '징크']
FLOOR_USES = ['사무실', '세대', '보관', '작업장', '병동', '병실', '주차장', '기계실', '전기실', '발전기실', '펌프실', '교실', '강의실', '로비', '외래', '객실', '계단실',
              '생활실', '창고', '물류창고', '매장', '판매장', '상가', '식당', '음식점', '회의실', '강당', '진료실', '수술실', '대기실', '휴게실', '숙소', '공장', '제조',
              '전시실', '체육관', '주거', '근린생활시설', '학원', '옥탑', '물탱크실', '저수조', '관리실', '방재실', '수위실', '화장실', '복도', '공용', '주방', '실험실', '연구실',
              '도서실', '열람실', '강의동', '기숙사', '요양실', '침실', '거실', '어린이집', '유치원', '판매시설', '업무시설', '숙박시설', '의료시설']
ETC_WORDS = ['피난층', '무창층', '지하층', '필로티', '옥상광장', '헬리포트', '특별피난계단', '비상용승강기']
HAZ_NAMES = ['제조소', '옥내저장소', '옥외저장소', '옥내탱크저장소', '옥외탱크저장소', '지하탱크저장소', '간이탱크저장소', '이동탱크저장소', '암반탱크저장소',
             '주유취급소', '판매취급소', '이송취급소', '일반취급소', '제1류', '제2류', '제3류', '제4류', '제5류', '제6류',
             'LPG저장(용기)', 'LPG저장', '도시가스정압기', '의료가스저장', '창고적재', '비상발전기연료탱크', '보일러실', '가스저장소']
HAZ_ITEMS = ['특수인화물', '제1석유류', '알코올류', '제2석유류', '제3석유류', '제4석유류', '동식물유류', '휘발유', '등유', '경유', '중유', '윤활유', '벤젠', '톨루엔',
             '액화석유가스', '도시가스(LNG)', '도시가스', 'LNG', 'LPG', '산소', '수소', '아세틸렌', '질소', '면화류', '나무껍질및대팻밥', '넝마및종이부스러기', '사류', '볏짚류',
             '가연성고체류', '석탄·목탄류', '가연성액체류', '목재가공품및나무부스러기', '고무류·플라스틱류', '합성수지류(발포)', '합성수지류', '고무류', '플라스틱류']
WATER_KINDS = ['지상식', '지하식', '급수탑', '저수조', '비상소화장치']
FLOOR_PREFIX = ['지하', '지상', '옥탑']


def fac_vocab():
    from fire3dbeam.fac_catalog import CATALOG
    v = set()
    for c in CATALOG.values():
        for a in [c['name']] + c['aliases']: v.add(re.sub(r'\s+', '', a))
    v |= {'방화셔터', '방화문', '배연창', '방화댐퍼', '비상구', '피난구', '옥상광장', '방화구획', '비상방송설비', '자동화재탐지기(감)', '자동화재탐지기(회)'}
    return sorted(v)


_FAC_V = []


def fix_word(s, vocab, thr=0.72):
    """글자 → 어휘 중 가장 가까운 것(유사도 thr 이상, 한글 비교) — (결과, 유사도)"""
    if not s: return s, 0.0
    h = re.sub(r'\s+', '', s)
    if h in vocab: return h, 1.0
    b, sc, sc2 = best_match(h, vocab)
    if b is not None and sc >= thr and sc - sc2 > 0.02: return b, sc
    return h, sc


# ───────────────────────────── 값 보정(칸 종류별)
def _digits(s): return re.sub(r'\D', '', s or '')


def num_text(t, kind):
    """숫자 칸 글자 → (정규 글자, 값) — int: 정수, area: 소수 둘째 자리까지(㎡ 가 숫자로 오독된 꼬리 자름). 못 읽으면 (None, None)"""
    t = (t or '').replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    if kind == 'int':
        m = re.search(r'\d+', t)
        return (m.group(0), int(m.group(0))) if m else (None, None)
    m = re.search(r'\d[\d,.\s]*', t)
    if not m: return None, None
    s = re.sub(r'\s', '', m.group(0)).rstrip('.,')
    m2 = re.match(r'^([\d,.]*?)[.,](\d{1,2})\d*$', s)        # 마지막 구분점 뒤 1~2자리 = 소수
    if m2 and re.search(r'[.,]', s) and len(re.sub(r'\D', '', m2.group(1))) >= 1:
        # 구분점이 여러 개면 마지막 것만 소수점(12.350.29 → 12350.29). 소수 셋째 이하 = ㎡ 오독 꼬리
        parts = re.split(r'[.,]', s)
        # 소수점 후보: 뒤에서 첫 구분점 뒤가 3자리면 천 단위(1,312 → 1312), 아니면 소수
        if len(parts[-1]) == 3 and len(parts) >= 2 and not re.search(r'\.\d{1,2}', s):
            v = float(''.join(parts))
        else:
            ip = ''.join(parts[:-1]); fp = parts[-1][:2]
            v = float(f'{ip or 0}.{fp}')
    else:
        v = float(re.sub(r'\D', '', s) or 'nan')
    if v != v: return None, None
    return f'{v:,.2f}', v


def fix_floor(t):
    """⑩ 층 칸 → '지상9' / '지하1층' / '옥탑1' (양식 표기, '층' 이 있으면 유지). (글자, 신뢰도)"""
    s = re.sub(r'\s+', '', t or '')
    m = re.match(r'^(.{1,3}?)(\d{1,3})(.?)', s)
    if not m:
        if s and jsim(s[:2], '옥탑') > 0.6: return '옥탑1' + ('층' if '층' in s or '충' in s else ''), 0.5
        return None, 0.0
    pre, num, post = m.groups()
    p, sc, _ = best_match(hangul(pre) or pre, FLOOR_PREFIX)
    if sc < 0.4: return None, 0.0
    suf = '층' if post and (post in '층충증흥쯩' or jsim(post, '층') >= 0.6) else ''
    return f'{p}{int(num)}{suf}', min(1.0, sc)


def fix_multiword(t, vocab, sep='·', thr=0.7):
    """'보관 작업장'·'기계실.주차장' → 낱말마다 어휘 보정 후 '·' 로 잇기"""
    s = (t or '').strip()
    if not s: return '', 1.0
    whole, sc = fix_word(s, vocab, thr)
    if sc >= thr: return whole, sc
    parts = [p for p in re.split(r'[\s·ㆍ.,:;/]+', s) if p]
    if len(parts) >= 2:
        fx = [fix_word(p, vocab, thr) for p in parts]
        if all(f[1] >= thr for f in fx): return sep.join(f[0] for f in fx), min(f[1] for f in fx)
    # 붙어서 읽힌 두 낱말(보관작업장) → 가능한 자리에서 나눠 보기
    h = re.sub(r'[^가-힣]', '', s)
    best = (re.sub(r'\s+', '', s), sc)
    for k in range(2, len(h) - 1):
        a, sa = fix_word(h[:k], vocab, thr); b, sb = fix_word(h[k:], vocab, thr)
        if sa >= thr and sb >= thr and min(sa, sb) > best[1]: best = (a + sep + b, min(sa, sb))
    return best


FAC_SUFFIX = ['헤드', '스피커', '감', '회', '방수구', '송수구', '부속실', '거실', '쌍구형', '단구형', '안테나', '접속단자', '감지기', '회로', '옥내']


def _fix_fac_name(nm):
    global _FAC_V
    if not _FAC_V: _FAC_V = fac_vocab()
    nm = re.sub(r'\s+', '', nm).replace('（', '(').replace('）', ')').strip('.,·ㆍ')
    nm = re.sub(r'[\u3131-\u318E]', '', nm)                          # 떨어진 자모(ㅁ·ㅣ 등 잡음) 제거
    nm = re.sub(r'^[5$§]/', 'S/', nm).replace('『', 'P').replace('「', 'P')    # kor 판독의 영문 오독(S/P헤드)
    if nm in _FAC_V: return nm, 1.0
    b, sc, sc2 = best_match(nm, _FAC_V)
    if sc >= 0.8: return b, sc
    # 꼬리 괄호(쌍구형 등) 분리 후 재시도
    m = re.match(r'^(.*?)(\(([^()]*)\)?)$', nm)
    if m and m.group(1):
        base, sfx = m.group(1), m.group(3) or ''
        b1, s1 = _fix_fac_name(base) if base != nm else (base, 0)
        sf, ss, _ = best_match(sfx, FAC_SUFFIX)
        if s1 >= 0.8 and ss >= 0.6:
            cand = f'{b1}({sf})'
            if cand in _FAC_V: return cand, min(s1, ss)
            return cand, min(s1, ss) * 0.95
    if b is not None and sc >= 0.68 and sc - sc2 > 0.04: return b, sc
    return nm, sc


FAC_SEP = r"[-–—~_=÷'`ㅡ一―]"                                       # ':' 오독 문자(가는 두 점이 한 획·물결로 읽힘)


def _split_fac(nm):
    """이름 자리 글자에 'A<구분>N B' 꼴(A·B = 소방시설 어휘, N = 수량)이 붙어 있으면 [(A,N,유사도), …, (B,None,유사도)] — 못 나누면 None.
    문법: 시설명 [-:;.,~=…]? 숫자 시설명 — 양쪽 이름이 어휘와 0.8 이상 맞을 때만(숫자 든 이름 'CO2'·'할론1301' 은 어휘 그대로 지킴)"""
    best = None
    for m in re.finditer(r'(?:[-–—~_=÷\'`ㅡ一―:;.,]?)(\d{1,6})', nm):
        a, b = nm[:m.start()].strip(',.·'), nm[m.end():].strip(',.·:;')
        if not a or not b or not re.search(r'[가-힣A-Za-z)]$', a) or not re.match(r'^[가-힣A-Za-z(]', b): continue
        fa, sa = _fix_fac_name(a)
        if sa < 0.8: continue
        rb = _split_fac(b)
        fb, sb = (rb[-1][0], rb[-1][2]) if rb else _fix_fac_name(b)
        if sb < 0.8: continue
        cand = [(fa, int(m.group(1)), sa)] + (rb[:-1] if rb else []) + [(fb, None, sb)]
        if best is None or min(c[2] for c in cand) > min(c[2] for c in best): best = cand
    return best


def fix_fac(t, conf=90.0):
    """소방시설 칸 OCR 글자 → ('시설명:수량 …' 정규형, 신뢰도, 항목 수). 이름은 fac_catalog 별칭 어휘로 보정, 수량은 숫자.
    항목 경계 = 수량 숫자 뒤 공백/줄끝(이름 안 공백·줄바꿈은 붙임: '자동화재탐지기\n(감)')"""
    lines = [re.sub(r'\s+', ' ', x).strip() for x in (t or '').split('\n') if x.strip()]
    txt = ''
    for ln in lines:
        ln = ln.replace('：', ':').replace(';', ':')
        if not txt: txt = ln; continue
        txt += (' ' if re.search(r'[:]\s*\d+$', txt) else '') + ln
    txt = re.sub(r'할\s*론\s*1301', '할론\x00', txt)                 # 숫자 든 시설명(할론1301)은 수량 규칙에서 지킴
    txt = re.sub(r'(?<=[가-힣)])\s*[.,](?=\s*\d)', ':', txt)           # '유도등.53' → ':'
    txt = re.sub(r'(?<=[가-힣)\x00])\s*' + FAC_SEP + r'\s*(?=\d)', ':', txt)    # ':' 가 '-'·'~'·'=' 등으로 읽힘 '(헤드)-288'
    txt = re.sub(r'(?<=[가-힣)])(?=\d)', ':', txt)                      # 콜론 빠짐 '유도등53'
    # 수량 자리의 가는 '1' 이 ']'·'|'·'l' 로 읽히거나 아예 빠짐('완강기: 제연설비…') → '1' (빠진 경우는 신뢰도 낮춰 검수 플래그)
    txt = re.sub(r'(?<=:)\s*[\]\[|lI!](?=\s|$|[가-힣(])', '1', txt)
    guess = len(re.findall(r':(?=\s*(?:$|[가-힣(]))', txt))
    txt = re.sub(r':(?=\s*(?:$|[가-힣(]))', ':1 ', txt)
    out, scs, pos = [], [], 0
    for m in re.finditer(r'\s*:\s*(\d{1,6})(?=\s|$|[^\d])', txt):
        nm = re.sub(r'\s+', '', txt[pos:m.start()]).strip(',.·').replace('\x00', '1301')
        pos = m.end()
        if not nm: continue
        f, sc = _fix_fac_name(nm)
        if sc < 0.8:                                                  # 앞 항목의 ':수량' 이 이름에 섞여 붙음('…(헤드)·288옥내소화전') → 어휘로 다시 나눔
            sp = _split_fac(nm)
            if sp:
                for (f1, q1, s1) in sp[:-1]: out.append(f'{f1}:{q1}'); scs.append(s1)
                f, sc = sp[-1][0], sp[-1][2]
        out.append(f'{f}:{int(m.group(1))}'); scs.append(sc)
    cf = (min(scs) if scs else 0.0) * min(1.0, max(0.3, conf / 90.0))
    if guess: cf = min(cf, 0.6)
    return ' '.join(out), cf, len(out)


def fix_dist(t1, t2):
    """⑤ 거리 칸: 줄마다 소방서/관할/최단 거리(km) → parse_std 가 읽는 정규 글자. t1 = kor, t2 = eng 재독(숫자·Km)"""
    L1 = [x for x in (t1 or '').split('\n') if x.strip()]; L2 = [x for x in (t2 or '').split('\n') if x.strip()]
    def dec(line):
        c = [x for x in re.findall(r'\d+[.,]\d+', line) if not x.startswith('119')]
        return float(c[-1].replace(',', '.')) if c else None
    vals = []
    for i in range(3):
        a = dec(L2[i]) if i < len(L2) else None
        b = dec(L1[i]) if i < len(L1) else None
        v = a if a is not None else b
        vals.append((v, 0.95 if (a is not None and a == b) else (0.6 if v is not None else 0.0)))
    if len(L1) != 3 and len(L2) != 3:                          # 줄 수가 다르면 전체에서 소수 3개
        c1 = [float(x.replace(',', '.')) for x in re.findall(r'\d+[.,]\d+', ' '.join(L2 or L1))]
        if len(c1) == 3: vals = [(v, 0.6) for v in c1]
    a, b, c = [v[0] for v in vals]
    txt = f"소방서 {a} Km\n119 안전센터 {b} Km(관할)\n119 안전센터(119 지역대) {c} Km(최단)".replace('None', '')
    return txt, min(v[1] for v in vals)


def fix_water(t1, t2):
    """⑥ 소방용수: '종류-관서-동-번호 : 0.150Km' 항목들 '/' 로 구분. 종류는 어휘, 거리는 eng 재독 숫자"""
    s1 = re.sub(r'\s*\n\s*', ' ', t1 or '')
    parts = [p for p in re.split(r'\s*[/|]\s*|(?<=[Kk][Mm])\s+', s1) if p.strip()]
    ds = re.findall(r'(?<![\d.,])(\d{1,2}[.,]\d{2,3})(?![\d])', re.sub(r'\s*\n\s*', ' ', t2 or ''))
    if len(ds) < len(parts): ds = []                              # eng 재독이 줄을 놓치면 kor 판독에서(소수 3자리까지, 뒤 'Km' 오독 숫자 버림)
    out, scs = [], []
    for i, p in enumerate(parts):
        p = re.sub(r'\s+', '', p)
        m = re.match(r'^(.*?)[:：;](.*)$', p)
        head, tail = (m.group(1), m.group(2)) if m else (p, '')
        toks = [x for x in re.split(r'[-~_=—–]+', head) if x]
        if len(toks) < 2: continue
        kind, sk = fix_word(toks[0], WATER_KINDS, 0.5)
        num = _digits(toks[-1]) or toks[-1]
        mid = toks[1:-1] if len(toks) >= 4 else toks[1:-1] + ['']
        d = ds[i] if i < len(ds) else (re.findall(r'\d{1,2}[.,]\d{1,3}', tail) or [None])[0]
        if d is None: continue
        d = d.replace(',', '.')
        out.append(f"{kind}-{'-'.join(mid[:2])}-{num} : {d}Km")
        scs.append(min(sk, 0.95 if i < len(ds) else 0.6))
    return ' / '.join(out), (min(scs) if scs else 0.0)


def fix_addr(t):
    """④ 소재지: '(지번) …' / '(도로명) …' 표지 보정"""
    lines = [x.strip() for x in (t or '').split('\n') if x.strip()]
    out = []
    for ln in lines:
        m = re.match(r'^[\(\[\{]?\s*([가-힣]{1,4})\s*[\)\]\}]?\s*(.*)$', ln)
        if m:
            k, sc, _ = best_match(m.group(1), ['지번', '도로명'])
            if sc >= 0.5 and len(m.group(1)) <= 4: out.append(f'({k}) {m.group(2).strip()}'); continue
        out.append(ln)
    if len(out) == 2 and not out[0].startswith('(지번)') and not out[1].startswith('(도로명)'):
        out = ['(지번) ' + out[0], '(도로명) ' + out[1]]
    return '\n'.join(out)


def fix_qty(t, item=''):
    """⑨ 수량 '1,000ℓ'·'150kg'·'1,200㎥'·'-' — 단위 오독(ℓ→l/1/e) 보정"""
    s = re.sub(r'\s+', '', t or '')
    if not s or re.fullmatch(r'[-—–~_]+', s): return '-' if s else '', 0.9
    m = re.match(r'^(\d{1,3}(?:[,.]\d{3})*|\d+)(.*)$', s)
    if not m: return s, 0.3
    n, u = m.group(1).replace('.', ','), m.group(2)
    liquid = any(k in (item or '') for k in ('석유', '유류', '경유', '등유', '휘발유', '윤활유', '알코올', '중유'))
    gas = any(k in (item or '') for k in ('산소', '가스', '수소', '질소', 'LNG'))
    ul = u.lower()
    if 'kg' in ul or ul in ('k9', 'kq', 'ky'): return n + 'kg', 0.9
    if ul in ('t', '톤'): return n + '톤', 0.9
    if u in ('ℓ', 'L') : return n + 'ℓ', 0.9
    if u in ('㎥', 'm3', 'm'): return n + '㎥', 0.85
    if liquid: return n + 'ℓ', 0.7
    if gas: return n + '㎥', 0.6
    return n + u, 0.4


# ───────────────────────────── 쪽 기하(기울기·표·글줄)
def _bordered(hm, vm, b, ppt):
    """표 후보가 바깥 테두리(위·아래 가로선, 왼·오른 세로선)를 가졌나 + 전폭 가로선 수·전고 세로선 수"""
    x0, y0, x1, y1 = b
    H = hm[y0:y1, x0:x1] > 0; V = vm[y0:y1, x0:x1] > 0
    hr = H.mean(1); vc = V.mean(0)
    t = max(3, int(3 * ppt))
    top, bot = hr[:t].max(), hr[-t:].max(); lef, rig = vc[:t].max(), vc[-t:].max()
    full_h = len(_peaks(hr, 0.9, max(2, int(2.5 * ppt)))); full_v = len(_peaks(vc, 0.9, max(2, int(2.5 * ppt))))
    return min(top, bot) > 0.8 and min(lef, rig) > 0.8, full_h, full_v


def text_lines(ink, hm, vm, boxes, ppt):
    """표 밖 글줄 상자 [(x0,y0,x1,y1)] — 선 제거 후 가로로 이어 붙인 덩어리"""
    m = ink.copy()
    lines = cv2.dilate(((hm | vm) > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    m[lines > 0] = 0
    for x0, y0, x1, y1 in boxes: m[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = 0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(7 * ppt)), max(1, int(0.8 * ppt))))
    mm = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 3.5 * ppt or h > 40 * ppt or w < 4 * ppt: continue
        if m[y:y + h, x:x + w].sum() < 6 * ppt * ppt: continue
        out.append([x, y, x + w, y + h])
    # 같은 줄(세로 겹침 60% 이상) + 간격 15pt 이내 합치기
    out.sort(key=lambda b: (b[1], b[0]))
    merged = []
    for b in sorted(out, key=lambda b: b[0]):
        for c in merged:
            ov = min(c[3], b[3]) - max(c[1], b[1])
            cc, cb = (c[1] + c[3]) / 2, (b[1] + b[3]) / 2
            same = ov > 0.6 * min(c[3] - c[1], b[3] - b[1]) or (ov > 0 and abs(cc - cb) < 0.35 * max(c[3] - c[1], b[3] - b[1]))
            if same and b[0] - c[2] < 15 * ppt and b[0] >= c[0]:
                c[0], c[1], c[2], c[3] = min(c[0], b[0]), min(c[1], b[1]), max(c[2], b[2]), max(c[3], b[3]); break
        else: merged.append(list(b))
    return sorted([tuple(b) for b in merged], key=lambda b: (b[1], b[0]))


def page_geom(page):
    t0 = time.time()
    img, ppt = page_image(page)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = flatten(gray); ink = ink_mask(g)
    ang = skew_angle(ink)
    if abs(ang) >= 0.04:
        g = rotate(g, ang); img = rotate(img, ang); ink = ink_mask(g)
    hm, vm = line_masks(ink, ppt)
    tabs = []
    raw_tabs = find_tables(hm, vm, ppt)
    for b in raw_tabs:
        ok, fh, fv = _bordered(hm, vm, b, ppt)
        if ok: tabs.append(dict(box=b, full_h=fh, full_v=fv, frame=(fh <= 2 and fv <= 2 and (b[3] - b[1]) > 40 * ppt)))
    tl = text_lines(ink, hm, vm, [t['box'] for t in tabs], ppt)
    return dict(img=img, g=g, ink=ink, hm=hm, vm=vm, ppt=ppt, angle=ang, tables=tabs, raw_tables=raw_tabs, tlines=tl, t_geom=time.time() - t0,
                w_pt=page.rect.width, h_pt=page.rect.height)


FIG_WORDS = ['평면도', '위치도', '정면사진', '측면사진', '후면사진', '사진', '진압작전도']


def norm_line(t):
    """표 밖 글줄 OCR → 절 제목·⑪ 제목·그림 캡션 정규형(text-layer 와 같은 모양). (글자, 종류) 종류: sec2|title11|cap|None"""
    s = re.sub(r'\s+', ' ', (t or '').strip())
    ns = re.sub(r'\s+', '', s)
    h = hangul(ns)
    # ⑪ 제목 / 2. 절 제목: '동별위치도사진및층별평면도'
    key = '동별위치도사진및층별평면도'
    if len(h) >= 8 and jsim(h[:len(key) + 2][:len(key)], key) >= 0.7 or ('동별위치도' in h and '평면도' in h):
        m = re.search(r'동\s*명\s*[:：;]?\s*(\S+)\s*$', s)
        lead = ns[:3]
        if re.match(r'^[2２][.,]', lead) or (lead[:1] == '2' and not lead[1:2].isdigit()):
            return '2. 동별 위치도, 사진 및 층별 평면도', 'sec2'
        name = None
        if m: name = m.group(1)
        elif '동명' in h:
            name = s.split('동명')[-1].strip(' :：;') or None
        return '⑪ 동별 위치도, 사진 및 층별 평면도' + (f' — 동명: {name}' if name else ''), 'title11'
    # 평면도 캡션: 짧은 줄 + '(지하|지상|옥탑…' (번호·'평면도' 글자 오독 허용)
    if len(ns) <= 32 and not re.match(r'^[※#*]', ns) and re.search(r'\(\s*(지\s*[하상삿싱]|옥\s*탑)', s) and not re.search(r'적습니다|처럼', s):
        mnum = re.match(r'^\s*(\d{1,3})', s)
        inner, ok = plan_caption_floors(s[s.index('('):])
        return f"{mnum.group(1) if mnum else '0'}. 평면도({inner})", ('cap' if ok else 'bad_cap')
    # 캡션: 번호 + 그림 종류 (번호 오독 '고.'·'!'·'요.' 허용: 짧은 줄의 평면도·위치도 캡션만)
    m = re.match(r'^[\[\(]?\s*([0-9OoIl]{1,3})\s*[.,]?\s*(.*)$', s)
    if not m and len(ns) <= 30 and not re.match(r'^[※#*]', ns):
        m2 = re.match(r'^(\S{1,2}?)\s*[.,!]?\s*(\(?\s*지.*|평.*|위\s*치\s*도.*)$', s)
        if m2 and not hangul(m2.group(1))[:1] in ('측', '후', '정', '관', '아'):
            m = re.match(r'^(.*)$', '0 ' + m2.group(2)); m = re.match(r'^([0-9])\s*(.*)$', '0 ' + m2.group(2))
    if m and m.group(2):
        num = m.group(1).replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        rest = m.group(2); rh = hangul(rest.split('(')[0])
        if '평면' in rest or '면도' in rest or jsim(hangul(rest)[:5], '평면도') >= 0.6 or re.search(r'평.?면.?도', rest) or \
                (len(rest) < 40 and re.search(r'\(\s*지|지\s*[하상]', rest) and re.search(r'.도', rest)):
            rh = '평면도'
        if rh:
            w, sc, _ = best_match(rh[:6], FIG_WORDS)
            if sc >= 0.6:
                if w == '평면도':
                    inner, ok = plan_caption_floors(rest)
                    return f'{num}. 평면도({inner})', ('cap' if ok else 'bad_cap')
                if w == '진압작전도': return f'{num} 진압작전도', 'cap'
                if w == '위치도': return f'{num}. 위치도', 'cap'
                return f'{num} {w}', 'cap'
    return s, None


def plan_caption_floors(rest):
    """평면도 캡션 나머지 글자(순서가 뒤섞여도: '(지하평면도 2 충)') → ('지하2층' | '지상2층~지상9층 기준층', 층 번호 모두 읽음)"""
    s = re.sub(r'평\s*면\s*도|[뒤며졌]\s*[면먼]\s*도|면\s*도', ' ', rest)
    s = s.replace('〜', '~').replace('∼', '~')
    toks = list(re.finditer(r'(지\s*[하상삿싱]|옥\s*탑|저\s*상|져\s*상)\s*([0-9]{1,3})?', s))
    fl, ok = [], bool(toks)
    for m in toks:
        pre, sc, _ = best_match(re.sub(r'\s', '', m.group(1)), FLOOR_PREFIX)
        num = m.group(2)
        if num is None:
            after = s[m.end():m.end() + 6]
            d = re.search(r'\d{1,3}', after)
            if d and not re.search(r'지|옥', after[:d.start()]): num = d.group(0)
        if num is None:
            if pre == '옥탑': num = '1'
            else: ok = False; continue
        fl.append(f'{pre}{int(num)}층')
    if not fl: return fix_caption_floors(rest), False
    base = '~'.join(fl[:2]) if (len(fl) >= 2 and ('~' in s or '-' in s)) else ','.join(fl)
    if '기준' in s or jsim(hangul(s)[-3:], '기준층') > 0.6: base += ' 기준층'
    return base, ok


def fix_caption_floors(t):
    """'지하2층' / '지상2층~지상9층 기준층' OCR 보정"""
    s = re.sub(r'\s+', '', t or '').replace('〜', '~').replace('∼', '~').replace('-', '~') if '~' not in (t or '') and re.search(r'\d\s*층?\s*-\s*[지]', t or '') else re.sub(r'\s+', '', t or '').replace('〜', '~').replace('∼', '~')
    toks = list(re.finditer(r'([가-힣]{1,3}?)(\d{1,3})\s*([가-힣]?)', s))
    out, last = '', 0
    for m in toks:
        pre, sc, _ = best_match(hangul(m.group(1))[-2:] or m.group(1), FLOOR_PREFIX)
        if sc < 0.4: continue
        out += s[last:m.start()] + f'{pre}{int(m.group(2))}층'
        last = m.end()
    out += s[last:]
    out = re.sub(r'층층', '층', out)
    if '기준' in out or jsim(hangul(out)[-3:], '기준층') > 0.6:
        out = re.sub(r'(층)[^~층]*$', r'\1 기준층', out) if not out.endswith('기준층') else out.replace('기준층', ' 기준층').replace('  ', ' ')
    return out


# ───────────────────────────── 양식 표 OCR → 행
def _label_of(txt):
    """칸 글자 → (출력 칸 이름, 유사도, 무리) | (None, 유사도, None). 앞머리 동그라미 숫자 오독(®·%·2 …) 무시, 한글만 비교"""
    h = hangul(txt)
    if not h or len(h) > 16: return None, 0.0, None
    if re.search(r'[가-힣]\s*\d{1,3}\s*[가-힣]?\s*$', txt.strip()) and _floorish(txt): return None, 0.0, None     # '지하1층' = 층 값(이름 '지하 층수' 아님)
    sc = sorted(((jsim(h, l[0]), k) for k, l in enumerate(LABELS)), reverse=True)
    s1, k1 = sc[0]; s2 = sc[1][0]
    L = LABELS[k1]
    thr = 0.72 if len(L[0]) <= 2 else 0.66
    if s1 >= thr and s1 - s2 > 0.03: return L[1], s1, L[2]
    return None, s1, None


def form_rows(P, tables, pn, state=None):
    """양식 표들 → rows [(쪽, y_pt, [칸 글자|None])] + meta(칸별 종류·신뢰도)"""
    ppt, g = P['ppt'], P['g']
    grids, tiles, where = [], [], []
    for ti, tb in enumerate(tables):
        ys, xs, cells = cell_grid(P['hm'], P['vm'], tb['box'], ppt)
        grids.append((ys, xs, cells))
        for ci, c in enumerate(cells):
            t = clean_cell(g, c['box'], ppt)
            c['tile'] = t; c['ppt'] = ppt
            if t is not None: tiles.append(t); where.append((ti, ci))
    t0 = time.time()
    res = ocr_tiles(tiles, ppt, lang='kor')
    for (ti, ci), r in zip(where, res):
        c = grids[ti][2][ci]
        c['t1'], c['c1'], _ = lines_text(r)
    for ys, xs, cells in grids:
        for c in cells:
            if 't1' not in c: c['t1'], c['c1'] = '', -1.0
            c['label'], c['lsim'], c['lgrp'] = _label_of(c['t1']) if c['t1'] else (None, 0.0, None)
    # 행 구성(pymupdf find_tables 모양: 병합으로 가려진 칸 None)
    rows = []
    for (ys, xs, cells) in grids:
        nr, nc = len(ys) - 1, len(xs) - 1
        for r in range(nr):
            rc = sorted([c for c in cells if c['r0'] == r], key=lambda c: c['c0'])
            if not rc: continue
            arr = [None] * nc
            for c in rc: arr[c['c0']] = c
            rows.append(dict(y=ys[r] / ppt, cells=arr))
    _unlabel_values(rows)
    _template_fill(rows)
    kinds = _value_kinds(rows, state)
    # 2차: 숫자·영문 칸 재독(eng)
    numc = [c for c in kinds if c['kind'][0] in ('int', 'area', 'phone', 'farea', 'floor') and c['tile'] is not None]
    mixc = [c for c in kinds if c['kind'][0] in ('dist', 'water', 'id', 'hazqty', 'addr') and c['tile'] is not None]
    if numc:
        for c, r in zip(numc, ocr_tiles([c['tile'] for c in numc], ppt, lang='eng', wl='0123456789.,-')): c['t2'] = lines_text(r)[0]
    if mixc:
        for c, r in zip(mixc, ocr_tiles([c['tile'] for c in mixc], ppt, lang='eng')): c['t2'] = lines_text(r)[0]
    flc = [c for c in kinds if c['kind'][0] in ('floor', 'id', 'hazname', 'hazitem') and c['tile'] is not None]       # 한글+숫자 칸: kor+eng 3차
    if flc:
        for c, r in zip(flc, ocr_tiles([c['tile'] for c in flc], ppt, lang='kor+eng')): c['t3'] = lines_text(r)[0]
    t_ocr = time.time() - t0
    meta = []
    for row in rows:
        out = []
        for c in row['cells']:
            if c is None: out.append(None); continue
            if c.get('label'):
                out.append(c['label']); c['conf'] = min(1.0, c['lsim']); continue
            txt, cf = _fix_value(c, c.get('kind'))
            c['text'], c['conf'] = txt, cf
            out.append(txt)
        row['out'] = out
    return [(pn, row['y'], row['out']) for row in rows], rows, t_ocr


VALUE_AFTER = {'① 관리번호', '② 주 용 도', '③ 상 호', '④ 소 재 지', '⑤ 소방대상물과의거리', '⑥ 소방용수시설', '양 식', '구 조', '지 붕', '동 명', '성 명', '연 락 처'} | set(NUM_LABELS)


def _unlabel_values(rows):
    """값을 받는 칸 이름 바로 다음 칸은 값 — 다른 무리 이름으로 잘못 맞았으면(업무시설→'위험물 시설') 이름 해제(같은 글자 '양식' 은 유지)"""
    for row in rows:
        cs = [c for c in row['cells'] if c is not None]
        for i, c in enumerate(cs[:-1]):
            if c.get('label') in VALUE_AFTER:
                n = cs[i + 1]
                if n.get('label') and n['label'] != c['label'] and not (n['lgrp'] == c['lgrp'] and n['lsim'] >= 0.9 and c['lgrp'] not in ('top', 'rel')):
                    n['label'], n['lgrp'] = None, None
        # ⑨ 구분 칸 이름(위험물 시설 등)은 ⑨ 행 둘째 칸에만
        for i, c in enumerate(cs):
            if c.get('lgrp') == 'haz' and i > 1 and c['lsim'] < 0.95: c['label'], c['lgrp'] = None, None


def _template_fill(rows):
    """⑦ 격자 등: 한 행에서 같은 무리 칸 이름이 2개 이상 맞으면, 못 읽은 이름 칸(값 칸과 번갈아)을 양식 순서로 채움"""
    for row in rows:
        cs = [c for c in row['cells'] if c is not None]
        grp = [c['lgrp'] for c in cs if c.get('label')]
        for gname in ('s0', 's1', 's2', 's3', 'top', 'fh', 'hh'):
            if grp.count(gname) < 2 and not (gname == 's3' and grp.count(gname) == 1): continue
            tpl = TEMPLATE_ROWS[gname]
            idx = [i for i, c in enumerate(cs) if c.get('lgrp') == gname]
            if gname in ('fh', 'hh'):
                start = idx[0] - tpl.index(cs[idx[0]]['label'])
                for j, name in enumerate(tpl):
                    i = start + j
                    if 0 <= i < len(cs) and not cs[i].get('label') and name not in [c.get('label') for c in cs]:
                        cs[i]['label'], cs[i]['lsim'], cs[i]['lgrp'] = name, 0.5, gname
                continue
            # 이름·값 번갈아: 첫 맞은 이름 자리에서 양식 순서 역산
            i0 = idx[0]; k0 = tpl.index(cs[i0]['label'])
            for j, name in enumerate(tpl):
                i = i0 + 2 * (j - k0)
                if 0 <= i < len(cs) and not cs[i].get('label') and name not in [c.get('label') for c in cs] and hangul(cs[i]['t1']) != hangul(name):
                    if cs[i]['t1'] and len(hangul(cs[i]['t1'])) > len(hangul(name)) + 3: continue
                    cs[i]['label'], cs[i]['lsim'], cs[i]['lgrp'] = name, 0.5, gname
                elif 0 <= i < len(cs) and not cs[i].get('label') and hangul(cs[i]['t1']) == hangul(name):
                    cs[i]['label'], cs[i]['lsim'], cs[i]['lgrp'] = name, 1.0, gname


def _floorish(t):
    h = hangul(t)
    if not h or len(h) > 4 or len(re.sub(r'\s', '', t or '')) > 7: return False
    return best_match(h[:2], FLOOR_PREFIX)[1] >= 0.6


def _value_kinds(rows, state=None):
    """값 칸 → (종류, 부가) — 왼쪽 이름 칸 / ⑩ 머리행 열 / ⑧·⑨ 행 구조로 결정. state: 앞 쪽에서 이어지는 절(⑩ 이 쪽을 넘김·머리행 반복 없음)"""
    kinds = []
    sec, fhdr = (state or {}).get('sec'), None
    for row in rows:
        cells = row['cells']
        cs = [c for c in cells if c is not None]
        labs = [c.get('label') for c in cs]
        if any(l == '⑧ 관계인등' or (l in ('소유자', '관리자', '소방안전관리자', '위험물안전관리자')) for l in labs): sec = 'rel'
        if '구 분' in labs and '품 명' in labs: sec = 'haz'; continue
        if '층 별' in labs and '소방시설 설치현황' in labs:
            sec = 'floor'; fhdr = {c['c0']: c['label'] for c in cs if c.get('label')}; continue
        if any(l in ('① 관리번호', '④ 소 재 지', '⑥ 소방용수시설') for l in labs): sec = 'top'
        if any(c.get('lgrp') in ('s0', 's1', 's2', 's3') for c in cs): sec = 'struct'
        if sec == 'floor':
            fc = next((c for c in cs if not c.get('label') and _floorish(c['t1'])), None)
            if fc is None: continue
            fc['kind'] = ('floor', None)
            rest = [c for c in cs if c['c0'] > fc['c0'] and not c.get('label')]
            if fhdr:
                hcols = sorted(fhdr)
                for c in rest:
                    h = max([k for k in hcols if k <= c['c0']], default=None)
                    nm = fhdr.get(h)
                    c['kind'] = ({'용 도': ('fuse', None), '면 적(㎡)': ('farea', None), '소방시설 설치현황': ('fac', None), '기 타': ('fetc', None)}.get(nm, ('text', None)))
            else:
                for c, k in zip(rest, ('fuse', 'farea', 'fac', 'fetc')): c['kind'] = (k, None)
            continue
        if sec == 'haz':
            hk = next((c for c in cs if c.get('lgrp') == 'haz'), None)
            if hk is not None:
                rest = [c for c in cs if c['c0'] > hk['c0'] and not c.get('label')]
                for c, k in zip(rest, ('hazname', 'hazitem', 'hazqty')): c['kind'] = (k, rest)
            continue
        for i, c in enumerate(cs):
            if not c.get('label'): continue
            nxt = cs[i + 1] if i + 1 < len(cs) else None
            if nxt is None or nxt.get('label') and nxt['label'] != c['label']: continue
            L = c['label']
            k = {'① 관리번호': 'id', '② 주 용 도': 'use', '③ 상 호': 'name', '④ 소 재 지': 'addr', '⑤ 소방대상물과의거리': 'dist', '⑥ 소방용수시설': 'water',
                 '양 식': 'form', '구 조': 'struct', '지 붕': 'roof', '동 명': 'dongname', '성 명': 'pname', '연 락 처': 'phone'}.get(L)
            if L in NUM_LABELS: k = NUM_LABELS[L]
            if k:
                if nxt.get('label') == L: nxt['label'] = None           # 값이 이름과 같은 글자('양식')
                nxt['kind'] = (k, None)
    if state is not None: state['sec'] = sec
    return [c for row in rows for c in row['cells'] if c is not None and c.get('kind')]


def _glyph_count(tile):
    """숫자 칸 잉크 덩어리 수(자리수 추정)와 '가는 한 획(1)' 여부 → (개수, 한획) | None"""
    if tile is None: return None
    m = (tile < 140).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    H = tile.shape[0]
    comps = [st[k] for k in range(1, n) if st[k, 3] >= 0.35 * H and st[k, 4] >= 6]
    if not comps: return None
    comps.sort(key=lambda b: b[0])
    merged = []                                                   # 겹치는 열(같은 글자 조각) 합치기
    for b in comps:
        if merged and b[0] < merged[-1][0] + merged[-1][2] - 1: continue
        merged.append(b)
    thin = len(merged) == 1 and merged[0][2] < 0.42 * merged[0][3]
    return len(merged), thin


def _fix_haz(s, kind):
    V = HAZ_NAMES if kind == 'hazname' else HAZ_ITEMS
    if not s: return '', 0.0
    m = re.match(r'^(.*?)(\((.*)\))?$', s)
    base, par = (m.group(1), m.group(3)) if m else (s, None)
    b, sc = fix_word(base, V, 0.7)
    if par: p2, sp = fix_word(par, V + HAZ_ITEMS + ['용기', '제4류', '제1류', '제2류', '제3류', '제5류', '제6류', '발포'], 0.6); b = f'{b}({p2})'; sc = min(sc, sp)
    return b, sc


def patch_hyphens(s, t2):
    """kor 판독에서 빠진 숫자 사이 하이픈을 eng 재독에서 되살림('11455' → '1-1455')"""
    for m in re.finditer(r'\d+(?:-\d+)+', t2 or ''):
        key = m.group(0).replace('-', '')
        s = re.sub(r'(?<![\d-])' + re.escape(key) + r'(?![\d-])', m.group(0), s)
    return s


def _fix_value(c, k):
    """칸 값 보정 → (글자, 신뢰도 0~1)"""
    t1, c1 = c.get('t1') or '', max(0.0, c.get('c1', -1)) / 100.0
    t2 = c.get('t2')
    kind = k[0] if k else 'text'
    if not t1.strip() and not (t2 or '').strip(): return '', 1.0
    if kind == 'int':                                  # 정수 칸(층수·계단수·승강기): 두 판독 + 글자 덩어리 수로 자리수 확인, 가는 한 획 = '1'
        n = _glyph_count(c.get('tile'))
        cand = []
        for t, w in ((t2, 1.0), (t1, 0.9)):
            d = _digits((t or '').replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1'))
            if d and len(d) <= 3: cand.append((d, w))
        if n and n[0] == 1 and n[1] and not any(len(d) == 1 for d, _ in cand): cand.append(('1', 0.8))
        if (not cand or len({d for d, _ in cand}) > 1 or (n and all(len(d) != n[0] for d, _ in cand))) and c.get('tile') is not None:
            psm = 10 if n and n[0] == 1 else 8                      # 한 글자/한 낱말 판독(칸마다 1회 — 흔하지 않음)
            d = _digits(lines_text(ocr_tiles([c['tile']], c.get('ppt', 2.4), lang='eng', wl='0123456789', psm=psm, target=6.0)[0])[0])
            if d and len(d) <= 3: cand.append((d, 1.1))
        if not cand: return (t1 or '').strip(), 0.2
        score = {}
        for d, w in cand: score[d] = score.get(d, 0) + w + (0.6 if n and len(d) == n[0] else 0)
        best = max(score, key=score.get)
        agree = len({d for d, _ in cand}) == 1 and len(cand) >= 2
        return str(int(best)), (0.97 if agree else (0.7 if n and len(best) == n[0] else 0.45))
    if kind in ('area', 'farea'):
        nk = 'area'
        a, va = num_text(t2, nk) if t2 else (None, None)
        b, vb = num_text(t1, nk)
        v, s = (a, va) if a is not None else (b, vb)
        if v is None: return t1, 0.2
        agree = va is not None and vb is not None and abs(va - vb) < 1e-6
        cf = 0.97 if agree else 0.55
        if kind == 'farea': return f'{v}㎡', cf
        return v if nk == 'area' else str(s), cf
    if kind == 'phone':
        s = re.sub(r'[^\d-]', '', t2 or t1)
        ok = re.fullmatch(r'\d{2,4}-\d{3,4}-\d{4}', s) is not None
        return s, (0.9 if ok and _digits(t1) == _digits(s) else (0.7 if ok else 0.3))
    if kind == 'floor':
        t3 = c.get('t3') or ''
        f1, s1 = fix_floor(t1); f3, s3 = fix_floor(t3)
        d1, d3, d2 = _digits(t1), _digits(t3), _digits(t2)
        # 층 번호 투표: kor · kor+eng 가 같으면 확실. 숫자 전용 재독(t2)은 한글을 숫자로 오독해 앞에 붙기 쉬움('지상6' → '76') → 끝자리 일치로만 지지
        votes = {}
        for d, w in ((d1, 1.0), (d3, 1.0)):
            if d and len(d) <= 3: votes[d] = votes.get(d, 0) + w
        for d in list(votes):
            if d2 and d2.endswith(d): votes[d] += 0.5
        if not votes and d2 and len(d2) <= 2: votes[d2] = 0.5
        if not votes: return (f1 or f3 or t1), 0.2
        num = max(votes, key=lambda d: (votes[d], -len(d)))
        h = hangul(t1) or hangul(t3)
        p, sp, _ = best_match(h[:2], FLOOR_PREFIX)
        p3, sp3, _ = best_match(hangul(t3)[:2], FLOOR_PREFIX)
        if sp3 > sp: p, sp = p3, sp3
        suf = '층' if any((f or '').endswith('층') for f in (f1, f3)) or (len(h) >= 3 and jsim(h[-1], '층') >= 0.6) else ''
        cf = min(sp, 0.97) * (1.0 if votes[num] >= 2 else (0.75 if votes[num] >= 1.5 else 0.5))
        return f'{p}{int(num)}{suf}', cf
    if kind == 'fac':
        s, cf, n = fix_fac(t1, c.get('c1', 90))
        return s, cf
    if kind == 'fuse':
        return fix_multiword(t1, FLOOR_USES)[0], fix_multiword(t1, FLOOR_USES)[1] * min(1, c1 + 0.2)
    if kind == 'fetc':
        if not hangul(t1): return re.sub(r'[^\w·]', '', t1), 0.8
        w, sc = fix_word(t1, ETC_WORDS, 0.6); return w, sc
    if kind == 'use':
        w, sc = fix_word(t1, USES, 0.7); return w, sc
    if kind == 'struct':
        w, sc = fix_word(t1, STRUCTS, 0.7); return w, sc
    if kind == 'roof':
        w, sc = fix_word(t1, ROOFS, 0.7); return w, sc
    if kind == 'dist': return fix_dist(t1, t2)
    if kind == 'water': return fix_water(t1, t2)
    if kind == 'addr':
        s = fix_addr(patch_hyphens(t1, t2))
        jb = next((ln for ln in s.split('\n') if ln.startswith('(지번)')), '')
        # 지번 본번·부번은 각 4자리 이하 → 하이픈 없는 5자리 이상 숫자 = 가는 '-' 를 놓친 것(어디서 끊을지는 모름 → 고치지 않고 검수 플래그)
        if re.search(r'(?<![\d-])\d{5,}(?![\d-])', jb): c1 = min(c1, 0.5)
        return s, c1
    if kind == 'id':
        s = re.sub(r'\s+', '', t1)
        s = re.sub(r'(?<=[가-힣])[.,·_~=—–](?=\d)|(?<=\d)[.,·_~=—–](?=[가-힣])', '-', s)     # 한글↔숫자 사이 구분점 오독 → '-'
        n2 = max((t2 or '').count('-'), (c.get('t3') or '').count('-'))
        while s.count('-') < n2:                                                     # eng 재독의 하이픈 수만큼 한글↔숫자 경계에 넣기
            m = re.search(r'(?<=[가-힣A-Za-z])(?=\d)|(?<=\d)(?=[가-힣])', s)
            if not m: break
            s = s[:m.start()] + '-' + s[m.start():]
        d2 = _digits(t2); ok = d2 and _digits(s) and (_digits(s) in d2 or d2.endswith(_digits(s)[-4:]))
        return s, (min(0.95, c1 + 0.1) if ok else c1 * 0.8)
    if kind in ('hazname', 'hazitem'):                     # kor / kor+eng 두 판독 중 어휘에 더 가까운 것('LPG저장(용기)')
        a = _fix_haz(re.sub(r'\s+', '', t1), kind); b = _fix_haz(re.sub(r'\s+', '', c.get('t3') or ''), kind) if c.get('t3') else ('', 0)
        return a if a[1] >= b[1] else b
    if kind == 'hazname':
        s = re.sub(r'\s+', '', t1)
        m = re.match(r'^(.*?)(\((.*)\))?$', s)
        base, par = (m.group(1), m.group(3)) if m else (s, None)
        b, sc = fix_word(base, HAZ_NAMES, 0.7)
        if par: p2, sp = fix_word(par, HAZ_NAMES, 0.6); b = f'{b}({p2})'; sc = min(sc, sp)
        return b, sc
    if kind == 'hazitem':
        s = re.sub(r'\s+', '', t1)
        m = re.match(r'^(.*?)(\((.*)\))?$', s)
        base, par = (m.group(1), m.group(3)) if m else (s, None)
        b, sc = fix_word(base, HAZ_ITEMS, 0.7)
        if par: p2, sp = fix_word(par, HAZ_ITEMS, 0.6); b = f'{b}({p2})'; sc = min(sc, sp)
        return b, sc
    if kind == 'hazqty':
        rest = k[1] or []
        item = next((x.get('text') or x.get('t1') for x in rest if x is not c and x['c0'] < c['c0'] and x is rest[1]), '') if len(rest) > 1 else ''
        q, sc = fix_qty(t2 if t2 and re.search(r'\d', t2) else t1, item or '')
        return q, sc
    if kind in ('pname', 'dongname', 'name', 'form'):
        return re.sub(r'\s+', ' ', re.sub(r'[“”"\'‘’`´′″|]', '', t1)).strip(), c1
    return re.sub(r'[ \t]+', ' ', t1).strip(), c1


# ───────────────────────────── 항목별 신뢰도(필드 이름) — 값 글자는 담지 않음(관계인 개인정보)
_KFIELD = {'id': '관리번호', 'use': '주용도', 'name': '상호', 'addr': '소재지', 'dist': '거리', 'water': '소방용수', 'form': '건물구조.양식', 'struct': '건물구조.구조',
           'roof': '건물구조.지붕', 'dongname': '건물구조.동명'}
_NUMF = {'지하 층수': '지하층수', '지상 층수': '지상층수', '연면적(㎡)': '연면적', '바닥면적(㎡)': '바닥면적', '특별피난계단수': '특별피난계단수', '피난계단수': '피난계단수',
         '일반계단수': '일반계단수', '옥외계단수': '옥외계단수', '승강기(비상)': '승강기_비상', '승강기(일반)': '승강기_일반'}


def field_confs(all_rows):
    """form_rows 의 행(칸 dict) 들 → [(동 번호, 필드, 신뢰도, 개인정보 여부)] 쪽 순서대로"""
    from fire3dbeam.card import norm_floor
    out, dong = [], -1
    for row in all_rows:
        cs = [c for c in row['cells'] if c is not None]
        if any(c.get('label') == '① 관리번호' for c in cs): dong += 1
        d = max(dong, 0)
        role = next((c['label'] for c in cs if c.get('lgrp') == 'role'), None)
        hz = next((hangul(c['label']) for c in cs if c.get('lgrp') == 'haz'), None)
        fl = next((c for c in cs if (c.get('kind') or ('',))[0] == 'floor'), None)
        fid = norm_floor(fl.get('text') or '') if fl is not None else None
        prev = None
        for c in cs:
            k = (c.get('kind') or (None,))[0]
            if k is None:
                prev = c; continue
            if k in _KFIELD: f = _KFIELD[k]
            elif k in ('int', 'area') and prev is not None and prev.get('label') in _NUMF: f = '건물구조.' + _NUMF[prev['label']]
            elif k == 'pname': f = f'관계인.{role}.성명'
            elif k == 'phone': f = f'관계인.{role}.연락처'
            elif k in ('hazname', 'hazitem', 'hazqty'): f = f"위험물.{hz}.{dict(hazname='시설명', hazitem='품명', hazqty='수량')[k]}"
            elif k in ('floor', 'fuse', 'farea', 'fac', 'fetc'): f = f"층별현황.{fid}.{dict(floor='층', fuse='용도', farea='면적', fac='소방시설', fetc='기타')[k]}"
            else: f = k
            if (c.get('text') or '') == '' and k not in ('floor',): prev = c; continue      # 빈 칸
            out.append((d, f, round(float(c.get('conf', 0.0)), 3), k in ('pname', 'phone')))
            prev = c
    return out


# ───────────────────────────── 문서 분석(캐시)
def doc_key(doc):
    h = hashlib.sha1()
    for p in doc:
        if is_image_page(p): h.update(page_key(p).encode())
        else: h.update(b'T' + hashlib.sha1((p.get_text() or '').encode('utf-8', 'replace')).digest())
    return h.hexdigest()


def has_image_pages(doc):
    return any(is_image_page(p) for p in doc)


def analyze(doc):
    """문서 전체(글자층 없는 쪽만 OCR) → dict(pages=[쪽별 None|{rows, lines, frames, sec2_y, …}], text, fields, seconds).
    같은 내용 문서는 캐시(프로세스 안)"""
    key = doc_key(doc)
    if key in _CACHE: return _CACHE[key]
    t_all = time.time()
    pages, after, all_rows, texts, per_page, state = [], False, [], [], [], {}
    for page in doc:
        if not is_image_page(page):
            pages.append(None)
            if SEC2_RE.search(page.get_text() or ''): after = True
            continue
        t0 = time.time()
        P = page_geom(page); ppt = P['ppt']
        tl = P['tlines']
        tiles = [line_tile(P, b) for b in tl]
        lines = []
        retry = 0
        for b, tile, r in zip(tl, tiles, ocr_tiles(tiles, ppt, lang='kor', target=LINE_PPT) if tiles else []):
            raw, cf, _ = lines_text(r, ' ')
            if not raw.strip() and retry < 6 and (b[3] - b[1]) >= 5 * ppt and (tile < 140).mean() > 0.04:
                # 쌓은 한 장 판독에서 통째로 빠진 글줄(굵은 밑줄 캡션 '00 정면사진' 이 사라지는 경우) → 이 줄만 다시(한 줄 모드)
                retry += 1
                raw, cf, _ = lines_text(ocr_tiles([tile], ppt, lang='kor', target=LINE_PPT, psm=7)[0], ' ')
            txt, kind = norm_line(raw)
            if kind == 'bad_cap':                                   # 평면도 캡션인데 층 번호를 못 읽음 → 다른 배율로 다시
                for tg in (3.2, 5.5, 4.0):
                    raw2 = lines_text(ocr_tiles([tile], ppt, lang='kor', target=tg)[0], ' ')[0]
                    t2, k2 = norm_line(raw2)
                    if k2 == 'cap': txt, kind = t2, k2; break
            if not txt: continue
            lines.append(dict(bbox=[b[0] / ppt, b[1] / ppt, b[2] / ppt, b[3] / ppt], text=txt, kind=kind, conf=cf))
        sec2 = [l['bbox'][1] for l in lines if l['kind'] == 'sec2']
        if not sec2 and not after: sec2 = [l['bbox'][1] for l in lines if l['kind'] == 'title11'][:1]
        sec2_y = min(sec2) if sec2 else None
        form_t, frames = [], []
        for tb in P['tables']:
            y0 = tb['box'][1] / ppt
            if after or (sec2_y is not None and y0 > sec2_y) or tb['frame']: frames.append(tb)
            else: form_t.append(tb)
        rows, rraw, t_ocr = form_rows(P, form_t, page.number, state) if form_t else ([], [], 0.0)
        all_rows += rraw
        figs = []
        if after or sec2_y is not None:                           # ⑪ 그림 영역: 표 선 틀 + 내용 덩어리(틀 선이 흐린 사진·위치도)
            regs = figure_regions(P, sec2_y if not after else None)
        else:
            regs = [tb['box'] for tb in frames]
        regs = [b for b in regs if not any(o is not b and _inside(b, o) for o in regs)]     # 틀 안의 틀(사진 속 사각형·작전도 테두리) 제외
        wb = white_balance(P) if regs else None
        for x0, y0, x1, y1 in sorted(regs, key=lambda b: b[1]):
            t = max(3, int(2.0 * ppt))
            if (y1 - y0) < 20 * ppt: continue
            crop = fig_crop(wb, (x0 + t, y0 + t, x1 - t, y1 - t), ppt)
            figs.append(dict(bbox=[x0 / ppt, y0 / ppt, x1 / ppt, y1 / ppt], img=crop))
        if FIG_DROP and figs:                                         # 빈 쪽·설명(※) 쪽 과검출 억제: 캡션 없는 영역 중 잉크가 거의 없거나(종이 얼룩·잡음) 낮은 띠(글줄 덩어리)
            prev_cap = bool(pages) and pages[-1] is not None and bool(pages[-1]['lines']) and pages[-1]['lines'][-1]['kind'] in ('cap', 'bad_cap')
            keep = []
            for i, f in enumerate(figs):
                cap = any(l['kind'] in ('cap', 'bad_cap') and 0 <= f['bbox'][1] - l['bbox'][3] < 90 for l in lines) or (i == 0 and prev_cap)
                gi = cv2.cvtColor(f['img'], cv2.COLOR_BGR2GRAY) if f['img'].ndim == 3 else f['img']
                ink = float((gi < 160).mean())
                if not cap and (ink < FIG_MIN_INK or f['bbox'][3] - f['bbox'][1] < FIG_MIN_H_PT): continue
                keep.append(f)
            figs = keep
        for f in figs:                                                # 그림 바로 위(60pt 안) 캡션이 없고 짧은 글줄만 있으면 = 못 읽은 평면도 캡션 후보
            above = [l for l in lines if 0 <= f['bbox'][1] - l['bbox'][3] < 60]
            if above and not any(l['kind'] in ('cap', 'bad_cap', 'title11', 'sec2') for l in above):
                l = max(above, key=lambda l: l['bbox'][3])
                if len(re.sub(r'\s', '', l['text'])) <= 20 and re.search(r'[도층충총(]', l['text']) and not re.search(r'범례|사진|위치', l['text']):
                    l['text'], l['kind'] = '0. 평면도()', 'bad_cap'
        for r in rows: texts.append(' '.join(c for c in r[2] if c))
        texts += [l['text'] for l in lines]
        pages.append(dict(rows=rows, lines=lines, frames=figs, sec2_y=sec2_y, angle=round(P['angle'], 2), ppt=round(ppt, 3),
                          n_cells=sum(1 for r in rows for c in r[2] if c is not None)))
        dt = time.time() - t0
        per_page.append(round(dt, 1))
        log(f"쪽 {page.number + 1}: 기울기 {P['angle']:+.2f}° · 표 칸 {pages[-1]['n_cells']} · 글줄 {len(lines)} · 그림 {len(figs)} · {dt:.1f}초")
        if sec2_y is not None: after = True
    fix_plan_captions(pages, all_rows)
    texts += [l['text'] for p in pages if p for l in p['lines'] if l.get('fixed')]
    res = dict(pages=pages, text='\n'.join(texts), fields=field_confs(all_rows), seconds=round(time.time() - t_all, 1), per_page=per_page,
               n_image_pages=sum(1 for p in pages if p is not None))
    if len(_CACHE) >= 4: _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = res
    return res


def figure_regions(P, y_min_pt=None):
    """⑪ 그림 영역(쪽 화소 상자): 표 선으로 둘러싼 틀 + 색·잉크 내용 덩어리(틀 선이 흐리거나 끊긴 사진·위치도). 캡션 글줄은 제외"""
    ppt, img, g = P['ppt'], P['img'], P['g']
    H, W = g.shape
    y_min = int((y_min_pt or 0) * ppt)
    boxes = [tuple(int(v) for v in b) for b in P['raw_tables'] if (b[3] - b[1]) > 30 * ppt and (b[2] - b[0]) > 60 * ppt and b[1] >= y_min]
    ib = cv2.GaussianBlur(img, (0, 0), 1.2)                       # 스캔 잡음(색 점) 누르기
    hsv_s = ib.max(2).astype(np.int16) - ib.min(2).astype(np.int16)
    m = ((cv2.GaussianBlur(g, (0, 0), 1.2) < 190) | (hsv_s > 45)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m[:y_min] = 0
    for b in P['tlines']:                                        # 표 밖 글줄(캡션·설명)은 그림 아님 — 단 큰 그림 안 글자도 지워지지만 덩어리는 남음
        if (b[3] - b[1]) < 16 * ppt: m[b[1]:b[3], b[0]:b[2]] = 0
    m[:int(0.03 * H)] = 0; m[int(0.96 * H):] = 0                    # 쪽 번호·스캔 가장자리
    m[:, :int(0.01 * W)] = 0; m[:, int(0.99 * W):] = 0
    k = max(3, int(4 * ppt))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    blobs = [[st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]] for i in range(1, n)
             if st[i, 2] > 40 * ppt and st[i, 3] > 25 * ppt and st[i, 4] > 0.08 * st[i, 2] * st[i, 3]]
    # 가까운 덩어리 합치기(8pt)
    changed = True
    while changed:
        changed = False
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                a, b = blobs[i], blobs[j]; gap = 8 * ppt
                if a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]:
                    blobs[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]; blobs.pop(j); changed = True; break
            if changed: break
    out = list(boxes)
    for b in blobs:
        if (b[3] - b[1]) < 40 * ppt or (b[2] - b[0]) < 40 * ppt: continue
        if b[0] < 0.015 * W or b[2] > 0.985 * W or (b[3] - b[1]) > 0.85 * H: continue      # 쪽 가장자리에 붙은 덩어리 = 스캔 테두리·그림자
        hit = [o for o in boxes if _overlap(b, o) > 0.5]
        if hit: continue                                          # 틀이 이미 있음(틀 상자가 더 정확)
        p = int(2 * ppt)
        out.append((max(0, b[0] - p), max(0, b[1] - p), min(W, b[2] + p), min(H, b[3] + p)))
    return out


FIG_DROP = os.environ.get('FIRE3DBEAM_OCR_FIGDROP', '1') != '0'       # 캡션 없는 빈·낮은 그림 영역 버림(0 = 예전)
FIG_MIN_INK = 0.008            # 캡션 없는 영역의 최소 잉크 비율(정규화한 그림에서 회색 < 160) — dev 정상 그림 최소 0.013
FIG_MIN_H_PT = 120             # 캡션 없는 영역의 최소 높이(pt) — dev 정상 그림 최소 217pt, 과검출 글줄 덩어리 45~92pt
FIG_NORM = os.environ.get('FIRE3DBEAM_OCR_FIGNORM', '1') != '0'
FIG_PPT = float(os.environ.get('FIRE3DBEAM_OCR_FIGPPT', '0') or 0)       # >0 이면 그림을 이 화소/pt 로 다시 표본화(0 = 스캔 해상도 그대로)


def white_balance(P):
    """쪽 색 보정: 종이색(평탄화 영상에서 바탕인 화소들의 원래 색 중앙값) → 흰색, 잉크(0.5% 분위) → 검정. 채널별 선형"""
    img = P['img']
    if not FIG_NORM: return img
    bgm = P['g'] > 245
    if bgm.sum() < 1000: return img
    paper = np.median(img[bgm], axis=0).astype(np.float32)
    if os.environ.get('FIRE3DBEAM_OCR_WBLO', '0') == '1':
        lo = np.percentile(img.reshape(-1, 3)[::7], 0.5, axis=0).astype(np.float32)
        lo = np.minimum(lo, paper - 60)
    else:
        lo = np.zeros(3, np.float32)
    out = (img.astype(np.float32) - lo) / np.maximum(paper - lo, 1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


FIG_SHARP = os.environ.get('FIRE3DBEAM_OCR_FIGSHARP', '0') != '0'
FIG_GAMMA = float(os.environ.get('FIRE3DBEAM_OCR_FIGGAMMA', '1.5') or 0)


def clear_border_lines(c, ppt):
    """잘라낸 그림 가장자리(6pt 안)의 틀 선 조각 지우기 — 도면 외곽·면적 추정이 틀 선을 건물로 오인하지 않게"""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    d = (g < 200).astype(np.uint8)
    H, W = g.shape; e = max(3, int(6 * ppt))
    for i in range(min(e, H)):
        if d[i].mean() > 0.5: c[:i + 2] = 255
        if d[H - 1 - i].mean() > 0.5: c[H - 3 - i:] = 255
    for j in range(min(e, W)):
        if d[:, j].mean() > 0.5: c[:, :j + 2] = 255
        if d[:, W - 1 - j].mean() > 0.5: c[:, W - 3 - j:] = 255
    return c


def fig_crop(img, b, ppt):
    x0, y0, x1, y1 = b
    c = img[y0:y1, x0:x1].copy()
    if c.size == 0: return c
    if FIG_NORM: c = clear_border_lines(c, ppt)
    if FIG_SHARP:                                               # 스캔 잡음·흐림 보정(가벼움): 중앙값 3 → 윤곽 선명화
        c = cv2.medianBlur(c, 3)
        bl = cv2.GaussianBlur(c, (0, 0), 1.0)
        c = cv2.addWeighted(c, 1.6, bl, -0.6, 0)
    if FIG_GAMMA and FIG_NORM:                                  # 흐림으로 옅어진 가는 회색 선(그리드·문 호·벽) 되살리기: 무채색 화소만 감마
        sat = c.max(2).astype(np.int16) - c.min(2).astype(np.int16)
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** FIG_GAMMA, 0, 255).astype(np.uint8)
        g2 = cv2.LUT(c, lut)
        c = np.where((sat < 40)[..., None], g2, c)
    if FIG_PPT and c.size:
        f = FIG_PPT / ppt
        c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return c


def _overlap(a, b):
    """a 가 b 와 겹치는 비율(a 넓이 기준)"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def _inside(a, b, tol=4):
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol and (a[2] - a[0]) * (a[3] - a[1]) < (b[2] - b[0]) * (b[3] - b[1])


def line_tile(P, b):
    """표 밖 글줄 타일: 밑줄(캡션 밑줄 등 가로선) 지운 회색"""
    x0, y0, x1, y1 = b
    pad = int(2 * P['ppt'])
    ya, yb, xa, xb = max(0, y0 - pad), y1 + pad, max(0, x0 - pad), x1 + pad
    return P['g'][ya:yb, xa:xb].copy()          # 밑줄은 그대로(지우면 굵은 캡션 글자 아랫부분이 같이 지워져 숫자 오독)


def fix_plan_captions(pages, all_rows):
    """평면도 캡션 층 보정 — ⑩ 층별현황(같은 문서 OCR)의 층 목록으로: ① 없는 층 번호에서 오독 자리 빼기('지상54'→지상5)
    ② 못 읽은 캡션이 앞뒤 캡션 사이 빠진 층 하나뿐이면 그 층(평면도는 층 순서로 실림)"""
    from fire3dbeam.card import norm_floor, level_of, expand_floors
    valid = set()
    for row in all_rows:
        for c in row['cells']:
            if c is not None and (c.get('kind') or ('',))[0] == 'floor':
                f = norm_floor(c.get('text') or '')
                if f: valid.add(f)
    caps = [l for p in pages if p for l in p['lines'] if l.get('kind') in ('cap', 'bad_cap') and '평면도' in l['text']]
    if not valid: caps_fix = []
    else: caps_fix = caps
    def ko(f): return f'지하{f[1:]}층' if f.startswith('B') else (f'지상{f[:-1]}층' if f.endswith('F') and f[:-1].isdigit() else '옥탑1층')
    for l in caps_fix:
        m = re.match(r'^(\S+)\s*평면도\((.*)\)$', l['text'])
        if not m: l['floors'] = None; continue
        fl = expand_floors(m.group(2))
        fixed = []
        for f in fl:
            if f in valid: fixed.append(f); continue
            k = re.match(r'^(B|)(\d+)(F|)$', f)
            cands = []
            if k:
                n = k.group(2)
                for alt in {n[:-1], n[1:], n[:1]} - {''}:
                    g = f'{k.group(1)}{int(alt)}{k.group(3)}'
                    if g in valid: cands.append(g)
            fixed.append(cands[0] if len(set(cands)) == 1 else None)
        l['floors'] = fixed if fl and all(fixed) else None
        if l['floors'] is not None and l['floors'] != fl and len(fl) == 1:
            l['text'] = f"{m.group(1)} 평면도({ko(l['floors'][0])})"; l['fixed'] = True
    for i, l in enumerate(caps_fix):                                # ② 사이 빠진 층 하나
        if l.get('floors'): continue
        prev = next((c['floors'] for c in reversed(caps[:i]) if c.get('floors')), None)
        nxt = next((c['floors'] for c in caps[i + 1:] if c.get('floors')), None)
        if not prev and not nxt: continue
        a = max(level_of(f) for f in prev) if prev else -1e9
        b = min(level_of(f) for f in nxt) if nxt else 1e9
        between = [f for f in valid if a < level_of(f) < b and f != 'RF' and not any(f in (c.get('floors') or []) for c in caps)]
        if len(between) == 1:
            num = re.match(r'^(\S+)', l['text']).group(1)
            l['text'] = f'{num} 평면도({ko(between[0])})'; l['kind'] = 'cap'; l['fixed'] = True; l['floors'] = between
    for l in caps:
        if l.get('kind') == 'bad_cap':
            if l.get('floors') is None and re.search(r'평면도\(\)$', l['text']):
                l['text'], l['kind'] = '평면도 캡션 층 미확인', None           # 층을 모르는 평면도 → 그림 종류 '기타'(3D 에 넣지 않음)
            else: l['kind'] = 'cap'


QUICK_SIZE = os.environ.get('FIRE3DBEAM_QUICK_SIZE', '1') != '0'
SEC2_RE = re.compile(r'2\s*\.\s*동\s*별\s*위\s*치\s*도')


def quick_is_card(page):
    """첫 쪽이 스캔본 정보카드인가(분류용, 가벼운 검사): 표 밖 글줄에 '소방활동 정보카드'·'건물개황' 또는 양식 칸 이름 4개 이상"""
    P = page_geom(page); ppt = P['ppt']
    tl = P['tlines'][:12]
    if tl:
        for r in ocr_tiles([line_tile(P, b) for b in tl], ppt, lang='kor'):
            h = hangul(lines_text(r)[0])
            if ('정보카드' in h or jsim(h[-6:], '활동정보카드') > 0.7 or jsim(h, '건물개황') > 0.7) and len(h) < 20: return True
    # 제목을 못 읽음 → 첫 쪽 표 칸 이름 OCR(양식 칸 이름 4개 이상)
    if QUICK_SIZE and max(page.rect.width, page.rect.height) > 1.2 * 842:
        return False                                   # A4(표준양식) 보다 훨씬 큰 쪽(A3·A2·A1 도면 시트) — 제목도 없으면 카드 아님, 표 칸 OCR(쪽당 ≈4 s) 생략 (2026-10-03)
    form_t = [t for t in P['tables'] if not t['frame']]
    if not form_t: return False
    rows, rraw, _ = form_rows(P, form_t, page.number)
    labs = {c['label'] for r in rraw for c in r['cells'] if c is not None and c.get('label') and c['lsim'] >= 0.8}
    return len(labs) >= 6
