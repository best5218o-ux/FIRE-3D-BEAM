# -*- coding: utf-8 -*-
"""정보카드 파이프라인의 '실제 도면' 경로 (2026-10-03, 공개 실제 도면 종단 시험 후속).

정보카드 ⑪ 평면도(또는 별도 도면 파일)가 **소방 기호 범례·기호가 없는 실제 CAD 평면도**이면(pipeline 의 `_nosym` 조건)
가상 도면용 규칙(색 범례 기호·디딤판+비상구 근접)을 쓸 수 없다 → 이 모듈이 그 평면도의 축척·벽·문·창·계단·승강기·실명·외곽을 만든다.

  1) 축척 = 여러 출처 합의(신뢰도 가중):
     sheet  도면 시트 글자층(또는 OCR)의 'SCALE 1/150 (A3)' + 용지 크기 → px/m   (review/attach.py 가 캡션에 '[도면축척 … ppm=…]' 로 넘김)
     dims   치수 사슬(fire3dbeam/realplan 치수 판독·합의)
     tags   창호 태그 '1.1X2.2'(폭×높이 m) ↔ 그 옆 STR 모델 문·창 개체 폭(px) — 3개 이상 같은 값이면 강함
     grid   다른 층(강한 축척)과 그리드선 간격 비교(같은 도면 세트)
     tread  계단 디딤판 간격(SPA 계단실 안 등간격 평행선) ÷ 0.28 m
     doors  STR 모델 문 폭 중앙값 ÷ 1.04 m(realplan.DOOR_MM) — 약함
     area   정보카드 층 면적 ↔ **건물 덩어리 외곽**(아래 3) — 마지막 수단
     출처끼리 ±10% 넘게 갈리면 플래그 scale_disagree(검수).
  2) 벽·창·문 = STR 분할 모델(fire3dbeam/str_seg, xcheck.model_plan 캐시 공유) 벡터, 계단·승강기 = SPA v3.1 하이브리드(fire3dbeam/spa_seg)
     + 규칙(UP/DN·EV 글자 확인, fire3dbeam/realsym), 실명 = realsym OCR(비주거 어휘 포함).
  3) 건물 덩어리 외곽(mass): 벽∪창∪문 장벽 → 빈 공간 성분 → 바깥(그림 가장자리에 닿음)에서 시작해 '마당형' 성분을 벗김
     (바깥과 맞닿은 경계에 창·문이 하나도 없고 바깥 접촉 비율 ≥0.3 → 대지선·주차구획·조경·옥상정원 구획). 창 사이 얇은 외벽도 창∪벽으로 닫힘.
     외부 출입문 = 바깥(벗긴 마당 포함)과 실내를 잇는 모델 문 개체.
  4) 층 정합 기준점(anchors) = 계단·승강기 중심 + 그리드선(+ 그리드 버블 이름 X1·Y1) → fire3dbeam/align.align_real.

끄기: FIRE3DBEAM_REALPLAN=0 (예전 동작: 범례 없는 도면도 가상 규칙 경로). 가상 정보카드 도면(범례·기호 있음)은 이 모듈을 부르지 않는다."""
import os, re, math
os.environ.setdefault('OMP_THREAD_LIMIT', '1')     # tesseract OpenMP 끔: 2코어 공유 시 스레드 회전 대기로 4~5배 느려짐(판독 결과 동일, 2026-10-03)
import numpy as np, cv2

ENABLED = os.environ.get('FIRE3DBEAM_REALPLAN', '1') != '0'
TREAD_M = 0.28            # 계단 디딤판 깊이 표준값(건축물 피난·방화구조 규칙: 공공 0.26~0.30 m)
DOOR_M = 1.04             # STR 모델 문 개체 폭 중앙값(realplan.DOOR_MM, obj2 개발 60장)
W = dict(sheet=1.0, sheet_guess=0.55, dims=0.8, dims_weak=0.4, tags3=0.8, tags2=0.5, tags1=0.2, grid=0.6, tread=0.3, doors=0.2, area=0.15)
AGREE = 0.08              # 같은 축척으로 보는 상대 차(±8%)
DISAGREE = 0.10           # 강한 출처끼리 이만큼 넘게 다르면 scale_disagree
RERUN = 0.12              # 최종 축척이 모델을 돌린 축척과 이만큼 넘게 다르면 STR·SPA 다시
OUTLIER_REJECT = os.environ.get('FIRE3DBEAM_SCALE_OUTLIER', '1') != '0'   # 한 층 혼자 갈리는 약한 축척 후보(태그 짝 등) 버리고 플래그 안 띄움(2026-10-03 2차)
FUSE_TRIM = float(os.environ.get('FIRE3DBEAM_FUSE_TRIM', '0.04'))   # 축척 합의: 명시 출처 없을 때 가중 중앙값 ±4% 안만 평균(2026-10-03 2차) — 0 = 예전(무리 전체 가중 평균)
TAG_V2 = os.environ.get('FIRE3DBEAM_TAG_V2', '1') != '0'          # 창호 태그 판독 2차(fire3dbeam/realtag.py: 칸 정리·어휘 맞춤·본보기 투표) — 0 = 예전 read_tags
BUBBLE_V2 = os.environ.get('FIRE3DBEAM_BUBBLE_V2', '1') != '0'    # 그리드 버블 2차(축별 이름 꼴·순서 맞춤) — 0 = 예전 read_bubbles
STR_SINGLE_PX = float(os.environ.get('FIRE3DBEAM_STR_SINGLE_MPX', '0')) * 1e6   # (시험용, 기본 끔) STR 한 번 실행 화소 예산(백만) — S 층당 STR 20→10 s 이지만 AI허브 STR 보류 30장 화소 F1 이 벽 .886→.885·문 .913→.910 로
#   조금 낮고(겹친 타일 평균 효과가 사라짐) Z 1층 외곽이 402→369 ㎡ 로 흔들려 채택 안 함. 대신 글자층 치수 범위(STR_ROI)만 돌림
DIMS_MISS_STOP = int(os.environ.get('FIRE3DBEAM_DIMS_MISS_STOP', '2'))   # 치수 OCR 이 이만큼 층에서 실패하면 나머지 층은 치수 판독 생략(0 = 끔)
_DIMS_MISS = {}
AREA_SKIP = os.environ.get('FIRE3DBEAM_AREA_SKIP', '1') != '0'
STR_ROI = os.environ.get('FIRE3DBEAM_STR_ROI', '1') != '0'
REAL_TILE = int(os.environ.get('FIRE3DBEAM_REAL_TILE', '0')) or None   # (시험용) STR 타일 크기(기본 = 모델 .json 1024)
SKIP_BLANK = os.environ.get('FIRE3DBEAM_SKIP_BLANK', '1') != '0'   # STR 빈 종이 타일(잉크 < 0.02%) 건너뜀 — 결과 같음(배경), 여러 평면 시트 칸 여백(2026-10-03)
EVEN_TILES = os.environ.get('FIRE3DBEAM_EVEN_TILES', '0') == '1'   # (시험용, 기본 끔) STR 타일을 고르게 나눔 — S 층당 23→18 s 이지만 작은 타일은 문맥이 줄어 옥탑 외곽이 96→188 ㎡ 로 흔들려 채택 안 함(2026-10-03)
GROUP_DIMS = os.environ.get('FIRE3DBEAM_GROUP_DIMS', '1') != '0'   # 용지 없는 같은 시트 축척 묶음은 한 층 치수로 확인(2026-10-03)
LAZY_DIMS = os.environ.get('FIRE3DBEAM_LAZY_DIMS', '1') != '0'
PDF_TEXT = os.environ.get('FIRE3DBEAM_PDF_TEXT', '1') != '0'
TEXT_AXES = os.environ.get('FIRE3DBEAM_TEXT_AXES', '1') != '0'    # 글자층의 원 없는 그리드 이름 줄을 버블처럼(2026-10-04, 새 세트 E 지붕층 정합)      # 벡터 도면 글자층을 OCR 대신(2026-10-03 2차 강건화, fire3dbeam/pdftext.py)
S0_TREAD = os.environ.get('FIRE3DBEAM_S0_TREAD', '1') != '0'      # 임시 축척(모델 입력)에 규칙 디딤판 간격 사용   # 용지 명시 시트 축척이 있으면 치수 판독은 출처가 갈릴 때만
WORKERS = int(os.environ.get('FIRE3DBEAM_REAL_WORKERS', '2'))     # 층별 1·2단계 동시 실행 스레드 수(OCR 은 별도 프로세스라 2코어를 함께 씀). 1 = 순차


def available():
    if not ENABLED: return False
    try:
        import onnxruntime  # noqa: F401
        from fire3dbeam import str_seg as SS, spa_seg as SP
        return os.path.exists(SS.MODEL) and SP.available()
    except Exception:
        return False


def applicable(syms, legend, text=None):
    """범례·기호가 없는 평면도(실제 CAD 도면) + 모델 사용 가능.
    범례가 없고 '기호'가 모두 색 글자(공사범위 주기 '계단 난간 철거 후 신설' 같은 파란·빨간 글씨)로 설명되면 기호 없음으로 봄(2026-10-04, 새 세트 E)"""
    if legend or not available(): return False
    return not syms or (MARKUP_TEXT and markup_only(syms, text))


MARKUP_TEXT = os.environ.get('FIRE3DBEAM_MARKUP_TEXT', '1') != '0'


def markup_only(syms, text=None):
    """기호 검출이 전부 색 글자인가: ① 글자층 줄 상자(±2 px) 안 또는 ② 같은 종류 4개 이상이 한 줄(가운데 높이 차 < 0.35×높이, 간격 < 2.5×높이)로 늘어선 글자 꼴"""
    B = [q.get('box') for q in syms]
    if not B or any(b is None or len(b) < 4 for b in B): return False
    expl = [False] * len(B)
    for i, b in enumerate(B):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for l in text or ():
            lb = l['box']
            if lb[0] - 2 <= cx <= lb[2] + 2 and lb[1] - 2 <= cy <= lb[3] + 2: expl[i] = True; break
    kinds = [q.get('type') or q.get('kind') for q in syms]
    for o in (0, 1):                                   # 0 = 가로 줄, 1 = 세로 줄 — 같은 종류를 한 줄씩 묶어(가운데 차 < 0.6×중앙 높이) 간격 4×높이로 끊고, 4개 이상 무리 = 글자 줄
        hh = lambda j: max(1.0, (B[j][3] - B[j][1]) if o == 0 else (B[j][2] - B[j][0]))
        cc_ = lambda j: ((B[j][1] + B[j][3]) / 2) if o == 0 else ((B[j][0] + B[j][2]) / 2)
        lo = lambda j: B[j][0] if o == 0 else B[j][1]
        hi = lambda j: B[j][2] if o == 0 else B[j][3]
        for kd in set(kinds):
            J = sorted([j for j in range(len(B)) if kinds[j] == kd], key=cc_)
            lines, cur = [], []
            for j in J:
                if cur and abs(cc_(j) - cc_(cur[0])) > 0.6 * float(np.median([hh(q) for q in cur])): lines.append(cur); cur = []
                cur.append(j)
            if cur: lines.append(cur)
            for ln in lines:
                ln.sort(key=lo); mh = float(np.median([hh(q) for q in ln]))
                runs, run = [], [ln[0]]
                for a_, b_ in zip(ln, ln[1:]):
                    if lo(b_) - hi(a_) < 4 * mh: run.append(b_)
                    else: runs.append(run); run = [b_]
                runs.append(run)
                for r_ in runs:
                    if len(r_) < 4: continue
                    for j in ln:                            # 줄 끝에서 10×높이 안의 같은 줄 글자 조각도
                        if lo(r_[0]) - 10 * mh <= lo(j) and hi(j) <= hi(r_[-1]) + 10 * mh: expl[j] = True
    return all(expl)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ───────────────────────── 원형 태그(창호·그리드 버블) ─────────────────────────
def find_circles(g, rmin=8, rmax=80):
    """닫힌 원(창호 태그·그리드 버블) → [(x, y, r)] (원 둘레 80% 이상 잉크)"""
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    cs = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, dp=1, minDist=max(10, rmin * 2), param1=120, param2=22, minRadius=rmin, maxRadius=rmax)
    if cs is None: return []
    H, Wd = g.shape
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    out = []
    for x, y, r in cs[0]:
        hit = np.zeros(72, bool)
        for dr in (-1.5, -0.5, 0.5, 1.5):
            xx = np.round(x + (r + dr) * ca).astype(int); yy = np.round(y + (r + dr) * sa).astype(int)
            ok = (xx >= 0) & (xx < Wd) & (yy >= 0) & (yy < H)
            v = np.full(72, 255)
            v[ok] = g[yy[ok], xx[ok]]
            hit |= v < 200
        if hit.mean() > 0.8: out.append((float(x), float(y), float(r)))
    out.sort(key=lambda c: -c[2])
    keep = []
    for c in out:
        if all(math.hypot(c[0] - k[0], c[1] - k[1]) > 0.8 * max(c[2], k[2]) for k in keep): keep.append(c)
    return keep


def _circle_tile(g, x, y, r, part='top'):
    """원 안 글자만(둘레 지움): part='top'(창호 태그 위 칸) | 'all'(그리드 버블)"""
    R = int(r + 3); x0, y0 = int(x - R), int(y - R)
    c = g[max(0, y0):int(y + R), max(0, x0):int(x + R)].copy()
    if c.size == 0: return None
    yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    d = np.hypot(xx + max(0, x0) - x, yy + max(0, y0) - y)
    c[d > r - 2.5] = 255
    if part == 'top':
        cy = int(y - max(0, y0))
        c = c[int(0.15 * r):max(int(0.15 * r) + 2, cy - int(0.12 * r))]
    elif part == 'bottom':
        cy = int(y - max(0, y0))
        c = c[cy + int(0.12 * r):max(cy + int(0.12 * r) + 2, c.shape[0] - int(0.15 * r))]
    if c.size == 0 or c.shape[0] < 3: return None
    U = cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(U, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


TAG_M = re.compile(r'(?<![.])(\d)\s*[.,]\s*(\d{1,2})\s*[Xx×*]+\s*\d')          # 1.1X2.2 · 2.28X3.5 (m)  ('Xx' 겹읽기 허용)
TAG_MM = re.compile(r'(?<!\d)([1-9]\d{2,3})\s*[Xx×*]+\s*([1-9]\d{2,3})(?!\d)')   # 1100X2200 (mm) — 높이 ≥ 500 mm 만


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


DOOR_CODE = re.compile(r'(ASD|FSD|SSD|SD|FD|AD|WD|PD|SSWD|D)$')        # 창호 기호 아래 칸: …D = 문(SD 강제문·ASD 자동문·FD 접이문·WD 목재문), …W = 창(AW·PW)


def read_tags(g, circles):
    """창호 태그 OCR: 원 위 칸 '폭X높이'(→ W m) + 아래 칸 기호(ASD·SD·FD → 문 / AW → 창) → [{x,y,r,W|None,text,code,door}] — tesseract 2회(모자이크)"""
    from fire3dbeam.realsym import _ocr_mosaic
    tiles, bots, meta = [], [], []
    for x, y, r in circles:
        t = _circle_tile(g, x, y, r, 'top'); b = _circle_tile(g, x, y, r, 'bottom')
        if t is not None and b is not None: tiles.append(t); bots.append(b); meta.append((x, y, r))
    if not tiles: return []
    txt = _ocr_mosaic(tiles, lang='eng', whitelist='0123456789.Xx')
    cod = _ocr_mosaic(bots, lang='eng', whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    out = []
    for (t, conf), (c, _), (x, y, r) in zip(txt, cod, meta):
        w = parse_tag(t)
        c = re.sub(r'[^A-Z]', '', (c or '').upper())
        # 문 기호: 끝이 D(ASD·SD·FD …) — 작은 글자 OCR 이 D 를 O·B·0 으로, 앞 글자를 빠뜨리는 일이 잦아 폭이 읽힌 태그는 W(창) 없는 기호면 문으로 봄
        door = bool(c) and 'W' not in c and (bool(DOOR_CODE.search(c)) or (w is not None and c not in ('A',)))
        if w or c: out.append(dict(x=x, y=y, r=r, W=w, text=t, code=c, door=door))
    return out


def read_bubbles(g, circles, vl, hl):
    """그리드 버블(그리드선 연장 위의 원, 'X1'·'Y2'·'1'·'A') → {('v', x): 이름, ('h', y): 이름}"""
    from fire3dbeam.realsym import _ocr_mosaic
    tiles, meta = [], []
    for x, y, r in circles:
        on = [('v', v) for v in vl if abs(x - v) <= max(2.0, 0.25 * r)] + [('h', h) for h in hl if abs(y - h) <= max(2.0, 0.25 * r)]
        if len(on) != 1: continue
        t = _circle_tile(g, x, y, r, 'all')
        if t is not None: tiles.append(t); meta.append(on[0])
    if not tiles: return {}
    txt = _ocr_mosaic(tiles, lang='eng', whitelist='ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
    out = {}
    for (t, conf), key in zip(txt, meta):
        t = re.sub(r'[^A-Z0-9]', '', (t or '').upper())
        if 1 <= len(t) <= 3 and (re.fullmatch(r'[XY]\d{1,2}', t) or re.fullmatch(r'\d{1,2}', t) or re.fullmatch(r'[A-Z]\d?', t)): out[key] = t
    seen = {}
    for (o, c), t in out.items(): seen[(o, t)] = seen.get((o, t), 0) + 1
    return {k: t for k, t in out.items() if seen[(k[0], t)] == 1}          # 같은 축에 같은 이름 둘 이상 = 오독(D·C → 0) → 버림


# ───────────────────────── 그리드선 ─────────────────────────
def grid_lines(g, min_frac=0.55, edge=0.15):
    """실제 도면 그리드선(가늘고 길게 도면 대부분을 가로지르는 선, 일점쇄선 포함) → (세로선 x 목록, 가로선 y 목록) px.
    길이 ≥ min_frac × 그림 변, 한쪽 끝이 그림 가장자리 edge 안까지 감(건물 밖으로 나감), 굵기 ≤ 4 px."""
    H, Wd = g.shape
    ink = (g < 215).astype(np.uint8)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thin = ink & (1 - cv2.dilate(thick, np.ones((3, 3), np.uint8)))
    out = []
    for o in ('v', 'h'):
        L = H if o == 'v' else Wd
        k1 = (1, max(9, L // 60)) if o == 'v' else (max(9, L // 60), 1)
        k2 = (1, int(min_frac * L)) if o == 'v' else (int(min_frac * L), 1)
        m = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k2))
        prof = m.sum(0 if o == 'v' else 1)
        idx = np.nonzero(prof >= min_frac * L)[0]
        lines = []
        for grp in np.split(idx, np.nonzero(np.diff(idx) > 2)[0] + 1) if len(idx) else []:
            if len(grp) > 4: continue                                   # 굵은 것(벽 겹선 덩어리)
            c = int(round(grp.mean()))
            col = m[:, max(0, grp[0]):grp[-1] + 1].max(1) if o == 'v' else m[max(0, grp[0]):grp[-1] + 1, :].max(0)
            nz = np.nonzero(col)[0]
            if not len(nz): continue
            if nz[0] > edge * L and nz[-1] < (1 - edge) * L: continue      # 건물 안에서만 긴 선(외벽)은 그리드 아님
            lines.append(c)
        # 4~10 px 안에 붙은 두 선 = 벽 겹선(그리드 아님) → 둘 다 버림
        drop = set()
        for a, b in zip(lines, lines[1:]):
            if b - a <= 10: drop |= {a, b}
        out.append([c for c in lines if c not in drop])
    return out[0], out[1]


def grid_from_bubbles(g, circles, margin=0.3):
    """그리드 버블(도면 가장자리 쪽에 같은 크기로 한 줄로 선 원) → 그 중심을 지나는 그리드선 (세로선 x 목록, 가로선 y 목록) px.
    선 자체가 흐리거나 일점쇄선이라 길이 검출(grid_lines)이 놓칠 때 보충. 버블에서 도면 안쪽으로 원 지름 5배 구간의 잉크가 25% 이상이어야 함"""
    H, Wd = g.shape
    ink = (g < 200)
    C = [c for c in circles if 8 <= c[2] <= 0.05 * max(H, Wd)]
    vl, hl = [], []
    def inked(x, y, r, o, sgn):
        L = int(10 * r); pts = 0; hit = 0
        for t in range(int(1.3 * r), int(1.3 * r) + L, 2):
            xx, yy = (int(x), int(y + sgn * t)) if o == 'v' else (int(x + sgn * t), int(y))
            if not (0 <= xx < Wd and 0 <= yy < H): break
            pts += 1; hit += bool(ink[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3].any())
        return pts >= 10 and hit / pts >= 0.25
    outer = {}
    if BUBBLE_V2:                                                 # 2차: 변마다 가장 바깥 줄(같은 크기 원 2개 이상)만 버블 줄 — 그 안쪽 창호 태그 줄 제외
        for o in ('v', 'h'):
            Lm = H if o == 'v' else Wd
            for side in (0, 1):
                rows = []
                for a in C:
                    pos = a[1] if o == 'v' else a[0]
                    if (side == 0 and pos > margin * Lm) or (side == 1 and pos < (1 - margin) * Lm): continue
                    row = [b for b in C if abs((b[1] if o == 'v' else b[0]) - pos) <= 0.3 * a[2] and abs(b[2] / a[2] - 1) < 0.2]
                    if len(row) >= 2: rows.append(pos)
                if rows: outer[(o, side)] = min(rows) if side == 0 else max(rows)
    for o in ('v', 'h'):
        for a in C:
            pos = a[1] if o == 'v' else a[0]                       # 세로선 버블 = 위·아래 가장자리 줄(같은 y)
            Lm = H if o == 'v' else Wd
            if margin * Lm < pos < (1 - margin) * Lm: continue
            if BUBBLE_V2:
                op = outer.get((o, 0 if pos < Lm / 2 else 1))
                if op is None or abs(pos - op) > 0.6 * a[2]: continue
            row = [b for b in C if abs((b[1] if o == 'v' else b[0]) - pos) <= 0.3 * a[2] and abs(b[2] / a[2] - 1) < 0.2]
            if len(row) < 2: continue
            sgn = 1 if pos < Lm / 2 else -1
            # 2차: 가장자리 한 줄에 같은 크기 원이 3개 이상이면 그리드 버블 줄(선이 흐린 일점쇄선이라 잉크 검사를 못 넘어도)
            if inked(a[0], a[1], a[2], o, sgn) or (BUBBLE_V2 and len(row) >= 3): (vl if o == 'v' else hl).append(int(round(a[0] if o == 'v' else a[1])))
    return sorted(set(vl)), sorted(set(hl))


def merge_lines(A, B, tol=4):
    out = sorted(A)
    for b in B:
        if all(abs(b - a) > tol for a in out): out.append(b)
    return sorted(out)


def grid_spacings(L):
    return [float(b - a) for a, b in zip(L, L[1:]) if b - a > 10]


# ───────────────────────── 축척 출처 ─────────────────────────
SHEET_RE = re.compile(r'\[도면축척[^\]]*?ppm=([\d.]+)([^\]]*)\]')


def sheet_ppm_from_caption(cap):
    """review/attach.py 가 붙인 캡션 꼬리 '[도면축척 1/150(A3) ppm=52.49]'(ppm = 그림 px/m) → (px/m, 용지 명시 여부). 없으면 (None, False)"""
    t = (cap or '').replace(' ', '')
    m = SHEET_RE.search(t)
    if not m: return None, False
    return float(m.group(1)), '추정' not in (m.group(0) or '')


# ── 도면 시트 글자(표제란 'SCALE 1/150 (A3)', '축척 1:100', 'A1: 75 / A3: 150') → 축척 ──
ISO_A = {0: 1189.0, 1: 841.0, 2: 594.0, 3: 420.0, 4: 297.0}         # A 계열 긴 변(mm)
STD_N = {20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 125, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1200}
_RATIO = re.compile(r'1\s*[/:]\s*(\d{2,4})(?:\s*\(?\s*A\s*([0-4]))?')
_NUM = re.compile(r'^[=:]?\(?(\d{2,4})\)?(?:\(?A([0-4])\)?)?[,.]?$')
_PAPER = re.compile(r'^\(?A([0-4])\)?[:：]?$')
_LABEL = re.compile(r'(SCALE|축척|S\s*=)', re.I)


def paper_of(long_mm, tol=0.04):
    for k, L in ISO_A.items():
        if abs(long_mm / L - 1) < tol: return k
    return None


def sheet_scale_words(words, page_long_mm, unit_mm=25.4 / 72):
    """words: [(x0,y0,x1,y1,글자)] (좌표 단위 unit_mm mm) → (도면 축척 분모 N(그 용지 기준), 근거 문자열, 용지 명시 여부) | None.
    용지가 적힌 축척(1:150(A3), A1: 75)은 실제 쪽 용지로 환산(A1 1/75 를 A3 로 출력 → 1/150)."""
    page_p = paper_of(page_long_mm)
    cen = [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in words]
    labels = [c for c, w in zip(cen, words) if _LABEL.search(w[4])]
    near = lambda c, mm: any(math.hypot(c[0] - l[0], c[1] - l[1]) * unit_mm < mm for l in labels)
    Ns, papers = [], []
    for c, w in zip(cen, words):
        t = w[4].replace(' ', '')
        m = _RATIO.search(t)
        if m and int(m.group(1)) in STD_N:
            Ns.append([int(m.group(1)), int(m.group(2)) if m.group(2) else None, c]); continue
        m = _NUM.match(t)
        if m and int(m.group(1)) in STD_N and near(c, 40):
            Ns.append([int(m.group(1)), int(m.group(2)) if m.group(2) else None, c]); continue
        m = _PAPER.match(t)
        if m: papers.append((int(m.group(1)), c))
    if not Ns: return None
    free = [n for n in Ns if n[1] is None]
    for n in free:                                       # 용지 표기 짝: 9 mm 안 가장 가까운 것(같은 칸), 둘씩이면 작은 N ↔ 큰 용지
        cand = [(math.hypot(n[2][0] - c[0], n[2][1] - c[1]) * unit_mm, k) for k, c in papers]
        cand = [x for x in cand if x[0] < 9]
        if cand: n[1] = min(cand)[1]
    pf = [n for n in free if n[1] is not None]
    if len(pf) >= 2 and len({n[1] for n in pf}) == len(pf):
        for n, k in zip(sorted(pf, key=lambda n: n[0]), sorted({n[1] for n in pf})): n[1] = k
    conv = []
    for N, k, c in Ns:
        if k is not None and page_p is not None: conv.append((N * ISO_A[k] / ISO_A[page_p], f'1/{N}(A{k})', True))
        elif k is None: conv.append((float(N), f'1/{N}', False))
    if not conv: return None
    stated = [c for c in conv if c[2]]
    pool = stated or conv
    vals = [round(c[0]) for c in pool]
    best = max(set(vals), key=lambda v: (vals.count(v), -v))
    why = next(c[1] for c in pool if round(c[0]) == best)
    return float(best), why + (f'=A{page_p} 1/{best}' if stated and page_p is not None and why.find(f'A{page_p})') < 0 else ''), bool(stated)


def sheet_scale_pdf_page(page, img_w_px):
    """벡터 PDF 쪽 글자층 → (px/m(그림 img_w_px 기준), 캡션 꼬리) | None"""
    import pymupdf  # noqa: F401
    r = page.rect
    W_pt = r.width                                      # get_pixmap 은 회전 반영 → 그림 가로 = 회전된 쪽 가로
    long_mm = max(r.width, r.height) * 25.4 / 72
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text('words')]
    res = sheet_scale_words(words, long_mm)
    if not res: return None
    N, why, stated = res
    ppm = img_w_px / (W_pt * 25.4 / 72) * 1000.0 / N
    return ppm, f"[도면축척 {why} ppm={ppm:.2f}{'' if stated else ' 용지추정'}]"


def sheet_scale_ocr(img):
    """그림 도면 시트(글자층 없음, 도곽=용지 가정) OCR → (px/m, 캡션 꼬리) | None — 축척에 용지(A1·A3)가 적혀 있을 때만"""
    import subprocess, tempfile
    g = _gray(img)
    H, Wd = g.shape
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, g); fn = tf.name
    try:
        out = subprocess.run(['tesseract', fn, '-', '-l', 'eng', '--psm', '11', 'tsv'], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None
    finally:
        os.unlink(fn)
    words = []
    for ln in out.splitlines()[1:]:
        p = ln.split('\t')
        if len(p) == 12 and p[11].strip():
            x, y, w, h = map(int, p[6:10]); words.append((x, y, x + w, y + h, p[11].strip()))
    long_px = max(H, Wd)
    # 용지 = 그림 전체(도곽 가장자리) 가정: 용지 긴 변을 A 계열 중 축척 표기 용지로
    res = sheet_scale_words(words, ISO_A[3], unit_mm=ISO_A[3] / long_px)
    if not res or not res[2]: return None
    N, why, _ = res
    ppm = long_px / ISO_A[3] * 1000.0 / N
    return ppm, f"[도면축척 {why} ppm={ppm:.2f} OCR]"


def dims_scale(g):
    """치수 사슬 → px/m (realplan 치수 판독·합의, 원 시트 300dpi 가정 → 작은 그림은 확대)"""
    from fire3dbeam import realplan as RP
    up = 1.0
    if max(g.shape) < 3000: up = min(2.0, 3000.0 / max(g.shape))
    gF = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else g
    H, Wd = gF.shape
    try:
        pre = RP.region_prep(gF, [0, 0, Wd // 4, H // 4], None, True)
        s, conf, dims = RP.consensus(pre['raw'])
    except Exception:
        return None, 0.0, 0
    if not s: return None, 0.0, 0
    return float(s) * 1000.0 / up, float(conf), len(dims or [])



DIM_TXT = re.compile(r'^(?:\d{1,3}(?:,\d{3})+|\d{3,5})$')


def text_dims_scale(lines, tol=0.025):
    """벡터 도면 글자층 치수 숫자(18,700 · 6800 …) → (px/m, 신뢰도, 지지 수) | (None, 0, 0).
    같은 줄(가로 글자는 같은 y, 세로 글자는 같은 x)에 나란한 치수 글자 이웃 쌍은 이어진 치수 사슬의 두 칸 가운데 →
    가운데 사이 거리(px) = 축척 × (앞 칸 + 뒤 칸)/2 (m). 모든 줄의 쌍에서 ±2.5% 최대 무리(3쌍 이상). OCR 없음."""
    D = []
    for l in lines or ():
        t = l['t'].replace(' ', '')
        if not DIM_TXT.match(t): continue
        v = int(t.replace(',', ''))
        if not 250 <= v <= 120000: continue
        b = l['box']; hor = abs(l['dir'][0]) >= abs(l['dir'][1])
        th = (b[3] - b[1]) if hor else (b[2] - b[0])
        D.append(dict(v=v, hor=hor, c=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), th=max(th, 1.0), L=(b[2] - b[0]) if hor else (b[3] - b[1])))
    vals = []
    for hor in (True, False):
        G = sorted([d for d in D if d['hor'] == hor], key=lambda d: d['c'][1] if hor else d['c'][0])
        rows, cur = [], []
        for d in G:
            k = d['c'][1] if hor else d['c'][0]
            if cur and abs(k - (cur[-1]['c'][1] if hor else cur[-1]['c'][0])) > 0.35 * d['th']:
                rows.append(cur); cur = []
            cur.append(d)
        if cur: rows.append(cur)
        for row in rows:
            row = sorted(row, key=lambda d: d['c'][0] if hor else d['c'][1])
            for a, b in zip(row, row[1:]):
                dist = (b['c'][0] - a['c'][0]) if hor else (b['c'][1] - a['c'][1])
                if dist <= 0.5 * (a['L'] + b['L']): continue
                vals.append(dist / ((a['v'] + b['v']) / 2000.0))
    if len(vals) < 3: return None, 0.0, 0
    s, n = ratio_cluster(vals, tol)
    if not s or n < 3: return None, 0.0, n
    return s, round(n / len(vals), 2), n


def tag_ratios(tags, openings):
    """태그 ↔ 모델 개구부 짝(서로 가장 가까운 짝만, 원 반지름 4배 안) → 비 px/m 목록"""
    out = []
    if not tags or not openings: return out
    D = np.array([[math.hypot(o['x'] - t['x'], o['y'] - t['y']) for o in openings] for t in tags])
    for i, t in enumerate(tags):
        j = int(np.argmin(D[i]))
        if D[i, j] > 4.0 * t['r'] or int(np.argmin(D[:, j])) != i: continue        # 그 개구부에 더 가까운 다른 태그가 있으면 버림
        o = openings[j]
        out.append(dict(ratio=o['w'] / t['W'], W=t['W'], w_px=o['w'], kind=o['kind'], tag=t['text'], code=t.get('code') or t.get('code0'), d=round(float(D[i, j] / t['r']), 2)))
    return out


TAG_PAIR2 = os.environ.get('FIRE3DBEAM_TAG_PAIR2', '1') != '0'   # 태그 짝 축척 2차(2026-10-03): 창 짝 위주·홑문 짝 제외·기호 종류 어긋난 짝 제외·폭 종류 수로 가중


def tag_scale(pairs, tol=AGREE):
    """태그 짝 → (px/m, 지지 수, 지지한 폭 종류 수). 모델 '문' 개체 폭은 문틀·문짝 꼴 때문에 홑문(폭 < 1.6 m)에서 명목 폭보다 15~40% 넓게 나와
    (시험 도면 B 글자층 태그 52쌍에서 확인) 홑문 짝은 쓰지 않음. 기호가 문(…D)인데 모델이 창으로 본 짝(또는 반대)도 뺌. 폭 3 m 넘는 태그(연창·전면 유리)는 개체가 갈라져 뺌."""
    use = []
    for p in pairs:
        c = (p.get('code') or '').upper()
        if c and ((c.endswith('D') and p['kind'] == 'window') or (c.endswith('W') and p['kind'] == 'door')): continue
        if p['W'] > 3.0: continue
        if p['kind'] == 'door' and p['W'] < 1.6: continue
        use.append(p)
    if not use: return None, 0, 0
    v = np.asarray([p['ratio'] for p in use], float)
    best = None
    for a in v:
        sel = np.abs(np.log(v / a)) < math.log(1 + tol)
        k = len({round(p['W'], 2) for p, m in zip(use, sel) if m})
        key = (int(sel.sum()) + 0.5 * k, -float(np.std(np.log(v[sel]))))
        if best is None or key > best[0]: best = (key, float(np.exp(np.mean(np.log(v[sel])))), int(sel.sum()), k)
    return best[1], best[2], best[3]


def ratio_cluster(vals, tol=AGREE):
    """값 목록 → (대표값, 지지 수) — 서로 ±tol 안의 최대 무리"""
    if not vals: return None, 0
    v = np.asarray(vals, float)
    best = None
    for a in v:
        sel = v[np.abs(np.log(v / a)) < math.log(1 + tol)]
        if best is None or len(sel) > best[1] or (len(sel) == best[1] and np.std(np.log(sel)) < best[2]):
            best = (float(np.exp(np.mean(np.log(sel)))), len(sel), float(np.std(np.log(sel))))
    return best[0], best[1]


def fuse(cands, tol=AGREE):
    """[{s, w, src}] → (s, src 문자열, 신뢰도 0~1, 갈림 여부). 가중 지지가 가장 큰 무리의 가중 기하평균."""
    C = [c for c in cands if c.get('s') and c['s'] > 0 and c.get('w', 0) > 0]
    if not C: return None, None, 0.0, False
    if BUBBLE_V2:                                          # 이 층 자체의 명시 축척(시트·치수)이 서로 맞으면(±3%) 다른 층에서 옮겨 온 후보가 뒤집지 못함
        E = [c for c in C if c['src'] in ('sheet', 'dims') and c['w'] >= 0.55]
        if E and (len(E) >= 2 and max(abs(math.log(c['s'] / E[0]['s'])) for c in E) < math.log(1.03) or (len(E) == 1 and E[0]['w'] >= 0.8)):
            C = [c for c in C if abs(math.log(c['s'] / E[0]['s'])) < math.log(1 + tol)] or C
    best = None
    for c in C:
        grp = [d for d in C if abs(math.log(d['s'] / c['s'])) < math.log(1 + tol)]
        sup = sum(d['w'] for d in grp); top = max(d['w'] for d in grp)
        if best is None or (sup, top) > (best[0], best[1]): best = (sup, top, grp)
    sup, top, grp = best
    exp_ = [d for d in grp if d['src'] in ('sheet', 'dims') and d['w'] >= 0.8]      # 명시 출처(용지가 적힌 시트 축척·치수 합의)가 무리 안에 있으면 그 값
    s = math.exp(sum(d['w'] * math.log(d['s']) for d in (exp_ or grp)) / sum(d['w'] for d in (exp_ or grp)))
    if FUSE_TRIM and not exp_ and len(grp) >= 3:              # 명시 출처 없음: 무리(±8%) 안 가중 중앙값 ±4% 만 남겨 평균(한쪽으로 치우친 태그·면적 후보가 끌어당기지 않게)
        G = sorted(grp, key=lambda d: d['s']); tw = sum(d['w'] for d in G); acc = 0.0; med = G[-1]['s']
        for d in G:
            acc += d['w']
            if acc >= 0.5 * tw: med = d['s']; break
        sub = [d for d in G if abs(math.log(d['s'] / med)) < math.log(1 + FUSE_TRIM)]
        if sum(d['w'] for d in sub) >= 0.5 * tw:
            s = math.exp(sum(d['w'] * math.log(d['s']) for d in sub) / sum(d['w'] for d in sub))
    tot = sum(c['w'] for c in C)
    srcs = '+'.join(sorted({d['src'] for d in grp}, key=lambda k: -max(d['w'] for d in grp if d['src'] == k)))
    strong = [c for c in C if c['w'] >= 0.5 and abs(math.log(c['s'] / s)) > math.log(1 + DISAGREE)]
    return s, srcs, round(min(1.0, top) * sup / tot, 3), bool(strong)


# ───────────────────────── 건물 덩어리 외곽(mass) ─────────────────────────
MASS_MODE = os.environ.get('FIRE3DBEAM_MASS_MODE', 'fill')     # fill(기본: 닫고 채우기 − 벗긴 마당) | inside(실내 성분 모으기)
OPEN_FRAC = 0.25          # 바깥 쪽 경계 길이 중 창·문 비율이 이보다 작으면 '개구부 없는' 공간(마당형 후보)
YARD_M2 = 20.0            # 바깥 쪽 개구부 없는 공간 중 이 면적 이상은 출입문이 있어도 마당형(실이 아님)으로 벗김
EXT_M = 0.6              # 외곽 닫기용 벽 끝 연장(m)
BRIDGE_M = 8.0            # 같은 선 위 벽 사이 틈(모델이 문·창으로 못 본 접이문·셔터·유리벽) 이 길이 이하면 외곽 닫기용 가상 벽


def bridge_lines(walls, ppm, max_m=BRIDGE_M, tol_m=0.3, with_len=False):
    """축 벽 선분(지역 m, y 위가 +) → 같은 선(중심선 차 ≤ tol) 위 이웃 벽 사이 틈(≤ max_m)을 잇는 선분 목록 [(x0,y0,x1,y1) 작업 px]"""
    out = []
    for o in ('h', 'v'):
        S = sorted([w for w in walls if w['o'] == o], key=lambda w: (round(w['c'] / tol_m), w['a']))
        for i, p in enumerate(S):
            best = None
            for q in S[i + 1:]:
                if abs(q['c'] - p['c']) > tol_m or q['a'] <= p['b']: continue
                g = q['a'] - p['b']
                if g <= max_m and (best is None or g < best[0]): best = (g, q)
            if best:
                q = best[1]; c = (p['c'] + q['c']) / 2
                if o == 'h': out.append((p['b'] * ppm, -c * ppm, q['a'] * ppm, -c * ppm))
                else: out.append((c * ppm, -p['b'] * ppm, c * ppm, -q['a'] * ppm))
                if with_len: out[-1] = out[-1] + (min(p['b'] - p['a'], q['b'] - q['a']),)
    return out


BRIDGE_INK = os.environ.get('FIRE3DBEAM_BRIDGE_INK', '1') != '0'
WORD_PEEL = os.environ.get('FIRE3DBEAM_WORD_PEEL', '1') != '0'     # 옥상·조경·방수 주기 글자만 있는 문 없는 공간 벗기기(2026-10-03 2차, 글자층 도면에서 특히)
ROOFNOTE = re.compile(r'(방수|파라펫|PARAPET|우레탄|드레인|DRAIN|도포|잔디|식재|수목|자갈)', re.I)
RIM_KEEP = os.environ.get('FIRE3DBEAM_RIM_KEEP', '1') != '0'   # 바깥 쪽이 장벽으로 막힌 좁은 띠는 남김(AI허브 보류 아파트 설비 공간, 2026-10-03)
STRIP_PEEL = os.environ.get('FIRE3DBEAM_STRIP_PEEL', '1') != '0'
ROOMISH = re.compile(r'(실|방|홀|강당|로비|사무|화장|창고|복도|주방|거실|침실|현관|계단|교실|센터|공간|ROOM|HALL|WC)', re.I)
LANDISH = re.compile(r'(조경|화단|식재|정원|주차|데크|DECK|마당|필로티|옥상|지붕|ROOF|외부|대지)', re.I)


def gap_ink(src, x0, y0, x1, y1, ppm, band_m=0.45):
    """작업 px 선분(벽 틈) → 원본 그림에서 그 틈을 따라 곧게 이어진 선(유리·셔터·벽 선)의 덮음 비율(0~1).
    벽 중심선이 원본 선과 조금 어긋날 수 있어(겹선 벽 두께) ±band_m 안의 평행 위치마다 덮음을 재고 가장 큰 값. src=(회색 원본, 원본px/작업px)"""
    g, k = src
    H, W = g.shape
    n = max(8, int(math.hypot(x1 - x0, y1 - y0) / 2))
    b = max(1, int(round(band_m * ppm * k)))
    horiz = abs(y1 - y0) < abs(x1 - x0)
    cols = []
    for f in np.linspace(0.05, 0.95, n):
        x, y = (x0 + f * (x1 - x0)) * k, (y0 + f * (y1 - y0)) * k
        xi, yi = int(round(x)), int(round(y))
        if not (b <= xi < W - b and b <= yi < H - b): continue
        win = g[yi - b:yi + b + 1, xi] if horiz else g[yi, xi - b:xi + b + 1]
        cols.append(win < 170)
    if not cols: return 0.0
    M = np.array(cols)                                      # 표본 × 평행 위치
    M = M | np.roll(M, 1, axis=1) | np.roll(M, -1, axis=1)  # ±1 px
    return float(M.mean(0).max())


def mass(lab, ppm, protect=(), close_m=0.25, peel=True, walls=None, neck_m=1.5, drop=(), words=(), src=None):
    """STR 라벨(작업 해상도, ppm px/m) → dict(mask=건물 덩어리(채움), exterior=바깥 빈 공간, peeled=[벗긴 마당형 성분 면적 m²], door_ext=[문 개체별 바깥 접함]).
    protect: 벗기지 않을 상자(작업 px) — 계단·승강기(코어)."""
    bar = (lab > 0).astype(np.uint8)
    if drop and walls:                                            # 건물 벽 꼴이 아닌 '선'(대지선·주차구획·연석·화단) 지우기 — 창·문 화소는 남김
        rm = np.zeros_like(bar)
        for i in drop:
            w = walls[i]; th = max(3, int(round((w['t'] + 0.2) * ppm)))
            if w['o'] == 'h': p0, p1 = (w['a'] * ppm, -w['c'] * ppm), (w['b'] * ppm, -w['c'] * ppm)
            else: p0, p1 = (w['c'] * ppm, -w['a'] * ppm), (w['c'] * ppm, -w['b'] * ppm)
            cv2.line(rm, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), 1, th)
        bar[(rm > 0) & (lab == 1)] = 0
        keep = set(range(len(walls))) - set(drop)
        walls = [w for i, w in enumerate(walls) if i in keep]
    if walls:                                                     # 같은 선 위 벽 틈(≤8 m) 잇기 — 접이문·셔터·유리 출입구가 모델 문·창으로 안 잡혀도 외곽이 닫히게
        th = max(2, int(round(0.15 * ppm)))
        for x0, y0, x1, y1, lm in bridge_lines(walls, ppm, with_len=True):
            # 짧은 벽 조각(1.5 m 미만, 대지선·주차선 끝 등)에서 시작하는 틈은 원본 그림에 그 틈을 잇는 선(유리·셔터·문)이 있을 때만 잇기
            if src is not None and BRIDGE_INK and lm < 1.5 and gap_ink(src, x0, y0, x1, y1, ppm) < 0.5: continue
            cv2.line(bar, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, th)
        e = EXT_M * ppm                                           # 벽 끝 늘리기: ㄱ자 모서리에서 두 벽 끝이 조금 모자라 생긴 틈 닫기(geom.FP_EXT 와 같은 생각)
        for w in walls:
            a, b = w['a'] * ppm - e, w['b'] * ppm + e
            if w['o'] == 'h': p0, p1 = (a, -w['c'] * ppm), (b, -w['c'] * ppm)
            else: p0, p1 = (w['c'] * ppm, -a), (w['c'] * ppm, -b)
            cv2.line(bar, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), 1, th)
    bar_raw = bar.copy()
    k = max(1, int(round(close_m * ppm)))
    bar = cv2.morphologyEx(bar, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1)))
    free = (1 - bar).astype(np.uint8)
    # 좁은 목(< neck_m: 모서리 틈·문 개체 없는 출입구)으로만 이어진 공간은 끊음 → 방 하나가 작은 틈으로 바깥과 이어져 통째로 '바깥'이 되는 것 방지
    r = max(1, int(round(neck_m * ppm / 2)))
    free0 = free
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n0, cc0 = cv2.connectedComponents(free0, connectivity=4)
    b0 = set(np.unique(np.r_[cc0[0], cc0[-1], cc0[:, 0], cc0[:, -1]]).tolist()) - {0}
    xmask = (np.isin(cc0, list(b0)) & (free0 > 0) & (free == 0)).astype(np.uint8)    # 열기로 지워진 좁은 바깥 띠(대지선과 벽 사이 1~1.5 m 등) = 바깥 취급
    n, cc, st, _ = cv2.connectedComponentsWithStats(free, 4)
    H, Wd = free.shape
    border = set(np.unique(np.r_[cc[0], cc[-1], cc[:, 0], cc[:, -1]]).tolist()) - {0}
    ext = np.zeros(n, bool)
    for i in border: ext[i] = True
    area = st[:, 4] / ppm / ppm
    ops = []                                                      # 개구부(창·문) 개체: (부류, 성분 마스크 상자, 화소)
    for cls_, kind in ((2, 'window'), (3, 'door')):
        m = (lab == cls_).astype(np.uint8)
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, nn):
            if ss[j, 4] < 6: continue
            x, y, w, h = ss[j, :4]
            ops.append(dict(kind=kind, box=(x, y, x + w, y + h), mask=(ll[y:y + h, x:x + w] == j)))
    d = max(2, int(round(0.6 * ppm)))                             # 벽 너머 이웃 성분 탐색 거리(벽 두께 + 여유)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    d2 = max(2, int(round(0.4 * ppm)))                            # 개구부가 닿는 공간: 개구부에서 0.4 m 안(벽 두께 + 여유)
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d2 + 1, 2 * d2 + 1))

    def touch(op):                                               # 개구부가 닿는 빈 공간 성분 목록
        x0, y0, x1, y1 = op['box']
        X0, Y0, X1, Y1 = max(0, x0 - d - 2), max(0, y0 - d - 2), min(Wd, x1 + d + 2), min(H, y1 + d + 2)
        m = np.zeros((Y1 - Y0, X1 - X0), np.uint8); m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0][op['mask']] = 1
        m = cv2.dilate(m, ker2)
        u = cc[Y0:Y1, X0:X1][m > 0]
        T = set(np.unique(u).tolist()) - {0}
        if xmask[Y0:Y1, X0:X1][m > 0].any(): T.add(-1)          # -1 = 좁은 바깥 띠
        return T
    op_touch = [touch(o) for o in ops]
    prot = set()
    for b in protect:                                            # 계단·승강기 상자(작업 px)를 주로 차지하는 공간(상자 면적의 4배 이하)은 벗기지 않음
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        sub = cc[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if not sub.size: continue
        u, c = np.unique(sub[sub > 0], return_counts=True)
        for j, k in zip(u, c):
            if k >= 0.3 * sub.size and st[j, 4] <= 4 * sub.size: prot.add(int(j))
    peeled, peel_set = [], set()
    cand = [i for i in range(1, n) if not ext[i] and area[i] >= 1.0]
    nb = {}
    for i in cand:                                                # 이웃(벽 너머) 성분 화소 수
        x, y, w, h = st[i, :4]
        X0, Y0, X1, Y1 = max(0, x - d), max(0, y - d), min(Wd, x + w + d), min(H, y + h + d)
        sub = (cc[Y0:Y1, X0:X1] == i).astype(np.uint8)
        ring = cv2.dilate(sub, ker) & (1 - sub)
        u, c = np.unique(cc[Y0:Y1, X0:X1][ring > 0], return_counts=True)
        nb[i] = {int(a): int(b) for a, b in zip(u, c) if a != 0 and a != i}
        nx = int((xmask[Y0:Y1, X0:X1][ring > 0]).sum())
        if nx: nb[i][-1] = nx
        hole = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, ker)
        cn, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill = np.zeros_like(sub); cv2.drawContours(fill, cn, -1, 1, -1)
        nb[i]['_holes'] = float((fill & (1 - sub)).sum()) / ppm / ppm         # 이 성분이 둘러싼(구멍) 면적 m² — 건물을 두른 고리형 마당
    wl_land, wl_room = set(), set()                               # 성분 안 글자: 옥상·조경·주차·방수 주기(바깥) / 실 이름(실내)
    if WORD_PEEL and src is not None and words:
        kq = src[1]
        for w_ in words:
            t_ = (w_.get('text') or '').replace(' ', '')
            x_, y_ = int((w_['box'][0] + w_['box'][2]) / 2 / kq), int((w_['box'][1] + w_['box'][3]) / 2 / kq)
            if not (0 <= x_ < Wd and 0 <= y_ < H) or cc[y_, x_] == 0: continue
            t2_ = ROOFNOTE.sub('', t_)                                  # '방수'의 '방'을 실 이름으로 보지 않게
            if (LANDISH.search(t_) or ROOFNOTE.search(t_)) and not ROOMISH.search(t2_): wl_land.add(int(cc[y_, x_]))
            elif ROOMISH.search(t2_) and not NOT_ROOM.search(t2_): wl_room.add(int(cc[y_, x_]))
    changed = peel
    while changed:                                                # 한 바퀴마다 같은 상태로 판정 후 한꺼번에 벗김(순서 무관)
        changed = False
        to_peel = []
        for i in cand:
            if ext[i] or i in prot: continue
            N = nb[i]
            e = sum(c for j, c in N.items() if j != '_holes' and (j == -1 or ext[j]))
            tot = sum(c for j, c in N.items() if j != '_holes')
            if tot == 0 or e < 0.3 * tot: continue
            # 바깥 쪽 경계의 창·문 길이 비율(개구부 길이 ÷ 바깥과 맞닿은 경계 길이): 외벽 실은 창·문이 이어짐, 마당·옥상정원 난간에는 모델이 가끔 잘못 본 짧은 조각뿐
            L_e = e / d / ppm
            L_o = sum(max(o['box'][2] - o['box'][0], o['box'][3] - o['box'][1]) for o, T in zip(ops, op_touch) if i in T and any((j == -1 or ext[j]) for j in T if j != i)) / ppm
            if L_o >= OPEN_FRAC * max(L_e, 0.5): continue           # 바깥 쪽 경계에 창·문 → 건물 실
            doors_in = sum(1 for o, T in zip(ops, op_touch) if o['kind'] == 'door' and i in T and any((j != -1 and not ext[j]) for j in T if j != i))
            # 바깥 쪽에 개구부가 없는 20㎡ 이상 공간(출입문이 있어도: 마당·주차장·옥상정원) · 건물을 두른 고리. 작은 공간은 두어(발코니 옆 침실처럼 모델이 창을 놓친 실 보호) 조금 크게 잡힘
            # + 문 없는 작은 공간이 대부분(≥60%) 바깥에 둘러싸임(벗긴 옥상정원 안의 화단·조경 조각, 주차구획 조각)
            x_, y_, w_, h_ = st[i, :4]
            strip = STRIP_PEEL and min(w_, h_) < 1.6 * ppm and max(w_, h_) >= 4 * min(w_, h_) and area[i] >= 3.0 and doors_in <= 1     # 길쭉한 띠(화단·연석·난간 사이, 폭 < 1.6 m) — 바깥쪽 개구부 없음
            land_w = i in wl_land and i not in wl_room and doors_in == 0 and area[i] >= 3.0       # 옥상·조경·방수 주기 글자만 든 문 없는 공간(옥탑 지붕 평면 삽도·화단 등)
            if area[i] >= YARD_M2 or N['_holes'] >= 20 or strip or land_w or (doors_in == 0 and e >= 0.6 * tot and any(j in peel_set for j in N if j != '_holes')): to_peel.append(i)
        for i in to_peel:
            ext[i] = True; peeled.append(round(float(area[i]), 1)); changed = True; peel_set.add(i)
    def thin_strips():
        """좁아서(목 열기로 사라진) 닫힌 빈 띠 중 바깥(벗긴 공간 포함)과 30% 넘게 맞닿고 창이 닿지 않는 길쭉한 것 → 마스크"""
        outm = np.zeros((H, Wd), np.uint8)
        extm = (np.isin(cc, np.nonzero(ext)[0]) & (cc > 0)) | (xmask > 0)
        win = cv2.dilate((lab == 2).astype(np.uint8), ker2)
        n0_, l0_, s0_, _ = cv2.connectedComponentsWithStats(free0, 4)
        for j in range(1, n0_):
            if j in b0: continue
            x_, y_, w_, h_, a_ = s0_[j]
            if a_ / ppm / ppm < 3.0 or min(w_, h_) >= 1.6 * ppm or max(w_, h_) < 4 * min(w_, h_): continue
            X0, Y0, X1, Y1 = max(0, x_ - d), max(0, y_ - d), min(Wd, x_ + w_ + d), min(H, y_ + h_ + d)
            comp = (l0_[Y0:Y1, X0:X1] == j)
            if (cc[Y0:Y1, X0:X1][comp] > 0).any(): continue          # 넓은 부분이 있는 공간은 성분 판정에 맡김
            ring = cv2.dilate(comp.astype(np.uint8), ker) & (~comp)
            if ring.sum() == 0 or extm[Y0:Y1, X0:X1][ring > 0].mean() < 0.3 or win[Y0:Y1, X0:X1][comp].any(): continue
            outm[Y0:Y1, X0:X1][comp] = 1
        return outm.astype(bool)
    # 기본 외곽 = realplan 모델 경로와 같은 '닫고 채우기'(벽∪창∪문 + 같은 선 틈 잇기, 닫기 0.3/0.6/1.3 m 중 채움 85% 유지되는 가장 작은 것)
    #   − 벗긴 마당형 공간(측지 팽창). 채우기가 실패하면 아래 실내 성분 방식(MASS_MODE='inside')으로
    from fire3dbeam import realplan as RP
    from fire3dbeam.str_seg import wall_thickness
    k3 = np.ones((3, 3), np.uint8)
    pg = np.isin(cc, list(peel_set)).astype(np.uint8)
    for _ in range(r + 2): pg = cv2.dilate(pg, k3) & free0
    if MASS_MODE == 'fill' and sum(area[i] for i in peel_set) < YARD_M2:      # 마당형을 크게 벗긴 층(대지·옥상정원)은 실내 성분 방식이 더 정확 → 아래로
        tw = int(max(3, round(wall_thickness((lab == 1).astype(np.uint8))))) if (lab == 1).any() else 3
        try:
            _, F1 = RP.outline(bar_raw, tw, ppm / 1000.0, RP.M_CLOSE_MM, RP.M_OPEN_MM)
        except Exception:
            F1 = None
        if F1 is not None and F1.sum() > 0:
            fill = (F1 & (1 - pg)).astype(np.uint8)
            if STRIP_PEEL:                                        # 닫기로 채워진 바깥 공간(그림 가장자리와 이어진 빈 공간 3 ㎡ 이상) 다시 빼기 — 화단·난간·연석 사이 띠, 오목한 마당
                E = (np.isin(cc, np.nonzero(ext)[0]) & (cc > 0) & (F1 > 0)).astype(np.uint8)
                ne_, le_, se_, _ = cv2.connectedComponentsWithStats(E, 4)
                winD = cv2.dilate(((lab == 2) | (lab == 3)).astype(np.uint8), ker2)
                kk_ = src[1] if src is not None else None
                wpts = [((w_['box'][0] + w_['box'][2]) / 2 / kk_, (w_['box'][1] + w_['box'][3]) / 2 / kk_) for w_ in (words or ())
                        if kk_ and ROOMISH.search((w_.get('text') or '').replace(' ', '')) and not LANDISH.search((w_.get('text') or '').replace(' ', ''))]
                for j in range(1, ne_):
                    if se_[j, 4] / ppm / ppm >= 3.0:
                        comp = (le_ == j).astype(np.uint8)
                        # 창·문이 닿았거나 실 이름 글자가 든 공간은 틈으로 새어 바깥과 이어진 '실'(강당 등) → 빼지 않음
                        if (winD[comp > 0] > 0).mean() > 0.02 or any(0 <= int(y) < H and 0 <= int(x) < Wd and comp[int(y), int(x)] for x, y in wpts): continue
                        for _ in range(r + 2): comp = cv2.dilate(comp, k3) & free0          # 목 열기로 깎인 가장자리까지(빈 공간 안에서만)
                        fill[comp > 0] = 0
                X = ((xmask | thin_strips()) & F1).astype(np.uint8)
                nx_, lx_, sx_, _ = cv2.connectedComponentsWithStats(X, 8)
                outside = cv2.dilate((1 - F1).astype(np.uint8), np.ones((3, 3), np.uint8))
                nearo = cv2.dilate((1 - F1).astype(np.uint8), ker2) > 0          # 덩어리 밖에서 0.4 m 안
                for j in range(1, nx_):
                    x_, y_, w_, h_, a_ = sx_[j]
                    if a_ / ppm / ppm < 3.0 or min(w_, h_) >= 1.6 * ppm or max(w_, h_) < 4 * min(w_, h_): continue
                    comp = lx_ == j
                    if not (outside[comp] > 0).any(): continue
                    # 바깥 쪽 테두리(띠와 바깥 사이 장벽)가 주로 창·문이면 발코니(창이 군데군데 빠져 바깥과 이어진 것) → 남김. 화단·난간 띠는 바깥 쪽이 벽·난간선이거나 없음
                    ring_ = cv2.dilate(comp.astype(np.uint8), ker2) > 0
                    rim = ring_ & nearo & (lab > 0)
                    if rim.sum() >= 4 and np.isin(lab[rim], (2, 3)).mean() >= 0.3: continue
                    # 바깥 쪽이 띠 길이의 70% 넘게 장벽(벽·창·문)으로 막힘 = 두 벽 사이 좁은 실(설비 공간·발코니)이 끝만 트인 것 → 남김
                    #   (실제 도면 경로에서는 건물 벽과 꼴이 다른 난간·화단 선을 먼저 빼므로(wall_styles) 화단 띠는 바깥이 트여 여기서 빠짐)
                    ax_ = 0 if w_ >= h_ else 1
                    rr = rim[max(0, y_ - d2):y_ + h_ + d2, x_:x_ + w_] if ax_ == 0 else rim[y_:y_ + h_, max(0, x_ - d2):x_ + w_ + d2]
                    cov = rr.any(axis=0).mean() if ax_ == 0 else rr.any(axis=1).mean()
                    if RIM_KEEP and cov >= 0.7: continue
                    fill[cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
            k2 = max(3, int(round(0.5 * ppm)) | 1)
            fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2)))
            nn, ll, ss, _ = cv2.connectedComponentsWithStats(fill, 8)
            if nn > 1:
                big = int(np.argmax(ss[1:, 4])) + 1
                fill = np.isin(ll, [j for j in range(1, nn) if ss[j, 4] >= 0.2 * ss[big, 4] and ss[j, 4] / ppm / ppm >= 6]).astype(np.uint8)
            door_ext = [bool(any((j == -1 or ext[j]) for j in T)) and bool(any((j != -1 and not ext[j]) for j in T)) for T in op_touch]
            return dict(mask=fill, exterior=np.isin(cc, np.nonzero(ext)[0]) & (cc > 0), peeled=peeled, ops=ops, op_ext=door_ext, mode='fill')
    inside = np.isin(cc, [i for i in range(1, n) if not ext[i]]).astype(np.uint8)
    inside = inside & (cc > 0)
    # 열기 전 빈 공간 성분 중 바깥과 안 이어진 것: 실내 성분만 품었거나(열기로 깎인 모서리·좁은 곳 되살림) 아무 성분도 없는(폭 < neck_m 발코니·창고·샤프트) → 통째로 실내
    pairs = np.c_[cc0[free0 > 0], cc[free0 > 0]]
    has = {}
    for a, b in np.unique(pairs, axis=0):
        if a in b0 or a == 0: continue
        has.setdefault(int(a), set())
        if b: has[int(a)].add(int(b))
    good = [a for a, S in has.items() if all(not ext[b] for b in S)]
    if good: inside = (inside | (np.isin(cc0, good) & (free0 > 0))).astype(np.uint8)
    for _ in range(r + 2):                                        # 열기로 깎인 방 모서리 되살리기: 빈 공간 안에서만(벽 너머로 안 번짐) r+2 px 측지 팽창
        inside = cv2.dilate(inside, k3) & free0
    # 실내 성분 + 그에 붙은 장벽(벽·창·문) = 건물 덩어리
    dd = max(2, int(round(0.45 * ppm)))
    grow = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dd + 1, 2 * dd + 1)))
    M = ((inside > 0) | ((grow > 0) & (bar > 0))).astype(np.uint8)
    kc = max(3, int(round(neck_m * 1.1 * ppm)) | 1)               # 좁은 실(화장실 칸·샤프트)·벽으로 갈라진 조각 잇기(열기로 지워진 폭 < neck_m 공간 메움)
    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    cn, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(M); cv2.drawContours(fill, cn, -1, 1, -1)
    k2 = max(3, int(round(0.5 * ppm)) | 1)                       # 0.5 m 열기: 덩어리에 붙어 삐져나온 가는 선 제거
    fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2)))
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(fill, 8)
    if nn > 1:
        big = int(np.argmax(ss[1:, 4])) + 1
        keepc = [j for j in range(1, nn) if ss[j, 4] >= 0.2 * ss[big, 4] and ss[j, 4] / ppm / ppm >= 6]
        fill = np.isin(ll, keepc).astype(np.uint8)
    door_ext = []
    for o, T in zip(ops, op_touch):
        door_ext.append(bool(any((j == -1 or ext[j]) for j in T)) and bool(any((j != -1 and not ext[j]) for j in T)))
    return dict(mask=fill, exterior=np.isin(cc, np.nonzero(ext)[0]) & (cc > 0), peeled=peeled, ops=ops, op_ext=door_ext, mode='inside')


WALL_STYLE = os.environ.get('FIRE3DBEAM_WALL_STYLE', '1') != '0'     # 모델 '벽' 중 건물 벽과 그림 꼴이 다른 선(대지선·주차구획·연석·화단) 빼기(2026-10-03)


def wall_profile(g, w, sm, n=12):
    """모델 벽 선분(지역 m, 모델 축척 sm px/m) → 원본 그림 단면 특징 dict(span 잉크 폭 m, core 가운데 60% 잉크 비율, runs 잉크 줄 수)."""
    H, Wd = g.shape
    t = max(w['t'], 0.08)
    half = int(round(max(0.45, 1.5 * t) * sm))
    spans, cores, runs = [], [], []
    for f in np.linspace(0.15, 0.85, n):
        u = (w['a'] + f * (w['b'] - w['a'])) * sm
        c = w['c'] * sm
        if w['o'] == 'h': x, y = int(round(u)), int(round(-c))
        else: x, y = int(round(c)), int(round(-u))
        if w['o'] == 'h':
            if not (0 <= x < Wd): continue
            prof = g[max(0, y - half):min(H, y + half + 1), max(0, x - 1):x + 2].min(1)
        else:
            if not (0 <= y < H): continue
            prof = g[max(0, y - 1):y + 2, max(0, x - half):min(Wd, x + half + 1)].min(0)
        ink = prof < 170
        if not ink.any(): spans.append(0.0); cores.append(0.0); runs.append(0); continue
        # 선분 중심(모델 벽 중심)에 가장 가까운 잉크 덩어리 묶음: 중심 ±(t/2 + 0.12 m) 안 잉크의 처음~끝
        mid = len(prof) // 2; r_ = int(round((t / 2 + 0.12) * sm))
        idx = np.nonzero(ink[max(0, mid - r_):mid + r_ + 1])[0] + max(0, mid - r_)
        if not len(idx): spans.append(0.0); cores.append(0.0); runs.append(0); continue
        a0, a1 = idx.min(), idx.max()
        sp = a1 - a0 + 1
        spans.append(sp / sm)
        k0, k1 = a0 + int(0.2 * sp), a1 - int(0.2 * sp)
        cores.append(float(ink[k0:k1 + 1].mean()) if k1 >= k0 and sp >= 4 else 1.0)
        seg = ink[a0:a1 + 1].astype(np.int8)
        runs.append(int((np.diff(np.r_[0, seg]) == 1).sum()))
    if not spans: return None
    return dict(span=float(np.median(spans)), core=float(np.median(cores)), runs=float(np.median(runs)), span_px=float(np.median(spans)) * sm)


def wall_styles(g, walls, sm, ops, near_m=0.5):
    """모델 벽마다 그림 꼴 → 건물 벽 꼴과 다른 '선'(대지선·주차·연석·화단) 목록(인덱스).
    본보기(앵커) = 창·문 개구부 0.5 m 안에 닿는 벽(건물 외벽·칸막이 확실). 앵커 꼴이 채움(빗금·검정, core ≥ 0.15)이면 채움 없는 겹선·홑선을,
    앵커가 겹선(core 낮음)이면 앵커 폭의 35% 미만 홑선만 '선'으로. 개구부에 닿는 벽·1.5 m 미만 짧은 벽은 남김. 앵커 5개 미만이면 아무것도 안 뺌."""
    if not walls: return [], {}
    P = [wall_profile(g, w, sm) for w in walls]
    def near_op(w):
        lo, hi = min(w['a'], w['b']), max(w['a'], w['b'])
        for o in ops:
            x, y, r = o['x'], o['y'], o.get('w', 0.9) / 2 + near_m          # 개구부 폭 끝에서 near_m 안
            if w['o'] == 'h':
                if x + r >= lo and x - r <= hi and abs(y - w['c']) <= near_m: return True
            else:
                if y + r >= lo and y - r <= hi and abs(x - w['c']) <= near_m: return True
        return False
    anc = [i for i, w in enumerate(walls) if P[i] and near_op(w) and (w['b'] - w['a']) >= 0.6]
    info = dict(n_anchor=len(anc))
    if len(anc) < 5: return [], info
    sp = float(np.median([P[i]['span'] for i in anc])); co = float(np.median([P[i]['core'] for i in anc]))
    info.update(anchor_span=round(sp, 3), anchor_core=round(co, 3))
    if sp * sm < 6: info['low_res'] = True; return [], info      # 벽 두께가 6 px 미만으로 보이는 저해상도: 그림 꼴로 벽·선을 가릴 수 없음 → 빼지 않음
    drop = []
    for i, w in enumerate(walls):
        if i in anc or not P[i] or (w['b'] - w['a']) < 1.5 or near_op(w): continue
        q = P[i]
        thin = q['span'] < 0.35 * sp or q['span_px'] <= 2.5
        unfilled = co >= 0.15 and q['core'] < 0.25 * co and q['span'] < 1.6 * sp
        if thin or unfilled: drop.append(i)
    info['dropped'] = len(drop)
    return drop, info


def mass_checked(lab, ppm, protect, drop=(), info=None, **kw):
    """mass(drop=꼴 다른 선 빼기) — 단, 빼서 건물 창이 덩어리 밖으로 밀려나면(창 2개 이상 또는 20% 이상 잃음) 빼기를 무름(외벽 일부를 선으로 잘못 본 경우)"""
    m1 = mass(lab, ppm, protect, drop=drop, **kw)
    if not drop: return m1
    m0 = mass(lab, ppm, protect, drop=(), **kw)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * ppm) + 1,) * 2)
    a0, a1 = cv2.dilate(m0['mask'].astype(np.uint8), k), cv2.dilate(m1['mask'].astype(np.uint8), k)
    win = (lab == 2).astype(np.uint8)
    n, ll, st, _ = cv2.connectedComponentsWithStats(win, 8)
    have0 = [j for j in range(1, n) if st[j, 4] >= 6 and (a0[ll == j] > 0).mean() > 0.5]
    lost = [j for j in have0 if (a1[ll == j] > 0).mean() <= 0.5]
    if info is not None: info.update(win0=len(have0), win_lost=len(lost))
    a0_, a1_ = float(m0['mask'].sum()), float(m1['mask'].sum())
    if len(lost) >= max(3, 0.3 * len(have0)) or (len(lost) >= 2 and a1_ < 0.5 * a0_):     # 창 여럿(3개·30% 이상)을 잃거나, 덩어리가 절반 넘게 줄며 창 2개 이상 잃음 → 외벽을 잘못 뺀 것
        if info is not None: info['reverted'] = True
        return m0
    return m1


NAMED_ROOM = os.environ.get('FIRE3DBEAM_NAMED_ROOM', '1') != '0'    # 한 변이 트인(셔터·차고문 점선) 이름 있는 실 되살리기(2026-10-03, 보류 N 진단 후)
ROOM_END = re.compile(r'(실|방|홀|당|대|고|소|장|관|청|과|팀|반|국|센터|ROOM|HALL)$', re.I)
NOT_ROOM = re.compile(r'(평면|축척|도면|공사|범위|확인|설치|시공|기존|철거|입면|단면|주기|일람|상세|부분|위치|캐노피|테라스|발코니|옥외|도로|인접|경사|램프|진입|광장|공원|운동장|놀이)', re.I)


def named_rooms(mask, lab, words, kk, ppm, max_m=9.0):
    """덩어리 밖 실 이름 글자(구조대·차고·창고 …)가 세 변이 건물 벽(덩어리에 붙은 장벽)으로 둘린 칸 안에 있으면 그 칸을 덩어리에 더함.
    한 변이 셔터·차고문(얇은 점선 — 모델이 벽으로 안 봄)이라 닫고 채우기로 못 닫힌 실. 트인 변은 양옆 벽이 끝나는 곳까지.
    mask·lab 작업 px(ppm), words 원본 px(kk = 작업/원본). → (새 mask, 더한 칸 [(x0,y0,x1,y1) 작업 px])"""
    H, W = mask.shape
    bar = (lab > 0)
    near0 = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * ppm) + 1,) * 2)) > 0
    nb_, lb_ = cv2.connectedComponents(cv2.dilate(bar.astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)
    near = np.isin(lb_, np.unique(lb_[near0 & bar])) & bar          # 덩어리에 닿은 장벽과 이어진 벽(건물 벽)
    L = int(max_m * ppm); band = max(1, int(0.25 * ppm)); gapmax = int(1.5 * ppm)
    out = mask.copy(); added = []
    for w in words or ():
        t = (w.get('text') or '').replace(' ', '')
        hang = sum(1 for ch in t if '가' <= ch <= '힣')
        if not (2 <= hang <= 6) or any(ch.isdigit() for ch in t) or len(t) > hang + 1: continue
        if not (ROOMISH.search(t) or ROOM_END.search(t)) or LANDISH.search(t) or NOT_ROOM.search(t): continue
        cx = int((w['box'][0] + w['box'][2]) / 2 * kk); cy = int((w['box'][1] + w['box'][3]) / 2 * kk)
        if not (0 <= cx < W and 0 <= cy < H) or out[cy, cx] or bar[cy, cx]: continue
        hits = {}
        for nm, dx, dy in (('l', -1, 0), ('r', 1, 0), ('u', 0, -1), ('d', 0, 1)):
            for k_ in range(1, L):
                x, y = cx + dx * k_, cy + dy * k_
                if not (0 <= x < W and 0 <= y < H): break
                if bar[y, x]:
                    if near[y, x]: hits[nm] = (x, y)
                    break
        if len(hits) != 3: continue                       # 네 변이 다 막혔으면 닫고 채우기 몫(밖에 남았다면 건물 아님), 두 변 이하는 마당
        op = ({'l', 'r', 'u', 'd'} - set(hits)).pop()
        def run(px, py, dx, dy):                          # 옆 벽 선을 따라 트인 쪽으로 벽이 이어지는 끝(틈 1.5 m 까지 허용)
            last, gap = 0, 0
            for k_ in range(1, L):
                x, y = px + dx * k_, py + dy * k_
                if not (0 <= x < W and 0 <= y < H): break
                xs = slice(max(0, x - band), x + band + 1) if dx == 0 else slice(x, x + 1)
                ys = slice(max(0, y - band), y + band + 1) if dy == 0 else slice(y, y + 1)
                if bar[ys, xs].any(): last, gap = k_, 0
                else:
                    gap += 1
                    if gap > gapmax: break
            return last
        if op in ('u', 'd'):
            dy = -1 if op == 'u' else 1
            e1, e2 = run(hits['l'][0], cy, 0, dy), run(hits['r'][0], cy, 0, dy)
            e = min(e1, e2)
            if e < 1.0 * ppm: continue
            x0, x1 = hits['l'][0], hits['r'][0]
            y0, y1 = (cy - e, hits['d'][1]) if op == 'u' else (hits['u'][1], cy + e)
        else:
            dx = -1 if op == 'l' else 1
            e1, e2 = run(cx, hits['u'][1], dx, 0), run(cx, hits['d'][1], dx, 0)
            e = min(e1, e2)
            if e < 1.0 * ppm: continue
            y0, y1 = hits['u'][1], hits['d'][1]
            x0, x1 = (cx - e, hits['r'][0]) if op == 'l' else (hits['l'][0], cx + e)
        a = (x1 - x0) * (y1 - y0) / ppm / ppm
        if not (4.0 <= a <= 120.0) or min(x1 - x0, y1 - y0) < 1.5 * ppm: continue
        if out[y0:y1 + 1, x0:x1 + 1].mean() > 0.5: continue
        out[y0:y1 + 1, x0:x1 + 1] = 1; added.append((int(x0), int(y0), int(x1), int(y1), t))
    return out, added


def mask_poly(mask, ppm, eps_m=0.15):
    """덩어리 마스크(작업 px) → shapely 다각형(지역 m: x=px/ppm, y=-py/ppm)"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    cn, hi = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    if hi is None: return None
    for k, c in enumerate(cn):
        if hi[0][k][3] != -1 or cv2.contourArea(c) < 4 * ppm * ppm: continue
        ap = cv2.approxPolyDP(c, max(1.0, eps_m * ppm), True)[:, 0, :]
        if len(ap) < 3: continue
        holes = []
        ch = hi[0][k][2]
        while ch != -1:
            if cv2.contourArea(cn[ch]) >= 4 * ppm * ppm:
                hp = cv2.approxPolyDP(cn[ch], max(1.0, eps_m * ppm), True)[:, 0, :]
                if len(hp) >= 3: holes.append([(x / ppm, -y / ppm) for x, y in hp])
            ch = hi[0][ch][0]
        pg = Polygon([(x / ppm, -y / ppm) for x, y in ap], holes).buffer(0)
        if pg.area > 4: polys.append(pg)
    return unary_union(polys) if polys else None


# ───────────────────────── 평면도 1장 ─────────────────────────
def text_roi(lines, shape, s, pad_m=2.5):
    """글자층 치수 숫자들이 건물을 둘러싸면(가로 치수 줄이 위·아래, 세로 치수 줄이 왼쪽·오른쪽에 모두) 그 범위 + pad_m → STR 을 돌릴 영역 [x0,y0,x1,y1] | None.
    시트에 비해 건물이 작은 도면(1/200 학교 등)에서 빈 종이·주기 타일을 건너뜀 — 타일 크기·겹침은 그대로라 모델 출력 꼴은 같음"""
    H, W = shape[:2]
    hs, vs = [], []
    for l in lines or ():
        t = l['t'].replace(' ', '')
        if not DIM_TXT.match(t) or not 250 <= int(t.replace(',', '')) <= 120000: continue
        b = l['box']; c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        (hs if abs(l['dir'][0]) >= abs(l['dir'][1]) else vs).append(c)
    if len(hs) < 4 or len(vs) < 4: return None
    hy = [c[1] for c in hs]; vx = [c[0] for c in vs]
    if max(hy) - min(hy) < 0.3 * H or max(vx) - min(vx) < 0.3 * W: return None         # 치수가 한쪽에만 → 건물 범위를 못 정함
    xs = [c[0] for c in hs + vs]; ys = [c[1] for c in hs + vs]
    p = pad_m * s
    x0, y0, x1, y1 = max(0, min(xs) - p), max(0, min(ys) - p), min(W, max(xs) + p), min(H, max(ys) + p)
    if (x1 - x0) * (y1 - y0) > 0.8 * W * H: return None                               # 아낄 게 적음
    return [x0, y0, x1, y1]


def _model(img, s, roi=None):
    from fire3dbeam import xcheck as X
    if roi is not None and STR_ROI: return X.model_plan(img, s, tile=REAL_TILE, even=EVEN_TILES, skip_blank=SKIP_BLANK, roi=roi)
    tile = REAL_TILE
    if STR_SINGLE_PX and not tile:                                # 작업 해상도 그림이 화소 예산 안이면 타일로 나누지 않고 한 번에(2026-10-03 2차: S 층당 STR 20→10 s, 겹친 타일 중복 계산 제거)
        from fire3dbeam.str_seg import WORK_PPM_1_100
        k = WORK_PPM_1_100 / s
        h, w = img.shape[0] * k, img.shape[1] * k
        if h * w <= STR_SINGLE_PX: tile = int(max(h, w)) + 64
    return X.model_plan(img, s, tile=tile, even=EVEN_TILES, skip_blank=SKIP_BLANK)


def _openings(P, s):
    """STR 벡터 문·창 개체 → 원본 px [{x,y,w,kind}]"""
    from fire3dbeam.str_seg import WORK_PPM_1_100
    k = WORK_PPM_1_100 / s
    out = []
    for kind in ('doors', 'windows'):
        for d in P['v'][kind]:
            out.append(dict(x=d['x'] * s, y=-d['y'] * s, w=d['w'] * s, kind=kind[:-1]))
    return out


def _spa(g, s):
    from fire3dbeam import spa_seg as SP
    return SP.detect(g, ppm=s)


def stage1(rp, info):
    """1단계(층마다): 글자·원·그리드·태그 + 임시 축척으로 STR·SPA → 축척 후보. 반환 R(dict)"""
    img = rp['img']; g = _gray(img)
    from fire3dbeam import realsym as RS
    from fire3dbeam.vision import masks, footprint_px_area
    R = dict(g=g, floors=rp['floors'], caption=rp.get('caption', ''), cands=[])
    fa = [info['floors'][f]['area'] for f in rp['floors'] if f in info['floors'] and info['floors'][f].get('area')]
    R['card_area'] = fa[0] if fa else None
    sp, stated = sheet_ppm_from_caption(rp.get('caption'))
    if sp: R['cands'].append(dict(s=sp, w=W['sheet'] if stated else W['sheet_guess'], src='sheet'))
    R['sheet_sp'], R['sheet_stated'] = sp, bool(stated)
    TL = rp.get('text') if PDF_TEXT else None                     # 벡터 도면 글자층(fire3dbeam/pdftext.py) — 있으면 치수·실명·태그·버블 OCR 대신
    R['text_src'] = 'pdf' if TL else 'ocr'
    td = text_dims_scale(TL) if TL else (None, 0.0, 0)
    if td[0]:                                                     # 글자층 치수 사슬(OCR 없음, 층당 수 ms) — 치수 판독(OCR ≈8 s)을 대신
        sd, cd, nd = td[0], max(td[1], 0.5 if td[2] >= 4 else 0.3), td[2]; R['dims_text'] = True
    elif sp and LAZY_DIMS and (stated or GROUP_DIMS):     # 용지까지 적힌 시트 축척이 있으면 치수 판독(층당 ≈8 s)은 미룸 — 다른 출처와 갈릴 때만 2단계 전에 읽음
        sd, cd, nd = None, 0.0, 0; R['dims_lazy'] = True  # 용지 없는 시트 축척은 process 에서 같은 축척 시트 묶음마다 한 층만 치수로 확인(GROUP_DIMS)
    elif DIMS_MISS_STOP and _DIMS_MISS.get(id(info), 0) >= DIMS_MISS_STOP:      # 같은 건물 앞 층 2개에서 치수를 못 읽음(창호 평면도처럼 치수가 없는 도면 세트) → 이 층도 건너뜀(층당 ≈7 s OCR)
        sd, cd, nd = None, 0.0, 0; R['dims_skipped'] = True
    else:
        sd, cd, nd = dims_scale(g)
        if not sd and id(info) in _DIMS_MISS: _DIMS_MISS[id(info)] += 1
    if sd: R['cands'].append(dict(s=sd, w=W['dims'] if cd >= 0.5 else W['dims_weak'], src='dims', n=nd, conf=round(cd, 2)))
    vl, hl = grid_lines(g)
    circ = find_circles(g); R['n_circles'] = len(circ)
    from fire3dbeam import realtag as RT
    R['tags'] = RT.read_tags(g, circ, text=TL) if TAG_V2 else read_tags(g, circ)
    if TAG_V2:
        tagc = [c for c in circ if any(math.hypot(c[0] - t['x'], c[1] - t['y']) < 2 and (t['W'] or t.get('code0') or len(t.get('raw') or '') >= 2) for t in R['tags'])]
    else:
        tagc = [c for c in circ if any(math.hypot(c[0] - t['x'], c[1] - t['y']) < 2 and (t['W'] or 'W' in t['code'] or 'D' in t['code']) for t in R['tags'])]
    rest = [c for c in circ if c not in tagc]                       # 창호 태그 원은 그리드 버블 후보에서 뺌
    bv, bh = grid_from_bubbles(g, rest)
    vl, hl = merge_lines(vl, bv), merge_lines(hl, bh); R['vl'], R['hl'] = vl, hl
    if BUBBLE_V2: R['bubbles'], R['bubble_dbg'] = RT.read_bubbles(g, rest, vl, hl, return_debug=True, text=TL)
    else: R['bubbles'] = read_bubbles(g, rest, vl, hl)
    if TL and TEXT_AXES:                                          # 원 없이 글자로만 적은 그리드 이름 줄('|1 |2 … |19') → 원 버블이 2개 미만인 축에만 보탬
        for k_, nm in RT.text_axes(TL).items():
            if sum(1 for kk in R['bubbles'] if kk[0] == k_[0]) < 2: R.setdefault('_tax', {})[k_] = nm
        if R.get('_tax'): R['bubbles'] = {**R['bubbles'], **R.pop('_tax')}
    if TL:
        from fire3dbeam import pdftext as PT
        rooms, words = RS.select_rooms(PT.words_of(TL))
    else:
        up = 2.0 if max(g.shape) <= 2600 else 1.0
        rooms, words = RS.detect_rooms(g, None, up=up)
    R['rooms_ocr'], R['words'] = rooms, words
    # 임시 축척 s0: 시트 > 치수 > 정보카드 면적(잉크 외곽) — 모델은 ±30% 축척 증강으로 학습되어 이 정도면 개구부·외곽이 나온다
    s0 = sp or sd
    if not s0 and S0_TREAD:                                   # 시트·치수 축척이 없으면 규칙 계단 디딤판 간격(÷0.28 m)을 임시 축척으로 — 모델을 엉뚱한 축척으로 돌렸다가 다시 돌리는 일 줄임
        # 디딤판 꼴: 7단 이상 + 디딤판 폭/깊이 비 3.5~12(폭 1~3 m ÷ 깊이 0.28 m) — 빗금·점 무늬(비 1.5~3)는 제외
        gp = sorted(c['gap'] for c in RS.detect_stairs(g, None, [], ppm=None) if c['n'] >= 7 and 3.5 <= c['tread_len'] / max(c['gap'], 1e-3) <= 12)
        gm, ng = ratio_cluster(gp, 0.08) if gp else (None, 0)
        if gm and ng >= 1:
            st_ = gm / TREAD_M
            if 15 <= st_ <= 400: s0 = st_; R['s0_src'] = 'tread_rule'
    if not s0 and R['card_area']:
        px = footprint_px_area(masks(img)); s0 = math.sqrt(px / R['card_area']) if px > 0 else None
    s0 = s0 or 40.0
    R['roi'] = text_roi(TL, g.shape, s0) if TL else None
    for it in range(2):                                       # 문 폭으로 임시 축척 한 번 보정(크게 틀린 면적 축척 대비)
        P = _model(img, s0, R['roi'])
        ops = _openings(P, s0)
        dw = [o['w'] for o in ops if o['kind'] == 'door']
        s_d = float(np.median(dw)) / DOOR_M if len(dw) >= 3 else None
        if it == 0 and not (sp or sd) and R.get('s0_src') != 'tread_rule' and s_d and abs(math.log(s_d / s0)) > math.log(1.35): s0 = s_d; continue
        break
    R['s0'], R['P0'] = s0, P
    if s_d: R['cands'].append(dict(s=s_d, w=W['doors'], src='doors', n=len(dw)))
    tr = tag_ratios([t for t in R['tags'] if t['W']], ops); R['tag_pairs'] = tr
    if TAG_PAIR2:
        st, n, k = tag_scale(tr)
        if st and n >= 1:
            R['cands'].append(dict(s=st, w=W['tags3'] if (n >= 3 and k >= 2) else W['tags2'] if (n >= 2 and k >= 2) or n >= 3 else W['tags1'], src='tags', n=n, of=len(tr), kinds=k))
    else:
        st, n = ratio_cluster([t['ratio'] for t in tr])
        if st and n >= 1:
            R['cands'].append(dict(s=st, w=W['tags3'] if n >= 3 else W['tags2'] if n == 2 else W['tags1'], src='tags', n=n, of=len(tr)))
    seg = _spa(g, s0); R['seg0'] = seg
    gaps = []
    if seg['stairs']:
        for c in RS.detect_stairs(g, None, words, ppm=None):
            b = c['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(q['box'][0] <= cx <= q['box'][2] and q['box'][1] <= cy <= q['box'][3] for q in seg['stairs']) and c['n'] >= 4: gaps.append(c['gap'])
    if gaps: R['cands'].append(dict(s=float(np.median(gaps)) / TREAD_M, w=W['tread'], src='tread', n=len(gaps)))
    from fire3dbeam.str_seg import WORK_PPM_1_100
    if R['card_area'] and not (AREA_SKIP and ((sp and stated) or R.get('dims_text'))):   # 면적 후보(가중 0.15)는 명시 축척(용지 적힌 시트·글자층 치수)이 있으면 결과를 못 바꿈 → 건너뜀(층당 ≈1 s)
        prot = [[v * WORK_PPM_1_100 / s0 for v in q['box']] for q in seg['stairs'] + seg['elevators']]
        Mm = mass(P['lab'], WORK_PPM_1_100, prot, walls=P['v']['walls'], peel=above_ground(rp['floors']))
        apx = float(Mm['mask'].sum()) * (s0 / WORK_PPM_1_100) ** 2           # 원본 px²
        if apx > 0: R['cands'].append(dict(s=math.sqrt(apx / R['card_area']), w=W['area'], src='area'))
    return R


def named_ratio(A, B, tol=0.02):
    """두 층 그리드 버블 이름(같은 축·같은 이름) 사이 거리 비 → (r, 짝 수) | (None, 0). 이름이 같은 두 버블 거리 = 같은 실거리"""
    rs = []
    for o in ('v', 'h'):
        a = {nm: c for (oo, c), nm in (A or {}).items() if oo == o}
        b = {nm: c for (oo, c), nm in (B or {}).items() if oo == o}
        com = sorted(set(a) & set(b))
        for i in range(len(com)):
            for j in range(i + 1, len(com)):
                da, db = a[com[j]] - a[com[i]], b[com[j]] - b[com[i]]
                if abs(db) > 20 and abs(da) > 20 and da / db > 0: rs.append(da / db)
    if len(rs) < 2: return None, 0
    r, n = ratio_cluster(rs, tol)
    return (r, n) if n >= 2 and n >= 0.6 * len(rs) else (None, 0)


def link_floors(Rs, tol=0.02, min_n=3):
    """같은 도면 세트(그리드 간격 무늬가 배율 r 로 겹침, 간격 3개 이상 일치)인 층끼리 축척 후보를 나눠 씀:
    층 R 의 후보에 다른 층 Q 의 후보(area 제외) × r(R/Q) 를 가중 0.8배로 더함(src 'grid:원출처') → 한 층만 태그를 잘못 짝지어도 건물 합의가 이김.
    2차(2026-10-03): 두 층에 같은 이름 그리드 버블이 2쌍 이상이면 버블 거리 비를 r 로(간격 무늬보다 우선), 버블 이름이 없으면 간격 무늬 —
    단 두 층 모두 명시 축척(시트·치수)이 있으면 그 비와 5% 안일 때만 잇는다(간격 무늬 오짝으로 엉뚱한 축척이 옮겨 가는 것 방지)."""
    sp = [grid_spacings(R['vl']) + grid_spacings(R['hl']) for R in Rs]
    own = [list(R['cands']) for R in Rs]
    def explicit(C):
        E = [c for c in C if c['src'] in ('sheet', 'dims') and c['w'] >= 0.55]
        return math.exp(sum(c['w'] * math.log(c['s']) for c in E) / sum(c['w'] for c in E)) if E else None
    for a, R in enumerate(Rs):
        R['linked'] = []
        for b, Q in enumerate(Rs):
            if a == b: continue
            r, n, how = None, 0, 'grid'
            if BUBBLE_V2:
                r, n = named_ratio(R.get('bubbles'), Q.get('bubbles')); how = 'bubble'
            if not r:
                how = 'grid'
                if len(sp[a]) < 2 or len(sp[b]) < 2: continue
                rs = [x / y for x in sp[a] for y in sp[b] if 0.4 < x / y < 2.5]
                r, n = ratio_cluster(rs, tol)
                if not r or n < min_n: continue
                ea, eb = explicit(own[a]), explicit(own[b])
                if BUBBLE_V2 and ea and eb and abs(math.log(r / (ea / eb))) > math.log(1.05): continue
            R['linked'].append(dict(floors=Q['floors'], r=round(r, 4), n=n, how=how))
            for c in own[b]:
                if c['src'] in ('area',): continue
                R['cands'].append(dict(s=c['s'] * r, w=round(c['w'] * 0.8, 3), src='grid:' + c['src'], n=c.get('n')))


def stage2(R, rp, info, s):
    """2단계: 최종 축척 s 로 벽·창·문·덩어리 외곽·계단·승강기·실명(평면도 지역좌표 m / 원본 px)"""
    from fire3dbeam import realsym as RS, geom
    from fire3dbeam.str_seg import WORK_PPM_1_100 as WPPM
    img, g = rp['img'], R['g']
    rerun = abs(math.log(s / R['s0'])) > math.log(1 + RERUN)
    P = _model(img, s, R.get('roi')) if rerun else R['P0']
    sm = s if rerun else R['s0']                       # 모델·SPA 를 돌린 축척(좌표 환산은 sm 기준 → 지역 m 는 s 로 다시)
    seg = _spa(g, s) if rerun else R['seg0']
    # 계단·승강기: SPA + 규칙(글자 확인) — realsym.fuse_model
    rule_st = RS.detect_stairs(g, None, R['words'], ppm=s)
    out = dict(rooms=list(R['rooms_ocr']), stairs=list(rule_st), elevators=RS.detect_elevators(g, None, R['words'], ppm=s))
    RS.fuse_model(out, seg)
    ev_words = [w for w in R['words'] if EV_STRONG.search((w.get('text') or '').replace(' ', '')) and not EV_NOT.search((w.get('text') or '').replace(' ', ''))]
    st_words = [w for w in R['words'] if STAIR_TXT.search((w.get('text') or '').replace(' ', ''))]
    def m2(b): return (b[2] - b[0]) / s, (b[3] - b[1]) / s
    def wc(w): return (w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2
    stairs = []
    for d in out['stairs']:
        a, c = m2(d['box'])
        if min(a, c) >= 0.9 and a * c >= 3.0 and max(a, c) <= 14:
            b = d['box']; m = 0.5 * s
            tr = max([q['n'] for q in rule_st if _ov([q['box'][0] - m, q['box'][1] - m, q['box'][2] + m, q['box'][3] + m], b)] or [0])
            lab = d.get('label') or any(q.get('label') for q in rule_st if _ov(q['box'], b)) or \
                any(b[0] - m <= wc(w)[0] <= b[2] + m and b[1] - m <= wc(w)[1] <= b[3] + m for w in st_words)
            ext_w = any(b[0] - m <= wc(w)[0] <= b[2] + m and b[1] - m <= wc(w)[1] <= b[3] + m and OUT_STAIR.search((w.get('text') or '').replace(' ', '')) for w in st_words)
            stairs.append(dict(box=[float(v) for v in b], score=d.get('score', 0.5), src=d.get('src'), treads=int(tr), label=bool(lab), out_word=bool(ext_w)))
    if FAINT_STAIR:                                      # UP·DN 글자 옆 흐린(회색·점선) 계단: 모델·규칙(진한 선)이 놓친 기존·옥외 계단
        for b, n_ in faint_stairs(g, st_words, s, [q['box'] for q in stairs]):
            ext_w = any(b[0] - 0.5 * s <= wc(w)[0] <= b[2] + 0.5 * s and b[1] - 0.5 * s <= wc(w)[1] <= b[3] + 0.5 * s and OUT_STAIR.search((w.get('text') or '').replace(' ', '')) for w in st_words)
            stairs.append(dict(box=[float(v) for v in b], score=0.5, src='faint', treads=int(n_), label=True, out_word=bool(ext_w)))
    if SPLIT_STAIRS: stairs = split_merged_stairs(stairs, st_words, s)
    elevs = []
    for d in merge_frags(out['elevators'], 1.0 * s, 4.5 * s):          # 승강기 안 글자('ELEV 15인승')에 갈라진 모델 조각 합치기
        a, c = m2(d['box'])
        b = d['box']
        word = any(b[0] - 5 <= wc(w)[0] <= b[2] + 5 and b[1] - 5 <= wc(w)[1] <= b[3] + 5 for w in ev_words)
        if (min(a, c) >= 1.2 and max(a, c) <= 4.5 and max(a, c) / max(min(a, c), 0.1) <= 2.2 and d.get('score', 0.6) >= 0.6) or (word and min(a, c) >= 0.9):
            if not any(_ovf(b, q['box']) > 0.3 for q in stairs): elevs.append(dict(box=[float(v) for v in b], score=d.get('score', 0.5), src=d.get('src'), word=word, cert='model'))
    kk = WPPM / sm
    # 승강기 글자('E/V'·'ELEV'·'13인승' …)만 있는 승강기(X 표시·모델 검출 없음): 글자를 둘러싼 1.0~3.5 m 닫힌 칸
    #   ① 원본 그림 잉크(가는 선 카 윤곽 포함, 글자 지움)로 닫힌 칸 → ② 모델 벽·창·문 장벽을 0 / 0.6 / 1.0 m 닫아 가며 닫힌 칸 → ③ 없으면 글자 중심 2.2 m 칸(확실성 'text')
    if ev_words:
        cells = ev_cells(g, P['lab'], kk, s, [w for w in ev_words if not any(_ov([w['box'][0] - 4, w['box'][1] - 4, w['box'][2] + 4, w['box'][3] + 4], q['box']) for q in elevs)], R['words'])
        for b, how in cells:
            if not any(_ovf(b, q['box']) > 0.3 for q in elevs + stairs): elevs.append(dict(box=[float(v) for v in b], score=0.6, src='word', word=True, cert='word' if how != 'nominal' else 'text'))
    # 건물 덩어리 외곽·외부 출입문(작업 px = 원본 px × WPPM / sm)
    prot = [[v * kk for v in q['box']] for q in stairs + elevs]
    drop, sinfo = wall_styles(g, P['v']['walls'], sm, P['v']['doors'] + P['v']['windows']) if WALL_STYLE else ([], {})
    Mm = mass_checked(P['lab'], WPPM, prot, walls=P['v']['walls'], peel=above_ground(rp['floors']), drop=drop, words=R.get('words') or (), src=(g, sm / WPPM), info=sinfo)
    if NAMED_ROOM and R.get('words'):
        Mm['mask'], nr_ = named_rooms(Mm['mask'], P['lab'], R['words'], kk, WPPM)
        if nr_: sinfo['named_rooms'] = [a[-1] for a in nr_]
    fac = sm / s                                         # 모델 지역 m(sm 기준) → 최종 축척 s 기준 지역 m
    def scl(w):
        w = dict(w); w['c'] *= fac; w['a'] *= fac; w['b'] *= fac; w['t'] *= fac; return w
    poly = mask_poly(Mm['mask'], WPPM)
    if poly is not None and fac != 1.0:
        from shapely import affinity
        poly = affinity.scale(poly, fac, fac, origin=(0, 0))
    walls = [scl(w) for w in P['v']['walls'] if (w['b'] - w['a']) * fac >= 0.3]
    if poly is not None: walls = geom.clip_walls(walls, poly, pad=0.8)
    # 창호 태그 기호가 문(ASD·SD·FD …)인 개구부를 모델이 창으로 본 경우(유리 자동문·접이문) → 문으로(가장 가까운 짝, 원 반지름 4배 안)
    if TAG_V2:                                           # 창호 기호가 창(AW·PW …)인 태그가 가장 가까운 '문' 개체 → 창(모델이 큰 창을 문으로 본 경우)
        wt = [t for t in R['tags'] if t.get('window')]
        for o in Mm['ops']:
            if o['kind'] != 'door' or not wt: continue
            x0, y0, x1, y1 = o['box']; cx, cy = (x0 + x1) / 2 / kk, (y0 + y1) / 2 / kk
            t = min(wt, key=lambda t: math.hypot(t['x'] - cx, t['y'] - cy))
            if math.hypot(t['x'] - cx, t['y'] - cy) <= 3.0 * t['r'] and min(R['tags'], key=lambda q: math.hypot(q['x'] - cx, q['y'] - cy)) is t:
                o['kind'] = 'window'; o['by_tag'] = t['code']
    dt = [t for t in R['tags'] if t.get('door')]
    for o in Mm['ops']:
        if o['kind'] != 'window' or not dt: continue
        x0, y0, x1, y1 = o['box']; cx, cy = (x0 + x1) / 2 / kk, (y0 + y1) / 2 / kk
        t = min(dt, key=lambda t: math.hypot(t['x'] - cx, t['y'] - cy))
        if math.hypot(t['x'] - cx, t['y'] - cy) <= 4.0 * t['r']:
            near_any = min(R['tags'], key=lambda q: math.hypot(q['x'] - cx, q['y'] - cy))
            if near_any is t: o['kind'] = 'door'; o['by_tag'] = t['code']
    if EXT_BY_MASK:                                      # 외부 문 = 문 양쪽(벽 수직 ±0.7 m) 중 한쪽은 건물 덩어리 밖·한쪽은 안 (빈 공간 성분은 틈으로 새어 '바깥'이 될 수 있어 최종 외곽으로 판정)
        Mk = Mm['mask']; Hm, Wm = Mk.shape; off = 0.7 * WPPM; LABm = P['lab']
        def side_ext(o):
            x0, y0, x1, y1 = o['box']; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            horiz = (x1 - x0) >= (y1 - y0)
            def inside(x, y):
                xi, yi = int(round(x)), int(round(y))
                return bool(0 <= xi < Wm and 0 <= yi < Hm and Mk[yi, xi])
            def split(hz, fs=(1.0, 1.8), clear=False):
                for f_ in fs:                       # 외곽이 벽 바깥으로 조금 나간 경우(처마·캐노피 선) 1.26 m 까지
                    d_ = off * f_
                    a_ = inside(cx, cy - d_) if hz else inside(cx - d_, cy)
                    b_ = inside(cx, cy + d_) if hz else inside(cx + d_, cy)
                    if a_ != b_:
                        if not clear: return True
                        sg = -1 if not a_ else 1          # 밖 쪽으로 가는 길에 모델 '벽'이 없어야(문이 그 벽의 개구부) — 벽 가까운 실내 문 제외
                        n_ = int(d_) + 1
                        xs_ = np.clip(np.round(cx + (0 if hz else sg) * np.linspace(0, d_, n_)).astype(int), 0, Wm - 1)
                        ys_ = np.clip(np.round(cy + (sg if hz else 0) * np.linspace(0, d_, n_)).astype(int), 0, Hm - 1)
                        return int(np.isin(LABm[ys_, xs_], (1, 2)).sum()) < 2     # 벽·창을 지나면 실내 문
                    if not a_ and not b_: return False
                return False
            if split(horiz): return True
            # 정사각에 가까운 문 개체(문짝 + 열림 호를 함께 잡음 — 밖여닫이 문은 외곽이 열림 호까지 불룩): 상자 꼴로 벽 방향을 못 정함 →
            #   다른 축에서 안/밖이 갈리고(밖으로 가는 길에 벽 없음), 상자 축은 양쪽이 같은 상태일 때 외부 문
            if max(x1 - x0, y1 - y0) <= ARC_SQ * max(1, min(x1 - x0, y1 - y0)) and split(not horiz, clear=True):
                d_ = off
                a_ = inside(cx, cy - d_) if horiz else inside(cx - d_, cy)
                b_ = inside(cx, cy + d_) if horiz else inside(cx + d_, cy)
                return a_ == b_
            return False
        Mm['op_ext'] = [side_ext(o) for o in Mm['ops']]
    if ARC_DOOR:                                         # 바깥과 맞닿은 '창'인데 원본에 문 열림 호(1/4 원)가 그려져 있으면 문(모델이 유리문을 창으로 본 경우)
        for o, ext in zip(Mm['ops'], Mm['op_ext']):
            if o['kind'] != 'window' or not ext: continue
            b = [v / kk for v in o['box']]
            if swing_arc(g, b, s): o['kind'] = 'door'; o['by_arc'] = True
    doors = []
    for o, ext in zip(Mm['ops'], Mm['op_ext']):
        if o['kind'] != 'door': continue
        x0, y0, x1, y1 = o['box']
        ys, xs = np.nonzero(o['mask'])
        if len(xs) < 6: continue
        (cx, cy), (ww, hh), ang = cv2.minAreaRect(np.c_[xs + x0, ys + y0].astype(np.float32))
        if ww < hh: ww, hh, ang = hh, ww, ang + 90
        wm = ww / WPPM * fac
        if wm < 0.5 or wm > 9.0: continue
        a = ang % 180
        ori = 'h' if (a < 45 or a > 135) else 'v'
        x, y = cx / WPPM * fac, -cy / WPPM * fac
        if poly is not None:
            from shapely.geometry import Point
            if not poly.buffer(0.8).contains(Point(x, y)): continue
        why = (['tag:' + o['by_tag']] if o.get('by_tag') else []) + (['arc'] if o.get('by_arc') else []) + ([] if (o.get('by_tag') or o.get('by_arc')) else ['model'])
        doors.append(dict(o=ori, x=round(x, 2), y=round(y, 2), w=round(wm, 2), kind='door', ext=bool(ext and wm >= 0.75), src='model', why=why,
                          **({'tag': o['by_tag']} if o.get('by_tag') else {})))
    if TAG_V2 and poly is not None:                      # 문 기호 태그(FD·ASD …)인데 그 자리에 모델 문·창 개체가 없음(접이문·셔터를 벽으로 봄) → 외곽선 위 가장 가까운 점에 태그 폭의 외부 문
        from shapely.geometry import Point
        bnd = poly.boundary
        for tg in R['tags']:
            if not tg.get('door'): continue
            if not tg.get('W') or tg['W'] < 1.5: continue         # 폭을 읽은 넓은 문(접이문·셔터 — 모델이 벽으로 보는 것)만. 폭 모르는 보통 문 태그는 실내 방화문일 때가 많음
            tx_, ty_ = tg['x'] / s, -tg['y'] / s
            P_ = Point(tx_, ty_)
            if bnd.distance(P_) > max(1.5, 3.0 * tg['r'] / s): continue
            near_op = any(math.hypot(((o['box'][0] + o['box'][2]) / 2 / kk) - tg['x'], ((o['box'][1] + o['box'][3]) / 2 / kk) - tg['y']) <= 3.0 * tg['r'] for o in Mm['ops'])
            if near_op: continue
            q = bnd.interpolate(bnd.project(P_))
            wd = tg['W'] or 1.0
            if any(math.hypot(d['x'] - q.x, d['y'] - q.y) <= max(1.0, wd / 2) for d in doors): continue
            a_ = bnd.interpolate(max(0.0, bnd.project(P_) - 0.3)); b_ = bnd.interpolate(bnd.project(P_) + 0.3)
            ori = 'h' if abs(b_.x - a_.x) >= abs(b_.y - a_.y) else 'v'
            doors.append(dict(o=ori, x=round(q.x, 2), y=round(q.y, 2), w=round(wd, 2), kind='door', ext=True, src='tag', why=['tag-only:' + str(tg['code'])], tag=tg['code']))
    if EXT_BY_MASK:                                      # 같은 벽 선 위 이웃 외부 문 조각(유리 자동문·접이문이 둘로 갈린 것, 틈 < 0.6 m) 합치기
        E = sorted([d for d in doors if d['ext']], key=lambda d: (d['o'], d['y'] if d['o'] == 'h' else d['x'], d['x'] if d['o'] == 'h' else d['y']))
        drop_ = set()
        for i, a in enumerate(E):
            for b in E[i + 1:]:
                if id(b) in drop_ or id(a) in drop_ or a['o'] != b['o']: continue
                ca, cb = (a['y'], b['y']) if a['o'] == 'h' else (a['x'], b['x'])
                ua, ub = (a['x'], b['x']) if a['o'] == 'h' else (a['y'], b['y'])
                if abs(ca - cb) <= 0.6 and abs(ua - ub) - (a['w'] + b['w']) / 2 <= 0.6:
                    lo = min(ua - a['w'] / 2, ub - b['w'] / 2); hi = max(ua + a['w'] / 2, ub + b['w'] / 2)
                    keep, gone = (a, b) if a['w'] >= b['w'] else (b, a)
                    keep['w'] = round(hi - lo, 2)
                    if keep['o'] == 'h': keep['x'] = round((lo + hi) / 2, 2)
                    else: keep['y'] = round((lo + hi) / 2, 2)
                    keep['why'] = sorted(set(keep.get('why', []) + gone.get('why', [])))
                    drop_.add(id(gone))
        doors = [d for d in doors if id(d) not in drop_]
    wins = []
    for o in Mm['ops']:
        if o['kind'] != 'window': continue
        x0, y0, x1, y1 = o['box']
        wins.append(dict(x=round((x0 + x1) / 2 / WPPM * fac, 2), y=round(-(y0 + y1) / 2 / WPPM * fac, 2), w=round(max(x1 - x0, y1 - y0) / WPPM * fac, 2)))
    # 실명: OCR(비주거 어휘 포함) — 모델(SPA) 주거 부류 이름은 주거 용도일 때만
    resi = is_residential(info, rp['floors'])
    labels = []
    for r in out['rooms']:
        if r.get('src') == 'model':
            nm = r.get('name')
            if not resi:
                nm = {'화장실': '화장실', '엘리베이터홀': '홀', '계단실': None}.get(nm)
            if not nm or nm in ('기타', '계단실', '엘리베이터'): continue
            labels.append(dict(name=nm, px=r['xy'][0], py=r['xy'][1], src='model'))
        else:
            labels.append(dict(name=r.get('name') or r.get('text'), text=r.get('text'), px=r['xy'][0], py=r['xy'][1], src='ocr', score=r.get('score')))
    for q in stairs:                                     # 옥외 계단: 계단 상자 중심이 건물 덩어리 밖(0.3 m 넘게) 또는 '옥외계단' 글자
        cx, cy = (q['box'][0] + q['box'][2]) / 2 / s, -(q['box'][1] + q['box'][3]) / 2 / s
        from shapely.geometry import Point as _Pt
        q['outdoor'] = bool(q.get('out_word')) or (poly is not None and not poly.buffer(0.3).contains(_Pt(cx, cy)))
    return dict(walls=walls, doors=doors, windows=wins, fp=poly, stairs=stairs, elevs=elevs, labels=labels,
                peeled=Mm['peeled'], model_s=round(sm, 3), rerun=bool(rerun), n_walls_model=len(P['v']['walls']), wall_style=sinfo)


SPLIT_STAIRS = os.environ.get('FIRE3DBEAM_SPLIT_STAIRS', '1') != '0'     # 붙은 계단 둘이 한 상자로 잡힌 것 나누기(2026-10-03 2차)


def split_merged_stairs(stairs, words, s, min_sep_m=2.5):
    """계단 상자 하나 안에 같은 글자(UP 또는 DN)가 2.5 m 넘게 떨어져 둘 → 붙은 계단 두 개로 보고 두 글자 가운데에서 긴 축으로 나눔(나눈 쪽이 1.2 m 이상일 때)"""
    out = []
    for d in stairs:
        b = d['box']; m = 0.3 * s; done = False
        for key in ('UP', 'DN'):
            P = [((w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2) for w in words
                 if (w.get('text') or '').replace(' ', '').replace('.', '').upper() == key]
            P = [p for p in P if b[0] - m <= p[0] <= b[2] + m and b[1] - m <= p[1] <= b[3] + m]
            if len(P) < 2: continue
            a_, c_ = max(((p, q) for i, p in enumerate(P) for q in P[i + 1:]), key=lambda pq: math.dist(*pq))
            if math.dist(a_, c_) < min_sep_m * s: continue
            if abs(a_[0] - c_[0]) >= abs(a_[1] - c_[1]):
                cut = (a_[0] + c_[0]) / 2; b1, b2 = [b[0], b[1], cut, b[3]], [cut, b[1], b[2], b[3]]
            else:
                cut = (a_[1] + c_[1]) / 2; b1, b2 = [b[0], b[1], b[2], cut], [b[0], cut, b[2], b[3]]
            if min(min(q[2] - q[0], q[3] - q[1]) for q in (b1, b2)) < 1.2 * s: continue
            out += [dict(d, box=[float(v) for v in b1], split=key), dict(d, box=[float(v) for v in b2], split=key)]; done = True; break
        if not done: out.append(d)
    return out


def ev_cells(g, lab, kk, s, words, all_words=()):
    """승강기 글자마다 → [(상자 원본 px, 방법)] (방법 'ink'|'model'|'nominal')"""
    out = []
    H, Wd = g.shape
    ink = (g < 170).astype(np.uint8)
    for w in all_words or ():                              # 글자 자체는 지움(글자 획이 칸을 막지 않게)
        x0, y0, x1, y1 = [int(round(v)) for v in w['box']]
        ink[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = 0
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    nI, ccI, stI, _ = cv2.connectedComponentsWithStats(1 - ink, 4)
    bar0 = (lab > 0).astype(np.uint8)
    used = []
    for w in words:
        cx, cy = (w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2
        if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b, _ in out): continue
        got = None
        ix, iy = int(cx), int(cy)
        if 0 <= iy < H and 0 <= ix < Wd and ccI[iy, ix]:
            j = ccI[iy, ix]; x, y, bw, bh, ar = stI[j]
            a_, c_ = bw / s, bh / s
            if 1.0 <= min(a_, c_) and max(a_, c_) <= 3.5 and max(a_, c_) / min(a_, c_) <= 2.0 and ar / s / s <= 10: got = ([x, y, x + bw, y + bh], 'ink')
        if got is None:
            for km in (0.0, 0.6, 1.0):
                k = max(3, int(round(km * WORK_PPM())) | 1)
                bar = cv2.morphologyEx(bar0, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
                n_, cc_, st_, _ = cv2.connectedComponentsWithStats(1 - bar, 4)
                jx, jy = int(cx * kk), int(cy * kk)
                if not (0 <= jy < cc_.shape[0] and 0 <= jx < cc_.shape[1]) or cc_[jy, jx] == 0: continue
                j = cc_[jy, jx]; x, y, bw, bh, ar = st_[j]
                a_, c_ = bw / WORK_PPM(), bh / WORK_PPM()
                if 1.0 <= min(a_, c_) and max(a_, c_) <= 3.5 and ar / WORK_PPM() ** 2 <= 10:
                    got = ([x / kk, y / kk, (x + bw) / kk, (y + bh) / kk], 'model'); break
        if got is None and re.search(r'(E/V|E\.V|ELEV|\d{1,2}인승)', (w.get('text') or '').replace(' ', ''), re.I):
            r = 1.1 * s; got = ([cx - r, cy - r, cx + r, cy + r], 'nominal')
        if got: out.append(got)
    return out


ARC_DOOR = os.environ.get('FIRE3DBEAM_ARC_DOOR', '1') != '0'
EXT_BY_MASK = os.environ.get('FIRE3DBEAM_EXT_BY_MASK', '1') != '0'
ARC_SQ = float(os.environ.get('FIRE3DBEAM_EXT_SQ', '1.35'))           # 문 개체 상자 긴 변/짧은 변 ≤ 이 값이면 벽 방향을 상자 꼴로 안 정함(두 축 검사, 2026-10-03 보류 N 진단 후)


def swing_arc(g, box, s):
    """개구부 상자(원본 px) → 문 열림 호(경첩에서 문 폭 반지름 1/4 원, 외여닫이·양여닫이)가 그려져 있나"""
    H, Wd = g.shape
    x0, y0, x1, y1 = box
    horiz = (x1 - x0) >= (y1 - y0)
    L = (x1 - x0) if horiz else (y1 - y0)
    if L < 0.6 * s or L > 4.0 * s: return False
    c = (y0 + y1) / 2 if horiz else (x0 + x1) / 2
    ink = g < 170
    def cov(hx, hy, R, a0):
        th = np.radians(np.linspace(a0 + 8, a0 + 82, 24))
        xs = np.round(hx + R * np.cos(th)).astype(int); ys = np.round(hy + R * np.sin(th)).astype(int)
        ok = (xs >= 1) & (xs < Wd - 1) & (ys >= 1) & (ys < H - 1)
        if ok.sum() < 18: return 0.0
        hit = [ink[y - 1:y + 2, x - 1:x + 2].any() for x, y in zip(xs[ok], ys[ok])]
        return float(np.mean(hit))
    # 경첩 = 개구부 양 끝, 호는 벽 양쪽 중 한쪽으로
    if horiz: hinges = [((x0, c), 0, 1), ((x1, c), 180, -1)]       # (점, 개구부 안쪽 방향 각, ...)
    else: hinges = [((c, y0), 90, 1), ((c, y1), 270, -1)]
    best = 0.0
    for (hx, hy), a_in, _ in hinges:
        for side in (-1, 1):                                       # 벽 위·아래(왼·오른) 쪽
            a0 = a_in if side == 1 else a_in - 90
            best = max(best, cov(hx, hy, L, a0 % 360))             # 외여닫이(문 폭 = 개구부)
    dbl = []
    for side in (-1, 1):
        v = [cov(hx, hy, L / 2, (a_in if side == 1 else a_in - 90) % 360) for (hx, hy), a_in, _ in hinges]
        dbl.append(min(v))                                         # 양여닫이: 두 경첩 모두
    return best >= 0.7 or max(dbl) >= 0.6


def ev_at(img, box, s, pad_m=0.3):
    """원본 도면 상자(px) 안에 승강기 표기(글자 'ELEV·E/V·승강기·N인승' 또는 승강로 X 대각선)가 있나 — 이웃 층 승강기 샤프트 잇기용(작은 영역만 OCR)"""
    from fire3dbeam import realsym as RS
    g = _gray(img); H, Wd = g.shape
    p = pad_m * s
    x0, y0, x1, y1 = int(max(0, box[0] - p)), int(max(0, box[1] - p)), int(min(Wd, box[2] + p)), int(min(H, box[3] + p))
    if x1 - x0 < 0.8 * s or y1 - y0 < 0.8 * s: return False
    if RS.detect_elevators(g, [x0, y0, x1, y1], [], ppm=s): return True
    sub = g[y0:y1, x0:x1]
    import subprocess, tempfile
    for Lm, dil in ((1.0, 1), (0.6, 0)):                          # 긴 직선(그리드선·벽·카 윤곽)을 지우고 글자만 검정으로 → 4배 → OCR
        ink = (sub < 215).astype(np.uint8); L = max(9, int(Lm * s))
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) | \
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        if dil: lines = cv2.dilate(lines, np.ones((1 + 2 * dil, 1 + 2 * dil), np.uint8))
        up = max(1.0, min(4.0, 60.0 / max(8.0, 0.25 * s)))
        u = cv2.resize(np.where((ink & (1 - lines)) > 0, 0, 255).astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
        u = np.where(u < 200, 0, 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
            cv2.imwrite(tf.name, u); fn = tf.name
        try:
            txt = subprocess.run(['tesseract', fn, '-', '-l', 'kor+eng', '--psm', '11'], capture_output=True, text=True, timeout=60).stdout
        except Exception:
            txt = ''
        finally:
            os.unlink(fn)
        if EV_FUZZY.search(txt.replace(' ', '')): return True
    return False


def stair_at(img, box, s, pad_m=0.4):
    """원본 도면의 상자(px) 안에 계단 디딤판(등간격 평행선 4개 이상, 간격 0.2~0.4 m, 길이 0.6~3.2 m — 점선·흐린 선 포함)이 있나"""
    from fire3dbeam import realsym as RS
    g = _gray(img); H, Wd = g.shape
    p = pad_m * s
    x0, y0, x1, y1 = int(max(0, box[0] - p)), int(max(0, box[1] - p)), int(min(Wd, box[2] + p)), int(min(H, box[3] + p))
    if x1 - x0 < 0.8 * s or y1 - y0 < 0.8 * s: return False
    sub = g[y0:y1, x0:x1]
    ink = (sub < 215).astype(np.uint8)
    k = max(3, int(0.25 * s))
    for o in ('h', 'v'):
        lm = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if o == 'h' else (1, k)))   # 점선 잇기
        segs = RS._segments(lm, o, max(8, int(0.5 * s)))
        for a0, a1, c0, c1, n, gap in RS._runs(segs, 4):
            if n >= 4 and 0.2 <= gap / s <= 0.42 and 0.6 <= (a1 - a0) / s <= 3.4: return True
    if FAINT_STAIR:                                       # 회색(≈230) 디딤판·화살표에 가린 줄 빠짐 허용(faint_stairs 와 같은 방식)
        ink = (sub < 242).astype(np.uint8); Ll = int(3.6 * s)
        ink = ink & (1 - (cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (Ll, 1))) |
                          cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, Ll)))))
        k = max(3, int(0.1 * s))
        for o in ('h', 'v'):
            lm = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if o == 'h' else (1, k)))
            lm = cv2.dilate(lm, np.ones((3, 1) if o == 'h' else (1, 3), np.uint8))
            for a0, a1, c0, c1, n, gap in _faint_runs(RS._segments(lm, o, max(8, int(0.5 * s))), s, min_n=4 if FAINT_MIN_M < 0.8 else 5, skip3=FAINT_MIN_M < 0.8):
                if FAINT_MIN_M <= (a1 - a0) / s <= 3.4: return True             # (이웃 층에 확정 계단이 있는 자리만 보므로 화살표·끊김선에 가린 두 줄 빠짐까지 허용)
    return False


FAINT_MIN_M = float(os.environ.get('FIRE3DBEAM_FAINT_STAIR_MIN', '0.55'))   # 이웃 층 같은 자리 흐린 디딤판 최소 길이(m) — 좁은 옥외 철골 계단(한 흐름 0.6~0.8 m) 포함(2026-10-03 2차, 예전 0.8)


FAINT_STAIR = os.environ.get('FIRE3DBEAM_FAINT_STAIR', '1') != '0'
UPDN = re.compile(r'^(UP|DN|U\.P|D\.N)$', re.I)


def _updn_ocr(g, b, s, pad_m=1.0):
    """상자 둘레(pad_m) 흐린 글자까지 OCR(긴 선 지움, 4배) → UP·DN 이 있나"""
    import subprocess, tempfile
    H, Wd = g.shape; p = pad_m * s
    x0, y0, x1, y1 = int(max(0, b[0] - p)), int(max(0, b[1] - p)), int(min(Wd, b[2] + p)), int(min(H, b[3] + p))
    sub = g[y0:y1, x0:x1]
    ink = (sub < 235).astype(np.uint8); L = max(9, int(0.6 * s))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    up = max(1.0, min(4.0, 60.0 / max(8.0, 0.25 * s)))
    u = cv2.resize(np.where((ink & (1 - lines)) > 0, 0, 255).astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    u = np.where(u < 200, 0, 255).astype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        cv2.imwrite(tf.name, u); fn = tf.name
    try:
        txt = subprocess.run(['tesseract', fn, '-', '--psm', '11', '-c', 'tessedit_char_whitelist=UPDN'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        txt = ''
    finally:
        os.unlink(fn)
    return bool(re.search(r'(^|\s)(UP|DN)(\s|$)', txt))


def _faint_runs(segs, s, min_n=4, skip3=False):
    """흐린 디딤판 묶음: 비슷한 구간(a0,a1 ±25%)의 평행선이 간격 g(0.2~0.42 m)로, 화살표·글자에 가린 한 줄 빠짐(2g)까지 허용해 이어지는 것 → [(a0,a1,c0,c1,n,g)]"""
    segs = sorted(segs, key=lambda t: t[2]); out = []; used = set()
    for i, (p0, p1, pc) in enumerate(segs):
        if i in used: continue
        L = p1 - p0; tol = 0.25 * L + 3
        sim = [j for j, t in enumerate(segs) if abs(t[0] - p0) <= tol and abs(t[1] - p1) <= tol and t[2] >= pc - 0.5]
        cs, js = [], []
        for j in sim:
            if not cs or segs[j][2] - cs[-1] > 3: cs.append(segs[j][2]); js.append(j)
        best = None
        gs = [cs[k] - cs[0] for k in range(1, len(cs))]
        if skip3: gs += [b_ - a_ for a_, b_ in zip(cs, cs[1:])]           # 첫 줄 다음이 가려진 경우: 이웃 간격도 간격 후보로
        for g in gs:
            if not (0.2 * s <= g <= 0.42 * s): continue
            ch, last = [0], cs[0]
            for m in range(1, len(cs)):
                d = cs[m] - last
                if 0.75 * g <= d <= 1.25 * g or 1.75 * g <= d <= 2.25 * g or (skip3 and 2.75 * g <= d <= 3.25 * g): ch.append(m); last = cs[m]
                elif d > (3.25 if skip3 else 2.25) * g: break
            if best is None or len(ch) > len(best[0]): best = (ch, g)
        if best and len(best[0]) >= min_n and (cs[best[0][-1]] - cs[0]) >= 3.5 * best[1]:
            idx = [js[m] for m in best[0]]; used.update(idx)
            out.append((min(segs[j][0] for j in idx), max(segs[j][1] for j in idx), cs[0], cs[best[0][-1]], len(idx), float(best[1])))
    return out


def faint_stairs(g, words, s, have, max_ocr=6):
    """모델·진한 선 규칙이 놓친 흐린(회색·점선) 계단 — 기존·옥외 계단을 배경처럼 옅게 그린 도면.
    흐린 선까지(잉크 < 215, 점선 잇기) 등간격 평행선(디딤판 5개+, 간격 0.2~0.42 m, 길이 0.8~3.4 m, 직교 등간격 선이 겹치면 타일·격자라 제외) 묶음 중
    이미 찾은 계단 밖의 것 → 1 m 안에 계단 글자(UP·DN·계단)가 있거나, 둘레를 흐린 글자까지 다시 OCR 해 UP·DN 이 읽히면 계단. → [(상자 원본 px, 디딤판 수)]"""
    from fire3dbeam import realsym as RS
    ink = (g < 242).astype(np.uint8)                                   # 회색(≈230) 선까지
    Ll = int(3.6 * s)                                                  # 디딤판보다 긴 선(그리드선·벽·대지선) 지움 — 디딤판끼리 이어져 한 덩어리가 되는 것 막기
    ink = ink & (1 - (cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (Ll, 1))) |
                      cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, Ll)))))
    k = max(3, int(0.1 * s)); R_ = {}                                  # 짧은 끊김만 잇기(위·아래 계단 흐름의 디딤판이 이어지지 않게)
    for o in ('h', 'v'):
        lm = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if o == 'h' else (1, k)))
        lm = cv2.dilate(lm, np.ones((3, 1) if o == 'h' else (1, 3), np.uint8))   # 2 px 겹선(디딤판 코) 한 줄로
        R_[o] = []
        for a0, a1, c0, c1, n, gap in _faint_runs(RS._segments(lm, o, max(8, int(0.5 * s))), s):
            if 0.8 <= (a1 - a0) / s <= 3.4 and (c1 - c0) / s <= 6.0:
                R_[o].append(([a0, c0, a1, c1] if o == 'h' else [c0, a0, c1, a1], n))
    def ovl(a, b, m=0): return a[0] - m <= b[2] and b[0] - m <= a[2] and a[1] - m <= b[3] and b[1] - m <= a[3]
    cand = []
    for o, oth in (('h', 'v'), ('v', 'h')):
        for b, n in R_[o]:
            if any(ovl(b, q, 0.5 * s) for q in have): continue
            if any(ovl(b, q) for q, _ in R_[oth]): continue          # 직교 등간격 선과 겹침 = 타일·격자·창살
            cand.append((b, n))
    wc = [((w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2) for w in words or ()]
    out = []
    for b, n in sorted(cand, key=lambda t: -t[1]):
        if any(ovl(b, q, 1.0 * s) for q, _ in out):                    # 같은 계단의 다른 흐름 — 합침
            for e in out:
                if ovl(b, e[0], 1.0 * s): e[0] = [min(e[0][0], b[0]), min(e[0][1], b[1]), max(e[0][2], b[2]), max(e[0][3], b[3])]; e[1] = max(e[1], n); break
            continue
        ok = any(b[0] - s <= x <= b[2] + s and b[1] - s <= y <= b[3] + s for x, y in wc)
        if not ok and max_ocr > 0:
            max_ocr -= 1; ok = _updn_ocr(g, b, s)
        if ok: out.append([list(b), n])
    return [(b, n) for b, n in out if (b[2] - b[0]) * (b[3] - b[1]) / s / s >= 1.2]


def WORK_PPM():
    from fire3dbeam.str_seg import WORK_PPM_1_100
    return WORK_PPM_1_100


def merge_frags(dets, gap_px, max_px):
    """가까운(틈 ≤ gap_px) 검출 상자 조각 합치기(합친 상자 긴 변 ≤ max_px) — 점수는 최댓값"""
    D = [dict(d, box=[float(v) for v in d['box']]) for d in dets]
    merged = True
    while merged:
        merged = False
        for i in range(len(D)):
            for j in range(i + 1, len(D)):
                a, b = D[i]['box'], D[j]['box']
                gx = max(a[0], b[0]) - min(a[2], b[2]); gy = max(a[1], b[1]) - min(a[3], b[3])
                u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                if max(gx, gy) <= gap_px and max(u[2] - u[0], u[3] - u[1]) <= max_px:
                    D[i] = dict(D[i], box=u, score=max(D[i].get('score', 0), D[j].get('score', 0))); D.pop(j); merged = True; break
            if merged: break
    return D


def above_ground(floors):
    """지하층 도면은 마당형 벗기기 안 함(옹벽·저수조 등 창 없는 외벽 실이 정상) — 대지선·조경은 지상층 도면에 그려짐"""
    from fire3dbeam.card import level_of
    return bool(floors) and min(level_of(f) for f in floors) >= 0


def _ovf(a, b):
    """상자 a 중 b 와 겹치는 비율"""
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, iw) * max(0.0, ih) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def _ov(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


EV_STRONG = re.compile(r'(E/V|E\.V|ELEV|엘리베|승강기|\d{1,2}인승)', re.I)     # 승강기 글자(장애인·비상용 단독은 화장실·계단 이름에도 쓰여 제외)
EV_FUZZY = re.compile(r'(E\s*/\s*V|[ECF]L[IL]?[EF]V|엘리베|승강기|\d?인승)', re.I)     # 작은 영역 OCR 용(ELEV→ELIEV·CLEV 오독 허용)
EV_NOT = re.compile(r'(홀|HALL|전실|로비|LOBBY|기계실|피트|PIT)', re.I)              # '승강기홀'·'E/V 전실'·'승강기 기계실' 은 승강기 칸이 아님
STAIR_TXT = re.compile(r'(UP|DN|D\.N|U\.P|계단|STAIR|올라|내려)', re.I)
OUT_STAIR = re.compile(r'(옥외|외부|OUTDOOR|EXT)', re.I)
RESI_WORDS = ('공동주택', '아파트', '연립', '다세대', '다가구', '단독주택', '주택', '기숙사', '오피스텔')


def is_residential(info, floors=()):
    """이 평면도 층의 용도(정보카드 ⑩)가 주거면 True — 층 용도가 없으면 주용도(①)로"""
    fu = ' '.join(((info.get('floors') or {}).get(f) or {}).get('use') or '' for f in floors)
    u = fu if fu.strip() else (info.get('use') or info.get('주용도') or '')
    return any(w in u for w in RESI_WORDS)


def _pmap(fn, items):
    """층별 작업을 스레드 WORKERS 개로(순서 유지). tesseract 는 별도 프로세스, onnx·OpenCV 는 GIL 을 놓아 2코어에서 겹쳐 돈다."""
    if WORKERS <= 1 or len(items) <= 1: return [fn(*a) for a in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda a: fn(*a), items))


def process(raw_plans, info, log=None):
    """pipeline.run 에서 부름: 실제 도면 평면도들 [(i, rp)] → {i: 결과}. 층마다 1단계 → 층간 그리드 축척 → 합의 → 2단계"""
    import time
    cache = os.environ.get('FIRE3DBEAM_REAL_S1CACHE')                # 개발용: 1단계 결과 캐시 폴더(영상·캡션·층 지문) — 2단계만 고칠 때 반복 시간 단축
    def s1(i, rp):
        key = None
        if cache:
            import hashlib, pickle
            h = hashlib.sha1(_gray(rp['img']).tobytes()); h.update(repr((rp.get('caption'), rp.get('floors'))).encode())
            key = os.path.join(cache, h.hexdigest()[:20] + '.pkl')
            if os.path.exists(key):
                R = pickle.load(open(key, 'rb')); R['i'] = i; return R
        t0 = time.time(); R = stage1(rp, info); R['i'] = i; R['t1'] = time.time() - t0
        if key:
            os.makedirs(cache, exist_ok=True); pickle.dump(R, open(key, 'wb'))
        return R
    _DIMS_MISS[id(info)] = 0
    try:
        Rs = _pmap(s1, list(raw_plans))
    finally:
        _DIMS_MISS.pop(id(info), None)
    for R in Rs:                                                 # 글자층 치수가 용지 없는 시트 축척과 2% 안으로 맞으면 시트 축척을 확인된 것으로
        if R.get('dims_text') and R.get('sheet_sp') and not R.get('sheet_stated'):
            dd = [c for c in R['cands'] if c['src'] == 'dims']
            if dd and abs(math.log(dd[0]['s'] / R['sheet_sp'])) < math.log(1.02):
                for c in R['cands']:
                    if c['src'] == 'sheet': c['w'] = W['sheet']; c['by'] = 'dims_text'
    if GROUP_DIMS and LAZY_DIMS:                                 # 용지 없는 시트 축척(A1 원도 축척을 A3 에 그대로 적었을 수 있음): 같은 축척 시트 묶음에서 한 층씩 치수를 읽어
        groups = {}                                              #   시트 축척과 2% 안으로 맞으면 그 묶음의 나머지 층은 시트 축척을 확인된 것(용지 명시와 같은 무게)으로 — 치수 판독 층당 ≈8 s 절약
        for R in Rs:
            if R.get('dims_lazy') and not R.get('sheet_stated'): groups.setdefault(round(R['sheet_sp'], 2), []).append(R)
        for sp_, G in groups.items():
            rest = list(G)
            while rest:
                R0 = rest.pop(0)
                sd, cd, nd = dims_scale(R0['g']); R0['dims_lazy'] = 'read'
                if sd: R0['cands'].append(dict(s=sd, w=W['dims'] if cd >= 0.5 else W['dims_weak'], src='dims', n=nd, conf=round(cd, 2)))
                if sd and cd >= 0.5 and abs(math.log(sd / sp_)) < math.log(1.02):
                    for R in rest:
                        for c in R['cands']:
                            if c['src'] == 'sheet': c['w'] = W['sheet']; c['by'] = 'dims@' + '/'.join(R0['floors'])
                    break
    if TAG_V2:                                                   # 창호 기호: 건물 모든 층 태그를 모아 어휘 맞춤·본보기 투표
        from fire3dbeam import realtag as RT
        sheet = set()
        for R in Rs: sheet |= RT.sheet_codes_from_words(R.get('words'))
        RT.resolve_codes([t for R in Rs for t in R['tags']], sheet)
    link_floors(Rs)
    def s2(i, rp, R):
        t0 = time.time()
        if R.get('dims_lazy'):                                   # 미룬 치수 판독: 이 층(또는 옮겨 온) 다른 출처가 시트 축척과 10% 넘게 갈리면 읽어서 판정에 넣음
            sh = [c for c in R['cands'] if c['src'] == 'sheet']
            oth = [c for c in R['cands'] if c['src'] in ('tags', 'tread', 'grid:tags', 'grid:dims') and c['w'] >= 0.3]
            if sh and any(abs(math.log(c['s'] / sh[0]['s'])) > math.log(1 + DISAGREE) for c in oth):
                sd, cd, nd = dims_scale(R['g'])
                if sd: R['cands'].append(dict(s=sd, w=W['dims'] if cd >= 0.5 else W['dims_weak'], src='dims', n=nd, conf=round(cd, 2)))
                R['dims_lazy'] = 'read'
        s, src, conf, dis = fuse(R['cands'])
        if not s: s, src, conf = R['s0'], 'default', 0.0
        own_strong = [c for c in R['cands'] if not c['src'].startswith('grid:') and c['w'] >= 0.5]
        expl = any(c['src'] in ('sheet', 'dims') and c['w'] >= 0.8 and abs(math.log(c['s'] / s)) < math.log(1 + AGREE) for c in R['cands'])
        outl = []
        def rejected(c):                                         # 외톨이 거르기: 명시 출처(시트·치수)가 아닌 이 층 후보 하나가 혼자 갈리고, 최종값을 서로 다른 출처 2종 이상이
            if not OUTLIER_REJECT or c['src'] in ('sheet', 'dims'): return False     #   그 후보 가중의 2배 넘게 받치면 그 후보를 버린 것으로 보고 검수 플래그를 띄우지 않음(scale_outliers 에 남김)
            sup = [d for d in R['cands'] if d is not c and abs(math.log(d['s'] / s)) < math.log(1 + AGREE)]
            kinds = {d['src'].split(':')[-1] for d in sup}
            ok = sum(d['w'] for d in sup) >= 2 * c['w'] and len(kinds) >= 2
            if ok: outl.append(dict(src=c['src'], s=round(c['s'], 2), n=c.get('n')))
            return ok
        dis = any(abs(math.log(c['s'] / s)) > math.log(1 + DISAGREE) and not rejected(c) for c in own_strong
                  if not expl or c['src'] in ('sheet', 'dims'))     # 이 층 자체의 강한 출처가 최종값과 갈림 → 검수(명시 출처로 정해졌으면 명시 출처끼리 갈릴 때만)
        res = stage2(R, rp, info, s)
        if outl: res['scale_outliers'] = outl
        res.update(s=float(s), scale_src='real:' + src, scale_conf=conf, scale_disagree=dis, vl=R['vl'], hl=R['hl'], bubbles=R['bubbles'],
                   scale_cands=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in R['cands']],
                   tags=[dict(W=t['W'], text=t['text'], code=t.get('code'), **({'code_src': t.get('code_src')} if t.get('code_src') else {})) for t in R['tags']],
                   n_tag_pairs=len(R['tag_pairs']))
        res['t_s'] = round(R['t1'] + time.time() - t0, 1)
        return i, res
    return dict(_pmap(s2, [(i, rp, R) for (i, rp), R in zip(raw_plans, Rs)]))
