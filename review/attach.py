# -*- coding: utf-8 -*-
"""정보카드 PDF + 별도 평면도·진압작전도 파일(PDF/PNG/JPG/TIF) → 한 PDF(입력 합본).
정보카드에 평면도가 없거나 일부 층만 있을 때, 따로 받은 도면 파일을 '평면도(지상N층)' 캡션 쪽으로 카드 뒤에 붙인다
(엔진·카드 해석기는 그대로 — 캡션으로 층을 읽는 기존 경로를 탄다).

층 = 파일 이름(예: 지상1층.png, 1F.pdf, B1.jpg, 지하2층.tif, 2F~9F 기준층.png, 옥탑.png).
여러 쪽 PDF 는 쪽마다 글자('… 평면도', 'N층')에서 층을 찾는다. 이름에 '작전도'·'sitemap' 이 있으면 진압작전도.
정보카드에 이미 있는 층·진압작전도는 건너뛴다(카드 원본 우선). 합본은 결과 폴더에만 저장(반출 금지)."""
import os, re, unicodedata

IMG_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
SKIP_NAMES = ('대장', 'daejang', '일람도', 'ilramdo', 'site', '위치도', '사진', 'photo')
PLAN_DPI = 200                      # 벡터 도면 PDF 를 그림으로 바꿀 해상도


def _nfc(s): return unicodedata.normalize('NFC', s or '')


def is_card_pdf(path):
    """정보카드(표준양식 또는 기존 근사 배치)인가: 양식 칸 이름이 글자층에 있음"""
    import pymupdf
    from fire3dbeam.card import is_std
    try:
        doc = pymupdf.open(path)
        t = '\n'.join(doc[k].get_text() for k in range(min(3, doc.page_count)))
    except Exception:
        return False
    ns = re.sub(r'\s+', '', t)
    if is_std(t) or ('정보카드' in ns and ('층별' in ns or '소방시설' in ns)) or ('관리번호' in ns and '주용도' in ns): return True
    # 스캔본(글자층 없는 그림 쪽): 첫 쪽을 가볍게 OCR — 제목 '소방활동 정보카드'·'건물개황' 또는 양식 칸 이름(fire3dbeam/card_ocr.py)
    try:
        from fire3dbeam import card_ocr
        if os.environ.get('FIRE3DBEAM_OCR', '1') != '0' and card_ocr.is_image_page(doc[0]): return bool(card_ocr.quick_is_card(doc[0]))
    except Exception:
        return False
    return False


def classify(path):
    """'card' | 'plan' | 'sitemap' | None"""
    b = _nfc(os.path.basename(path)).lower()
    ext = os.path.splitext(b)[1]
    if ext not in IMG_EXT + ('.pdf',): return None
    if ext == '.pdf' and is_card_pdf(path): return 'card'
    if any(k in b for k in SKIP_NAMES): return None          # 건축물대장·경계구역 일람도·부지 자료 등(카드 옆 보조 입력 — 파이프라인이 이름으로 따로 찾음)
    if '작전도' in b or 'sitemap' in b: return 'sitemap'
    return 'plan'


TOK = re.compile(r'(지하\s*(\d+)\s*층?)|((?<![A-Za-z])[Bb]\s*(\d+)\s*(?:F|층)?(?!\d))|((?:지상\s*)?(\d+)\s*(?:층|F|f)(?![A-Za-z]))|(옥탑|PH|RF|R층)')


def floors_from_text(t):
    """글자 → 층 id 목록(['1F'], ['B1'], ['2F',…,'9F'], ['RF']). 못 찾으면 []"""
    from fire3dbeam.card import expand_floors
    t = _nfc(t)
    toks = []
    for m in TOK.finditer(t):
        if m.group(1): toks.append((m.start(), m.end(), f'지하{int(m.group(2))}'))
        elif m.group(3): toks.append((m.start(), m.end(), f'지하{int(m.group(4))}'))
        elif m.group(5): toks.append((m.start(), m.end(), f'지상{int(m.group(6))}'))
        elif m.group(7): toks.append((m.start(), m.end(), '옥탑1'))
    if not toks: return []
    if len(toks) == 2 and re.search(r'[~〜∼\-–]|to', t[toks[0][1]:toks[1][0]]):
        return expand_floors(f'{toks[0][2]}~{toks[1][2]}')
    out = []
    for _, _, k in toks:
        for f in expand_floors(k):
            if f not in out: out.append(f)
    return out


def ko_caption(fids):
    """층 id → 캡션 글자('지상1층', '지상2층~지상9층', '지하1층', '옥탑1층')"""
    from fire3dbeam.card import level_of
    def one(f):
        if f.startswith('B'): return f'지하{f[1:]}층'
        if f.endswith('F') and f[:-1].isdigit(): return f'지상{f[:-1]}층'
        return '옥탑1층'
    fids = sorted(fids, key=level_of)
    if len(fids) > 2 and all(f.endswith('F') and f[:-1].isdigit() for f in fids):
        nums = [int(f[:-1]) for f in fids]
        if nums == list(range(nums[0], nums[-1] + 1)): return f'{one(fids[0])}~{one(fids[-1])}'
    return ','.join(one(f) for f in fids)


def _page_image(page):
    """PDF 한 쪽 → BGR 영상: 쪽 대부분을 덮는 그림 한 장(스캔본)이면 원본 화소 그대로, 아니면 PLAN_DPI 로 그리기"""
    import numpy as np, pymupdf
    infos = [i for i in page.get_image_info(xrefs=True) if i.get('xref')]
    pa = page.rect.width * page.rect.height
    big = [i for i in infos if (i['bbox'][2] - i['bbox'][0]) * (i['bbox'][3] - i['bbox'][1]) > 0.6 * pa]
    if len(big) == 1 and len(infos) == 1:
        pix = pymupdf.Pixmap(page.parent, big[0]['xref'])
    else:
        pix = page.get_pixmap(dpi=PLAN_DPI)
    if pix.n - pix.alpha >= 4 or pix.n - pix.alpha == 1: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
    return arr[..., ::-1].copy()


def crop_sheet(img):
    """CAD 도면 시트(도곽 테두리 + 표제란)면 가장 큰 도면 영역만 잘라냄 → (영상, 상자|None).
    표제란 로고(색)·도곽선이 소방 기호·외벽으로 잡히고 외곽 면적 축척이 시트 전체로 커지는 것을 막는다(2026-10-03 공개 도면 시험).
    시트 판정 = 가장자리 8% 띠 안 긴 도곽선이 3면 이상(fire3dbeam/realplan.frame_lines). 영역 = realplan.sheet_layout 가장 큰 영역(시트의 4~85%).
    끄기: FIRE3DBEAM_SHEET_CROP=0"""
    import cv2, numpy as np
    if os.environ.get('FIRE3DBEAM_SHEET_CROP', '1') == '0': return img, None
    from fire3dbeam import sheetcrop as SCR
    if SCR.ENABLED:                                     # 2차(2026-10-03): 도곽·표제란 띠·범례 표 찾아 빼기 — fire3dbeam/sheetcrop.py
        A = SCR.analyze(img)
        if A is None: return img, None
        im, bx, _ = SCR.crop(img, A)
        return im, bx
    from fire3dbeam import realplan as RP
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    f = max(1, int(round(max(H, W) / 1240)))           # sheet_layout 조정 해상도(A3 시트 ≈ 75 dpi, 긴 변 ≈ 1240 px)
    q = RP.minpool(g, f) if f > 1 else g
    Hq, Wq = q.shape
    fr = RP.frame_lines((q < 190).astype(np.uint8))
    bx, by = max(2, int(0.08 * Wq)), max(2, int(0.08 * Hq))
    sides = [fr[:, :bx].sum() > 0.5 * Hq, fr[:, -bx:].sum() > 0.5 * Hq, fr[:by].sum() > 0.5 * Wq, fr[-by:].sum() > 0.5 * Wq]
    if sum(sides) < 3: return img, None
    regs, _, _ = RP.sheet_layout(q)
    if not regs: return img, None
    b = max(regs, key=lambda r: (r['box'][2] - r['box'][0]) * (r['box'][3] - r['box'][1]))['box']
    frac = (b[2] - b[0]) * (b[3] - b[1]) / float(Wq * Hq)
    if not 0.04 < frac < 0.85: return img, None
    pad = int(0.015 * Wq)
    x0, y0 = max(0, (b[0] - pad) * f), max(0, (b[1] - pad) * f)
    x1, y1 = min(W, (b[2] + pad) * f), min(H, (b[3] + pad) * f)
    return img[y0:y1, x0:x1].copy(), [x0, y0, x1, y1]


def _read_image(path):
    import cv2, numpy as np
    buf = np.fromfile(path, dtype=np.uint8)            # 한글 경로 대비(cv2.imread 는 Windows 한글 경로를 못 엶)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise ValueError('그림을 읽을 수 없음')
    return img


def _sheet_scale(pg, img):
    """벡터 PDF 쪽 글자층의 도면 축척(표제란 'SCALE 1/150(A3)' 등) → 캡션 꼬리 '[도면축척 … ppm=…]' | None (fire3dbeam/realcard.py 가 축척 후보로 씀).
    그림 쪽(스캔본·글자층 없음)은 None. 끄기 FIRE3DBEAM_SHEET_SCALE=0"""
    if os.environ.get('FIRE3DBEAM_SHEET_SCALE', '1') == '0': return None
    try:
        from fire3dbeam import realcard as RC
        if len(pg.get_text('words')) < 5: return None
        r = RC.sheet_scale_pdf_page(pg, img.shape[1])
        return r[1] if r else None
    except Exception:
        return None


def _page_text(pg, img):
    """벡터 쪽 글자층 → 그림(img) 좌표 글자 줄 | None (fire3dbeam/pdftext.py — realcard 가 OCR 대신 씀)"""
    try:
        from fire3dbeam import pdftext as PT
        if not PT.ENABLED or len(pg.get_text('words')) < PT.MIN_LINES: return None
        return PT.page_lines(pg, img.shape[1] / pg.rect.width, None, (img.shape[1], img.shape[0]))
    except Exception:
        return None


def caption_lines(pg):
    """벡터 PDF 쪽 글자층의 '…평면도' 줄 → [(층 목록, 줄 상자 [x0,y0,x1,y1] 회전 반영 pt)]"""
    import pymupdf
    M = pg.rotation_matrix
    lines = {}
    for w in pg.get_text('words'):
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in lines.values():
        txt = ''.join(w[4] for w in sorted(ws, key=lambda w: w[0]))
        if '평면도' not in txt: continue
        if any(k in txt for k in ('천정', '천장', '지붕', '구조', '창호일람', '확대', '부분', '상세')): continue      # 천정·지붕·구조·부분 확대 평면도는 층 평면이 아님
        fl = floors_from_text(txt)
        if not fl: continue
        pts = [pymupdf.Point(x, y) * M for w in ws for x, y in ((w[0], w[1]), (w[2], w[3]))]
        out.append((fl, [min(p.x for p in pts), min(p.y for p in pts), max(p.x for p in pts), max(p.y for p in pts)]))
    return out


def _split_page(pg, log, name):
    """한 쪽에 서로 다른 층 평면도 캡션이 2개 이상 → [(kind, floors, img, 설명, 축척 꼬리)] | None"""
    if os.environ.get('FIRE3DBEAM_SHEET_SPLIT', '1') == '0': return None
    caps = caption_lines(pg)
    if len({tuple(f) for f, _ in caps}) < 2: return None
    try:
        from fire3dbeam import sheetcrop as SCR
        im = _page_image(pg)
        z = im.shape[1] / pg.rect.width
        parts = SCR.split_regions(im, [(f, [v * z for v in b]) for f, b in caps])
    except Exception as e:
        log(f'  ! {name} p{pg.number + 1}: 평면도 나누기 실패({e!r})'); return None
    if not parts: return None
    sc = _sheet_scale(pg, im)
    log(f'  · {name} p{pg.number + 1}: 한 쪽에 평면도 {len(parts)}개 → 층별로 나눔(' + ', '.join(ko_caption(f) for f, _, _ in parts) + ')')
    from fire3dbeam import pdftext as PT
    TL = _page_text(pg, im)
    return [('plan', f, img, f'{name} p{pg.number + 1} 영역{k + 1} {bx}', sc, PT.crop_lines(TL, bx)) for k, (f, img, bx) in enumerate(parts)]


def load_extra(path, kind, log=print):
    """→ [(kind, floors, img, 설명, 축척 캡션 꼬리|None)]"""
    import pymupdf
    name = _nfc(os.path.basename(path))
    fl_name = floors_from_text(os.path.splitext(name)[0]) if kind == 'plan' else []
    out = []
    if path.lower().endswith('.pdf'):
        doc = pymupdf.open(path)
        for pg in doc:
            fl = fl_name
            if kind == 'plan' and not fl:
                lines = [ln for ln in pg.get_text().splitlines() if '평면도' in ln]      # 도면 제목 줄만(다른 문서의 '1층' 글자 오인 방지)
                for ln in lines:
                    fl = floors_from_text(ln)
                    if fl: break
            if kind == 'plan' and not fl_name:                 # 한 쪽에 평면도 여럿(캡션 '1층 평면도'·'2층 평면도') → 층별로 나눔(fire3dbeam/sheetcrop.split_regions)
                parts = _split_page(pg, log, name)
                if parts:
                    out += parts; continue
            if kind == 'plan' and fl_name and doc.page_count > 1:
                im = _page_image(pg)
                log(f'  ! {name}: 여러 쪽인데 파일 이름의 층({ko_caption(fl_name)})만 알 수 있어 첫 쪽만 사용')
                out.append((kind, fl, im, f'{name} p1', _sheet_scale(pg, im) if kind == 'plan' else None, _page_text(pg, im) if kind == 'plan' else None)); break
            im = _page_image(pg)
            out.append((kind, fl, im, f'{name} p{pg.number + 1}', _sheet_scale(pg, im) if kind == 'plan' else None, _page_text(pg, im) if kind == 'plan' else None))
            if kind == 'sitemap': break
    else:
        out.append((kind, fl_name, _read_image(path), name, None, None))
    return out


def combine(card_pdf, extras, out_pdf, log=print):
    """extras: 파일 경로 목록. 붙인 것이 없으면 None(원본 그대로 쓰면 됨), 있으면 out_pdf 경로.
    반환 (경로|None, 붙인 항목 설명 목록)"""
    import cv2, pymupdf
    from fire3dbeam.card import parse_card, extract_plans, extract_figures
    info, doc = parse_card(card_pdf)
    have = {f for p in extract_plans(doc) for f in p.get('floors') or []}
    has_sitemap = False
    try:
        has_sitemap = any(f['kind'] == '진압작전도' for f in extract_figures(doc)) if info.get('양식') == 'std' else False
    except Exception:
        pass
    if info.get('양식') == 'std' and len(info.get('동목록') or []) > 1:
        log('  ! 여러 동 정보카드: 붙이는 도면은 마지막 동의 ⑪ 에 들어갑니다(동 선택 FIRE3DBEAM_DONG 확인)')
    add, notes, n = [], [], len(extract_plans(doc))
    texts = {}                                           # 캡션 → 그 평면도 그림 좌표 글자 줄(벡터 도면 글자층, fire3dbeam/pdftext.py)
    for p in extras:
        kind = classify(p)
        if kind not in ('plan', 'sitemap'): log(f'  건너뜀(도면 파일 아님): {p}'); continue
        try:
            items = load_extra(p, kind, log)
        except Exception as e:
            log(f'  ✗ 읽기 실패 {os.path.basename(p)}: {e}'); continue
        for k, fl, img, desc, sc, TL in items:
            if k == 'sitemap':
                if has_sitemap: log(f'  건너뜀: {desc} — 정보카드에 진압작전도가 이미 있음'); continue
                has_sitemap = True; add.append(('0 진압작전도', img)); notes.append(f'진압작전도 ← {desc}'); continue
            if not fl: log(f'  ✗ 층을 알 수 없음: {desc} — 파일 이름에 층을 넣으세요(예: 지상1층.png, B1.pdf)'); continue
            dup = [f for f in fl if f in have]
            if dup: log(f'  건너뜀: {desc} — 정보카드에 이미 있는 층({ko_caption(dup)})'); continue
            full = img
            try:
                img, bx = crop_sheet(img)
            except Exception as e:
                bx = None; log(f'  ! {desc}: 도면 시트 영역 찾기 실패({e!r}) → 쪽 전체 사용')
            if bx: desc += f' (도곽·표제란 제외 영역 {bx[2] - bx[0]}×{bx[3] - bx[1]}px)'
            from fire3dbeam import pdftext as PT
            TL = PT.crop_lines(TL, bx)
            if sc is None and bx and os.environ.get('FIRE3DBEAM_SHEET_SCALE', '1') != '0':      # 그림 도면 시트(글자층 없음): 표제란 OCR — 축척에 용지(A1·A3)가 적혀 있을 때만
                try:
                    from fire3dbeam import realcard as RC
                    r = RC.sheet_scale_ocr(full); sc = r[1] if r else None
                except Exception:
                    sc = None
            have |= set(fl); n += 1
            cap = f'{n}. 평면도({ko_caption(fl)})' + (f' {sc}' if sc else '')      # 축척 꼬리 '[도면축척 … ppm=…]' → fire3dbeam/realcard.py 축척 후보(sheet)
            add.append((cap, img)); notes.append(f'{cap} ← {desc}' + (f' · 글자층 {len(TL)}줄' if TL else ''))
            if TL: texts[cap] = dict(w=int(img.shape[1]), h=int(img.shape[0]), lines=TL)
    if not add: return None, notes
    for cap, img in add:
        h, w = img.shape[:2]
        pw, margin = 595.0, 40.0
        iw = pw - 2 * margin; ih = iw * h / w
        pg = doc.new_page(width=pw, height=max(842.0, ih + 120))
        pg.insert_text((margin, 56), cap, fontname='korea', fontsize=11)
        ok, png = cv2.imencode('.png', img)                                     # 무손실(도면 선·글자 보존)
        pg.insert_image(pymupdf.Rect(margin, 70, margin + iw, 70 + ih), stream=png.tobytes())
    if texts:
        from fire3dbeam import pdftext as PT
        PT.embed(doc, texts)
    doc.save(out_pdf, garbage=3, deflate=True)
    for s in notes: log('  + ' + s)
    return out_pdf, notes
