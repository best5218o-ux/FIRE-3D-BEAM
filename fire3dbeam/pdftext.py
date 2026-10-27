# -*- coding: utf-8 -*-
"""벡터 도면 PDF 의 글자층 → 평면도 그림 좌표 글자 줄 (2026-10-03 실제 도면 강건화 2차, '글자층 먼저').

CAD 에서 내보낸 벡터 PDF(나라장터 첨부 도면 등)는 실명·창호 기호(AW·ASD …)·그리드 이름(X1·A)·'13인승 E/V'·UP/DN 이 글자로 들어 있다.
review/attach.py 가 도면 쪽을 그림으로 그려 정보카드에 붙일 때, 그 쪽의 글자 줄을 잘라낸 그림 좌표(px)로 옮겨
합본 PDF 에 내장 파일(fire3dbeam_text.json, 캡션별)로 함께 넣는다 → pipeline 이 평면도 rp['text'] 로 넘기고
fire3dbeam/realcard.py 는 OCR(실명·창호 태그·그리드 버블) 대신 이 글자를 쓴다(빠르고 정확). 글자층 없는 그림 도면은 예전처럼 OCR.
끄기: FIRE3DBEAM_PDF_TEXT=0
"""
import json, os, re, unicodedata

ENABLED = os.environ.get('FIRE3DBEAM_PDF_TEXT', '1') != '0'
EMB_NAME = 'fire3dbeam_text.json'
MIN_LINES = 12                      # 이보다 적으면 글자층 없음으로 봄(글자를 선으로 바꾼 도면)


def capkey(cap):
    """캡션 → 비교 키(공백·NFC 차이 무시)"""
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', cap or ''))


def page_lines(pg, zoom, crop=None, img_wh=None):
    """PDF 쪽 → 글자 줄 [{t, box:[x0,y0,x1,y1] (그림 px), dir:[dx,dy] (그림 좌표, 글자 진행 방향), h(글자 크기 px)}].
    zoom = 그림 px / pt (회전 반영 쪽 좌표 기준), crop = 그림에서 잘라낸 상자 [x0,y0,x1,y1](px) — 상자 밖 줄은 버림."""
    import pymupdf
    M = pg.rotation_matrix
    D = pg.get_text('dict')
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    W = (crop[2] - crop[0]) if crop else (img_wh[0] if img_wh else None)
    H = (crop[3] - crop[1]) if crop else (img_wh[1] if img_wh else None)
    out, seen = [], set()
    for b in D.get('blocks', []):
        for ln in b.get('lines', []):
            t = ''.join(s.get('text', '') for s in ln.get('spans', []))
            t = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', t)).strip()
            if not t: continue
            x0, y0, x1, y1 = ln['bbox']
            P = [pymupdf.Point(x, y) * M for x, y in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
            bx = [min(p.x for p in P) * zoom - ox, min(p.y for p in P) * zoom - oy, max(p.x for p in P) * zoom - ox, max(p.y for p in P) * zoom - oy]
            if W is not None and (bx[2] <= 0 or bx[3] <= 0 or bx[0] >= W or bx[1] >= H): continue
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if W is not None and not (0 <= cx < W and 0 <= cy < H): continue
            dx, dy = ln.get('dir', (1.0, 0.0))
            d = pymupdf.Point(dx, dy) * M - pymupdf.Point(0, 0) * M
            sz = max((s.get('size', 0) for s in ln.get('spans', [])), default=0) * zoom
            key = (t, round(bx[0]), round(bx[1]))
            if key in seen: continue                          # CAD 가 같은 글자를 두 번 쓴 경우
            seen.add(key)
            out.append(dict(t=t, box=[round(v, 1) for v in bx], dir=[round(d.x, 3), round(d.y, 3)], h=round(sz, 1)))
    return merge_chars(out)


def crop_lines(lines, box):
    """그림을 box=[x0,y0,x1,y1] 로 잘랐을 때 글자 줄 좌표 옮기기(가운데가 상자 밖이면 버림)"""
    if lines is None: return None
    if not box: return lines
    x0, y0, x1, y1 = box
    out = []
    for l in lines:
        b = l['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x0 <= cx < x1 and y0 <= cy < y1:
            out.append(dict(l, box=[round(b[0] - x0, 1), round(b[1] - y0, 1), round(b[2] - x0, 1), round(b[3] - y0, 1)]))
    return out


def merge_chars(lines):
    """한 글자씩 따로 쓴 CAD 글자(‘축’ ‘척’)를 같은 방향·같은 크기·바로 옆이면 한 줄로 잇기"""
    if len(lines) < 2: return lines
    L = sorted(lines, key=lambda l: (round(l['dir'][1]), l['box'][1] if abs(l['dir'][0]) >= abs(l['dir'][1]) else l['box'][0]))
    used = [False] * len(L); out = []
    for i, a in enumerate(L):
        if used[i]: continue
        cur = dict(a); used[i] = True
        hor = abs(a['dir'][0]) >= abs(a['dir'][1])
        if len(a['t']) <= 2:
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(L):
                    if used[j] or len(b['t']) > 2 or b['dir'] != a['dir'] or abs(b['h'] - a['h']) > 0.15 * max(a['h'], 1): continue
                    A, B = cur['box'], b['box']; h = max(a['h'], 1)
                    if hor:
                        same = abs((A[1] + A[3]) - (B[1] + B[3])) / 2 < 0.3 * h
                        gap = B[0] - A[2] if a['dir'][0] > 0 else A[0] - B[2]
                    else:
                        same = abs((A[0] + A[2]) - (B[0] + B[2])) / 2 < 0.3 * h
                        gap = B[1] - A[3] if a['dir'][1] > 0 else A[1] - B[3]
                    if same and -0.2 * h <= gap <= 0.6 * h and all('가' <= ch <= '힣' for ch in cur['t'] + b['t']):
                        cur['t'] += b['t']; cur['box'] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                        used[j] = True; changed = True
        out.append(cur)
    return out


def embed(doc, mapping):
    """합본 PDF 에 {캡션 키: {w,h,lines}} 내장"""
    if not mapping: return
    data = json.dumps({capkey(k): v for k, v in mapping.items()}, ensure_ascii=False).encode('utf-8')
    try:
        if EMB_NAME in (doc.embfile_names() or []): doc.embfile_del(EMB_NAME)
    except Exception:
        pass
    doc.embfile_add(EMB_NAME, data, filename=EMB_NAME, desc='FIRE-3D BEAM 도면 글자층(평면도 그림 좌표)')


def load(doc):
    """합본 PDF 의 내장 글자 → {캡션 키: {w,h,lines}} (없으면 {})"""
    if not ENABLED: return {}
    try:
        if EMB_NAME not in (doc.embfile_names() or []): return {}
        return json.loads(doc.embfile_get(EMB_NAME).decode('utf-8'))
    except Exception:
        return {}


def for_plan(T, caption, img_shape):
    """캡션에 맞는 글자 줄(그림 크기가 다르면 비례 환산) | None"""
    v = T.get(capkey(caption)) if T else None
    if not v or len(v.get('lines') or []) < MIN_LINES: return None
    H, W = img_shape[:2]
    kx, ky = W / max(1, v['w']), H / max(1, v['h'])
    if abs(kx - 1) < 1e-3 and abs(ky - 1) < 1e-3: return v['lines']
    return [dict(l, box=[l['box'][0] * kx, l['box'][1] * ky, l['box'][2] * kx, l['box'][3] * ky], h=l['h'] * ky) for l in v['lines']]


def words_of(lines):
    """글자 줄 → realsym OCR 후보 꼴 [{text, box, xy, how, conf}] (how='h' — 두 줄 결합 대상)"""
    out = []
    for l in lines or ():
        b = l['box']
        out.append(dict(text=l['t'], box=list(b), xy=[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], how='h', conf=95.0, src='pdf'))
    return out
