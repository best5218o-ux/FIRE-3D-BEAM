# -*- coding: utf-8 -*-
"""소방활동 정보카드 PDF 해석 — 층별 현황 표, 계단·승강기 수, 도면 이미지(캡션 매칭)"""
import re
import numpy as np
import pymupdf
from fire3dbeam.fac_catalog import parse_items, merge_items

FLOOR_TOKEN = re.compile(r'^(지하\d+|지상\d+|옥탑\d*|PIT)(층)?$')


def norm_floor(tok):
    tok = tok.replace('층', '').strip()
    m = re.match(r'지하(\d+)', tok)
    if m: return f'B{int(m.group(1))}'
    m = re.match(r'지상(\d+)', tok)
    if m: return f'{int(m.group(1))}F'
    if tok.startswith('옥탑'): return 'RF'
    if tok.upper().startswith('PIT'): return 'PIT'
    return None


def level_of(fid):
    if fid == 'PIT': return -99
    if fid == 'RF': return 999
    if fid.startswith('B'): return -int(fid[1:])
    return int(fid[:-1]) - 1


def _join_cell(lines):
    """셀 안에서 줄바꿈으로 끊긴 토큰을 다시 붙인다 (예: '수동식소화기:4' + '6' → ':46')"""
    out = ''
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        if not out: out = ln; continue
        a, b = out[-1], ln[0]
        glue = (a.isdigit() and b.isdigit()) or a in ':(' or b in ':)(' or \
               ('가' <= a <= '힣' and '가' <= b <= '힣')
        out += ('' if glue else ' ') + ln
    return out


def parse_fac(txt):
    fac = {}
    for m in re.finditer(r'([가-힣A-Za-z()·]+?)\s*:\s*([\d,]+)', txt):
        k = m.group(1).strip(); v = int(m.group(2).replace(',', ''))
        fac[k] = fac.get(k, 0) + v
    return fac


def parse_card(pdf_path, dong=None):
    """정보카드 PDF → (info, doc). 양식 자동 판별: 표준양식(templates/소방활동정보카드_표준양식.docx) → parse_std, 아니면 기존(근사 배치) 해석.
    dong: 여러 동 카드에서 기존 키(use·floors·stairs…)를 채울 동 번호(기본 0 = 첫 동, 환경변수 FIRE3DBEAM_DONG)"""
    doc = pymupdf.open(pdf_path)
    ocr = ocr_of(doc)                                # 글자층 없는 쪽(스캔본·그림으로 내보낸 카드)이 있으면 OCR(fire3dbeam/card_ocr.py)
    full = doc_text(doc, ocr)
    if is_std(full):
        info = parse_std(doc, full, dong)
        if ocr: ocr_meta(info, ocr)
        return info, doc
    info = {'양식': 'legacy'}
    if ocr: info.update(source='ocr', ocr=dict(pages=ocr['n_image_pages'], seconds=ocr['seconds'], error='표준양식 칸 이름을 찾지 못함(스캔본 기존 배치 카드는 OCR 미지원)'))
    m = re.search(r'주\s*용\s*도\s*([^\s③]+)', full); info['use'] = m.group(1) if m else None
    m = re.search(r'상호\s*([^\n]+)', full); info['name'] = m.group(1).strip() if m else None
    def num(pat):
        m = re.search(pat + r'\s*:\s*(\d*)', full); return int(m.group(1)) if m and m.group(1) else 0
    info['stairs'] = dict(special=num('특별피난계단수'), evac=num(r'(?<!특별)피난계단수'), normal=num('일반계단수'), outdoor=num('옥외계단수'))
    info['elev'] = dict(emergency=num(r'승강기\(비상\)'), normal=num(r'승강기\(일반\)'))
    m = re.search(r'연면적\s*:\s*([\d,\.]+)', full); info['gfa'] = float(m.group(1).replace(',', '')) if m else None
    # ⑥ 소방용수시설: '소화전(지상식)-가상-1 : 0.150Km' (기능 24 부지 교차검증용, 추가 항목)
    info['water'] = [dict(kind=m.group(1), id=m.group(2), km=float(m.group(3))) for m in
                     re.finditer(r'(소화전\([^)]*\)|소화전|급수탑|저수조|비상소화장치)\s*-\s*([^\s:]+)\s*:\s*([\d.]+)\s*[Kk]m', full)]

    # 층별 현황 표: 단어 좌표로 열을 나눈다. 표가 여러 쪽에 걸치면(소방시설 항목이 많아 행이 길어짐) 쪽을 세로로 이어 붙여 한 표로 본다
    floors = {}
    words, yoff = [], 0.0
    for page in doc:
        for w in page.get_text('words'):          # x0,y0,x1,y1,word,...
            m = re.match(r'^([\d,]+\.\d+㎡)(.+)$', w[4])   # 칸 경계 없이 붙은 '면적㎡소방시설' 분리
            if m:
                k = len(m.group(1)) / len(w[4]); xm = w[0] + (w[2] - w[0]) * k
                words.append((w[0], w[1] + yoff, xm, w[3] + yoff, m.group(1))); words.append((xm + 0.5, w[1] + yoff, w[2], w[3] + yoff, m.group(2)))
            else:
                words.append((w[0], w[1] + yoff, w[2], w[3] + yoff, w[4]))
        yoff += page.rect.height + 20
    hdr = {}
    for w in words:
        if w[4] in ('층별', '용도', '면적') and w[4] not in hdr: hdr[w[4]] = w
    fac_hdr = [w for w in words if w[4].startswith('소방시설')]
    if '층별' in hdr and '면적' in hdr and fac_hdr:
        hy = hdr['층별'][3]
        x_floor_end = hdr['층별'][2] + 8
        rows = sorted([w for w in words if w[1] > hy and w[0] < x_floor_end and FLOOR_TOKEN.match(w[4])], key=lambda w: w[1])
        for i, r in enumerate(rows):
            y0 = r[1] - 2
            if i + 1 < len(rows): y1 = rows[i + 1][1] - 2
            else:                                   # 마지막 행: 다음 왼쪽 열 글자(다음 절 제목 등)까지 — 소방시설 줄이 길어도 자르지 않음
                nxt = [w[1] for w in words if w[1] > r[1] + 3 and w[0] < x_floor_end]
                y1 = min(nxt + [r[1] + 600]) - 2
            cell = sorted([w for w in words if y0 <= w[1] < y1 and w[0] > r[2]], key=lambda w: (round(w[1] / 3), w[0]))
            area_ws = [w for w in cell if re.fullmatch(r'[\d,]+\.\d+\s*(㎡|m2)?', w[4]) or re.fullmatch(r'[\d,]+\.\d+㎡', w[4])]
            if not area_ws: continue
            aw = area_ws[0]
            use = ' '.join(w[4] for w in cell if w[2] <= aw[0] + 1 and abs(w[1] - aw[1]) < 30)
            fac_words = [w for w in cell if w is not aw and w[0] > aw[0] + 1]
            lines, cur, ly = [], [], None
            for w in fac_words:
                if ly is not None and abs(w[1] - ly) > 3: lines.append(' '.join(cur)); cur = []
                cur.append(w[4]); ly = w[1]
            if cur: lines.append(' '.join(cur))
            fid = norm_floor(r[4])
            if fid:
                txt = _join_cell(lines)
                floors[fid] = dict(use=use, area=float(re.sub(r'[^\d.]', '', aw[4])), fac=parse_fac(txt), fac_txt=txt,
                                   fac_items=merge_items(parse_items(txt)))
    info['floors'] = floors
    _legacy_form_keys(info, full)
    return info, doc


CAP = re.compile(r'평면도\s*\(([^)]*)\)')


def expand_floors(txt):
    """'지상2층~지상10층 기준층' → [2F..10F], '지하1층' → [B1]"""
    toks = re.findall(r'(지하\d+|지상\d+|옥탑\d*|PIT)', txt)
    ids = [norm_floor(t) for t in toks if norm_floor(t)]
    if '~' in txt and len(ids) == 2 and ids[0][-1] == 'F' and ids[1][-1] == 'F':
        a, b = int(ids[0][:-1]), int(ids[1][:-1]); return [f'{k}F' for k in range(a, b + 1)]
    if '~' in txt and len(ids) == 2 and ids[0][0] == 'B' and ids[1][0] == 'B':
        a, b = int(ids[0][1:]), int(ids[1][1:]); return [f'B{k}' for k in range(min(a, b), max(a, b) + 1)]
    return ids


def extract_plans(doc):
    """이미지마다 가장 가까운 위쪽 캡션을 붙이고, '평면도(…)' 캡션만 층 도면으로 사용.
    표준양식이면 extract_figures(동 구분·쪽 넘김 캡션·설명 문단 건너뜀)의 평면도만 — 항목에 'dong'(0부터) 추가"""
    if is_std(doc_text(doc)):
        return [f for f in extract_figures(doc) if f['kind'] == '평면도']
    plans = []
    for page in doc:
        blocks = [b for b in page.get_text('blocks') if b[6] == 0]
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info['bbox']
            cands = [b for b in blocks if b[3] <= y0 + 2 and y0 - b[3] < 80]
            cap = max(cands, key=lambda b: b[3])[4].strip() if cands else ''
            m = CAP.search(cap)
            if not m: continue
            pix = pymupdf.Pixmap(doc, info['xref'])
            if pix.n - pix.alpha >= 4: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
            plans.append(dict(caption=cap, floors=expand_floors(m.group(1)), img=arr[..., ::-1].copy()))   # BGR (OpenCV)
    return plans



# ═════════════════════════════ 표준양식(소방활동 정보카드 표준양식) 해석
# 양식 항목명을 그대로 키로 쓴다 — 관리번호·주용도·상호·소재지·거리·소방용수·건물구조·관계인·위험물·층별현황(+ 동목록, 그림)
STRUCT_KEYS = {'양식': '양식', '구조': '구조', '지붕': '지붕', '동명': '동명', '지하층수': '지하층수', '지상층수': '지상층수', '연면적(㎡)': '연면적', '연면적': '연면적',
               '바닥면적(㎡)': '바닥면적', '바닥면적': '바닥면적', '특별피난계단수': '특별피난계단수', '피난계단수': '피난계단수', '일반계단수': '일반계단수',
               '옥외계단수': '옥외계단수', '승강기(비상)': '승강기_비상', '승강기(일반)': '승강기_일반'}
STRUCT_NUM = {'지하층수', '지상층수', '연면적', '바닥면적', '특별피난계단수', '피난계단수', '일반계단수', '옥외계단수', '승강기_비상', '승강기_일반'}
ROLES = ['소유자', '관리자', '소방안전관리자', '위험물안전관리자']
HAZ_KINDS = ['위험물시설', '가스취급현황', '특수가연물', '기타']
FR_WORDS = ('철근콘크리트', '철골철근', '철골콘크리트', '콘크리트', '프리캐스트')     # 주요구조부 내화(evac.py 와 같은 낱말)
SEC2 = re.compile(r'2\s*\.\s*동\s*별\s*위\s*치\s*도')
FIG_CAP = re.compile(r'^\s*\[?\s*(\d+)\s*\.?\s*(평\s*면\s*도\s*\(.*\)|위\s*치\s*도|(?:정\s*면|측\s*면|후\s*면)?\s*사\s*진|진\s*압\s*작\s*전\s*도)')


def _ns(t): return re.sub(r'\s+', '', t or '')


def is_std(full):
    """표준양식 판별: ⑦ 격자 칸 이름·⑧ 관계인 칸 이름(기존 근사 배치에는 없음)"""
    t = _ns(full)
    return sum(k in t for k in ('지하층수', '지상층수', '소방안전관리자', '위험물안전관리자', '층별현황', '소방용수시설')) >= 4


UNIT_SP = re.compile(r'(?<=\d) (?=(?:류|길|층|호|번지|석유류)(?![가-힣]))')


def _cell(t):
    """표 칸 글자: 줄바꿈으로 끊긴 토큰 다시 붙이고 공백 정리. 한글↔숫자 글꼴이 바뀌는 자리에 PDF 변환기(LibreOffice·한글)가 끼워 넣는
    공백 중 흔한 단위 앞 공백('제4 류'·'가상로5 길'·'지상1 층')은 지운다(그 밖은 비교 시 공백 무시)"""
    if t is None: return None
    return UNIT_SP.sub('', re.sub(r'\s+', ' ', _join_cell(t.split('\n'))).strip())


def _num_or(t, cast=float):
    m = re.search(r'-?[\d,]*\.?\d+', t or '')
    if not m: return None
    try: return cast(m.group(0).replace(',', '')) if cast is float else int(float(m.group(0).replace(',', '')))
    except ValueError: return None


def _fac_canon(t):
    """소방시설 칸 → 「시설명:수량」 띄어쓰기 정규형(글꼴 바뀜 자리의 끼어든 공백 제거: 'S/P 헤드:30' → 'S/P헤드:30')"""
    items = re.findall(r'([^\s:,;][^:;]*?)\s*:\s*(\d[\d,]*)', t or '')
    return ' '.join(f"{_ns(k)}:{v.replace(',', '')}" for k, v in items)


def _table_rows(doc):
    """1절(건물개황) 표 행 [(쪽, y, [칸 글자|None])] — '2. 동별 위치도…' 제목 위까지. 표 선(격자) 기반 pymupdf find_tables"""
    rows = []
    ocr = ocr_of(doc)
    for page in doc:
        O = ocr['pages'][page.number] if ocr else None
        if O is not None:                                   # 스캔본 쪽: OCR 표 행(pymupdf find_tables 와 같은 모양)
            ye = O['sec2_y'] if O['sec2_y'] is not None else 1e9
            rows += [r for r in O['rows'] if r[1] < ye]
            if O['sec2_y'] is not None: break
            continue
        y_end = 1e9
        for b in page.get_text('blocks'):
            if b[6] == 0 and SEC2.search(b[4]): y_end = min(y_end, b[1])
        try: tabs = page.find_tables().tables
        except Exception: tabs = []
        for tb in sorted(tabs, key=lambda t: t.bbox[1]):
            if tb.bbox[1] >= y_end: continue
            for r_, row in zip(tb.rows, tb.extract()):
                y = r_.bbox[1] if r_.bbox else tb.bbox[1]
                if y < y_end: rows.append((page.number, y, [_cell(c) for c in row]))
        if y_end < 1e9: break
    return rows


def _new_dong():
    return dict(관리번호=None, 주용도=None, 상호=None, 소재지=dict(지번=None, 도로명=None), 거리=dict(소방서=None, 관할센터=None, 최단센터=None),
                소방용수=[], 건물구조={v: None for v in dict.fromkeys(STRUCT_KEYS.values())}, 관계인=[], 위험물=[], 층별현황=[])


def _parse_water(t):
    out = []
    for part in re.split(r'\s/\s|/(?=\s*[가-힣])', t or ''):
        m = re.search(r'([^\s:/]+?)\s*-\s*([^:/]+?)\s*-\s*([^:/]+?)\s*-\s*([^:/\s]+)\s*:\s*([\d.]+)\s*[Kk]\s*[Mm]', part)
        if m: out.append(dict(종류=_ns(m.group(1)), 관서=_ns(m.group(2)), 동=_ns(m.group(3)), 번호=_ns(m.group(4)), 거리km=float(m.group(5))))
    return out


def _parse_dist(t):
    s = _ns(t)
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, s))
    return dict(소방서=g(r'소방서([\d.]+)[Kk][Mm]'), 관할센터=g(r'([\d]+\.?\d*)[Kk][Mm]\(관할\)'), 최단센터=g(r'([\d]+\.?\d*)[Kk][Mm]\(최단\)'))


def _floor_row(d, cells, hdr):
    """⑩ 한 행 → 층별현황 항목(층 칸을 찾아 그 오른쪽 칸을 머리행 열 순서대로)"""
    idx = next((i for i, c in enumerate(cells) if c and FLOOR_TOKEN.match(_ns(c))), None)
    if idx is None: return
    raw = _ns(cells[idx]); fid = norm_floor(raw)
    if not fid: return
    if hdr and len(hdr['n']) == len(cells) and hdr['층별'] == idx:
        get = lambda k: cells[hdr[k]] if hdr.get(k) is not None and hdr[k] < len(cells) else None
        use, area, fac, etc = get('용도'), get('면적'), get('소방시설'), get('기타')
    else:                                                    # 열 수가 다르면(쪽 넘김·병합) 층 칸 오른쪽 값 칸 순서대로
        rest = [c for c in cells[idx + 1:] if c is not None]
        use, area, fac, etc = (rest + [None] * 4)[:4]
    txt = _fac_canon(fac)
    d['층별현황'].append(dict(층=fid, 층_원문=raw, 용도=(use or '').strip(), 면적=_num_or(area), 소방시설_원문=txt, 기타=(etc or '').strip(),
                          소방시설=[dict(code=it['code'], name=it['name'], count=it['count'], raw=it['raw']) for it in merge_items(parse_items(txt))]))


def parse_std(doc, full=None, dong=None):
    """표준양식 PDF → info(양식 항목명 키 + 기존 키 호환). 여러 동이면 info['동목록'] 에 동별로, 기존 키·최상위 양식 키는 선택 동(기본 첫 동)"""
    import os
    dongs, d, mode, hdr = [], None, None, None
    for pg, y, cells in _table_rows(doc):
        n = [_ns(c) if c else '' for c in cells]
        if any(x.startswith('①관리번호') or x == '①관리번호' for x in n):
            d = _new_dong(); dongs.append(d); mode = None
        if d is None: continue
        i = 0
        while i < len(cells):                                        # 칸 이름 → 다음 값 칸(병합으로 None 인 칸은 건너뜀)
            c = n[i]
            if cells[i] is None or not c: i += 1; continue
            nxt = next((j for j in range(i + 1, len(cells)) if cells[j] is not None), None)
            val = cells[nxt] if nxt is not None else None
            key = re.sub(r'^[①-⑪]', '', c)
            if key in ('관리번호', '주용도', '상호') and mode is None:
                d[key] = val; i = (nxt or i) + 1; continue
            if key == '소재지':
                t = val or ''
                m1 = re.search(r'\(지번\)\s*(.*?)\s*(?=\(도로명\)|$)', t); m2 = re.search(r'\(도로명\)\s*(.*)$', t)
                d['소재지'] = dict(지번=(m1.group(1).strip() or None) if m1 else None, 도로명=(m2.group(1).strip() or None) if m2 else None)
                i = (nxt or i) + 1; continue
            if key.startswith('소방대상물과의거리'): d['거리'] = _parse_dist(val); i = (nxt or i) + 1; continue
            if key == '소방용수시설': d['소방용수'] = _parse_water(val); i = (nxt or i) + 1; continue
            if key in STRUCT_KEYS and mode != 'floor':
                k = STRUCT_KEYS[key]
                v = val
                if k in STRUCT_NUM: v = _num_or(val, float if k in ('연면적', '바닥면적') else int)
                d['건물구조'][k] = v if v != '' else None
                i = (nxt or i) + 1; continue
            if key in ROLES and '성명' in n:                               # ⑧ 관계인: 구분 | 성명 값 | 연락처 값
                j = n.index('성명'); k2 = n.index('연락처') if '연락처' in n else None
                nm = next((cells[t] for t in range(j + 1, len(cells)) if cells[t] is not None), None)
                ph = next((cells[t] for t in range(k2 + 1, len(cells)) if cells[t] is not None), None) if k2 is not None else None
                d['관계인'].append(dict(구분=key, 성명=nm or None, 연락처=ph or None)); break
            if key == '구분' and '품명' in n: mode = 'haz'; break
            if key in HAZ_KINDS and mode == 'haz':
                vals = [cells[t] or '' for t in range(i + 1, len(cells)) if cells[t] is not None][:3]
                vals += [''] * (3 - len(vals))
                if any(v.strip() for v in vals): d['위험물'].append(dict(구분=key, 시설명=vals[0], 품명=vals[1], 수량=vals[2]))
                break
            if key == '층별' and any(x.startswith('소방시설') for x in n):
                mode = 'floor'
                hdr = dict(n=n, 층별=i, 용도=next((t for t, x in enumerate(n) if x == '용도'), None), 면적=next((t for t, x in enumerate(n) if x.startswith('면적')), None),
                           소방시설=next((t for t, x in enumerate(n) if x.startswith('소방시설')), None), 기타=next((t for t, x in enumerate(n) if x == '기타'), None))
                break
            i += 1
        if mode == 'floor' and not any(x == '층별' for x in n): _floor_row(d, cells, hdr)
    for dd in dongs:
        for k in ('관리번호', '주용도', '상호'): dd[k] = (dd[k] or '').strip() or None
        for k in ('양식', '구조', '지붕', '동명'):
            if isinstance(dd['건물구조'].get(k), str): dd['건물구조'][k] = _ns(dd['건물구조'][k]) if k != '동명' else dd['건물구조'][k].strip()
        s = dd['건물구조'].get('구조')
        dd['내화구조'] = (any(w in s for w in FR_WORDS)) if s else None        # 주요구조부 내화 여부(정보카드 ⑦ 구조 글자 기준 추정)
        dd['위험물_있음'] = bool(dd['위험물'])
    if dong is None: dong = int(os.environ.get('FIRE3DBEAM_DONG', '0') or 0)
    dong = min(max(0, dong), max(0, len(dongs) - 1))
    sel = dongs[dong] if dongs else _new_dong()
    info = {'양식': 'std', '선택동': dong, '동목록': dongs}
    info.update({k: v for k, v in sel.items()})
    # ── 기존 키(하류 모듈 호환: pipeline·daejang·site·fac·export)
    S = sel['건물구조']
    iv = lambda k: S.get(k) or 0
    info.update(use=sel['주용도'], name=sel['상호'], gfa=S.get('연면적'),
                stairs=dict(special=iv('특별피난계단수'), evac=iv('피난계단수'), normal=iv('일반계단수'), outdoor=iv('옥외계단수')),
                elev=dict(emergency=iv('승강기_비상'), normal=iv('승강기_일반')),
                water=[dict(kind=(f"소화전({w['종류']})" if w['종류'] in ('지상식', '지하식') else w['종류']), id=f"{w['관서']}-{w['번호']}", id_full=f"{w['관서']}-{w['동']}-{w['번호']}",
                            km=w['거리km']) for w in sel['소방용수']],
                hazmat=' / '.join(' '.join(x for x in (h['시설명'], h['품명'], h['수량']) if x) for h in sel['위험물']))
    floors = {}
    for r in sel['층별현황']:
        if r['면적'] is None: continue
        floors[r['층']] = dict(use=r['용도'], area=r['면적'], fac=parse_fac(r['소방시설_원문']), fac_txt=r['소방시설_원문'],
                               fac_items=merge_items(parse_items(r['소방시설_원문'])), etc=r['기타'])
    info['floors'] = floors
    return info


def _legacy_form_keys(info, full):
    """기존(근사 배치) 카드도 양식 항목명 키 일부를 채움(하류 코드가 한 방식으로 읽도록)"""
    m = re.search(r'관\s*리\s*번\s*호\s*([^\s②]+)', full)
    st, ev = info.get('stairs') or {}, info.get('elev') or {}
    d = dict(관리번호=m.group(1) if m else None, 주용도=info.get('use'), 상호=info.get('name'), 소재지=dict(지번=None, 도로명=None),
             거리=dict(소방서=None, 관할센터=None, 최단센터=None), 소방용수=[], 관계인=[], 위험물=[],
             건물구조=dict(연면적=info.get('gfa'), 특별피난계단수=st.get('special'), 피난계단수=st.get('evac'), 일반계단수=st.get('normal'), 옥외계단수=st.get('outdoor'),
                       승강기_비상=ev.get('emergency'), 승강기_일반=ev.get('normal')),
             층별현황=[dict(층=f, 용도=v['use'], 면적=v['area'], 소방시설_원문=v['fac_txt']) for f, v in (info.get('floors') or {}).items()],
             내화구조=None, 위험물_있음=None)
    info.update(선택동=0, 동목록=[d]); info.update({k: v for k, v in d.items() if k not in info})


def _fig_kind(cap):
    c = _ns(cap)
    if '평면도(' in c: return '평면도'
    if '작전도' in c: return '진압작전도'
    if '위치도' in c: return '위치도'
    if '사진' in c: return '사진'
    return '기타'


def extract_figures(doc, min_pt=40):
    """표준양식 ⑪ 그림 전부 → [{kind(위치도|사진|진압작전도|평면도|기타), caption, floors, dong, page, bbox, img(BGR)}].
    캡션 = 그림 위 90pt 안의 '번호. 평면도(…)'·'0. 위치도'·'00 정면사진'·'0 진압작전도' 꼴 문단(설명 문단 '(측면·후면 사진은…)' 은 건너뜀),
    같은 쪽에 없으면 앞 쪽 맨 아래 캡션(쪽 넘김). 동 = '⑪ 동별 위치도…' 제목 순서(제목에 '동명: X' 가 있으면 그 이름)."""
    out, dong, names, started, carry = [], -1, [], False, None
    ocr = ocr_of(doc)
    for page in doc:
        O = ocr['pages'][page.number] if ocr else None
        if O is not None:                                   # 스캔본 쪽: OCR 글줄(정규화된 제목·캡션) + 틀 안 그림
            blocks = [(l['bbox'][0], l['bbox'][1], l['bbox'][2], l['bbox'][3], l['text'], k, 0) for k, l in enumerate(O['lines'])]
            imgs = [dict(bbox=tuple(f['bbox']), img=f['img']) for f in O['frames'] if f['bbox'][2] - f['bbox'][0] > min_pt and f['bbox'][3] - f['bbox'][1] > min_pt]
        else:
            blocks = sorted([b for b in page.get_text('blocks') if b[6] == 0], key=lambda b: b[1])
            imgs = sorted([i for i in page.get_image_info(xrefs=True) if i['bbox'][2] - i['bbox'][0] > min_pt and i['bbox'][3] - i['bbox'][1] > min_pt], key=lambda i: i['bbox'][1])
        ev = [('b', b[1], b) for b in blocks] + [('i', i['bbox'][1], i) for i in imgs]
        ev.sort(key=lambda e: (e[1], 0 if e[0] == 'b' else 1))
        for typ, y, obj in ev:
            if typ == 'b':
                t = obj[4].strip()
                if SEC2.search(t): started = True
                if t.startswith('⑪'):
                    started = True; dong += 1
                    m = re.search(r'동\s*명\s*[:：]\s*(\S+)', t); names.append(m.group(1) if m else None)
                t1 = re.sub(r'\s*\n\s*', ' ', t)
                if FIG_CAP.match(t1): carry = (re.sub(r'^(\d+\.?)', r'\1 ', _ns(t1)), page.number, obj[3])     # '1. 평면도(지하2층)' 꼴로 공백 정리
                continue
            if not started: continue
            x0, y0, x1, y1 = obj['bbox']
            cap = None
            if carry and (carry[1] == page.number and 0 <= y0 - carry[2] < 90 or carry[1] == page.number - 1): cap = carry[0]
            carry = None
            if 'img' in obj: arr = obj['img']
            else:
                pix = pymupdf.Pixmap(doc, obj['xref'])
                if pix.n - pix.alpha >= 4: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()[..., ::-1]
            kind = _fig_kind(cap or '')
            m = CAP.search(cap or '')
            out.append(dict(kind=kind, caption=cap or '', floors=expand_floors(m.group(1)) if (m and kind == '평면도') else [], dong=max(0, dong),
                            dong_name=names[dong] if 0 <= dong < len(names) else None, page=page.number, bbox=[round(v, 1) for v in obj['bbox']], img=arr.copy(),
                            **({'source': 'ocr'} if 'img' in obj else {})))
    return out


# ═════════════════════════════ 스캔본(글자층 없는 쪽) — fire3dbeam/card_ocr.py
OCR_FLAG_CONF = 0.75         # 이 신뢰도 미만 항목 → 검수 플래그(card_ocr)


def ocr_of(doc):
    """글자층 없는 쪽이 있으면 OCR 분석(캐시) → dict, 없으면 None. FIRE3DBEAM_OCR=0 이면 끔"""
    import os
    if os.environ.get('FIRE3DBEAM_OCR', '1') == '0': return None
    from fire3dbeam import card_ocr
    if not card_ocr.has_image_pages(doc): return None
    return card_ocr.analyze(doc)


def doc_text(doc, ocr=None):
    """문서 글자: 글자층 + (스캔본 쪽) OCR 글자"""
    t = '\n'.join(p.get_text() for p in doc)
    if ocr is None: ocr = ocr_of(doc)
    return t + ('\n' + ocr['text'] if ocr else '')


def ocr_meta(info, ocr):
    """OCR 결과 표시 + 항목별 신뢰도(선택 동) + 낮은 신뢰도 항목(검수 플래그 후보). 값 글자는 넣지 않음(관계인 개인정보)"""
    sel = info.get('선택동', 0)
    conf = {f: c for d, f, c, priv in ocr['fields'] if d == sel}
    low = [dict(field=f, conf=c, private=priv) for d, f, c, priv in ocr['fields'] if d == sel and c < OCR_FLAG_CONF]
    info['source'] = 'ocr'
    info['ocr'] = dict(pages=ocr['n_image_pages'], seconds=ocr['seconds'], per_page=ocr['per_page'], conf=conf, low=low,
                       conf_mean=round(sum(conf.values()) / max(1, len(conf)), 3), n_fields=len(conf))
    return info


def ocr_value(info, field):
    """'층별현황.1F.면적' 같은 필드 이름 → 해석된 값(표시용). 관계인은 None"""
    parts = field.split('.')
    if parts[0] == '관계인': return None
    if parts[0] == '층별현황':
        r = next((x for x in info.get('층별현황') or [] if x['층'] == parts[1]), None)
        if not r: return None
        return {'층': r.get('층_원문'), '용도': r.get('용도'), '면적': r.get('면적'), '소방시설': r.get('소방시설_원문'), '기타': r.get('기타')}.get(parts[2])
    if parts[0] == '위험물':
        r = next((x for x in info.get('위험물') or [] if _ns(x['구분']) == parts[1]), None)
        return r.get(parts[2]) if r else None
    v = info
    for k in parts:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def redact(info, include=None):
    """⑧ 관계인 성명·연락처(개인정보) 제거 — 기본값. 역할(구분)과 기재 여부만 남긴다.
    내부 PC 에서 지휘관이 관리자에게 전화해야 할 때만 FIRE3DBEAM_INCLUDE_CONTACTS=1 (또는 include=True) 로 그대로 둔다."""
    import os, copy
    if include is None: include = os.environ.get('FIRE3DBEAM_INCLUDE_CONTACTS', '').lower() in ('1', 'true', 'on', 'yes')
    if include: return dict(info, 개인정보='포함(FIRE3DBEAM_INCLUDE_CONTACTS=1)')
    out = copy.copy(info)
    red = lambda L: [dict(구분=r['구분'], 기재=bool(r.get('성명') or r.get('연락처'))) for r in (L or [])]
    if '관계인' in out: out['관계인'] = red(out['관계인'])
    if '동목록' in out: out['동목록'] = [dict(d, 관계인=red(d.get('관계인'))) for d in out['동목록']]
    out['개인정보'] = '제외(관계인 성명·연락처 — 기본값)'
    return out
