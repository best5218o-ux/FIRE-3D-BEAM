# -*- coding: utf-8 -*-
"""기능 18: 공공데이터(건축물대장) ↔ 소방활동 정보카드 ↔ 도면 추출값 3자 교차검증 (오프라인 입력).

입력: daejang.json — 건축HUB 건축물대장정보 서비스 응답 항목명 그대로
        {"brTitleInfo":[{표제부}], "brFlrOulnInfo":[{층별개요}, ...]}
      (또는 data.go.kr 원 응답 {"response":{"body":{"items":{"item":[...]}}}} 를 표제부/층별개요 두 파일로 저장한 것 → load() 가 합침)
      daejang.pdf — 건축물대장(갑) 출력물(텍스트 PDF)을 parse_pdf() 로 같은 구조로 읽음(스캔본은 미지원).
비교 항목(대장 항목명 → 정보카드 항목):
  totArea(연면적) ↔ ⑦ 연면적 / grndFlrCnt·ugrndFlrCnt(지상·지하 층수) ↔ ⑦ 지상 n층·지하 n층 및 ⑩ 층별 현황 행 수
  층별개요 area(층별 면적) ↔ ⑩ 층별 면적 ↔ 도면 외곽 면적(3자: 다수결로 어느 쪽이 틀렸는지 추정)
  mainPurpsCdNm(주용도) ↔ ② 주용도 / rideUseElvtCnt+emgenUseElvtCnt ↔ ⑦ 승강기 수 ↔ 도면 승강기
  useAprDay(사용승인일)·heit(높이)는 표시·층고 추정에 사용."""
import json, re
from fire3dbeam.sop import CFG
from fire3dbeam.card import level_of


def _items(x):
    if isinstance(x, dict) and 'response' in x:
        it = x['response']['body']['items']['item']; return it if isinstance(it, list) else [it]
    return x


def load(path, flr_path=None):
    if path.lower().endswith('.pdf'): return parse_pdf(path)
    d = json.load(open(path, encoding='utf-8'))
    if flr_path: d = dict(brTitleInfo=_items(d), brFlrOulnInfo=_items(json.load(open(flr_path, encoding='utf-8'))))
    d['brTitleInfo'] = _items(d.get('brTitleInfo') or [])
    d['brFlrOulnInfo'] = _items(d.get('brFlrOulnInfo') or [])
    return d


def fid_of(row):
    g, n = str(row.get('flrGbCdNm', '')), int(row.get('flrNo') or 0)
    if '지하' in g: return f'B{n}'
    if '옥탑' in g: return 'RF'
    if '지상' in g: return f'{n}F'
    m = re.match(r'(지하|지상)(\d+)', str(row.get('flrNoNm', '')))
    return (f'B{m.group(2)}' if m.group(1) == '지하' else f'{m.group(2)}F') if m else None


def _num(s): return float(str(s).replace(',', '')) if s not in (None, '') else None


def parse_pdf(path):
    """건축물대장(갑) 텍스트 PDF → load() 와 같은 구조"""
    import pymupdf
    doc = pymupdf.open(path); t = '\n'.join(p.get_text() for p in doc)
    g = lambda pat: (re.search(pat, t) or [None, None])[1]
    title = dict(platPlc=(g(r'대지위치\s*([^\n]+)') or '').strip(), bldNm=(g(r'명칭\s*([^\n]+)') or '').strip(),
                 archArea=_num(g(r'건축면적\s*([\d,\.]+)')), totArea=_num(g(r'연면적\s*([\d,\.]+)')),
                 strctCdNm=(g(r'주구조\s*([^\n]+)') or '').strip(), mainPurpsCdNm=(g(r'주용도\s*([^\n\s]+)') or '').strip(),
                 ugrndFlrCnt=int(g(r'지하\s*:\s*(\d+)\s*층') or 0), grndFlrCnt=int(g(r'지상\s*:\s*(\d+)\s*층') or 0),
                 heit=_num(g(r'높이\s*([\d\.]+)\s*m')), rideUseElvtCnt=int(g(r'승용\s*(\d+)') or 0), emgenUseElvtCnt=int(g(r'비상용\s*(\d+)') or 0))
    d = re.search(r'사용승인일\s*(\d{4})\.(\d{2})\.(\d{2})', t)
    title['useAprDay'] = ''.join(d.groups()) if d else None
    rows = []
    for m in re.finditer(r'(지하|지상|옥탑)(\d+)층\s+([^\n]+)\s+([^\n(]+)\(([^)]*)\)\s+([\d,]+\.\d+)', t):
        rows.append(dict(flrGbCdNm=m.group(1), flrNo=int(m.group(2)), flrNoNm=f'{m.group(1)}{m.group(2)}층', strct=m.group(3).strip(),
                         mainPurpsCdNm=m.group(4).strip(), etcPurps=m.group(5), area=_num(m.group(6))))
    return dict(source='건축물대장 출력물(PDF) 판독', brTitleInfo=[title], brFlrOulnInfo=rows)


def _close(a, b, tol): return a is not None and b is not None and abs(a - b) / max(abs(a), 1e-6) <= tol


def verdict(dj, cd, dw, cfg):
    """3자 다수결 — 대장(dj)·정보카드(cd)·도면(dw)"""
    t, td = cfg['daejang_area_rel'], cfg['drawing_area_rel']
    if dj is None: return '대장 미등재 층 — 증축·대장 미갱신 여부 확인'
    if cd is None: return '정보카드에 없는 층 — 정보카드 갱신 필요'
    if _close(dj, cd, t):
        return '일치' if dw is None or _close(cd, dw, max(td, 0.2)) else '대장·카드 일치 / 도면 축척·외곽 확인'
    if dw is not None and _close(dj, dw, td) and not _close(cd, dw, td): return '정보카드 면적 오기 의심(도면·대장 일치)'
    if dw is not None and _close(cd, dw, td) and not _close(dj, dw, td): return '대장 미갱신 의심(증축·용도변경 등, 도면·카드 일치)'
    return '3자 불일치 — 현장·서류 확인'


def compare(dj, info, floors, elevs_found=None, cfg=CFG):
    """dj: load() 결과, info: parse_card() 결과, floors: {fid: {'footprint_area'}} → (report, flags)"""
    T = (dj.get('brTitleInfo') or [{}])[0]
    rows = dj.get('brFlrOulnInfo') or []
    by = {}
    for r in rows:                       # 같은 층 여러 용도 행 → 면적 합
        f = fid_of(r)
        if not f: continue
        b = by.setdefault(f, dict(area=0.0, uses=[]))
        b['area'] += _num(r.get('area')) or 0; b['uses'].append(r.get('mainPurpsCdNm') or '')
    cf = info.get('floors') or {}
    flags, table = [], []
    for fid in sorted(set(by) | set(cf) | set(floors), key=level_of):
        a = round(by[fid]['area'], 2) if fid in by else None
        c = (cf.get(fid) or {}).get('area')
        d = (floors.get(fid) or {}).get('footprint_area')
        v = verdict(a, c, d, cfg)
        table.append(dict(floor=fid, daejang=a, card=c, drawing=d, verdict=v, use=', '.join(sorted(set(by[fid]['uses']))) if fid in by else None))
        if a is None and c is not None: flags.append(dict(kind='daejang_floor_missing', floor=fid, card=c, drawing=d, verdict=v))
        elif c is None and a is not None: flags.append(dict(kind='daejang_floor_extra', floor=fid, daejang=a, verdict=v))
        elif a is not None and c is not None and not _close(a, c, cfg['daejang_area_rel']):
            flags.append(dict(kind='daejang_area', floor=fid, daejang=a, card=c, drawing=d, ratio=round(c / a, 2), verdict=v))
    nb = sum(1 for f in cf if f.startswith('B')); nu = sum(1 for f in cf if f.endswith('F') and f != 'RF')
    g, u = T.get('grndFlrCnt'), T.get('ugrndFlrCnt')
    if g is not None and u is not None and (int(g), int(u)) != (nu, nb):
        nb_r = sum(1 for f in by if f.startswith('B')); nu_r = sum(1 for f in by if f.endswith('F') and f != 'RF')
        flags.append(dict(kind='daejang_floor_count', daejang=f'지하{u}/지상{g}', card=f'지하{nb}/지상{nu}', rows=f'지하{nb_r}/지상{nu_r}'))
    ta, gfa = _num(T.get('totArea')), info.get('gfa')
    if ta and gfa and not _close(ta, gfa, cfg['daejang_gfa_rel']):
        flags.append(dict(kind='daejang_gfa', daejang=ta, card=gfa, ratio=round(gfa / ta, 3)))
    mu, cu = (T.get('mainPurpsCdNm') or '').strip(), (info.get('use') or '').strip()
    if mu and cu and mu != cu: flags.append(dict(kind='daejang_use', daejang=mu, card=cu))
    ev_d = (T.get('rideUseElvtCnt') or 0) + (T.get('emgenUseElvtCnt') or 0) if ('rideUseElvtCnt' in T or 'emgenUseElvtCnt' in T) else None
    ev_c = sum((info.get('elev') or {}).values()) if info.get('elev') else None
    if ev_d is not None and ev_c is not None and int(ev_d) != int(ev_c):
        flags.append(dict(kind='daejang_elev', daejang=int(ev_d), card=ev_c, drawing=elevs_found))
    heit, g_ = _num(T.get('heit')), T.get('grndFlrCnt')
    report = dict(source=dj.get('source') or '건축물대장', title={k: T.get(k) for k in ('bldNm', 'platPlc', 'mainPurpsCdNm', 'strctCdNm', 'totArea', 'archArea',
                                                                                  'grndFlrCnt', 'ugrndFlrCnt', 'heit', 'useAprDay', 'rideUseElvtCnt', 'emgenUseElvtCnt')},
                  card=dict(use=info.get('use'), gfa=gfa, floors=f'지하{nb}/지상{nu}', elev=ev_c, elev_drawing=elevs_found),
                  floor_height=round(heit / int(g_), 2) if heit and g_ else None, table=table)
    return report, flags


KIND_TXT = dict(daejang_floor_missing='대장 미등재 층(증축 미반영 의심)', daejang_floor_extra='정보카드 누락 층(대장에만 있음)',
                daejang_area='층 면적 불일치(대장↔정보카드)', daejang_floor_count='층수 불일치(대장 표제부↔정보카드)',
                daejang_gfa='연면적 불일치(대장↔정보카드)', daejang_use='주용도 불일치(대장↔정보카드)', daejang_elev='승강기 수 불일치(대장↔정보카드)')


def issue_text(f, ko):
    k = f['kind']
    if k == 'daejang_floor_missing': return f"{ko(f['floor'])}: 정보카드 {f['card']:,.0f}㎡ · 도면 {('%.0f㎡' % f['drawing']) if f.get('drawing') else '-'} / 건축물대장 층별개요에 없음 → {f['verdict']}"
    if k == 'daejang_floor_extra': return f"{ko(f['floor'])}: 건축물대장 {f['daejang']:,.0f}㎡ / 정보카드·도면에 없음 → 정보카드 갱신 필요"
    if k == 'daejang_area': return f"{ko(f['floor'])}: 대장 {f['daejang']:,.0f}㎡ / 카드 {f['card']:,.0f}㎡ / 도면 {('%.0f㎡' % f['drawing']) if f.get('drawing') else '-'} → {f['verdict']}"
    if k == 'daejang_floor_count': return f"대장 표제부 {f['daejang']} / 정보카드 {f['card']} (대장 층별개요 {f['rows']})"
    if k == 'daejang_gfa': return f"대장 {f['daejang']:,.0f}㎡ / 정보카드 {f['card']:,.0f}㎡ (비율 {f['ratio']})"
    if k == 'daejang_use': return f"대장 '{f['daejang']}' / 정보카드 '{f['card']}' → 용도변경 여부·적용 소방시설 기준 확인"
    if k == 'daejang_elev': return f"대장 {f['daejang']}대 / 정보카드 {f['card']}대 / 도면 {f.get('drawing')}대"
    return json.dumps(f, ensure_ascii=False)
