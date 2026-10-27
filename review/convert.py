# -*- coding: utf-8 -*-
"""정보카드 PDF 1개(+선택: 별도 평면도·진압작전도 파일) → 결과 폴더
   (3D 지휘화면 HTML + IFC + 사전 활동계획 HTML + 검수화면 HTML + 결과 JSON). 엔진은 한 번만 실행.
사용: python -m review.convert 정보카드.pdf [평면도 파일 …] [--out 폴더] [--model | --rules]
기본 출력 폴더: PDF와 같은 곳의 '<이름>_변환결과' (쓸 수 없으면 프로그램 폴더의 '결과')."""
import argparse, datetime, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review.common import ROOT, save_json, sha256_file, slug, build_outputs

NAMES = dict(result='{}_결과.json', html='{}_3D지휘화면.html', ifc='{}.ifc', plan='{}_활동계획.html', review='{}_검수화면.html')


def building_stem(pdf):
    st = os.path.splitext(os.path.basename(pdf))[0]
    if st.lower() in ('card', '정보카드'): st = os.path.basename(os.path.dirname(os.path.abspath(pdf))) or st
    return slug(st)


def out_dir_for(pdf, out=None):
    st = building_stem(pdf)
    for d in ([out] if out else []) + [os.path.join(os.path.dirname(os.path.abspath(pdf)), st + '_변환결과'), os.path.join(ROOT, '결과', st + '_변환결과')]:
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, '.w'); open(t, 'w').close(); os.unlink(t)
            return d
        except OSError:
            continue
    raise SystemExit('출력 폴더를 만들 수 없음')


def model_available():
    """STR 분할 모델 교차검증을 쓸 수 있나(모델 파일 + onnxruntime) → (bool, 이유)"""
    from fire3dbeam import str_seg
    if not os.path.exists(str_seg.MODEL): return False, f'모델 파일 없음({os.path.basename(str_seg.MODEL)})'
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        return False, f'onnxruntime 불러오기 실패({e!r})'
    return True, ''


# ───────── 개인정보(관계인 성명·연락처) 유출 검사 — 결과 파일 글자 전체에서 원문 문자열 찾기
def contact_strings(pdf):
    from fire3dbeam.card import parse_card
    info, _ = parse_card(pdf)
    rows = list(info.get('관계인') or []) + [r for d in info.get('동목록') or [] for r in (d.get('관계인') or [])]
    out = set()
    for r in rows:
        for k in ('성명', '연락처'):
            v = (r.get(k) or '').strip()
            if len(re.sub(r'\W', '', v)) >= 2: out.add(v)
    return sorted(out)


_X2 = re.compile(r'\\X2\\((?:[0-9A-F]{4})+)\\X0\\')


def _ifc_text(t):
    return _X2.sub(lambda m: bytes.fromhex(m.group(1)).decode('utf-16-be', 'replace'), t)


def privacy_scan(pdf, files):
    """→ (검사한 문자열 수, [(파일 이름, 찾은 개수)])  — 문자열 자체는 돌려주지 않음(로그에 남기지 않게)"""
    needles = contact_strings(pdf)
    hits = []
    for p in files:
        if not p or not os.path.exists(p): continue
        with open(p, encoding='utf-8', errors='replace') as f: t = f.read()
        if p.lower().endswith('.ifc'): t = _ifc_text(t)
        n = sum(t.count(s) for s in needles)
        if n: hits.append((os.path.basename(p), n))
    return len(needles), hits


def aux_inputs(pdf):
    """정보카드 옆 보조 입력(건축물대장·부지·경계구역 일람도) — 파이프라인 기본 이름 + 한글 이름('…대장….pdf', '…일람도….json').
    합본 PDF 는 결과 폴더에 있으므로 원본 카드 폴더에서 찾아 명시적으로 넘긴다('-' = 없음)"""
    import glob
    d = os.path.dirname(os.path.abspath(pdf))
    def first(names, pats=()):
        for n in names:
            if os.path.exists(os.path.join(d, n)): return os.path.join(d, n)
        for pt in pats:
            h = sorted(x for x in glob.glob(os.path.join(d, pt)) if os.path.abspath(x) != os.path.abspath(pdf))
            if h: return h[0]
        return '-'
    return dict(daejang=first(['daejang.json', 'daejang.pdf'], ['*건축물대장*.json', '*건축물대장*.pdf', '*대장*.pdf']),
                site=first(['site.json']), ilramdo=first(['zones.json', 'ilramdo.json'], ['*일람도*.json']))


def convert(pdf, out=None, log=print, model=None, extras=None):
    """model: True/False/None(None = 모델 파일·onnxruntime 있으면 사용). extras: 별도 평면도·진압작전도 파일 목록"""
    from fire3dbeam.pipeline import run
    from review.make import make as make_review
    from review import plan_sheet
    t0 = time.time(); pdf = os.path.abspath(pdf)
    st = building_stem(pdf); d = out_dir_for(pdf, out)
    P = {k: os.path.join(d, v.format(st)) for k, v in NAMES.items()}
    if model is None:
        model, why = model_available()
        if not model: log(f'      (AI 모델 교차검증 생략: {why} → 규칙만)')
    src_pdf, notes = pdf, []
    if extras:
        from review.attach import combine
        log(f'[0/5] 별도 도면 {len(extras)}개 붙이기')
        cp, notes = combine(pdf, extras, os.path.join(d, st + '_입력합본.pdf'), log=log)
        if cp: src_pdf = cp
    log(f'[1/5] 정보카드 해석·도면 인식{" + AI 모델 교차검증" if model else ""}: {os.path.basename(pdf)}')
    aux = aux_inputs(pdf)
    for k, v in aux.items():
        if v != '-' and os.path.basename(v) not in ('daejang.json', 'daejang.pdf', 'site.json', 'zones.json', 'ilramdo.json'): log(f'      보조 입력({k}): {os.path.basename(v)}')
    res = run(src_pdf, keep=True, model=bool(model), **aux)
    imgs = [p['img'] for p in res['_plans']]
    clean = {k: v for k, v in res.items() if not k.startswith('_')}
    clean['source'] = dict(pdf=src_pdf, pdf_sha256=sha256_file(src_pdf), card_pdf=pdf, card_sha256=sha256_file(pdf),
                           extras=[dict(file=os.path.basename(x), sha256=sha256_file(x)) for x in (extras or []) if os.path.exists(x)], attached=notes,
                           model=bool(model), converted=datetime.datetime.now().isoformat(timespec='seconds'))
    save_json(clean, P['result'])
    fig = [f['kind'] for f in clean.get('figures') or []]
    site = clean.get('site') if isinstance(clean.get('site'), dict) else {}
    log(f"      양식 {'표준양식' if (clean.get('card') or {}).get('양식') == 'std' else '기존 배치'} · 층 {len(res['floors'])}개 · 계단 {len(res['stairs'])} · 승강기 {len(res['elevs'])}"
        f" · 교차검증 플래그 {len(res['flags'])}" + (f" · 진압작전도 {'인식' if site.get('src') == '진압작전도' else '있음(인식 실패 → ' + str(site.get('src') or site.get('error')) + ')'}" if '진압작전도' in fig else ''))
    oc = (clean.get('card') or {}).get('ocr') if (clean.get('card') or {}).get('source') == 'ocr' else None
    if oc:
        if oc.get('error'): log(f"      ! 정보카드 스캔본 OCR: {oc['error']}")
        else: log(f"      정보카드 = 스캔본(글자층 없음) → OCR {oc.get('pages')}쪽 {oc.get('seconds')}초(쪽당 최대 {max(oc.get('per_page') or [0])}초) · "
                  f"항목 {oc.get('n_fields')}개 평균 신뢰도 {oc.get('conf_mean')} · 확인 필요 {len(oc.get('low') or [])}항목(확인사항에 표시)")
    if isinstance(clean.get('evac'), dict) and clean['evac'].get('error'): log(f"      ! 대피로 계산 생략: {clean['evac']['error']}")
    for k, v in (clean.get('xcheck') or {}).items():
        if isinstance(v, dict) and v.get('error'): log(f'      ! 도면 {k} AI 모델 교차검증 실패 → 규칙 결과 사용: {v["error"][:160]}')
    log('[2/5] 3D 지휘화면 + IFC 생성')
    D, _ = build_outputs(res, imgs, P['html'], P['ifc'])
    log('[3/5] 사전 활동계획')
    plan_sheet.write(clean, D, P['plan'], html_name=os.path.basename(P['html']),
                     src_note=f"입력 SHA256 {clean['source']['card_sha256'][:12]}…" + (f" + 별도 도면 {len(notes)}개" if notes else ''))
    log('[4/5] 검수화면 생성')
    make_review(P['result'], pdf=src_pdf, out_html=P['review'], res=clean, imgs=imgs)
    n, hits = privacy_scan(pdf, [P['result'], P['html'], P['ifc'], P['plan'], P['review']])
    if os.environ.get('FIRE3DBEAM_INCLUDE_CONTACTS', '').lower() in ('1', 'true', 'on', 'yes'):
        log(f'      개인정보: 관계인 연락처 포함 모드(FIRE3DBEAM_INCLUDE_CONTACTS=1) — 결과 파일 취급 주의')
    elif hits:
        log(f'      ✗ 개인정보 검사: 관계인 성명·연락처가 결과에 남음 {hits} — 결과를 쓰지 말고 담당자에게 알리세요')
    else:
        log(f'      개인정보 검사: 관계인 성명·연락처 {n}건 모두 결과에 없음')
    log(f'[5/5] 완료 ({time.time() - t0:.0f}초) → {d}')
    P['_privacy_hits'] = hits
    return d, P


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='정보카드 PDF (+별도 평면도) → 결과 폴더')
    ap.add_argument('pdf'); ap.add_argument('extras', nargs='*', help='별도 평면도·진압작전도 파일(PDF/PNG/JPG/TIF, 이름에 층)')
    ap.add_argument('--out'); g = ap.add_mutually_exclusive_group(); g.add_argument('--model', action='store_true'); g.add_argument('--rules', action='store_true')
    a = ap.parse_args()
    convert(a.pdf, a.out, model=True if a.model else (False if a.rules else None), extras=a.extras)
