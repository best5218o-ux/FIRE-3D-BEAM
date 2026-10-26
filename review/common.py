# -*- coding: utf-8 -*-
"""검수·채점 공통: 결과 JSON 읽기, 원본 PDF 찾기, 도면 이미지, 좌표 변환(m ↔ 도면 px), 안전한 파일 쓰기.
엔진(firebim) 내부 구조 대신 결과 JSON 스키마(card/floors/stairs/elevs/flags/plans)에만 기대도록 작성."""
import base64, hashlib, json, os, re, shutil, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ['exit', 'hose', 'panel', 'gas', 'ev']
KO = {'exit': '비상구', 'hose': '옥내소화전', 'panel': '수신기', 'gas': '가스밸브', 'ev': '승강기'}
SCHEMA = 'firebim.review/1'


def load_json(p):
    with open(p, encoding='utf-8') as f: return json.load(f)


def save_json(obj, p, indent=None):
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)


def sha256_file(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(n), b''): h.update(b)
    return h.hexdigest()


def result_sha(res):
    """결과 JSON의 안정적 지문(교정 파일이 어느 결과에 대한 것인지 확인용)"""
    core = {k: res[k] for k in ('floors', 'plans') if k in res}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]


def stem_of(path):
    b = os.path.basename(path)
    for suf in ('.result.json', '_결과.json', '.corrected.json', '.json'):
        if b.endswith(suf): return b[: -len(suf)]
    return os.path.splitext(b)[0]


def find_pdf(result_path, res, pdf=None):
    """원본 PDF: 명시 → 결과에 기록된 경로 → 같은 폴더 같은 이름 → data/<이름>/card.pdf"""
    cand = [pdf, (res.get('source') or {}).get('pdf')]
    st = stem_of(result_path); d = os.path.dirname(os.path.abspath(result_path))
    cand += [os.path.join(d, st + '.pdf'), os.path.join(os.path.dirname(d), st + '.pdf'),
             os.path.join(ROOT, 'data', st, 'card.pdf'), os.path.join(d, '..', 'data', st, 'card.pdf')]
    for c in cand:
        if c and os.path.exists(c): return os.path.abspath(c)
    return None


def plan_images(pdf, res):
    """PDF에서 평면도 이미지(BGR)를 다시 뽑는다(OCR 없이 빠름). 개수가 안 맞으면 엔진 전체 실행(keep=True)."""
    from firebim.card import parse_card, extract_plans
    info, doc = parse_card(pdf)
    raw = extract_plans(doc)
    if info.get('양식') == 'std':                    # 여러 동 표준양식: 결과 JSON 의 선택 동 평면도만
        raw = [r for r in raw if r.get('dong', 0) == ((res.get('card') or {}).get('선택동') or 0)]
    if len(raw) == len(res.get('plans') or []):
        return [r['img'] for r in raw]
    from firebim.pipeline import run
    return [p['img'] for p in run(pdf, keep=True)['_plans']]


def plan_T(res, i):
    p = res['plans'][i]
    return float(p['s']), float(p['tx']), float(p['ty'])


def g2p(T, x, y):
    s, tx, ty = T
    return (x - tx) * s, (ty - y) * s


def p2g(T, px, py):
    s, tx, ty = T
    return px / s + tx, ty - py / s


def plan_floors(res):
    from firebim.card import level_of
    out = {}
    for fid, F in res['floors'].items(): out.setdefault(F['plan'], []).append(fid)
    return {k: sorted(v, key=level_of) for k, v in out.items()}


def img_data_uri(img, maxw=2400, q=85):
    import cv2
    h, w = img.shape[:2]
    if w > maxw: img = cv2.resize(img, (maxw, int(h * maxw / w)), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(b.tobytes()).decode()


def ascii_tmpdir():
    """Windows: OpenCV·Tesseract·IFC 라이브러리는 한글 경로를 못 여는 경우가 있어 ASCII 임시폴더 사용"""
    d = os.environ.get('FIREBIM_TMP') or tempfile.gettempdir()
    try: d.encode('ascii')
    except UnicodeEncodeError:
        d = os.path.join(os.environ.get('PUBLIC', 'C:\\Users\\Public'), 'FireBIM_tmp')
    os.makedirs(d, exist_ok=True)
    return d


def safe_export_ifc(D, BIM, path, accuracy_note=None):
    """IFC는 ASCII 임시 경로에 쓴 뒤 옮긴다(한글 폴더 대비). accuracy_note: 검수 반영 시 '모델정확도' 속성 갱신"""
    from firebim.export import export_ifc
    fd, tmp = tempfile.mkstemp(suffix='.ifc', dir=ascii_tmpdir()); os.close(fd)
    try:
        export_ifc(D, BIM, tmp)
        if accuracy_note:
            try:
                import ifcopenshell, ifcopenshell.api as api
                f = ifcopenshell.open(tmp)
                for ps in f.by_type('IfcPropertySet'):
                    if ps.Name == 'Pset_소방대상물':
                        api.run('pset.edit_pset', f, pset=ps, properties={'모델정확도': accuracy_note})
                f.write(tmp)
            except Exception as e:
                print('  (IFC 검수 표기 생략:', e, ')')
        shutil.move(tmp, path)
        try: os.chmod(path, 0o644)
        except OSError: pass
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return path


def build_outputs(res, imgs, html_path, ifc_path, name_hint=None, note=None):
    """결과 + 도면 이미지 → 3D 지휘화면 HTML + IFC (make_demo.py 와 같은 흐름)"""
    from firebim.export import build, to_viewer_bim
    from build_viewer import make as make_html
    plans = [dict(img=imgs[p['i']], s=p['s'], tx=p['tx'], ty=p['ty'], caption=p['caption']) for p in res['plans']]
    r = dict(res); r['_plans'] = plans
    D, BIM = build(r, plans, name_hint=name_hint)
    if note: D['building']['source'] = D['building'].get('source', '') + ' · ' + note
    safe_export_ifc(D, BIM, ifc_path, accuracy_note=note)
    VB = to_viewer_bim(BIM)
    dj = os.path.splitext(html_path)[0] + '.data.json'
    save_json(dict(D=D, BIM=VB), dj)
    try: make_html(dj, html_path)
    finally: os.unlink(dj)
    if note:      # 뷰어의 '검수 전' 표기를 검수 반영으로(뷰어 코드는 건드리지 않고 출력만 치환)
        with open(html_path, encoding='utf-8') as f: h = f.read()
        h = h.replace('자동 변환 · 검수 전', '자동 변환 · 검수 반영').replace('자동 추출한 값이며 검수 전입니다', '자동 추출한 뒤 검수자가 확인·수정한 값입니다')
        with open(html_path, 'w', encoding='utf-8') as f: f.write(h)
    return D, VB


def slug(s):
    return re.sub(r'[\\/:*?"<>|\s]+', '_', s).strip('_') or 'building'
