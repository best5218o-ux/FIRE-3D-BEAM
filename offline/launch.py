# -*- coding: utf-8 -*-
"""내부 PC(Windows, 인터넷 없음)용 실행기. .bat 파일이 이 스크립트를 부른다(bat 에는 한글 로직을 두지 않음).
  python launch.py check                      설치 확인(라이브러리·AI 모델·Tesseract·kor 언어팩·뷰어 번들)
  python launch.py selftest [--quick]         자체시험: 가상 표준양식 정보카드 1동 변환(AI 모델 교차검증·진압작전도·소방시설·대피로)
                                              + 개인정보·오프라인 검사 + 검수 반영 흐름(--quick: 설치 확인만)
  python launch.py convert <PDF|폴더|도면>... [--rules|--model] [--out 폴더]
                                              변환: 3D 지휘화면 + IFC + 사전 활동계획 + 검수화면
                                              정보카드 PDF 와 함께 놓은 평면도·진압작전도 파일(PDF/PNG/JPG/TIF, 이름에 층)은 그 카드에 붙인다
                                              스캔본(인쇄 후 스캔·그림으로 내보낸, 글자층 없는) 표준양식 카드 PDF 도 OCR 로 자동 해석(300 dpi 스캔 권장)
  python launch.py review [<파일|폴더>]        검수: *_corrections.json → 반영·재생성·학습라벨 / 결과폴더·PDF → 검수화면 열기
  python launch.py score                      채점 화면(가상 건물 자동 채점 + 검수 기반 성능)
모든 출력은 이 PC 안에만 저장된다. 실행 중 인터넷 접속을 하지 않는다(소켓 차단). 실제 정보카드·결과·라벨은 내부 PC 밖으로 반출 금지."""
import glob, os, re, shutil, subprocess, sys, tempfile, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)                                   # 코드 루트(firebim, review, …)
PKG = os.environ.get('FIREBIM_HOME') or APP                   # 설치 폴더(python\, tesseract\, app\ …) — .bat 이 지정
sys.path.insert(0, APP)
IS_WIN = os.name == 'nt'
LABELS = os.path.join(PKG, '학습데이터')
LOGDIR = os.path.join(PKG, 'logs')
SCOREBOARD = os.path.join(PKG, '채점현황.html')
CARD_EXT = ('.pdf',)
DOC_EXT = ('.docx', '.doc', '.hwp', '.hwpx', '.odt')
IMG_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def say(*a): print(*a, flush=True)


def is_ascii(s):
    try: s.encode('ascii'); return True
    except UnicodeEncodeError: return False


def block_network():
    """실행 중 외부 접속 차단(오프라인 원칙을 코드로 보장). 같은 PC 안(127.0.0.1) 연결만 허용. FIREBIM_ALLOW_NET=1 이면 끔(개발용)."""
    if os.environ.get('FIREBIM_ALLOW_NET') == '1': return
    import socket
    LOCAL = ('127.0.0.1', '::1', 'localhost', '')
    def _host(addr): return addr[0] if isinstance(addr, tuple) else addr
    def guard(what, host):
        if isinstance(host, (bytes, bytearray)): host = host.decode('ascii', 'replace')
        if host not in LOCAL and not str(host).startswith('127.'):
            raise OSError(f'[FireBIM] 네트워크 차단: {what} {host} — 이 프로그램은 인터넷을 쓰지 않습니다')
    _connect, _gai = socket.socket.connect, socket.getaddrinfo
    def connect(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6): guard('connect', _host(addr))
        return _connect(self, addr)
    def getaddrinfo(host, *a, **k):
        guard('getaddrinfo', host); return _gai(host, *a, **k)
    socket.socket.connect = connect; socket.getaddrinfo = getaddrinfo


def setup_env():
    """Tesseract 경로·언어팩, ASCII 임시폴더, 오프라인 뷰어 번들"""
    os.environ['FIREBIM_PREBUILT'] = '1'
    os.environ.setdefault('PYTHONUTF8', '1')
    # 임시 폴더: OpenCV·Tesseract 가 한글 경로를 못 여는 경우 대비
    tmp = tempfile.gettempdir()
    if not is_ascii(tmp):
        tmp = os.path.join(os.environ.get('PUBLIC', r'C:\Users\Public'), 'FireBIM_tmp')
    os.makedirs(tmp, exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = os.environ['FIREBIM_TMP'] = tmp; tempfile.tempdir = tmp
    # Tesseract: 동봉(tesseract\) → PATH → 기본 설치 위치
    cands = [os.path.join(PKG, 'tesseract', 'tesseract.exe' if IS_WIN else 'tesseract')]
    w = shutil.which('tesseract');  cands += [w] if w else []
    if IS_WIN:
        for base in (os.environ.get('ProgramFiles', r'C:\Program Files'), os.environ.get('LOCALAPPDATA', ''), r'C:\Program Files (x86)'):
            if base: cands += [os.path.join(base, 'Tesseract-OCR', 'tesseract.exe'), os.path.join(base, 'Programs', 'Tesseract-OCR', 'tesseract.exe')]
    exe = next((c for c in cands if c and os.path.exists(c)), None)
    if exe:
        d = os.path.dirname(os.path.abspath(exe))
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
        td = os.path.join(d, 'tessdata')
        if os.path.isdir(td):
            if not is_ascii(td):          # Tesseract 는 한글 경로의 언어팩을 못 읽음 → ASCII 위치로 복사
                dst = os.path.join(tmp, 'tessdata')
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(td):
                    if f.endswith('.traineddata') and not os.path.exists(os.path.join(dst, f)): shutil.copy2(os.path.join(td, f), dst)
                td = dst
            os.environ['TESSDATA_PREFIX'] = td
    return exe


def open_path(p):
    if os.environ.get('FIREBIM_NO_OPEN') == '1': return
    if IS_WIN:
        try: os.startfile(p)          # 탐색기/기본 브라우저
        except OSError as e: say('  (자동으로 열지 못함:', e, ')')
    else:
        say('  열기:', p)


def log_error(where):
    """오류 기록: 파이썬 추적 정보만(파일 경로의 사용자 폴더·PDF 이름은 가림) — 개인정보·카드 내용은 남기지 않음"""
    os.makedirs(LOGDIR, exist_ok=True)
    fn = os.path.join(LOGDIR, time.strftime('%Y%m%d_%H%M%S') + f'_{where}.log')
    tb = traceback.format_exc()
    home = os.path.expanduser('~')
    if home and len(home) > 3: tb = tb.replace(home, '%USERPROFILE%')
    tb = re.sub(r"[^\\/'\"\s]+\.(?:pdf|PDF|png|jpg|jpeg|tif|tiff|docx|hwp|hwpx)", '<입력파일>', tb)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(f'FireBIM {where} 오류 · Python {sys.version.split()[0]} · {sys.platform}\n{tb}')
    say(f'  ✗ 오류: {type(sys.exc_info()[1]).__name__}\n    자세한 내용(개인정보 없음): {fn}')


def soffice_exe():
    """(선택) LibreOffice — 정보카드가 docx/hwp 일 때 PDF 로 바꾸는 데만 사용. 없어도 변환 기능은 모두 동작."""
    c = [shutil.which('soffice'), shutil.which('soffice.exe')]
    if IS_WIN:
        for base in (os.environ.get('ProgramFiles', r'C:\Program Files'), r'C:\Program Files (x86)', os.environ.get('LOCALAPPDATA', '')):
            if base: c.append(os.path.join(base, 'LibreOffice', 'program', 'soffice.exe'))
    return next((x for x in c if x and os.path.exists(x)), None)


# ───────── 명령 ─────────
LIBS = (('pymupdf', 'PyMuPDF', True), ('cv2', 'OpenCV', True), ('numpy', 'NumPy', True), ('shapely', 'Shapely', True), ('PIL', 'Pillow', True),
        ('ifcopenshell', 'IfcOpenShell', True), ('onnxruntime', 'onnxruntime(AI 모델)', False), ('skimage', 'scikit-image(대피로)', False))


def cmd_check():
    exe = setup_env(); ok = True
    say('■ 설치 확인'); say(f'  Python {sys.version.split()[0]} ({sys.executable})')
    for mod, nm, need in LIBS:
        try:
            m = __import__(mod); say(f'  ✓ {nm} {getattr(m, "__version__", getattr(m, "version", ""))}')
        except Exception as e:
            ok = False; say(f'  ✗ {nm} 불러오기 실패: {e!r}')
            if mod == 'cv2' and 'DLL' in repr(e): say('    → Windows N/KN 에디션이면 "미디어 기능 팩" 설치 필요(설치안내.md 참고)')
            if 'DLL' in repr(e): say('    → Visual C++ 재배포 패키지(msvcp140.dll) 문제일 수 있음: install.bat 를 다시 실행하세요')
    for f, what in (('str_seg.onnx', '벽·창·문 분할 모델(교차검증)'), ('str_seg.json', '분할 모델 설정'), ('spa_seg.onnx', '실·계단·승강기 분할 모델(실제 도면)')):
        p = os.path.join(APP, 'models', f); e = os.path.exists(p); ok &= e
        say(f"  {'✓' if e else '✗'} {what}: models/{f}" + (f' ({os.path.getsize(p) / 1e6:.1f}MB)' if e else ' 없음'))
    tpl = os.path.join(APP, 'templates', '소방활동정보카드_표준양식.docx')
    say(f"  {'✓' if os.path.exists(tpl) else '·'} 정보카드 표준양식(templates/소방활동정보카드_표준양식.docx) {'' if os.path.exists(tpl) else '없음(참고용 — 변환에는 불필요)'}")
    if exe:
        try:
            out = subprocess.run([exe, '--list-langs'], capture_output=True, text=True, timeout=30)
            langs = (out.stdout + out.stderr).split()
            say(f'  ✓ Tesseract: {exe}'); say(f"  {'✓' if 'kor' in langs else '✗'} 한국어 언어팩(kor) {'있음' if 'kor' in langs else '없음 → tessdata 폴더에 kor.traineddata 복사'}")
            say(f"  {'✓' if 'eng' in langs else '✗'} 영어 언어팩(eng)")
            ok &= 'kor' in langs and 'eng' in langs
        except Exception as e:
            ok = False; say(f'  ✗ Tesseract 실행 실패: {e!r}')
    else:
        ok = False; say('  ✗ Tesseract 없음 → 설치안내.md 의 "Tesseract" 절 참고')
    js = os.path.join(APP, 'viewer', 'dist', 'app.js')
    say(f"  {'✓' if os.path.exists(js) else '✗'} 3D 뷰어 번들(viewer/dist/app.js)"); ok &= os.path.exists(js)
    so = soffice_exe()
    say(f"  · (선택) LibreOffice: {so if so else '없음 — 정보카드는 PDF로 넣으면 됨(docx/hwp 는 한글·Word 에서 PDF로 저장)'}")
    if not is_ascii(PKG): say(f'  ! 설치 폴더에 한글/특수문자가 있음({PKG}) — 동작은 하도록 했지만 C:\\FireBIM 같은 영문 경로를 권장')
    say('■ 결과:', '정상' if ok else '문제 있음'); return 0 if ok else 1


def cmd_setup_tesseract():
    """install.bat: 동봉 tesseract\ 확인 → 없으면 installers\ 의 설치파일을 조용히 실행 → 언어팩(kor, eng) 복사"""
    dst = os.path.join(PKG, 'tesseract'); inst = os.path.join(PKG, 'installers')
    if not os.path.exists(os.path.join(dst, 'tesseract.exe')):
        exe = sorted(glob.glob(os.path.join(inst, 'tesseract-ocr-w64-setup-*.exe')))
        if exe and IS_WIN:
            say('  Tesseract 설치 프로그램 실행(관리자 권한 창이 뜰 수 있음)...')
            subprocess.run(f'"{exe[-1]}" /S /D={dst}', shell=False)        # NSIS: /D 는 따옴표 없이 마지막에
        elif not exe:
            say('  동봉 Tesseract 없음 — PC에 설치된 Tesseract(C:\\Program Files\\Tesseract-OCR)를 찾아 사용합니다')
    exe = setup_env()
    if exe:
        td = os.path.join(os.path.dirname(exe), 'tessdata')
        for L in ('kor', 'eng'):
            src = os.path.join(inst, 'tessdata', L + '.traineddata')
            if os.path.exists(src) and not os.path.exists(os.path.join(td, L + '.traineddata')):
                try: os.makedirs(td, exist_ok=True); shutil.copy2(src, td); say(f'  언어팩 복사: {L}')
                except OSError as e: say(f'  ✗ 언어팩 복사 실패({e}) — {src} 를 {td} 에 직접 복사하세요')
        say('  Tesseract:', exe); return 0
    say('  ✗ Tesseract 를 찾지 못함 — 설치안내.md 「Tesseract 직접 설치」 참고'); return 1


# ───────── 자체시험 ─────────
NET_PAT = re.compile(r'''(?:src|href)\s*=\s*["']\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//|@import\s+["']?(?:https?:)?//|\bfetch\s*\(|XMLHttpRequest|new\s+WebSocket|sendBeacon''', re.I)


def offline_scan(paths):
    """결과 HTML 이 인터넷 자원을 부르지 않는지(외부 스크립트·글꼴·CDN·fetch) → 걸린 파일 목록"""
    bad = []
    for p in paths:
        with open(p, encoding='utf-8', errors='replace') as f: t = f.read()
        t = re.sub(r'data:[a-z/+.-]+;base64,[A-Za-z0-9+/=]+', '', t)
        m = NET_PAT.findall(t)
        if m: bad.append((os.path.basename(p), sorted(set(m))[:3]))
    return bad


def synthetic_corrections(result_path, pdf, out_path):
    """검수화면의 「검수 결과 저장」과 같은 형식(review/SCHEMA.md)으로 판정을 자동 생성 — 브라우저 없이 검수 반영 흐름 시험.
    첫 도면 '검수 완료', 모든 기호 '맞음', 교차검증(규칙↔AI) 항목 앞 4개에 모델/규칙 판정, 첫 교차검증 플래그 '불일치 맞음'"""
    import datetime, json
    from review.common import load_json, plan_images
    from review.make import build_data
    res = load_json(result_path)
    D = build_data(res, result_path, pdf, plan_images(pdf, res))
    now = datetime.datetime.now().isoformat(timespec='seconds')
    xit = [it for it in D['items'] if it['kind'] in ('xwall', 'xdoor')][:4]
    C = dict(schema=D['schema'], tool='launch-selftest', created=now, updated=now, building=D['building'], source=D['source'],
             summary=dict(items_total=len(D['items']), items_done=len(xit), plans_total=len(D['plans']), plans_done=1, auto_total=D['stats']['n_auto'],
                          xcheck_total=sum(1 for it in D['items'] if it['kind'] in ('xwall', 'xdoor')), xcheck_done=len(xit), vertical_added=0),
             plans={}, flags=[], vertical=[], vertical_added=[], xcheck=[], items=[], ops=[])
    for j, p in enumerate(D['plans']):
        C['plans'][str(p['i'])] = dict(floors=[f['id'] for f in p['floors']], caption=p['caption'], img_w=p['w'], img_h=p['h'], s=p['s'],
                                       status=dict(align='ok', scale='ok', done=(j == 0)), shift_m=[0, 0],
                                       symbols=[dict(q, src='auto', verified=True, orig=None) for q in p['symbols']], symbols_deleted=[],
                                       doors=[dict(d, src='auto') for d in p['doors']], doors_deleted=[], walls_deleted=[], walls_added=[])
    for k, it in enumerate(xit):
        dec = 'model' if k % 2 == 0 else 'rule'
        e = dict(id=it['id'], kind=it['kind'], sub=it.get('sub'), plan=it['plan'], p=it['p'], seg=it.get('seg'), loc=it.get('loc'), o=it.get('o'), ref=it.get('ref'), decision=dec)
        e.update(dict(tpx=it.get('tpx'), filled=bool(it.get('filled')), bridge=bool(it.get('bridge'))) if it['kind'] == 'xwall' else dict(wpx=it.get('wpx'), ext=bool(it.get('ext'))))
        C['xcheck'].append(e)
    xf = [f for f in D['flags'] if f['kind'] == 'xcheck_wall'][:1]
    C['flags'] = [dict(key=f['key'], kind=f['kind'], floor=f['floor'], decision='confirm') for f in xf]
    C['items'] = [dict(id=it['id'], kind=it['kind'], plan=it.get('plan'), ref=it.get('ref'), p=it['p'],
                       decision=next((x['decision'] for x in C['xcheck'] if x['id'] == it['id']), None)) for it in D['items']]
    with open(out_path, 'w', encoding='utf-8') as f: json.dump(C, f, ensure_ascii=False, indent=1)
    return len(xit)


def sample_cards():
    """자체시험용 가상 정보카드: samples/(표준양식·진압작전도 포함) → 없으면 data/(기존 배치)"""
    return sorted(glob.glob(os.path.join(APP, 'samples', '*', 'card.pdf'))) or sorted(glob.glob(os.path.join(APP, 'data', '*', 'card.pdf')))


def cmd_selftest(args=()):
    global LABELS, SCOREBOARD
    if cmd_check(): return 1
    if '--quick' in args: return 0
    from review.convert import convert, model_available
    pdfs = sample_cards()
    if not pdfs: say('가상 정보카드(samples/ 또는 data/)가 없어 변환 시험 생략'); return 0
    out = os.path.join(PKG, '결과', '_자체시험')
    shutil.rmtree(out, ignore_errors=True)
    use_model, why = model_available()
    say(f'\n■ 변환 시험(가상 표준양식 정보카드): {os.path.basename(os.path.dirname(pdfs[0]))} · AI 모델 {"사용" if use_model else "없음(" + why + ")"}')
    R = []                                       # (항목, 통과, 설명)
    def chk(name, ok, note=''): R.append((name, bool(ok), note)); say(f"  {'✓' if ok else '✗'} {name}" + (f' — {note}' if note else ''))
    t0 = time.time()
    try:
        d, P = convert(pdfs[0], out, log=lambda *a: say('   ', *a), model=use_model)
    except Exception:
        log_error('selftest'); return 1
    say(f'■ 결과 확인 ({time.time() - t0:.0f}초)')
    import json
    files = {k: p for k, p in P.items() if not k.startswith('_')}
    miss = [k for k, p in files.items() if not os.path.exists(p)]
    chk('결과 파일 5종(3D·IFC·활동계획·검수화면·JSON)', not miss, f'없음 {miss}' if miss else '')
    if miss: return 1
    res = json.load(open(files['result'], encoding='utf-8'))
    card = res.get('card') or {}
    chk('정보카드 표준양식 해석', card.get('양식') == 'std' or 'samples' not in pdfs[0], f"양식={card.get('양식')} · 층별현황 {len(card.get('층별현황') or [])}행")
    figs = [f['kind'] for f in res.get('figures') or []]
    site = res.get('site') if isinstance(res.get('site'), dict) else {}
    if '진압작전도' in figs: chk('진압작전도 인식 → 외부 부서(기능 24)', site.get('src') == '진압작전도', f"부지 출처={site.get('src') or site.get('error')} · 면 {len(site.get('facades') or [])} · 사다리차 후보 {len(site.get('aerial') or [])}")
    nf = sum(len(v) for v in ((res.get('fac') or {}).get('floors') or {}).values())
    chk('소방시설 레이어(분류·위치 확실성)', nf > 0, f'{nf}항목')
    ev = res.get('evac') or {}
    chk('대피로·진입로', 'floors' in ev and not ev.get('error'), ev.get('error') or f"진입구 {len(ev.get('entry') or [])} · 층 {len(ev.get('floors') or {})}")
    if use_model:
        xe = [k for k, v in (res.get('xcheck') or {}).items() if isinstance(v, dict) and v.get('error')]
        chk('AI 모델 교차검증(규칙 ↔ STR 모델)', res.get('xcheck') and not xe, f"도면 {len(res.get('xcheck') or {})}장" + (f' 실패 {xe}' if xe else ''))
    try:                                         # 인명검색 우선순위(기준값 sop.py SEARCH → 지휘화면) · 사전 활동계획 5-1절
        ph = open(files['plan'], encoding='utf-8').read(); vh = open(files['html'], encoding='utf-8').read()
        ok5 = '5-1. 인명검색 우선 실' in ph
        n5 = ph.split('5-1. 인명검색 우선 실')[1].split('</table>')[0].count('<tr>') - 1 if ok5 else 0
        chk('인명검색 우선순위(지휘화면 기준값·활동계획 5-1절)', '"sop_search"' in vh and '"sop_train"' in vh and ok5, f'활동계획 우선 실 {n5}행')
    except Exception as e:
        chk('인명검색 우선순위(지휘화면 기준값·활동계획 5-1절)', False, repr(e))
    chk('개인정보(관계인 성명·연락처) 결과에 없음', not P.get('_privacy_hits'), str(P.get('_privacy_hits') or ''))
    bad = offline_scan([files['html'], files['plan'], files['review']])
    chk('결과 화면 오프라인(외부 스크립트·글꼴·CDN 없음)', not bad, str(bad) if bad else '')
    try:
        import ifcopenshell
        f = ifcopenshell.open(files['ifc'])
        n = {t: len(f.by_type(t)) for t in ('IfcBuildingStorey', 'IfcWall', 'IfcDoor', 'IfcStair', 'IfcSpace', 'IfcFireSuppressionTerminal')}
        chk('IFC4 열기', f.schema == 'IFC4' and n['IfcBuildingStorey'] > 0, ' · '.join(f'{k[3:]} {v}' for k, v in n.items()))
    except Exception as e:
        chk('IFC4 열기', False, repr(e))
    # 검수 흐름: 판정 파일 → 검수.bat 와 같은 경로(cmd_review) → 3D·IFC·활동계획 재생성 + 학습데이터
    say('■ 검수 반영 시험(검수화면 저장 파일을 자동으로 만들어 검수.bat 경로로 반영)')
    try:
        cp = os.path.join(d, os.path.basename(files['result']).replace('_결과.json', '_corrections.json'))
        nx = synthetic_corrections(files['result'], res['source']['pdf'], cp)
        LABELS, keep, no_open = os.path.join(out, '_학습데이터_시험'), LABELS, os.environ.get('FIREBIM_NO_OPEN')
        SCOREBOARD, keep_sb = os.path.join(out, '채점현황_시험.html'), SCOREBOARD          # 실제 채점현황에 시험 검수가 섞이지 않게
        os.environ['FIREBIM_NO_OPEN'] = '1'                  # 시험 중에는 브라우저 창을 띄우지 않음
        try: rc = cmd_review([cp])
        finally:
            LABELS, SCOREBOARD = keep, keep_sb
            if no_open is None: os.environ.pop('FIREBIM_NO_OPEN', None)
            else: os.environ['FIREBIM_NO_OPEN'] = no_open
        st = os.path.basename(files['result']).replace('_결과.json', '')
        outs = [os.path.join(d, st + '_검수반영' + x) for x in ('.html', '.ifc', '_활동계획.html', '.json')]
        lab = glob.glob(os.path.join(out, '_학습데이터_시험', '**', '*.json'), recursive=True)
        chk('검수 반영 → 3D·IFC·활동계획 재생성 + 학습 라벨', rc == 0 and all(os.path.exists(x) for x in outs) and lab, f'교차검증 판정 {nx}건 · 라벨 파일 {len(lab)}개')
    except Exception:
        log_error('selftest_review'); chk('검수 반영', False, 'logs 폴더 참고')
    ok = all(r[1] for r in R)
    say(f"\n■ 자체시험: {'정상' if ok else '문제 있음'} ({sum(r[1] for r in R)}/{len(R)}) — 결과 폴더: {out}")
    return 0 if ok else 1


# ───────── 변환 ─────────
def doc_to_pdf(p, log=say):
    """docx/hwp 정보카드 → PDF (LibreOffice 가 있을 때만). 없으면 None + 안내"""
    so = soffice_exe()
    if not so:
        log(f'  ✗ {os.path.basename(p)}: 정보카드가 {os.path.splitext(p)[1]} 파일입니다. 한글·Word 에서 「PDF로 저장」한 뒤 PDF를 넣으세요'
            ' (이 PC에 LibreOffice 가 없어 자동 변환 불가 — 설치 불필요, PDF 로 넣으면 모든 기능 동작)')
        return None
    outdir = tempfile.mkdtemp(prefix='fb_doc_')
    try:
        subprocess.run([so, '--headless', '--norestore', '--convert-to', 'pdf', '--outdir', outdir, p], check=True, timeout=300, capture_output=True)
        q = glob.glob(os.path.join(outdir, '*.pdf'))
        if not q: raise RuntimeError('PDF 가 만들어지지 않음')
        dst = os.path.splitext(p)[0] + '.pdf'
        if os.path.exists(dst): dst = os.path.splitext(p)[0] + '_변환.pdf'
        shutil.move(q[0], dst); log(f'  문서 → PDF(LibreOffice): {os.path.basename(dst)}'); return dst
    except Exception as e:
        log(f'  ✗ {os.path.basename(p)} PDF 변환 실패({e!r}) — 한글·Word 에서 PDF로 저장해 넣으세요'); return None
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def gather(args):
    """끌어다 놓은 것들 → [(카드 PDF, [별도 도면 파일])]. 폴더는 안의 파일 전부. 도면은 같은 폴더의 카드(카드가 하나뿐이면 그 카드)에 붙임"""
    from review.attach import classify
    files = []
    for a in args:
        a = a.strip('"')
        if os.path.isdir(a):
            files += sorted(os.path.join(a, f) for f in os.listdir(a) if f.lower().endswith(CARD_EXT + DOC_EXT + IMG_EXT))
        elif os.path.exists(a): files.append(a)
        else: say('  건너뜀(없음):', a)
    cards, extras = [], []
    for p in files:
        low = p.lower()
        if os.path.basename(low).startswith('~$'): continue
        if low.endswith(DOC_EXT):
            q = doc_to_pdf(p)
            if q: cards.append(q)
            continue
        if not low.endswith(CARD_EXT + IMG_EXT): say('  건너뜀(PDF·그림 아님):', p); continue
        if re.search(r'_(변환결과|입력합본)', p): continue
        k = classify(p)
        if k is None: say(f'  도면 아님(건축물대장·부지·일람도·사진 — 대장·일람도는 카드와 같은 폴더에 있으면 보조 입력으로 자동 사용): {os.path.basename(p)}'); continue
        (cards if k == 'card' else extras).append(p)
    groups = [(c, []) for c in cards]
    for x in extras:
        same = [g for g in groups if os.path.dirname(os.path.abspath(g[0])) == os.path.dirname(os.path.abspath(x))]
        tgt = groups if len(groups) == 1 else same
        if len(tgt) == 1: tgt[0][1].append(x)
        else: say(f'  건너뜀: {os.path.basename(x)} — 어느 정보카드의 도면인지 알 수 없음(카드와 같은 폴더에 두거나 카드 하나와 함께 놓기)')
    return groups


def cmd_convert(args):
    setup_env()
    from review.convert import convert
    model = None
    if '--rules' in args or os.environ.get('FIREBIM_MODEL', '').lower() in ('0', 'false', 'off', 'no'): model = False
    if '--model' in args: model = True
    out = None
    if '--out' in args:
        i = args.index('--out'); out = args[i + 1] if i + 1 < len(args) else None; args = args[:i] + args[i + 2:]
    groups = gather([a for a in args if a not in ('--rules', '--model')])
    if not groups:
        say('정보카드 PDF 파일을 「변환.bat」 아이콘 위에 끌어다 놓으세요(여러 개·폴더 가능).\n'
            '  평면도·진압작전도를 따로 가지고 있으면 카드와 함께 놓으세요(파일 이름에 층: 지상1층.png, B1.pdf, 진압작전도.jpg).'); return 1
    done, fail = [], 0
    for i, (p, ex) in enumerate(groups, 1):
        say(f'\n━━ ({i}/{len(groups)}) {os.path.basename(p)}' + (f' + 별도 도면 {len(ex)}개' if ex else ''))
        try: done.append(convert(p, out=(os.path.join(out, os.path.splitext(os.path.basename(p))[0]) if out and len(groups) > 1 else out), log=say, model=model, extras=ex))
        except Exception: fail += 1; log_error('convert')
    say(f'\n■ 변환 완료 {len(done)}건' + (f', 실패 {fail}건' if fail else ''))
    for d, P in done:
        say(f'  {d}\n    - 3D 지휘화면: {os.path.basename(P["html"])}\n    - 사전 활동계획: {os.path.basename(P["plan"])}\n'
            f'    - 검수화면:   {os.path.basename(P["review"])}\n    - IFC:        {os.path.basename(P["ifc"])}')
    if len(done) == 1: open_path(done[0][1]['html']); open_path(done[0][1]['plan']); open_path(done[0][0])
    elif done: open_path(done[0][0])
    say('\n※ 실제 정보카드와 변환 결과는 내부 PC 밖으로 반출 금지.')
    return 1 if fail else 0


def find_result_for(corr_path, corr):
    from review.common import stem_of
    rp = (corr.get('source') or {}).get('result_json')
    if rp and os.path.exists(rp): return rp
    bid = (corr.get('building') or {}).get('id') or stem_of(corr_path).replace('_corrections', '')
    roots = [os.path.dirname(os.path.abspath(corr_path)), os.path.join(PKG, '결과'), os.path.expanduser('~')]
    for r in roots:
        for pat in (f'{bid}_결과.json', f'*/{bid}_결과.json', f'{bid}.json'):
            hits = glob.glob(os.path.join(r, pat))
            if hits: return hits[0]
    return None


def cmd_review(args):
    setup_env()
    from review.common import load_json
    if not args:
        say('사용법: 검수화면에서 「검수 결과 저장」으로 받은 *_corrections.json 파일을 「검수.bat」 위에 끌어다 놓으세요.\n'
            '        (변환 결과 폴더나 PDF를 놓으면 검수화면을 엽니다)')
        return 1
    rc = 0
    for a in (x.strip('"') for x in args):
        try:
            if a.lower().endswith('.json') and 'correction' in os.path.basename(a).lower():
                from review.apply import main as apply_main
                corr = load_json(a); rp = find_result_for(a, corr)
                if not rp: say('  ✗ 이 검수 파일의 변환 결과(*_결과.json)를 찾을 수 없음 — 결과 폴더에 검수 파일을 옮긴 뒤 다시 시도'); rc = 1; continue
                from review.common import stem_of
                st = stem_of(rp); d = os.path.dirname(os.path.abspath(rp)); outstem = os.path.join(d, st + '_검수반영')
                say(f'■ 검수 반영: {os.path.basename(a)} → {os.path.basename(rp)}')
                apply_main(rp, a, out_stem=outstem, labels=LABELS)
                rec = os.path.join(d, st + '_검수기록.json')
                if os.path.abspath(a) != os.path.abspath(rec): shutil.copy2(a, rec)
                from review.scoreboard import make as sb
                sb(os.path.join(APP, 'out', 'score_all.json'), os.path.join(APP, 'out', 'summary.json'), os.path.join(LABELS, 'review_metrics.json'), SCOREBOARD)
                say(f'  - 수정된 3D 지휘화면: {st}_검수반영.html\n  - 수정된 IFC: {st}_검수반영.ifc\n  - 수정된 사전 활동계획: {st}_검수반영_활동계획.html\n  - 학습데이터 누적: {LABELS}')
                open_path(outstem + '.html'); open_path(d)
            elif a.lower().endswith('.pdf'):
                rc |= cmd_convert([a])
            elif os.path.isdir(a):
                h = glob.glob(os.path.join(a, '*_검수화면.html'))
                if h: open_path(h[0])
                else: say('  이 폴더에 검수화면이 없음:', a); rc = 1
            elif a.lower().endswith('.json'):
                from review.make import make
                from review.common import stem_of
                h = make(a, out_html=os.path.join(os.path.dirname(os.path.abspath(a)), stem_of(a) + '_검수화면.html')); open_path(h)
            elif a.lower().endswith('.html'):
                open_path(a)
            else:
                say('  알 수 없는 파일:', a); rc = 1
        except Exception:
            log_error('review'); rc = 1
    return rc


def cmd_score(args):
    setup_env()
    n = len(glob.glob(os.path.join(APP, 'data', '*', 'card.pdf')))
    if n >= 3 and '--no-run' not in args:
        say(f'■ 가상 건물 {n}동 자동 변환·채점 (수 분 걸림)')
        env = dict(os.environ); env.pop('FIREBIM_MODEL', None)
        env.setdefault('WORKERS', str(max(1, min(4, (os.cpu_count() or 2) // 2))))       # 사무용 PC 메모리 보호
        r = subprocess.run([sys.executable, '-X', 'utf8', 'run_all.py', str(n // 3)], cwd=APP, env=env)
        if r.returncode: say('  ✗ 채점 실행 오류'); return 1
    from review.scoreboard import make as sb
    out = sb(os.path.join(APP, 'out', 'score_all.json'), os.path.join(APP, 'out', 'summary.json'), os.path.join(LABELS, 'review_metrics.json'), os.path.join(PKG, '채점현황.html'))
    open_path(out); return 0


def main(argv):
    if not argv: say(__doc__); return 1
    block_network()
    c, rest = argv[0], argv[1:]
    cmds = {'check': lambda: cmd_check(), 'selftest': lambda: cmd_selftest(rest), 'setup-tesseract': lambda: cmd_setup_tesseract(), 'convert': lambda: cmd_convert(rest),
            'review': lambda: cmd_review(rest), 'score': lambda: cmd_score(rest)}
    if c not in cmds: say(__doc__); return 1
    try:
        return cmds[c]()
    except Exception:
        log_error(c); return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
