# -*- coding: utf-8 -*-
"""build_bundle.sh 도우미(개발 PC 전용 — 설치판에는 넣지 않음).
  python bundle_tools.py app SRC DST WITH_SYNTH WITH_V3 SAMPLE      프로그램(app/) 복사 — 실행에 필요한 것만 + 금지 파일 검사
  python bundle_tools.py pth EMBED_ZIP DST PYTAG                   embeddable Python 풀기 + ._pth(site-packages, ..\\app)
  python bundle_tools.py report RESOLVE_JSON                       pip --dry-run 해결 결과 요약
  python bundle_tools.py tess-prune TESS_DIR                       휴대용 tesseract\\ 에서 쓰지 않는 exe·DLL·문서 제거(pefile)
  python bundle_tools.py manifest OUT PYFULL TESS_VER              MANIFEST.txt(구성·휠 목록)
  python bundle_tools.py zip OUT ZIPFILE                           UTF-8 파일명 zip(Windows 탐색기에서 한글 안 깨짐)
  python bundle_tools.py imports APP WHEELHOUSE                    실행 코드 import ↔ wheelhouse 휠 대조(빠진 라이브러리 검사)"""
import datetime, glob, os, re, shutil, sys, zipfile

RUNTIME_DIRS = ('firebim', 'review')
EVALR = ('__init__.py', 'score.py', 'card_eval.py', 'evac_score.py', 'xcheck_eval.py')        # 채점.bat(run_all.py) 에 필요한 것만
MODELS = ['str_seg.onnx', 'str_seg.json', 'spa_seg.onnx', 'spa_seg.json', 'spa_seg_v3.onnx', 'spa_seg_v3.json']   # STR = GPU 모델, SPA v3.1(+계단 전용 v3)
MODELS_V3 = ['str_seg_v3.onnx', 'str_seg_v3.json']
FORBID_DIR = ('aihub', 'real_cards', 'data_hold', 'data_std_hold', 'strseg', 'synth', 'spec', 'colab', 'out_')
FORBID_EXT = ('.pt', '.pth', '.ckpt', '.PNG', '.npz', '.npy')


def app(src, dst, synth, v3, sample):
    ign = shutil.ignore_patterns('__pycache__', '*.pyc', 'node_modules', 'build', '*.part')
    for d in RUNTIME_DIRS: shutil.copytree(os.path.join(src, d), os.path.join(dst, d), ignore=ign)
    os.makedirs(os.path.join(dst, 'evalr'))
    for f in EVALR: shutil.copy2(os.path.join(src, 'evalr', f), os.path.join(dst, 'evalr', f))
    os.makedirs(os.path.join(dst, 'models'))
    mods = MODELS + (MODELS_V3 if v3 else [])
    for f in mods: shutil.copy2(os.path.join(src, 'models', f), os.path.join(dst, 'models', f))
    os.makedirs(os.path.join(dst, 'templates'))
    for f in glob.glob(os.path.join(src, 'templates', '*.docx')): shutil.copy2(f, os.path.join(dst, 'templates'))
    os.makedirs(os.path.join(dst, 'offline'))
    for f in ('launch.py', 'bootstrap_pip.py', 'requirements-win.txt'): shutil.copy2(os.path.join(src, 'offline', f), os.path.join(dst, 'offline', f))
    for f in ('build_viewer.py', 'make_demo.py', 'run_all.py'): shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    os.makedirs(os.path.join(dst, 'viewer', 'src')); os.makedirs(os.path.join(dst, 'viewer', 'dist'))
    shutil.copy2(os.path.join(src, 'viewer', 'src', 'template.html'), os.path.join(dst, 'viewer', 'src'))
    shutil.copy2(os.path.join(src, 'viewer', 'dist', 'app.js'), os.path.join(dst, 'viewer', 'dist'))
    os.makedirs(os.path.join(dst, 'out'))
    for f in ('score_all.json', 'summary.json'):          # 가상 채점 기준값(비교용)
        if os.path.exists(os.path.join(src, 'out', f)): shutil.copy2(os.path.join(src, 'out', f), os.path.join(dst, 'out', f))
    # 자체시험 표본: 가상 표준양식 카드(진압작전도 포함) + 대장·부지 (정답 gt.json 제외)
    sd = os.path.join(src, sample); sn = os.path.basename(sample.rstrip('/'))
    if not os.path.exists(os.path.join(sd, 'card.pdf')): sys.exit(f'자체시험 표본 없음: {sd}/card.pdf (SAMPLE=… 지정)')
    os.makedirs(os.path.join(dst, 'samples', sn))
    for f in ('card.pdf', 'daejang.json', 'site.json'):
        if os.path.exists(os.path.join(sd, f)): shutil.copy2(os.path.join(sd, f), os.path.join(dst, 'samples', sn, f))
    if synth and os.path.isdir(os.path.join(src, 'data')):
        shutil.copytree(os.path.join(src, 'data'), os.path.join(dst, 'data'), ignore=shutil.ignore_patterns('*.html', '*.png', '__pycache__'))
    bad = []
    for r, _, fs in os.walk(dst):
        rel = os.path.relpath(r, dst)
        for f in fs:
            p = os.path.join(r, f)
            if f.endswith(FORBID_EXT) or any(k in rel.split(os.sep) or rel.startswith(k) for k in FORBID_DIR): bad.append(os.path.relpath(p, dst))
            elif f.endswith('.py') and re.search(r'^\s*(import torch|from torch)', open(p, encoding='utf-8').read(), re.M): bad.append(os.path.relpath(p, dst) + '(torch)')
    if bad: sys.exit(f'설치판에 들어가면 안 되는 파일: {bad[:10]}')
    print('  app 파일', sum(len(fs) for _, _, fs in os.walk(dst)), '개 · 모델', ', '.join(mods), '· 표본', sn, '· 채점 세트', 'data/' if synth else '없음')


def pth(z, dst, tag):
    zipfile.ZipFile(z).extractall(dst)
    p = os.path.join(dst, f'python{tag}._pth')
    # site-packages + 코드 폴더(..\app) 를 경로에 넣고 site 활성화 (pip 설치 가능하게)
    open(p, 'w', newline='\r\n').write(f'python{tag}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n')
    print('  ._pth:', ' | '.join(open(p).read().split()))


def report(js):
    import json
    r = json.load(open(js))
    names = sorted(f"{i['metadata']['name']}=={i['metadata']['version']}" for i in r['install'])
    print('  해결됨', len(names), '개:', ', '.join(names))


def tess_prune(d):
    os.chdir(d)
    for f in os.listdir('.'):
        if (f.endswith('.exe') and f != 'tesseract.exe') or f.endswith('.html') or f == 'doc': (shutil.rmtree if os.path.isdir(f) else os.remove)(f)
    for f in os.listdir('tessdata'):
        if f.endswith('.jar'): os.remove(os.path.join('tessdata', f))
    try:
        import pefile
    except ImportError:
        print('  (pefile 없음 — DLL 정리 생략: pip install pefile 하면 약 100MB 줄어듦)'); return
    have = {f.lower(): f for f in os.listdir('.')}; need, todo = set(), ['tesseract.exe']
    while todo:
        f = todo.pop()
        if f in need: continue
        need.add(f); pe = pefile.PE(have[f], fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY[k] for k in ('IMAGE_DIRECTORY_ENTRY_IMPORT', 'IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT')])
        for e in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []) + getattr(pe, 'DIRECTORY_ENTRY_DELAY_IMPORT', []):
            n = e.dll.decode().lower()
            if n in have: todo.append(n)
    drop = [have[f] for f in have if f.endswith('.dll') and f not in need]
    for f in drop: os.remove(f)
    print(f'  DLL {len(need) - 1}개 유지, {len(drop)}개 제거')


def manifest(out, pyfull, tess):
    wh = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, 'wheelhouse', '*.whl')))
    te = os.path.exists(os.path.join(out, 'tesseract', 'tesseract.exe'))
    L = [f'FireBIM 오프라인 설치판 구성 — 만든 날 {datetime.date.today().isoformat()}',
         f'Python {pyfull} (embeddable, Windows x64)' if os.path.isdir(os.path.join(out, 'python')) else 'Python: 동봉 안 됨(PC 의 Python 3.11 사용)',
         f"Tesseract {tess} (휴대용)" if te else 'Tesseract: 동봉 안 됨(installers/ 또는 PC 설치본)', 'Tesseract 언어팩: tessdata_fast kor, eng',
         'AI 모델: ' + ', '.join(sorted(os.listdir(os.path.join(out, 'app', 'models')))),
         '자체시험 표본: ' + ', '.join(sorted(os.listdir(os.path.join(out, 'app', 'samples')))),
         f'휠 {len(wh)}개(wheelhouse, win_amd64 · cp311):'] + ['  ' + w for w in wh]
    with open(os.path.join(out, 'MANIFEST.txt'), 'w', encoding='utf-8', newline='\r\n') as f: f.write('\n'.join(L) + '\n')
    print('  MANIFEST.txt', len(L), '줄')


def mkzip(src, z):
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as f:
        for r, _, fs in os.walk(src):
            for n in sorted(fs):
                p = os.path.join(r, n); f.write(p, os.path.relpath(p, os.path.dirname(src)))
    print('  zip', f'{os.path.getsize(z) / 1e6:.0f}MB', z)


# 실행 코드 import 이름 → 배포 이름(휠)
DIST = {'cv2': 'opencv_python_headless', 'PIL': 'pillow', 'pymupdf': 'pymupdf', 'fitz': 'pymupdf', 'skimage': 'scikit_image', 'numpy': 'numpy',
        'shapely': 'shapely', 'ifcopenshell': 'ifcopenshell', 'onnxruntime': 'onnxruntime', 'scipy': 'scipy'}


def imports(appdir, wh):
    """app/ 의 모든 .py 에서 import 를 모아, 표준 라이브러리·자체 패키지를 빼고 wheelhouse 휠과 대조"""
    std = set(sys.stdlib_module_names)
    own = {d for d in os.listdir(appdir) if os.path.isdir(os.path.join(appdir, d))} | {os.path.splitext(f)[0] for f in os.listdir(appdir) if f.endswith('.py')}
    found = {}
    for r, _, fs in os.walk(appdir):
        for f in fs:
            if not f.endswith('.py'): continue
            for m in re.finditer(r'^\s*(?:from\s+([A-Za-z_][\w]*)[\w.]*\s+import|import\s+([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w.]*)*))', open(os.path.join(r, f), encoding='utf-8').read(), re.M):
                names = [m.group(1)] if m.group(1) else [x.strip().split('.')[0].split(' ')[0] for x in m.group(2).split(',')]
                for n in names:
                    if n and n not in std and n not in own: found.setdefault(n, set()).add(os.path.relpath(os.path.join(r, f), appdir))
    have = {re.split(r'-\d', os.path.basename(w))[0].lower().replace('-', '_') for w in glob.glob(os.path.join(wh, '*.whl'))}
    miss = []
    for n, where in sorted(found.items()):
        d = DIST.get(n, n).lower()
        ok = d in have
        print(f"  {'✓' if ok else '✗'} {n:14s} → {d:24s} ({', '.join(sorted(where))[:90]})")
        if not ok: miss.append(n)
    if miss:
        print('  ✗ wheelhouse 에 없는 import:', miss); sys.exit(1)
    print('  실행 코드 import', len(found), '종 모두 wheelhouse 에 있음')


if __name__ == '__main__':
    c, a = sys.argv[1], sys.argv[2:]
    if c == 'app': app(a[0], a[1], a[2] == '1', a[3] == '1', a[4])
    elif c == 'pth': pth(*a)
    elif c == 'report': report(*a)
    elif c == 'tess-prune': tess_prune(*a)
    elif c == 'manifest': manifest(*a)
    elif c == 'zip': mkzip(*a)
    elif c == 'imports': imports(*a)
    else: print(__doc__); sys.exit(1)
