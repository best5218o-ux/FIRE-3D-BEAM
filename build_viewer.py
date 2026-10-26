# -*- coding: utf-8 -*-
"""뷰어 번들 + 데이터 → 오프라인 HTML 1개.  python build_viewer.py out/demo/x.data.json out/demo/x.html"""
import sys, json, subprocess, os
ROOT = os.path.dirname(os.path.abspath(__file__)); VD = os.path.join(ROOT, 'viewer')

def bundle():
    js = os.path.join(VD, 'dist', 'app.js')
    esb = os.path.join(VD, 'node_modules', '.bin', 'esbuild')
    if os.environ.get('FIREBIM_PREBUILT') or not os.path.exists(esb):     # 오프라인(내부 PC): node 없이 미리 빌드한 번들 사용
        if os.path.exists(js): return open(js, encoding='utf-8').read()
        raise SystemExit('viewer/dist/app.js 없음 — 인터넷 PC에서 offline/build_bundle.sh 로 번들을 먼저 만드세요')
    subprocess.run([os.path.join(VD, 'node_modules', '.bin', 'esbuild'), 'src/main.js', '--bundle', '--minify', '--format=iife', '--target=es2020',
                    '--legal-comments=none', f'--outfile={js}'], cwd=VD, check=True)
    return open(js, encoding='utf-8').read()

def make(data_json, out_html, js=None):
    js = js or bundle()
    d = json.load(open(data_json, encoding='utf-8'))
    tpl = open(os.path.join(VD, 'src', 'template.html'), encoding='utf-8').read()
    data = json.dumps(d, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = tpl.replace('__TITLE__', d['D']['building']['name'] + ' 소방활동 3D').replace('@@DATA@@', data).replace('__JS__', lambda_safe(js.replace('</script', '<\\/script')))
    open(out_html, 'w', encoding='utf-8').write(html)
    print(out_html, f'{len(html) / 1e6:.2f}MB')

def lambda_safe(s): return s

if __name__ == '__main__':
    make(sys.argv[1], sys.argv[2])
