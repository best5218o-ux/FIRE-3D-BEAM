# -*- coding: utf-8 -*-
"""정보카드 PDF → 3D 지휘화면(HTML 1개) + IFC4.   python make_demo.py data/xxx/card.pdf out/demo/name [--model]
--model (또는 FIREBIM_MODEL=1): STR 모델로 규칙 벽 교차검증·문/진입구 보충 → 불일치는 확인사항·검수 항목으로"""
import sys, os, json, time
from firebim.pipeline import run
from firebim.export import build, export_ifc, to_viewer_bim
from build_viewer import make as make_html

def main(pdf, stem, dong_files=None, **opt):
    t = time.time()
    res = run(pdf, keep=True, **opt)        # opt: daejang=, site=, ilramdo= (없으면 card.pdf 옆 파일 자동 사용), dong=
    D, BIM = build(res)
    if dong_files: D['building']['dong_files'] = dong_files      # 여러 동: 동 선택 링크(같은 폴더의 동별 HTML)
    export_ifc(D, BIM, stem + '.ifc')
    VB = to_viewer_bim(BIM)
    json.dump(dict(D=D, BIM=VB), open(stem + '.data.json', 'w'), ensure_ascii=False)
    make_html(stem + '.data.json', stem + '.html')
    print('ok', stem, f'{time.time()-t:.1f}s', 'floors', len(D['floors']), 'stairs', len(VB['stairs']), 'elevs', len(VB['elevs']), 'equip', len(VB['equip']), 'issues', len(D['issues']))

if __name__ == '__main__':
    # 선택: --daejang 대장.json|pdf  --site site.json  --zones 경계구역일람도.json
    if '--model' in sys.argv: sys.argv.remove('--model'); os.environ['FIREBIM_MODEL'] = '1'
    # 여러 동 표준양식 카드: --dong 2 (두 번째 동만) | --dong all (평면도가 있는 동마다 STEM_N동.html, 화면에 동 선택 링크)
    a = sys.argv[3:]; opt = {k[2:].replace('zones', 'ilramdo'): v for k, v in zip(a[::2], a[1::2])}
    dg = opt.pop('dong', None)
    if dg == 'all':
        from firebim.card import parse_card, extract_plans
        info, doc = parse_card(sys.argv[1])
        n = len(info.get('동목록') or [1]); have = {p.get('dong', 0) for p in extract_plans(doc)}
        base = os.path.basename(sys.argv[2])
        files = [f'{base}_{k + 1}동.html' if k in have else None for k in range(n)]
        for k in range(n):
            if k not in have: continue
            o = dict(opt) if k == 0 else dict({'daejang': '-', 'site': '-', 'ilramdo': '-'}, **opt)   # card.pdf 옆 대장·부지 파일은 첫 동 것 → 다른 동엔 자동 적용 안 함
            main(sys.argv[1], f'{sys.argv[2]}_{k + 1}동', dong_files=files, dong=k, **o)
    else:
        main(sys.argv[1], sys.argv[2], **(dict(opt, dong=int(dg) - 1) if dg else opt))
