# -*- coding: utf-8 -*-
"""Colab 노트북 코드 셀을 로컬에서 순서대로 실행(시험용). 셀은 순수 파이썬(매직 없음)이라 그대로 exec.
  FIRE3DBEAM_BASE=<가짜 Drive/FIRE-3D-BEAM> FIRE3DBEAM_WORK=<작업폴더> SPA_MINUTES=3 SPA_DEVICE=cpu SPA_BS=8 SPA_WORKERS=1 SPA_NVAL=5 \
  python colab/nb_local_run.py colab/FIRE-3D-BEAM_SPA_GPU학습.ipynb [0,1,2,3]      # 코드 셀 번호(0부터) 선택 가능"""
import json, sys, time
nb = json.load(open(sys.argv[1], encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
sel = range(len(cells)) if len(sys.argv) < 3 else [int(x) for x in sys.argv[2].split(',')]
G = {'__name__': '__main__'}
for i in sel:
    print(f'\n======== cell {i + 1}: {cells[i].splitlines()[0]}', flush=True)
    t0 = time.time(); exec(compile(cells[i], f'cell{i + 1}', 'exec'), G)
    print(f'======== cell {i + 1} {time.time() - t0:.1f}s', flush=True)
