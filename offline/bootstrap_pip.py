# -*- coding: utf-8 -*-
"""install.bat 용: 동봉 Python(embeddable, pip 없음)에 wheelhouse 의 pip 휠로 pip 를 오프라인 설치.
(.bat 의 for-와일드카드·경로 따옴표 문제를 피하려고 파이썬에서 처리)"""
import glob, os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <설치폴더>\app\offline\ → <설치폴더>
WH = os.path.join(ROOT, 'wheelhouse')
whl = sorted(glob.glob(os.path.join(WH, 'pip-*.whl')))
if not whl:
    print('[오류] wheelhouse 에 pip 휠이 없습니다:', WH); sys.exit(1)
sys.path.insert(0, whl[-1])                      # 휠(zip) 안의 pip 를 바로 실행
sys.argv = ['pip', 'install', '--no-index', '--find-links', WH, '--no-warn-script-location', '--disable-pip-version-check', 'pip']
runpy.run_module('pip', run_name='__main__', alter_sys=True)
