@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
rem ==========================================================================
rem  FIRE-3D BEAM offline installer (Windows 10/11 x64, no internet, no admin)
rem  - python\      : embeddable Python 3.11 (pre-extracted by build_bundle.sh)
rem  - wheelhouse\  : all Python packages as win_amd64 wheels (pip --no-index)
rem  - tesseract\   : portable Tesseract OCR + tessdata (kor, eng)
rem  - app\models\  : ONNX segmentation models (onnxruntime, CPU only - no GPU needed)
rem  Fallback: if python\ is missing, uses an installed Python 3.11 to make venv\
rem  Note: only "if errorlevel" is used (no && / ||) so it also runs under Wine cmd for testing.
rem ==========================================================================
set "ROOT=%~dp0"
set "FIRE3DBEAM_HOME=%ROOT:~0,-1%"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
echo.
echo ==== 소방 정보카드 변환프로그램 설치 (인터넷 불필요) ====
echo 설치 폴더: %ROOT%
echo.
if exist "%ROOT%wheelhouse\" goto have_wh
echo [오류] wheelhouse 폴더가 없습니다. 설치 파일을 통째로 복사했는지 확인하세요.
goto fail
:have_wh

set "PY=%ROOT%python\python.exe"
if exist "%PY%" goto have_embed

rem ---- fallback: installed Python 3.11 -> venv ----
echo [1/4] 동봉 Python 없음 - PC에 설치된 Python 3.11 로 가상환경을 만듭니다
py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 goto try_python
py -3.11 -m venv "%ROOT%venv"
goto venv_made
:try_python
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 goto nopython
python -m venv "%ROOT%venv"
:venv_made
if errorlevel 1 goto fail
set "PY=%ROOT%venv\Scripts\python.exe"
goto install_pkgs
:nopython
echo [오류] Python 3.11 을 찾지 못했습니다. 설치안내.md 의 "Python" 절을 참고하세요.
goto fail

:have_embed
echo [1/4] 동봉 Python: %PY%
"%PY%" -c "import sys; print('      Python ' + sys.version.split()[0])"
if errorlevel 1 goto fail
"%PY%" -m pip --version >nul 2>&1
if not errorlevel 1 goto install_pkgs
echo       pip 설치(오프라인)...
"%PY%" -X utf8 "%ROOT%app\offline\bootstrap_pip.py"
if errorlevel 1 goto fail

:install_pkgs
echo [2/4] 라이브러리 설치(오프라인 wheelhouse)...
"%PY%" -X utf8 -m pip install --no-index --find-links "%ROOT%wheelhouse" --no-warn-script-location -r "%ROOT%app\offline\requirements-win.txt"
if errorlevel 1 goto fail

echo [3/4] Tesseract OCR 확인...
"%PY%" -X utf8 "%ROOT%app\offline\launch.py" setup-tesseract

echo [4/4] 설치 확인 + 자체시험(가상 표준양식 정보카드 변환, AI 교차검증, 검수 반영 - 2~5분)...
"%PY%" -X utf8 "%ROOT%app\offline\launch.py" selftest
if errorlevel 1 goto fail
echo.
echo ==== 설치 완료 ====
echo  - 변환: 정보카드 PDF(+따로 받은 평면도 그림) 를 「변환.bat」 위에 끌어다 놓기
echo          결과: 3D 지휘화면 / 사전 활동계획 / 검수화면 / IFC
echo  - 검수: 검수화면에서 저장한 *_corrections.json 을 「검수.bat」 위에 끌어다 놓기
echo  - 채점: 「채점.bat」 더블클릭
echo  - 자체시험 다시: 「자체시험.bat」 더블클릭
echo  ※ 실제 정보카드와 결과물은 내부 PC 밖으로 반출 금지
echo.
pause
exit /b 0

:fail
echo.
echo ==== 설치 실패 - 위 메시지와 logs 폴더를 확인하세요 (설치안내.md 문제 해결 참고) ====
pause
exit /b 1
