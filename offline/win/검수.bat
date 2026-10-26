@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
rem ==========================================================================
rem  Review: drop *_corrections.json (apply + rebuild) or a result folder/PDF (open review page)
rem  All logic (Korean messages, paths) is in app\offline\launch.py
rem ==========================================================================
set "ROOT=%~dp0"
set "FIREBIM_HOME=%ROOT:~0,-1%"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PY=%ROOT%python\python.exe"
if not exist "%PY%" set "PY=%ROOT%venv\Scripts\python.exe"
if not exist "%PY%" goto nopy
"%PY%" -X utf8 "%ROOT%app\offline\launch.py" review %*
echo.
pause
exit /b

:nopy
echo [!] Python not found. 먼저 install.bat 을 실행하세요.
pause
exit /b 1
