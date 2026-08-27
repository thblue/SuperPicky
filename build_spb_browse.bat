@echo off
setlocal EnableExtensions

REM SPBBrowse.exe（结果浏览器独立封装）一键构建
REM One-shot build for SPBBrowse.exe (standalone results browser).
REM 产物 / output: dist_SPBBrowse\SPBBrowse\SPBBrowse.exe

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -m PyInstaller --noconfirm --distpath "%~dp0dist_SPBBrowse" --workpath "%~dp0build_dist_SPBBrowse" "%~dp0spb_browse_win.spec"
exit /b %ERRORLEVEL%
