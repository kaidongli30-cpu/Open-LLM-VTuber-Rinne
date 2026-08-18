@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not defined RINNE_GPT_SOVITS_ROOT set "RINNE_GPT_SOVITS_ROOT=%~dp0..\..\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"
"%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" "rinne_voice_manager.py" stop
echo.
pause
