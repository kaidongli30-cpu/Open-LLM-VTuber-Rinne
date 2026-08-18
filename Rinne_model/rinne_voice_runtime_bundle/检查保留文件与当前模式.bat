@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not defined RINNE_GPT_SOVITS_ROOT set "RINNE_GPT_SOVITS_ROOT=%~dp0..\..\GPT-SoVITS-v2pro-20250604"
if not exist "%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" set "RINNE_GPT_SOVITS_ROOT=%RINNE_GPT_SOVITS_ROOT%\GPT-SoVITS-v2pro-20250604"
if not exist "%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" (
  echo [错误] 未找到 GPT-SoVITS runtime\python.exe
  echo 请先按根目录 README 设置 RINNE_GPT_SOVITS_ROOT。
  pause
  exit /b 1
)
"%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" "rinne_voice_manager.py" status
echo.
pause
