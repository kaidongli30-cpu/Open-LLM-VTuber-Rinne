@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "RINNE_GPT_SOVITS_ROOT=%~dp0..\..\GPT-SoVITS-v2pro-20250604"
if not exist "%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" set "RINNE_GPT_SOVITS_ROOT=%RINNE_GPT_SOVITS_ROOT%\GPT-SoVITS-v2pro-20250604"

if not exist "%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" (
  echo [错误] 未找到 GPT-SoVITS 的 runtime\python.exe
  echo 请按根目录 README 将 GPT-SoVITS-v2pro-20250604 放到项目根目录。
  pause
  exit /b 1
)

echo 正在启动凛祢 V2 语音服务，请保持本窗口运行。
echo.
"%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" "rinne_voice_manager.py" switch-v2
echo.
pause
