@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not defined RINNE_GPT_SOVITS_ROOT set "RINNE_GPT_SOVITS_ROOT=%~dp0..\..\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"
if not exist "%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" (
  echo [错误] 未找到 GPT-SoVITS runtime\python.exe
  echo 请先按根目录 README 设置 RINNE_GPT_SOVITS_ROOT。
  pause
  exit /b 1
)
echo 将停止本保留包管理的 V4-B-R1 服务，并启动稳定 V2。
echo 如果另一个旧 GPT-SoVITS API 窗口仍在运行，请先关闭它。
echo.
"%RINNE_GPT_SOVITS_ROOT%\runtime\python.exe" "rinne_voice_manager.py" switch-v2
echo.
pause
