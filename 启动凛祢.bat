@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
echo ===== 桌宠启动器 =====
echo 启动主程序...
start "Server" cmd /k "cd /d "%PROJECT_ROOT%" && python -m uv run run_server.py"
if defined RINNE_TTS_DIR (
  echo 启动可选 TTS 服务...
  start "TTS" cmd /k "cd /d "%RINNE_TTS_DIR%" && runtime\python.exe api_v2.py"
) else (
  echo 未设置 RINNE_TTS_DIR，跳过外部 TTS 服务。
)
if defined RINNE_TUNNEL_DIR (
  echo 启动可选隧道服务...
  start "Tunnel" cmd /k "cd /d "%RINNE_TUNNEL_DIR%" && cloudflared.exe tunnel --config config.yml run"
)
echo 日记、周记和月记将在主程序启动时按当前配置处理。
echo ===== 启动命令已发出 =====
pause
