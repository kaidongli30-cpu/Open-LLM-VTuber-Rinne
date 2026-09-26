@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion

rem 只修改下面这一行：填写同时包含 api_v2.py 和 runtime 的语音目录。
set "TTS_DIR="

echo ===== 凛祢启动器 =====
if not defined TTS_DIR (
    echo [未配置] 请右键编辑本脚本，填写 TTS_DIR 后保存。
    goto failed
)
if not exist "%TTS_DIR%\api_v2.py" (
    echo [路径错误] 语音目录里没有 api_v2.py，请检查 TTS_DIR。
    goto failed
)
if not exist "%TTS_DIR%\runtime\python.exe" (
    echo [路径错误] 语音目录里没有 runtime\python.exe，请检查解压是否完整。
    goto failed
)
if not exist "%~dp0run_server.py" (
    echo [位置错误] 请把本脚本留在包含 run_server.py 的项目目录中。
    goto failed
)
where uv >nul 2>nul
if errorlevel 1 (
    echo [缺少软件] 找不到 uv，请按 README 安装，再重新打开脚本。
    goto failed
)
if /i "%~1"=="--check" (
    echo 启动路径检查通过，未启动任何服务。
    exit /b 0
)

rem 只检查，不关闭或接管已经占用端口的服务。
powershell.exe -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $t = $c.ConnectAsync('127.0.0.1', 9880); if ($t.Wait(800) -and $c.Connected) { exit 1 } } catch {} finally { $c.Dispose() }; exit 0"
if errorlevel 1 (
    echo [端口已占用] 9880 已有程序运行。请先确认并关闭旧语音窗口，不要重复启动。
    goto failed
)

echo 启动 TTS 语音服务...
start "凛祢 TTS" /D "%TTS_DIR%" cmd /d /k "runtime\python.exe api_v2.py"
echo 等待语音服务就绪，最多等待 180 秒，请保留 TTS 窗口...
powershell.exe -NoProfile -Command "$limit = [DateTime]::UtcNow.AddSeconds(180); while ([DateTime]::UtcNow -lt $limit) { try { $s = Invoke-RestMethod 'http://127.0.0.1:9880/openapi.json' -TimeoutSec 2; if ($s.paths.'/tts'.post -and $s.paths.'/set_gpt_weights') { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
    echo [未就绪] 语音服务未在等待时间内就绪，后端尚未启动。
    echo 请查看 TTS 窗口的报错；再次运行本脚本前，先关闭这次的 TTS 窗口。
    goto failed
)

echo 语音服务已就绪，正在启动后端...
start "凛祢后端" /D "%~dp0" cmd /d /k "uv run run_server.py"
echo 请等待后端窗口完成初始化；如要求审核日记，请按窗口提示操作。
echo 后端就绪后，再双击桌面上的凛祢客户端。
echo 使用期间请保留 TTS 和后端两个窗口，并确认 Ollama 已运行。
pause
exit /b 0

:failed
pause
exit /b 1
