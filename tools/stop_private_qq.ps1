[CmdletBinding()]
param([Parameter(Mandatory)] [string] $DataRoot)

$ErrorActionPreference = 'Stop'
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$stateRoot = Join-Path $dataPath 'qq_private\runtime'
$statusFile = Join-Path $stateRoot 'status.json'
$stopFile = Join-Path $stateRoot 'stop.request'
$pauseFile = Join-Path $stateRoot 'guardian.pause'

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
Set-Content -LiteralPath $pauseFile `
    -Value ([DateTimeOffset]::Now.ToString('o')) -Encoding utf8

if (-not (Test-Path -LiteralPath $statusFile)) {
    Write-Host '没有找到私人 QQ 凛祢的运行状态；未停止任何进程。'
    exit 0
}

$status = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
$supervisorPid = [int]$status.supervisor_pid
$supervisor = Get-CimInstance Win32_Process -Filter "ProcessId = $supervisorPid" -ErrorAction SilentlyContinue
if (-not $supervisor) {
    Write-Host '监护进程已经退出；未对未知进程执行停止操作。' -ForegroundColor Yellow
    exit 0
}
$expectedScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'private_qq_supervisor.ps1')).Path
if ($supervisor.CommandLine -notlike "*$expectedScript*") {
    throw "PID $supervisorPid 已属于其他程序；停止操作已中止。"
}

Set-Content -LiteralPath $stopFile -Value ([DateTimeOffset]::Now.ToString('o')) -Encoding utf8
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline) {
    if (-not (Get-Process -Id $supervisorPid -ErrorAction SilentlyContinue)) {
        Write-Host '私人 QQ 凛祢已停止；当前聊天 JSON 保持未封存，下一次手动启动会继续。' -ForegroundColor Green
        exit 0
    }
    Start-Sleep -Seconds 1
}

throw '监护进程未在 30 秒内安全退出；没有强制结束它，请检查运行日志。'
