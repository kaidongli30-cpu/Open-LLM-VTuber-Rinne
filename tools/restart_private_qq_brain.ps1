[CmdletBinding(SupportsShouldProcess)]
param([Parameter(Mandatory)] [string] $DataRoot)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$stateRoot = Join-Path $dataPath 'qq_private\runtime'
$statusFile = Join-Path $stateRoot 'status.json'
$supervisorScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'private_qq_supervisor.ps1')).Path

function Get-ExactListenerOwner {
    param([Parameter(Mandatory)] [int] $Port)
    $listeners = @(
        Get-NetTCPConnection -State Listen -LocalAddress '127.0.0.1' -LocalPort $Port `
            -ErrorAction SilentlyContinue
    )
    if ($listeners.Count -ne 1) {
        throw "端口 $Port 的本机监听进程不唯一或不存在；未停止任何进程。"
    }
    $listenerPid = [int]$listeners[0].OwningProcess
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid" `
        -ErrorAction SilentlyContinue
    if (-not $process) {
        throw "无法确认端口 $Port 的监听进程 PID $listenerPid；未停止任何进程。"
    }
    return $process
}

function Test-FreshStatus {
    param([object] $Status)
    try {
        $updated = [DateTimeOffset]::Parse([string]$Status.updated_at)
        return ([DateTimeOffset]::Now - $updated).TotalSeconds -le 30
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $statusFile)) {
    throw '私人 QQ 凛祢尚未启动；请先双击“启动私人QQ凛祢.bat”。'
}

$status = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($status.state -in @('stopped', 'failed') -or -not (Test-FreshStatus $status)) {
    throw '私人 QQ 监护状态不是正在运行；未停止任何进程。'
}
$runtimePath = [IO.Path]::GetFullPath([string]$status.runtime_root).TrimEnd('\')

$supervisorPid = [int]$status.supervisor_pid
$supervisor = Get-CimInstance Win32_Process -Filter "ProcessId = $supervisorPid" `
    -ErrorAction SilentlyContinue
if (-not $supervisor -or $supervisor.CommandLine -notlike "*$supervisorScript*") {
    throw '无法确认私人 QQ 监护进程；未停止任何进程。'
}

$backend = Get-ExactListenerOwner -Port 12394
$astrbot = Get-ExactListenerOwner -Port 6199
$napcat = Get-ExactListenerOwner -Port 6099

$expectedBackendEntry = Join-Path $projectRoot 'run_server.py'
if ($backend.CommandLine -notlike "*$expectedBackendEntry*") {
    throw "端口 12394 的 PID $($backend.ProcessId) 不是本项目后端；未停止任何进程。"
}
$expectedAstrBotRuntime = Join-Path $runtimePath 'python'
if ($astrbot.ExecutablePath -notlike "$expectedAstrBotRuntime\*" -or
    $astrbot.CommandLine -notmatch '(?i)astrbot') {
    throw "端口 6199 的 PID $($astrbot.ProcessId) 不是预期 AstrBot；未停止任何进程。"
}
$expectedNapCatExe = Join-Path $runtimePath 'napcat-node\node.exe'
if ([IO.Path]::GetFullPath([string]$napcat.ExecutablePath) -ne $expectedNapCatExe) {
    throw "端口 6099 的 PID $($napcat.ProcessId) 不是预期 NapCat；未停止任何进程。"
}

$oldBackendPid = [int]$backend.ProcessId
$oldAstrBotPid = [int]$astrbot.ProcessId
$oldNapCatPid = [int]$napcat.ProcessId
$description = "AstrBot PID $oldAstrBotPid 与 12394 后端 PID $oldBackendPid；保留 NapCat PID $oldNapCatPid"
if (-not $PSCmdlet.ShouldProcess($description, '重启私人 QQ 凛祢大脑')) {
    Write-Host "已确认目标：$description" -ForegroundColor Cyan
    exit 0
}

# Stop ingress first so AstrBot cannot submit a new turn while the backend is
# being replaced. The existing supervisor notices both missing listeners and
# recreates them with the same in-memory bridge token. NapCat is never stopped.
Stop-Process -Id $oldAstrBotPid -Force
Stop-Process -Id $oldBackendPid -Force

$deadline = [DateTime]::UtcNow.AddMinutes(2)
while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Seconds 2
    try {
        $newBackend = Get-ExactListenerOwner -Port 12394
        $newAstrBot = Get-ExactListenerOwner -Port 6199
        $currentNapCat = Get-ExactListenerOwner -Port 6099
        $currentStatus = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $oneBotConnected = [bool](
            Get-NetTCPConnection -State Established -LocalPort 6199 -ErrorAction SilentlyContinue
        )
        if ([int]$currentNapCat.ProcessId -ne $oldNapCatPid) {
            throw "NapCat 进程意外发生变化；请查看 $(Join-Path $stateRoot 'logs')。"
        }
        if ([int]$newBackend.ProcessId -ne $oldBackendPid -and
            [int]$newAstrBot.ProcessId -ne $oldAstrBotPid -and
            $currentStatus.state -eq 'ready' -and $oneBotConnected) {
            Write-Host '凛祢大脑已经重启，NapCat 登录保持不变。' -ForegroundColor Green
            Write-Host "12394 后端：PID $($newBackend.ProcessId)"
            Write-Host "AstrBot：PID $($newAstrBot.ProcessId)"
            Write-Host "NapCat（未重启）：PID $oldNapCatPid"
            exit 0
        }
    }
    catch {
        if ($_.Exception.Message -like 'NapCat 进程意外发生变化*') {
            throw
        }
        # Components may be between shutdown and supervisor recovery.
    }
}

throw "两分钟内未确认凛祢大脑恢复；NapCat 未被本脚本停止，请查看 $(Join-Path $stateRoot 'logs')。"
