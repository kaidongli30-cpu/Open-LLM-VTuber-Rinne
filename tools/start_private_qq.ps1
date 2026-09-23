[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $RuntimeRoot,
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $ConversationDataRoot,
    [string] $CloudflaredRoot,
    [string] $RecoveryPublicBaseUrl,
    [int] $AdoptNapCatPid = 0,
    [switch] $EnableRemoteRecoveryTunnel,
    [switch] $OpenRecoveryPage
)

$ErrorActionPreference = 'Stop'
if ($EnableRemoteRecoveryTunnel -and
    (-not $CloudflaredRoot -or -not $RecoveryPublicBaseUrl)) {
    throw '远程扫码恢复需要同时指定 CloudflaredRoot 和 RecoveryPublicBaseUrl。'
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runtimePath = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$conversationDataPath = if ($ConversationDataRoot) {
    (Resolve-Path -LiteralPath $ConversationDataRoot).Path
} else { $projectRoot }
$supervisorScript = Join-Path $PSScriptRoot 'private_qq_supervisor.ps1'
$openRecoveryScript = Join-Path $PSScriptRoot 'open_private_qq_recovery_page.ps1'
$stateRoot = Join-Path $dataPath 'qq_private\runtime'
$statusFile = Join-Path $stateRoot 'status.json'
$stopFile = Join-Path $stateRoot 'stop.request'
$pauseFile = Join-Path $stateRoot 'guardian.pause'
$logRoot = Join-Path $stateRoot 'logs'

if (-not (Test-Path -LiteralPath $supervisorScript)) {
    throw "缺少监护脚本：$supervisorScript"
}
if ($OpenRecoveryPage -and -not (Test-Path -LiteralPath $openRecoveryScript)) {
    throw "缺少本地扫码页打开脚本：$openRecoveryScript"
}
if ($dataPath -eq [IO.Path]::GetPathRoot($dataPath)) {
    throw 'DataRoot 不能是盘符根目录。'
}
if ($conversationDataPath -eq [IO.Path]::GetPathRoot($conversationDataPath)) {
    throw 'ConversationDataRoot 不能是盘符根目录。'
}
New-Item -ItemType Directory -Path $stateRoot,$logRoot -Force | Out-Null

$existing = $null
if (Test-Path -LiteralPath $statusFile) {
    try {
        $existing = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        # A stale or partial status file is replaced by the fresh launch below.
    }
}
if ($existing) {
    $existingSupervisorPid = 0
    $existingProcess = $null
    if ([int]::TryParse(
        [string]$existing.supervisor_pid,
        [ref]$existingSupervisorPid
    ) -and $existingSupervisorPid -gt 0) {
        $existingProcess = Get-CimInstance Win32_Process `
            -Filter "ProcessId = $existingSupervisorPid" `
            -ErrorAction SilentlyContinue
    }
    $isOwnedSupervisor = [bool](
        $existingProcess -and $existingProcess.CommandLine -and
        $existingProcess.CommandLine -like "*$supervisorScript*"
    )
    if ($isOwnedSupervisor -and $existing.state -notin @('stopped', 'failed')) {
        $runningRemoteTunnelMode = [bool](
            $existingProcess.CommandLine -match
                '(?i)(?:^|\s)-EnableRemoteRecoveryTunnel(?:\s|$)'
        )
        if ($EnableRemoteRecoveryTunnel -and -not $runningRemoteTunnelMode) {
            Write-Host '当前进程未监护远程扫码隧道，正在安全升级启动模式。' `
                -ForegroundColor Yellow
            Set-Content -LiteralPath $pauseFile `
                -Value ([DateTimeOffset]::Now.ToString('o')) -Encoding utf8
            Set-Content -LiteralPath $stopFile `
                -Value ([DateTimeOffset]::Now.ToString('o')) -Encoding utf8
            $upgradeDeadline = [DateTime]::UtcNow.AddSeconds(45)
            while ([DateTime]::UtcNow -lt $upgradeDeadline) {
                if (-not (Get-Process -Id ([int]$existingProcess.ProcessId) `
                    -ErrorAction SilentlyContinue)) {
                    break
                }
                Start-Sleep -Seconds 1
            }
            if (Get-Process -Id ([int]$existingProcess.ProcessId) `
                -ErrorAction SilentlyContinue) {
                Remove-Item -LiteralPath $pauseFile -Force `
                    -ErrorAction SilentlyContinue
                throw '现有私人 QQ 监护进程未能安全退出，未强制结束。'
            }
        }
        else {
            Write-Host "私人 QQ 凛祢已经在运行，当前状态：$($existing.state)" `
                -ForegroundColor Green
            if ($existing.state -eq 'scan_required') {
                if ($existing.qr_available) {
                    Write-Host "需要扫码：$($existing.qr_path)" -ForegroundColor Yellow
                }
                else {
                    Write-Host '需要重新认证，正在等待 NapCat 生成新二维码。' `
                        -ForegroundColor Yellow
                }
                if ($OpenRecoveryPage) {
                    & $openRecoveryScript -DataRoot $dataPath -RecoveryPublicBaseUrl $RecoveryPublicBaseUrl
                }
            }
            elseif ($existing.state -eq 'account_restricted') {
                Write-Host 'QQ 拒绝生成二维码：请先在最新版手机 QQ 恢复账号使用。' `
                    -ForegroundColor Yellow
                if ($OpenRecoveryPage) {
                    & $openRecoveryScript -DataRoot $dataPath -RecoveryPublicBaseUrl $RecoveryPublicBaseUrl
                }
            }
            exit 0
        }
    }
}

$instanceId = [Guid]::NewGuid().ToString('N')
Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $pauseFile -Force -ErrorAction SilentlyContinue
$pwshExe = Join-Path $PSHOME 'pwsh.exe'
$supervisorStdout = Join-Path $logRoot 'supervisor.stdout.log'
$supervisorStderr = Join-Path $logRoot 'supervisor.stderr.log'
foreach ($entry in @(
    @{ path = $supervisorStdout; stream = 'stdout' },
    @{ path = $supervisorStderr; stream = 'stderr' }
)) {
    $item = Get-Item -LiteralPath $entry.path -ErrorAction SilentlyContinue
    if ($item -and $item.Length -gt 0) {
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
        $archive = Join-Path $logRoot "supervisor.$stamp.$($entry.stream).log"
        Move-Item -LiteralPath $entry.path -Destination $archive
    }
}
$supervisorArguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', ('"' + $supervisorScript + '"'),
    '-ProjectRoot', ('"' + $projectRoot + '"'),
    '-RuntimeRoot', ('"' + $runtimePath + '"'),
    '-DataRoot', ('"' + $dataPath + '"'),
    '-ConversationDataRoot', ('"' + $conversationDataPath + '"'),
    '-InstanceId', $instanceId,
    '-AdoptNapCatPid', $AdoptNapCatPid
)
if ($EnableRemoteRecoveryTunnel) {
    $supervisorArguments += @(
        '-CloudflaredRoot', ('"' + $CloudflaredRoot + '"'),
        '-RecoveryPublicBaseUrl', ('"' + $RecoveryPublicBaseUrl + '"'),
        '-EnableRemoteRecoveryTunnel'
    )
}
$supervisor = Start-Process -FilePath $pwshExe -ArgumentList $supervisorArguments `
    -WindowStyle Hidden -RedirectStandardOutput $supervisorStdout `
    -RedirectStandardError $supervisorStderr -PassThru

Write-Host '正在启动私人 QQ 凛祢；冷启动需要加载长期记忆和本地模型，请保留此窗口。' `
    -ForegroundColor Cyan
$lastShownDetail = ''
$deadline = [DateTime]::UtcNow.AddMinutes(10)
while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Seconds 1
    if (-not (Get-Process -Id $supervisor.Id -ErrorAction SilentlyContinue)) {
        throw "私人 QQ 监护进程提前退出；请查看 $logRoot。"
    }
    if (-not (Test-Path -LiteralPath $statusFile)) {
        continue
    }
    try {
        $status = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        continue
    }
    if ($status.instance_id -ne $instanceId) {
        continue
    }
    $currentDetail = [string]$status.detail
    if ($currentDetail -and $currentDetail -ne $lastShownDetail) {
        Write-Host $currentDetail -ForegroundColor Cyan
        $lastShownDetail = $currentDetail
    }
    if ($status.state -eq 'ready') {
        Write-Host '私人 QQ 凛祢已启动，并由账号级监护进程保持在线。' -ForegroundColor Green
        Write-Host "聊天与记忆：$(Join-Path $conversationDataPath 'chat_history\rinne_01')"
        Write-Host "运行日志：$(Join-Path $stateRoot 'logs')"
        if ($EnableRemoteRecoveryTunnel) {
            Write-Host '远程扫码恢复页已经运行；如需再次打开，请进入“QQ凛祢维护工具”文件夹。'
        }
        else {
            Write-Host '本地扫码恢复页已经运行。'
        }
        exit 0
    }
    if ($status.state -eq 'scan_required') {
        Write-Host '运行组件已经就绪，但 QQ 登录需要扫码。' -ForegroundColor Yellow
        if ($status.qr_available) {
            Write-Host "二维码：$($status.qr_path)"
        }
        else {
            Write-Host 'NapCat 尚未生成本次登录的新二维码；监护进程会继续等待。'
        }
        if ($OpenRecoveryPage) {
            & $openRecoveryScript -DataRoot $dataPath -RecoveryPublicBaseUrl $RecoveryPublicBaseUrl
        }
        exit 0
    }
    if ($status.state -eq 'account_restricted') {
        Write-Host 'QQ 因账号安全限制拒绝生成二维码。' -ForegroundColor Yellow
        Write-Host '请先在最新版手机 QQ 按提示恢复账号使用，再停止并重新启动私人 QQ 凛祢。' `
            -ForegroundColor Yellow
        if ($OpenRecoveryPage) {
            & $openRecoveryScript -DataRoot $dataPath -RecoveryPublicBaseUrl $RecoveryPublicBaseUrl
        }
        exit 0
    }
    if ($status.state -eq 'failed') {
        throw "私人 QQ 凛祢启动失败：$($status.detail)"
    }
}

throw "十分钟内尚未取得 QQ 可回复状态；监护进程会继续记录状态，请查看 $logRoot。"
