[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $RuntimeRoot,
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $ConversationDataRoot,
    [string] $CloudflaredRoot,
    [string] $RecoveryPublicBaseUrl,
    [switch] $EnableRemoteRecoveryTunnel,
    [int] $PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($EnableRemoteRecoveryTunnel -and
    (-not $CloudflaredRoot -or -not $RecoveryPublicBaseUrl)) {
    throw '远程扫码恢复需要同时指定 CloudflaredRoot 和 RecoveryPublicBaseUrl。'
}
if ($PollSeconds -lt 10 -or $PollSeconds -gt 300) {
    throw 'PollSeconds 必须在 10 到 300 秒之间。'
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runtimePath = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$conversationDataPath = if ($ConversationDataRoot) {
    (Resolve-Path -LiteralPath $ConversationDataRoot).Path
} else { $projectRoot }
$startScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_private_qq.ps1')).Path
$supervisorScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'private_qq_supervisor.ps1')).Path
$stateRoot = Join-Path $dataPath 'qq_private\runtime'
$statusFile = Join-Path $stateRoot 'status.json'
$stopFile = Join-Path $stateRoot 'stop.request'
$pauseFile = Join-Path $stateRoot 'guardian.pause'
$logRoot = Join-Path $stateRoot 'logs'
$guardianLog = Join-Path $logRoot 'guardian.log'
$pwshExe = Join-Path $PSHOME 'pwsh.exe'

if ($dataPath -eq [IO.Path]::GetPathRoot($dataPath)) {
    throw 'DataRoot 不能是盘符根目录。'
}
New-Item -ItemType Directory -Path $stateRoot,$logRoot -Force | Out-Null

function Write-GuardianEvent {
    param([string] $Event, [string] $Detail = '')
    $item = Get-Item -LiteralPath $guardianLog -ErrorAction SilentlyContinue
    if ($item -and $item.Length -ge 5MB) {
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
        Move-Item -LiteralPath $guardianLog `
            -Destination (Join-Path $logRoot "guardian.$stamp.log")
    }
    $safeDetail = ($Detail -replace '[\r\n]+', ' ').Trim()
    if ($safeDetail.Length -gt 240) {
        $safeDetail = $safeDetail.Substring(0, 240)
    }
    $record = [ordered]@{
        at = [DateTimeOffset]::Now.ToString('o')
        event = $Event
        detail = $safeDetail
    } | ConvertTo-Json -Compress
    Add-Content -LiteralPath $guardianLog -Value $record -Encoding utf8
}

function Get-ExactSupervisorProcess {
    param([object] $Status)
    $pidValue = 0
    if (-not [int]::TryParse([string]$Status.supervisor_pid, [ref]$pidValue) -or
        $pidValue -le 0) {
        return $null
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" `
        -ErrorAction SilentlyContinue
    if (-not $process -or -not $process.CommandLine) {
        return $null
    }
    if ($process.CommandLine -notlike "*$supervisorScript*" -or
        $process.CommandLine -notlike "*$projectRoot*") {
        return $null
    }
    return $process
}

function ConvertTo-StatusTime {
    param([object] $Value)
    if ($Value -is [DateTimeOffset]) {
        return $Value.UtcDateTime
    }
    if ($Value -is [DateTime]) {
        return $Value.ToUniversalTime()
    }
    $parsed = [DateTimeOffset]::MinValue
    if ($Value -is [string] -and [DateTimeOffset]::TryParse(
        $Value,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind,
        [ref]$parsed
    )) {
        return $parsed.UtcDateTime
    }
    return $null
}

function Get-GuardianObservation {
    try {
        $status = Get-Content -LiteralPath $statusFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        return [pscustomobject]@{
            Healthy = $false
            ExactProcess = $null
            Reason = 'status_unavailable'
        }
    }
    $process = Get-ExactSupervisorProcess $status
    if (-not $process) {
        return [pscustomobject]@{
            Healthy = $false
            ExactProcess = $null
            Reason = 'supervisor_missing'
        }
    }
    $updatedAt = ConvertTo-StatusTime $status.updated_at
    if (-not $updatedAt -or
        [DateTime]::UtcNow - $updatedAt -gt [TimeSpan]::FromMinutes(2)) {
        return [pscustomobject]@{
            Healthy = $false
            ExactProcess = $process
            Reason = 'status_stale'
        }
    }
    if ($status.state -in @('failed', 'stopped')) {
        return [pscustomobject]@{
            Healthy = $false
            ExactProcess = $process
            Reason = [string]$status.state
        }
    }
    return [pscustomobject]@{
        Healthy = $true
        ExactProcess = $process
        Reason = [string]$status.state
    }
}

function Request-StaleSupervisorStop {
    param([object] $Process)
    Set-Content -LiteralPath $stopFile `
        -Value ([DateTimeOffset]::Now.ToString('o')) -Encoding utf8
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Get-Process -Id ([int]$Process.ProcessId) -ErrorAction SilentlyContinue)) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

Write-GuardianEvent -Event 'guardian_started'
while ($true) {
    try {
        if (Test-Path -LiteralPath $pauseFile) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $observation = Get-GuardianObservation
        if ($observation.Healthy) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        Write-GuardianEvent -Event 'recovery_requested' -Detail $observation.Reason
        if ($observation.Reason -eq 'status_stale' -and $observation.ExactProcess) {
            if (-not (Request-StaleSupervisorStop $observation.ExactProcess)) {
                Write-GuardianEvent -Event 'stale_supervisor_refused_to_stop'
                Start-Sleep -Seconds $PollSeconds
                continue
            }
        }

        $startArguments = @(
            '-NoProfile',
            '-ExecutionPolicy', 'Bypass',
            '-File', $startScript,
            '-RuntimeRoot', $runtimePath,
            '-DataRoot', $dataPath
        )
        $startArguments += @('-ConversationDataRoot', $conversationDataPath)
        if ($EnableRemoteRecoveryTunnel) {
            $startArguments += @(
                '-CloudflaredRoot', $CloudflaredRoot,
                '-RecoveryPublicBaseUrl', $RecoveryPublicBaseUrl,
                '-EnableRemoteRecoveryTunnel'
            )
        }
        & $pwshExe @startArguments
        if ($LASTEXITCODE -eq 0) {
            Write-GuardianEvent -Event 'recovery_started'
        }
        else {
            Write-GuardianEvent -Event 'recovery_start_failed' `
                -Detail "exit_code=$LASTEXITCODE"
        }
    }
    catch {
        Write-GuardianEvent -Event 'guardian_iteration_failed' `
            -Detail $_.Exception.GetType().Name
    }
    Start-Sleep -Seconds $PollSeconds
}
