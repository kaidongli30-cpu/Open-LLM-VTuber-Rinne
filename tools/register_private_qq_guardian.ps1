[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $RuntimeRoot,
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $ConversationDataRoot,
    [string] $CloudflaredRoot,
    [string] $RecoveryPublicBaseUrl,
    [switch] $EnableRemoteRecoveryTunnel,
    [string] $TaskName = 'RinnePrivateQQGuardian'
)

$ErrorActionPreference = 'Stop'
if ($EnableRemoteRecoveryTunnel -and
    (-not $CloudflaredRoot -or -not $RecoveryPublicBaseUrl)) {
    throw '远程扫码恢复需要同时指定 CloudflaredRoot 和 RecoveryPublicBaseUrl。'
}
$guardianScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'private_qq_guardian.ps1')).Path
$runtimePath = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$conversationDataPath = if ($ConversationDataRoot) {
    (Resolve-Path -LiteralPath $ConversationDataRoot).Path
} else { (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path }
$pwshExe = Join-Path $PSHOME 'pwsh.exe'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name

function Quote-TaskArgument {
    param([string] $Value)
    if ($Value.Contains('"')) {
        throw '计划任务参数包含不支持的双引号。'
    }
    return '"' + $Value + '"'
}

$arguments = @(
    '-NoProfile',
    '-WindowStyle', 'Hidden',
    '-ExecutionPolicy', 'Bypass',
    '-File', (Quote-TaskArgument $guardianScript),
    '-RuntimeRoot', (Quote-TaskArgument $runtimePath),
    '-DataRoot', (Quote-TaskArgument $dataPath)
)
$arguments += @('-ConversationDataRoot', (Quote-TaskArgument $conversationDataPath))
if ($EnableRemoteRecoveryTunnel) {
    $cloudflaredPath = (Resolve-Path -LiteralPath $CloudflaredRoot).Path
    $arguments += @(
        '-CloudflaredRoot', (Quote-TaskArgument $cloudflaredPath),
        '-RecoveryPublicBaseUrl', (Quote-TaskArgument $RecoveryPublicBaseUrl),
        '-EnableRemoteRecoveryTunnel'
    )
}
$arguments = $arguments -join ' '

$action = New-ScheduledTaskAction -Execute $pwshExe -Argument $arguments `
    -WorkingDirectory (Split-Path -Parent $guardianScript)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -RestartCount 99 `
    -RestartInterval ([TimeSpan]::FromMinutes(1)) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$task = New-ScheduledTask -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description 'Keeps the private QQ Rinne supervisor available after user logon.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "私人 QQ Windows 守护已安装并启动：$TaskName" -ForegroundColor Green
