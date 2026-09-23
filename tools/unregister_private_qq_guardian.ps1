[CmdletBinding()]
param([string] $TaskName = 'RinnePrivateQQGuardian')

$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host '没有找到私人 QQ Windows 守护计划任务。'
    exit 0
}
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "私人 QQ Windows 守护已移除：$TaskName" -ForegroundColor Green
