[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ProjectRoot,
    [Parameter(Mandatory)] [string] $RuntimeRoot,
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $ConversationDataRoot,
    [Parameter(Mandatory)] [string] $InstanceId,
    [string] $CloudflaredRoot,
    [string] $RecoveryPublicBaseUrl,
    [int] $AdoptNapCatPid = 0,
    [switch] $EnableRemoteRecoveryTunnel
)

$ErrorActionPreference = 'Stop'
if ($EnableRemoteRecoveryTunnel -and
    (-not $CloudflaredRoot -or -not $RecoveryPublicBaseUrl)) {
    throw '远程扫码恢复需要同时指定 CloudflaredRoot 和 RecoveryPublicBaseUrl。'
}
if ($EnableRemoteRecoveryTunnel) {
    $publicUri = $null
    if (-not [Uri]::TryCreate($RecoveryPublicBaseUrl, [UriKind]::Absolute, [ref]$publicUri) -or
        $publicUri.Scheme -ne 'https' -or $publicUri.AbsolutePath -ne '/' -or
        $publicUri.Query -or $publicUri.Fragment -or $publicUri.UserInfo -or
        -not $publicUri.IsDefaultPort) {
        throw 'RecoveryPublicBaseUrl 必须是 HTTPS 网站根地址，且不含路径、参数或自定义端口。'
    }
    $RecoveryPublicBaseUrl = $publicUri.GetLeftPart([UriPartial]::Authority)
}
$projectPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
$runtimePath = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$conversationDataPath = if ($ConversationDataRoot) {
    (Resolve-Path -LiteralPath $ConversationDataRoot).Path
} else { $projectPath }
if ($conversationDataPath -eq [IO.Path]::GetPathRoot($conversationDataPath)) {
    throw 'ConversationDataRoot 不能是盘符根目录。'
}
$stateRoot = Join-Path $dataPath 'qq_private\runtime'
$logRoot = Join-Path $stateRoot 'logs'
$statusFile = Join-Path $stateRoot 'status.json'
$transportHealthFile = Join-Path $stateRoot 'transport_health.json'
$stopFile = Join-Path $stateRoot 'stop.request'
$qrPath = Join-Path $dataPath 'napcat\cache\qrcode.png'
$recoverySecretRoot = Join-Path $dataPath 'qq_private\secrets'
$recoveryTokenFile = Join-Path $recoverySecretRoot 'qr_recovery_token.txt'
$recoveryUrlFile = Join-Path $recoverySecretRoot 'qr_recovery_url.txt'
$processTempRoot = Join-Path $dataPath 'qq_private\tmp'
$backendTempRoot = Join-Path $processTempRoot 'backend'
$astrbotTempRoot = Join-Path $processTempRoot 'astrbot'
$napcatTempRoot = Join-Path $processTempRoot 'napcat'
$recoveryTempRoot = Join-Path $processTempRoot 'recovery'
$cloudflaredTempRoot = Join-Path $processTempRoot 'cloudflared'

$pythonExe = Join-Path $projectPath '.venv\Scripts\python.exe'
$backendEntry = Join-Path $projectPath 'run_server.py'
$recoveryEntry = Join-Path $projectPath 'tools\private_qq_recovery_server.py'
$astrbotExe = Join-Path $runtimePath 'astrbot-tool\astrbot\Scripts\astrbot.exe'
$astrbotRoot = Join-Path $runtimePath 'astrbot-instance'
$napcatExe = Join-Path $runtimePath 'napcat-node\node.exe'
$napcatRoot = Join-Path $runtimePath 'napcat-node'
$pluginSource = Join-Path $projectPath 'integrations\astrbot_plugin_rinne_private'
$pluginTarget = Join-Path $astrbotRoot 'data\plugins\astrbot_plugin_rinne_private'
$pluginConfigPath = Join-Path $astrbotRoot 'data\config\astrbot_plugin_rinne_private_config.json'
$napcatWebUiConfig = Join-Path $dataPath 'napcat\config\webui.json'
$cloudflaredPath = $null
$cloudflaredExe = $null
$cloudflaredConfig = $null
if ($EnableRemoteRecoveryTunnel) {
    $cloudflaredPath = (Resolve-Path -LiteralPath $CloudflaredRoot).Path
    $cloudflaredExe = Join-Path $cloudflaredPath 'cloudflared.exe'
    $cloudflaredConfig = Join-Path $cloudflaredPath 'config.yml'
}

$backendProcess = $null
$astrbotProcess = $null
$napcatProcess = $null
$recoveryProcess = $null
$cloudflaredProcess = $null
$bridgeTokenBytes = [byte[]]::new(32)
[Security.Cryptography.RandomNumberGenerator]::Fill($bridgeTokenBytes)
$bridgeToken = [Convert]::ToHexString($bridgeTokenBytes).ToLowerInvariant()
$supervisorFailed = $false
$napcatStartedAtUtc = [DateTime]::MinValue
$astrbotStartedAtUtc = [DateTime]::MinValue
$napcatRecoveryHistory = @()
$lastAstrBotProbeRecoveryAtUtc = [DateTime]::MinValue

function Test-FreshNapCatQr {
    $qr = Get-Item -LiteralPath $qrPath -ErrorAction SilentlyContinue
    return [bool](
        $qr -and $qr.Length -gt 0 -and
        $qr.LastWriteTimeUtc -ge $script:napcatStartedAtUtc -and
        $qr.LastWriteTimeUtc -ge [DateTime]::UtcNow.AddMinutes(-3)
    )
}

function Test-NapCatAccountRestricted {
    # NapCat keeps the current process' output in these two files.  A QR can
    # legitimately expire while the process remains alive, so distinguish that
    # from QQ explicitly refusing to issue another QR for account-safety reasons.
    foreach ($stream in @('stdout', 'stderr')) {
        $path = Join-Path $logRoot "napcat.$stream.log"
        $tail = Get-Content -LiteralPath $path -Tail 240 -Encoding UTF8 `
            -ErrorAction SilentlyContinue | Out-String
        if ($tail -match '(?i)serverErrorCode["'']?\s*:\s*168' -or
            $tail -match '账号近期存在安全风险.*功能使用受限') {
            return $true
        }
    }
    return $false
}

function Write-RuntimeStatus {
    param(
        [Parameter(Mandatory)] [string] $State,
        [string] $Detail = ''
    )

    $processes = @()
    foreach ($item in @(
        @{ name = 'backend'; process = $script:backendProcess; port = 12394 },
        @{ name = 'astrbot'; process = $script:astrbotProcess; port = 6199 },
        @{ name = 'napcat'; process = $script:napcatProcess; port = 6099 },
        @{ name = 'recovery'; process = $script:recoveryProcess; port = 12395 }
    )) {
        if ($item.process) {
            $processes += [ordered]@{
                name = $item.name
                pid = [int]$item.process.Id
                port = [int]$item.port
            }
        }
    }
    $freshQr = $State -eq 'scan_required' -and (Test-FreshNapCatQr)
    $payload = [ordered]@{
        version = 2
        instance_id = $InstanceId
        state = $State
        detail = $Detail
        updated_at = [DateTimeOffset]::Now.ToString('o')
        project_root = $projectPath
        runtime_root = $runtimePath
        qr_available = [bool]$freshQr
        qr_path = if ($freshQr) { $qrPath } else { '' }
        supervisor_pid = $PID
        processes = $processes
    }
    $temporary = "$statusFile.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusFile -Force
}

function Test-ExactListener {
    param([string] $Address, [int] $Port)
    return [bool](Get-NetTCPConnection -State Listen -LocalAddress $Address -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Wait-ExactListener {
    param([string] $Name, [string] $Address, [int] $Port, [int] $TimeoutSeconds = 60)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-ExactListener -Address $Address -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name 未能在 $Address`:$Port 就绪。"
}

function Resolve-ExactListenerProcess {
    param([string] $Name, [string] $Address, [int] $Port)
    $listeners = @(
        Get-NetTCPConnection -State Listen -LocalAddress $Address -LocalPort $Port `
            -ErrorAction SilentlyContinue
    )
    if ($listeners.Count -ne 1) {
        throw "$Name 在 $Address`:$Port 的监听进程不唯一，拒绝记录未知进程。"
    }
    $listenerPid = [int]$listeners[0].OwningProcess
    $listenerProcess = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
    if (-not $listenerProcess) {
        throw "$Name 的监听进程 PID $listenerPid 已退出。"
    }
    return $listenerProcess
}

function Wait-PortAvailable {
    param([int] $Port, [int] $TimeoutSeconds = 15)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $listener) {
            Start-Sleep -Milliseconds 500
            $confirmation = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if (-not $confirmation) {
                return
            }
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    $ownerPid = [int]$listener.OwningProcess
    $owner = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    $ownerName = if ($owner) { $owner.ProcessName } else { 'unknown' }
    throw "端口 $Port 持续被占用（PID $ownerPid，进程 $ownerName）；拒绝接管未知进程。"
}

function Start-HiddenRuntimeProcess {
    param(
        [string] $Name,
        [string] $FilePath,
        [string[]] $ArgumentList,
        [string] $WorkingDirectory,
        [hashtable] $Environment
    )
    $stdoutPath = Join-Path $logRoot "$Name.stdout.log"
    $stderrPath = Join-Path $logRoot "$Name.stderr.log"
    Rotate-RuntimeLog -Path $stdoutPath -Name $Name -Stream 'stdout'
    Rotate-RuntimeLog -Path $stderrPath -Name $Name -Stream 'stderr'
    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -Environment $Environment `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
}

function Rotate-RuntimeLog {
    param([string] $Path, [string] $Name, [string] $Stream)
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $item -or $item.Length -eq 0) {
        return
    }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $archive = Join-Path $logRoot "$Name.$stamp.$Stream.log"
    for ($attempt = 0; $attempt -lt 10; $attempt += 1) {
        try {
            Move-Item -LiteralPath $Path -Destination $archive -ErrorAction Stop
            return
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }
    throw "无法轮转 $Name 的 $Stream 日志；拒绝覆盖旧故障证据。"
}

function Start-Backend {
    $script:backendProcess = Start-HiddenRuntimeProcess -Name 'backend' `
        -FilePath $pythonExe -ArgumentList @($backendEntry) `
        -WorkingDirectory $projectPath -Environment @{
            APPDATA = (Join-Path $dataPath 'appdata')
            TEMP = $backendTempRoot
            TMP = $backendTempRoot
            RINNE_RENDERER_PROFILE = 'live2d'
            RINNE_HEADLESS_PRIVATE_BRIDGE = 'true'
            RINNE_HEADLESS_PRIVATE_BRIDGE_PORT = '12394'
            RINNE_DATA_ROOT = $conversationDataPath
            RINNE_PRIVATE_BRIDGE_ENABLED = 'true'
            RINNE_PRIVATE_BRIDGE_TOKEN = $bridgeToken
            RINNE_PRIVATE_QQ_ALLOWED_USER_IDS = $script:allowedSenderId
            RINNE_PRIVATE_QQ_INBOX_ROOT = (Join-Path $dataPath 'qq_private\inbox')
            # QQ and desktop share the same configured conversation and library data.
            RINNE_LIBRARY_ROOT = (Join-Path $conversationDataPath 'rinne_library\rinne_01')
            RINNE_QQ_ENABLED = 'false'
            RINNE_PROACTIVE_ENABLED = 'false'
        }
    # Cold starts load the shared long-term memory and local retrieval models.
    # On this machine that can legitimately exceed 90 seconds after reboot.
    Wait-ExactListener -Name '凛祢私人桥接后端' -Address '127.0.0.1' -Port 12394 -TimeoutSeconds 300
    # uv/venv launchers may keep a small shim process in front of the Python
    # interpreter. Track the process that actually owns the listening socket so
    # a later component-only restart cannot leave an orphan backend behind.
    $script:backendProcess = Resolve-ExactListenerProcess `
        -Name '凛祢私人桥接后端' -Address '127.0.0.1' -Port 12394
}

function Start-RecoveryPage {
    $recoveryOrigin = if ($EnableRemoteRecoveryTunnel) {
        $RecoveryPublicBaseUrl
    }
    else {
        'http://127.0.0.1:12395'
    }
    $script:recoveryProcess = Start-HiddenRuntimeProcess -Name 'recovery' `
        -FilePath $pythonExe -ArgumentList @(
            $recoveryEntry,
            '--host', '127.0.0.1',
            '--port', '12395',
            '--status-file', $statusFile,
            '--qr-file', $qrPath,
            '--token-file', $recoveryTokenFile,
            '--url-file', $recoveryUrlFile,
            '--public-origin', $recoveryOrigin
        ) -WorkingDirectory $projectPath -Environment @{
            TEMP = $recoveryTempRoot
            TMP = $recoveryTempRoot
        }
    Wait-ExactListener -Name 'QQ 扫码恢复页' -Address '127.0.0.1' -Port 12395 -TimeoutSeconds 30
    $script:recoveryProcess = Resolve-ExactListenerProcess `
        -Name 'QQ 扫码恢复页' -Address '127.0.0.1' -Port 12395
}

function Test-CompatibleCloudflaredProcess {
    $expectedExe = [IO.Path]::GetFullPath($cloudflaredExe)
    return [bool](
        Get-CimInstance Win32_Process -Filter "Name = 'cloudflared.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ExecutablePath -and
                [IO.Path]::GetFullPath([string]$_.ExecutablePath) -eq $expectedExe -and
                $_.CommandLine -match '(?i)\btunnel\b' -and
                $_.CommandLine -match '(?i)\brun\b'
            } |
            Select-Object -First 1
    )
}

function Start-RecoveryTunnel {
    if (Test-CompatibleCloudflaredProcess) {
        $script:cloudflaredProcess = $null
        return
    }
    $script:cloudflaredProcess = Start-HiddenRuntimeProcess -Name 'cloudflared' `
        -FilePath $cloudflaredExe `
        -ArgumentList @('tunnel', '--config', $cloudflaredConfig, 'run') `
        -WorkingDirectory $cloudflaredPath -Environment @{
            TEMP = $cloudflaredTempRoot
            TMP = $cloudflaredTempRoot
        }
    Start-Sleep -Seconds 3
    if (-not (Test-OwnedProcessAlive $script:cloudflaredProcess)) {
        throw "Cloudflare Tunnel 启动失败；请查看 $(Join-Path $logRoot 'cloudflared.stderr.log')。"
    }
}

function Start-AstrBot {
    # Keep the runtime copy in sync whenever AstrBot is recovered or the brain
    # is restarted, not only when the supervisor itself first launches.
    Get-ChildItem -LiteralPath $pluginSource -File |
        Copy-Item -Destination $pluginTarget -Force
    $script:astrbotStartedAtUtc = [DateTime]::UtcNow
    $script:astrbotProcess = Start-HiddenRuntimeProcess -Name 'astrbot' `
        -FilePath $astrbotExe -ArgumentList @('run') `
        -WorkingDirectory $astrbotRoot -Environment @{
            RINNE_PRIVATE_BRIDGE_TOKEN = $bridgeToken
            RINNE_PRIVATE_BRIDGE_URL = 'http://127.0.0.1:12394/agent/private/v1/turn'
            NO_PROXY = 'localhost,127.0.0.1,::1'
            TEMP = $astrbotTempRoot
            TMP = $astrbotTempRoot
        }
    Wait-ExactListener -Name 'AstrBot WebUI' -Address '127.0.0.1' -Port 6185
    Wait-ExactListener -Name 'AstrBot OneBot' -Address '127.0.0.1' -Port 6199
    $script:astrbotProcess = Resolve-ExactListenerProcess `
        -Name 'AstrBot OneBot' -Address '127.0.0.1' -Port 6199
}

function Resolve-NapCatQuickAccount {
    $webUi = Get-Content -LiteralPath $napcatWebUiConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $configured = [string]$webUi.autoLoginAccount
    if ($configured -match '^\d+$') {
        return $configured
    }

    $accountIds = @(
        Get-ChildItem -LiteralPath (Join-Path $dataPath 'napcat\config') -File -Filter '*.json' |
            ForEach-Object {
                if ($_.Name -match '^(?:napcat|napcat_protocol|onebot11)_(\d+)\.json$') {
                    $Matches[1]
                }
            } |
            Sort-Object -Unique
    )
    if ($accountIds.Count -ne 1) {
        throw '无法从 NapCat 私密配置唯一确定自动登录账号。'
    }
    return [string]$accountIds[0]
}

function Start-NapCat {
    $environment = @{
        NAPCAT_DISABLE_MULTI_PROCESS = '1'
        NAPCAT_WORKDIR = (Join-Path $dataPath 'napcat')
        NAPCAT_QUICK_ACCOUNT = (Resolve-NapCatQuickAccount)
        TEMP = $napcatTempRoot
        TMP = $napcatTempRoot
    }
    $script:napcatStartedAtUtc = [DateTime]::UtcNow
    $script:napcatProcess = Start-HiddenRuntimeProcess -Name 'napcat' `
        -FilePath $napcatExe -ArgumentList @('./index.js') `
        -WorkingDirectory $napcatRoot -Environment $environment
    Wait-ExactListener -Name 'NapCat WebUI' -Address '127.0.0.1' -Port 6099 -TimeoutSeconds 60
    $script:napcatProcess = Resolve-ExactListenerProcess `
        -Name 'NapCat WebUI' -Address '127.0.0.1' -Port 6099
}

function Adopt-NapCat {
    if ($AdoptNapCatPid -le 0) {
        return $false
    }
    $listenerProcess = Resolve-ExactListenerProcess `
        -Name 'NapCat WebUI' -Address '127.0.0.1' -Port 6099
    if ([int]$listenerProcess.Id -ne $AdoptNapCatPid) {
        throw "6099 的监听 PID $($listenerProcess.Id) 与待接管 NapCat PID $AdoptNapCatPid 不一致。"
    }
    $expectedExe = [IO.Path]::GetFullPath($napcatExe)
    if (-not $listenerProcess.Path -or
        [IO.Path]::GetFullPath([string]$listenerProcess.Path) -ne $expectedExe) {
        throw "PID $AdoptNapCatPid 不是预期的 NapCat Node 进程，拒绝接管。"
    }
    $script:napcatProcess = $listenerProcess
    $script:napcatStartedAtUtc = $listenerProcess.StartTime.ToUniversalTime()
    return $true
}

function Stop-OwnedProcess {
    param([object] $Process)
    if (-not $Process) {
        return
    }
    $current = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
    if ($current) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Test-OwnedProcessAlive {
    param([object] $Process)
    if (-not $Process) {
        return $false
    }
    return [bool](Get-Process -Id $Process.Id -ErrorAction SilentlyContinue)
}

function Get-NapCatLoginStatus {
    $webUi = Get-Content -LiteralPath $napcatWebUiConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $plainToken = [string]$webUi.token
    if (-not $plainToken) {
        throw 'NapCat WebUI token 缺失。'
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($plainToken + '.napcat'))
        $hash = [Convert]::ToHexString($hashBytes).ToLowerInvariant()
        $login = Invoke-RestMethod -Method Post `
            -Uri 'http://127.0.0.1:6099/api/auth/login' `
            -ContentType 'application/json' `
            -Body (@{ hash = $hash } | ConvertTo-Json -Compress) -TimeoutSec 5
        $credential = [string]$login.data.Credential
        if ($login.code -ne 0 -or -not $credential) {
            throw 'NapCat WebUI 本机认证失败。'
        }
        $status = Invoke-RestMethod -Method Post `
            -Uri 'http://127.0.0.1:6099/api/QQLogin/CheckLoginStatus' `
            -Headers @{ Authorization = ('Bearer ' + $credential) } -TimeoutSec 5
        if ($status.code -ne 0) {
            throw 'NapCat 登录状态查询失败。'
        }
        return [pscustomobject]@{
            IsLogin = [bool]$status.data.isLogin
            IsOffline = [bool]$status.data.isOffline
        }
    }
    finally {
        if ($hashBytes) { [Array]::Clear($hashBytes, 0, $hashBytes.Length) }
        $plainToken = $null
        $hash = $null
        $credential = $null
        $sha.Dispose()
    }
}

function Test-OneBotEstablished {
    return [bool](Get-NetTCPConnection -State Established -LocalPort 6199 -ErrorAction SilentlyContinue)
}

function Wait-NapCatOnline {
    param([int] $TimeoutSeconds = 45)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $stableChecks = 0
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $status = Get-NapCatLoginStatus
            if ($status.IsLogin -and -not $status.IsOffline -and (Test-OneBotEstablished)) {
                $stableChecks += 1
                if ($stableChecks -ge 3) {
                    return $true
                }
            }
            else {
                $stableChecks = 0
            }
        }
        catch {
            # NapCat may still be starting; the supervisor retries until timeout.
            $stableChecks = 0
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function ConvertTo-UtcTimestamp {
    param([object] $Value)
    # PowerShell 7.5 ConvertFrom-Json converts ISO-8601 strings to DateTime
    # automatically. Older releases leave them as strings, so support both.
    if ($Value -is [DateTimeOffset]) {
        return $Value.UtcDateTime
    }
    if ($Value -is [DateTime]) {
        return $Value.ToUniversalTime()
    }
    if (-not ($Value -is [string]) -or -not $Value.Trim()) {
        return $null
    }
    $parsed = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse(
        [string]$Value,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind,
        [ref]$parsed
    )) {
        return $null
    }
    return $parsed.UtcDateTime
}

function Get-RemoteProbeStatus {
    $empty = [pscustomobject]@{
        HasAttempt = $false
        FreshSuccess = $false
        HardFailure = $false
        Stale = $false
        ConsecutiveFailures = 0
        LastAttemptUtc = $null
    }
    try {
        $payload = Get-Content -LiteralPath $transportHealthFile -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        $referenceStart = if ($script:napcatStartedAtUtc -gt $script:astrbotStartedAtUtc) {
            $script:napcatStartedAtUtc
        }
        else {
            $script:astrbotStartedAtUtc
        }
        $empty.Stale = [DateTime]::UtcNow - $referenceStart -gt [TimeSpan]::FromMinutes(3)
        return $empty
    }

    $lastAttempt = ConvertTo-UtcTimestamp $payload.last_probe_at
    $lastOk = ConvertTo-UtcTimestamp $payload.last_probe_ok_at
    $lastFailed = ConvertTo-UtcTimestamp $payload.last_probe_failed_at
    $failures = 0
    if ($payload.consecutive_probe_failures -is [int] -or
        $payload.consecutive_probe_failures -is [long]) {
        $failures = [Math]::Max(0, [int]$payload.consecutive_probe_failures)
    }
    $referenceStart = if ($script:napcatStartedAtUtc -gt $script:astrbotStartedAtUtc) {
        $script:napcatStartedAtUtc
    }
    else {
        $script:astrbotStartedAtUtc
    }
    $freshSuccess = [bool](
        $lastOk -and
        $lastOk -ge $referenceStart -and
        [DateTime]::UtcNow - $lastOk -le [TimeSpan]::FromMinutes(3) -and
        $failures -eq 0 -and
        (-not $lastFailed -or $lastOk -ge $lastFailed)
    )
    $hardFailure = [bool](
        $lastFailed -and
        $lastFailed -ge $referenceStart -and
        $failures -ge 3 -and
        (-not $lastOk -or $lastFailed -gt $lastOk)
    )
    $stale = [bool](
        [DateTime]::UtcNow - $referenceStart -gt [TimeSpan]::FromMinutes(3) -and
        (-not $lastAttempt -or
            $lastAttempt -lt [DateTime]::UtcNow.AddMinutes(-3) -or
            $lastAttempt -lt $referenceStart)
    )
    return [pscustomobject]@{
        HasAttempt = [bool]$lastAttempt
        FreshSuccess = $freshSuccess
        HardFailure = $hardFailure
        Stale = $stale
        ConsecutiveFailures = $failures
        LastAttemptUtc = $lastAttempt
    }
}

function Wait-RemoteProbe {
    param([int] $TimeoutSeconds = 75)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $probe = Get-RemoteProbeStatus
        if ($probe.FreshSuccess) {
            return $true
        }
        Write-RuntimeStatus -State 'recovering' `
            -Detail '本机组件已连接，正在等待 QQ 服务器的真实响应。'
        Start-Sleep -Seconds 5
    }
    return $false
}

function Test-CanRecoverNapCat {
    $cutoff = [DateTime]::UtcNow.AddHours(-1)
    $script:napcatRecoveryHistory = @(
        $script:napcatRecoveryHistory | Where-Object { $_ -ge $cutoff }
    )
    if ($script:napcatRecoveryHistory.Count -ge 2) {
        return $false
    }
    if ($script:napcatRecoveryHistory.Count -gt 0) {
        $last = $script:napcatRecoveryHistory[-1]
        if ([DateTime]::UtcNow - $last -lt [TimeSpan]::FromMinutes(10)) {
            return $false
        }
    }
    return $true
}

function Invoke-NapCatTransportRecovery {
    $script:napcatRecoveryHistory += [DateTime]::UtcNow
    Write-RuntimeStatus -State 'recovering' `
        -Detail 'QQ 远端连续探测失败，正在执行一次受限的 NapCat 快速重登。'
    Stop-OwnedProcess $script:napcatProcess
    Wait-PortAvailable -Port 6099 -TimeoutSeconds 20
    Start-NapCat
    if (-not (Wait-NapCatOnline -TimeoutSeconds 60)) {
        if (Test-NapCatAccountRestricted) {
            return 'account_restricted'
        }
        if (Test-FreshNapCatQr) {
            return 'scan_required'
        }
        return 'unverified'
    }
    if (Wait-RemoteProbe -TimeoutSeconds 75) {
        return 'verified'
    }
    if (Test-NapCatAccountRestricted) {
        return 'account_restricted'
    }
    if (Test-FreshNapCatQr) {
        return 'scan_required'
    }
    return 'unverified'
}

$requiredPaths = @(
    $pythonExe, $backendEntry, $recoveryEntry, $astrbotExe, $astrbotRoot,
    $napcatExe, $napcatRoot, $pluginSource, $pluginConfigPath,
    $napcatWebUiConfig
)
if ($EnableRemoteRecoveryTunnel) {
    $requiredPaths += $cloudflaredExe,$cloudflaredConfig
}
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "缺少必需路径：$requiredPath"
    }
}

$unsafeNapCatFallback = Join-Path $env:USERPROFILE '.config\QQ'
if (Test-Path -LiteralPath $unsafeNapCatFallback) {
    throw '检测到 NapCat 的 C 盘回退目录，已拒绝启动；请先人工检查。'
}

New-Item -ItemType Directory -Path $logRoot,$recoverySecretRoot -Force | Out-Null
$tempRoots = @($backendTempRoot, $astrbotTempRoot, $napcatTempRoot, $recoveryTempRoot)
if ($EnableRemoteRecoveryTunnel) {
    $tempRoots += $cloudflaredTempRoot
}
foreach ($tempRoot in $tempRoots) {
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
}
New-Item -ItemType Directory -Path $pluginTarget -Force | Out-Null
Get-ChildItem -LiteralPath $pluginSource -File | Copy-Item -Destination $pluginTarget -Force
$pluginConfig = Get-Content -LiteralPath $pluginConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$script:allowedSenderId = [string]$pluginConfig.allowed_sender_id
if (-not $script:allowedSenderId) {
    throw 'AstrBot 私人插件没有配置 allowed_sender_id。'
}

try {
    Write-RuntimeStatus -State 'starting' -Detail '正在检查私人 QQ 运行端口。'
    $portsThatMustBeFree = @(12394, 12395, 6185, 6199)
    if ($AdoptNapCatPid -le 0) {
        $portsThatMustBeFree += 6099
    }
    foreach ($port in $portsThatMustBeFree) {
        Wait-PortAvailable -Port $port
    }
    Write-RuntimeStatus -State 'starting' -Detail '正在启动手机扫码恢复页。'
    Start-RecoveryPage
    if ($EnableRemoteRecoveryTunnel) {
        Write-RuntimeStatus -State 'starting' -Detail '正在启动远程扫码恢复通道。'
        Start-RecoveryTunnel
    }
    Write-RuntimeStatus -State 'starting' `
        -Detail '正在加载凛祢大脑与长期记忆；冷启动可能需要数分钟。'
    Start-Backend
    Write-RuntimeStatus -State 'starting' -Detail '凛祢大脑已就绪，正在启动 AstrBot。'
    Start-AstrBot
    if (-not (Adopt-NapCat)) {
        Write-RuntimeStatus -State 'starting' -Detail '正在启动 NapCat 并恢复 QQ 登录。'
        Start-NapCat
    }

    Write-RuntimeStatus -State 'starting' -Detail '本地组件已就绪，正在验证 QQ 登录。'
    $scanRequired = -not (Wait-NapCatOnline -TimeoutSeconds 60)
    if ($scanRequired -and (Test-NapCatAccountRestricted)) {
        Write-RuntimeStatus -State 'account_restricted' `
            -Detail 'QQ 因账号安全限制拒绝生成二维码；请先在最新版手机 QQ 恢复账号使用。'
    }
    elseif ($scanRequired) {
        Write-RuntimeStatus -State 'scan_required' -Detail 'QQ 登录失效，需要扫码恢复。'
    }
    elseif (Wait-RemoteProbe -TimeoutSeconds 75) {
        Write-RuntimeStatus -State 'ready' -Detail 'QQ 服务器静默探测已通过。'
    }
    else {
        Write-RuntimeStatus -State 'recovering' -Detail 'QQ 本地组件已连接，但远端尚未通过验证。'
    }

    $offlineChecks = 0
    while (-not (Test-Path -LiteralPath $stopFile)) {
        if (-not (Test-OwnedProcessAlive $recoveryProcess) -or -not (Test-ExactListener '127.0.0.1' 12395)) {
            Stop-OwnedProcess $recoveryProcess
            Write-RuntimeStatus -State 'recovering' -Detail '正在恢复扫码恢复页。'
            Start-RecoveryPage
        }
        if ($EnableRemoteRecoveryTunnel -and -not (Test-CompatibleCloudflaredProcess)) {
            Stop-OwnedProcess $cloudflaredProcess
            Write-RuntimeStatus -State 'recovering' -Detail '正在恢复 Cloudflare Tunnel。'
            Start-RecoveryTunnel
        }
        if (-not (Test-OwnedProcessAlive $backendProcess) -or -not (Test-ExactListener '127.0.0.1' 12394)) {
            Stop-OwnedProcess $backendProcess
            Write-RuntimeStatus -State 'recovering' -Detail '正在恢复凛祢后端。'
            Start-Backend
        }
        if (-not (Test-OwnedProcessAlive $astrbotProcess) -or -not (Test-ExactListener '127.0.0.1' 6199)) {
            Stop-OwnedProcess $astrbotProcess
            Write-RuntimeStatus -State 'recovering' -Detail '正在恢复 AstrBot。'
            Start-AstrBot
        }
        if (-not (Test-OwnedProcessAlive $napcatProcess) -or -not (Test-ExactListener '127.0.0.1' 6099)) {
            Stop-OwnedProcess $napcatProcess
            Write-RuntimeStatus -State 'recovering' -Detail '正在恢复 NapCat。'
            Start-NapCat
            $offlineChecks = 0
        }

        $online = $false
        try {
            $loginStatus = Get-NapCatLoginStatus
            $online = $loginStatus.IsLogin -and -not $loginStatus.IsOffline
        }
        catch {
            $online = $false
        }

        if ($online -and (Test-OneBotEstablished)) {
            $offlineChecks = 0
            $probe = Get-RemoteProbeStatus
            if ($probe.FreshSuccess -and $scanRequired) {
                $scanRequired = $false
                Write-RuntimeStatus -State 'ready' -Detail '扫码恢复完成，QQ 远端探测已通过。'
            }
            elseif ($probe.FreshSuccess) {
                Write-RuntimeStatus -State 'ready' -Detail 'QQ 服务器静默探测已通过。'
            }
            elseif ($scanRequired) {
                Write-RuntimeStatus -State 'scan_required' -Detail '自动登录失败，需要扫码恢复。'
            }
            elseif ($probe.HardFailure) {
                if (Test-CanRecoverNapCat) {
                    $recovery = Invoke-NapCatTransportRecovery
                    if ($recovery -eq 'verified') {
                        $scanRequired = $false
                        Write-RuntimeStatus -State 'ready' -Detail 'NapCat 快速重登成功，QQ 远端探测已恢复。'
                    }
                    elseif ($recovery -eq 'scan_required') {
                        $scanRequired = $true
                        Write-RuntimeStatus -State 'scan_required' -Detail '快速重登失败，需要扫码恢复。'
                    }
                    elseif ($recovery -eq 'account_restricted') {
                        $scanRequired = $true
                        Write-RuntimeStatus -State 'account_restricted' `
                            -Detail 'QQ 因账号安全限制拒绝生成二维码；请先在最新版手机 QQ 恢复账号使用。'
                    }
                    else {
                        Write-RuntimeStatus -State 'recovering' `
                            -Detail '快速重登后仍未取得 QQ 远端响应；已进入冷却，避免循环重启。'
                    }
                }
                else {
                    Write-RuntimeStatus -State 'recovering' `
                        -Detail 'QQ 远端连续探测失败；自动重登处于十分钟冷却期。'
                }
            }
            elseif ($probe.Stale) {
                if ([DateTime]::UtcNow - $lastAstrBotProbeRecoveryAtUtc -ge [TimeSpan]::FromMinutes(5)) {
                    $lastAstrBotProbeRecoveryAtUtc = [DateTime]::UtcNow
                    Write-RuntimeStatus -State 'recovering' -Detail 'QQ 远端探针停止更新，正在恢复 AstrBot。'
                    Stop-OwnedProcess $astrbotProcess
                    Wait-PortAvailable -Port 6199 -TimeoutSeconds 20
                    Start-AstrBot
                }
                else {
                    Write-RuntimeStatus -State 'recovering' -Detail 'QQ 远端探针尚未更新，正在等待恢复。'
                }
            }
            else {
                Write-RuntimeStatus -State 'recovering' -Detail '正在验证 QQ 服务器连接。'
            }
        }
        else {
            $offlineChecks += 1
            if (Test-NapCatAccountRestricted) {
                $scanRequired = $true
                Write-RuntimeStatus -State 'account_restricted' `
                    -Detail 'QQ 因账号安全限制拒绝生成二维码；请先在最新版手机 QQ 恢复账号使用。'
            }
            elseif ($scanRequired) {
                Write-RuntimeStatus -State 'scan_required' -Detail '自动登录失败，需要扫码恢复。'
            }
            elseif ($offlineChecks -eq 1) {
                Write-RuntimeStatus -State 'recovering' -Detail 'QQ 账号或 OneBot 连接离线，正在复核。'
            }
            if (-not $scanRequired -and $offlineChecks -ge 3) {
                if (Test-FreshNapCatQr) {
                    $scanRequired = $true
                    Write-RuntimeStatus -State 'scan_required' -Detail 'QQ 登录已失效，需要扫码恢复；NapCat 进程保持运行。'
                }
                elseif (Test-CanRecoverNapCat) {
                    $recovery = Invoke-NapCatTransportRecovery
                    if ($recovery -eq 'verified') {
                        $scanRequired = $false
                        $offlineChecks = 0
                        Write-RuntimeStatus -State 'ready' -Detail 'NapCat 快速重登成功，QQ 远端探测已恢复。'
                    }
                    elseif ($recovery -eq 'scan_required') {
                        $scanRequired = $true
                        Write-RuntimeStatus -State 'scan_required' -Detail '快速重登失败，需要扫码恢复。'
                    }
                    elseif ($recovery -eq 'account_restricted') {
                        $scanRequired = $true
                        Write-RuntimeStatus -State 'account_restricted' `
                            -Detail 'QQ 因账号安全限制拒绝生成二维码；请先在最新版手机 QQ 恢复账号使用。'
                    }
                    else {
                        Write-RuntimeStatus -State 'recovering' `
                            -Detail '快速重登后仍未取得 QQ 响应；已进入冷却。'
                    }
                }
                else {
                    # A brief QQ/MSF or OneBot outage is not evidence that the
                    # local credential is invalid. Killing NapCat here can turn
                    # a recoverable network wobble into a forced QR login.
                    Write-RuntimeStatus -State 'recovering' `
                        -Detail 'QQ 或 OneBot 仍然离线；自动重登处于十分钟冷却期。'
                }
            }
        }
        Start-Sleep -Seconds 10
    }
}
catch {
    $supervisorFailed = $true
    Write-RuntimeStatus -State 'failed' -Detail $_.Exception.Message
}
finally {
    Stop-OwnedProcess $cloudflaredProcess
    Stop-OwnedProcess $recoveryProcess
    Stop-OwnedProcess $napcatProcess
    Stop-OwnedProcess $astrbotProcess
    Stop-OwnedProcess $backendProcess
    Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
    if ($bridgeTokenBytes) {
        [Array]::Clear($bridgeTokenBytes, 0, $bridgeTokenBytes.Length)
    }
    $bridgeToken = $null
    if (-not $supervisorFailed -and (Test-Path -LiteralPath $statusFile)) {
        Write-RuntimeStatus -State 'stopped'
    }
}
