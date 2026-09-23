[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $RecoveryPublicBaseUrl
)

$ErrorActionPreference = 'Stop'
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
$urlFile = Join-Path $dataPath 'qq_private\secrets\qr_recovery_url.txt'
if (-not (Test-Path -LiteralPath $urlFile)) {
    throw '扫码页地址还没有生成。请先启动私人 QQ 凛祢。'
}

$recoveryUrl = (Get-Content -LiteralPath $urlFile -Raw -Encoding UTF8).Trim()
$parsed = $null
if (-not [Uri]::TryCreate($recoveryUrl, [UriKind]::Absolute, [ref]$parsed)) {
    throw '扫码页地址格式异常，已拒绝打开。'
}
$expectedPath = $parsed.AbsolutePath.StartsWith(
    '/qq-recovery/',
    [StringComparison]::Ordinal
)
$localPage = [bool](
    $parsed.Scheme -eq 'http' -and
    $parsed.Host -eq '127.0.0.1' -and
    $parsed.Port -eq 12395
)
$remotePage = $false
if ($RecoveryPublicBaseUrl) {
    $publicUri = $null
    if (-not [Uri]::TryCreate($RecoveryPublicBaseUrl, [UriKind]::Absolute, [ref]$publicUri) -or
        $publicUri.Scheme -ne 'https' -or $publicUri.AbsolutePath -ne '/' -or
        $publicUri.Query -or $publicUri.Fragment -or $publicUri.UserInfo -or
        -not $publicUri.IsDefaultPort) {
        throw 'RecoveryPublicBaseUrl 必须是 HTTPS 网站根地址。'
    }
    $remotePage = [bool](
        $parsed.Scheme -eq 'https' -and
        $parsed.Host -eq $publicUri.Host -and
        $parsed.IsDefaultPort
    )
}
if (-not $expectedPath -or (-not $localPage -and -not $remotePage)) {
    throw '扫码页地址不属于允许的本地或远程恢复入口，已拒绝打开。'
}
if ($localPage) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync('127.0.0.1', 12395)
        $available = $connection.Wait(1500) -and $client.Connected
    }
    catch {
        $available = $false
    }
    finally {
        $client.Dispose()
    }
    if (-not $available) {
        throw '本地扫码页尚未运行。请先双击“启动私人QQ凛祢.bat”。'
    }
}

Start-Process -FilePath $parsed.AbsoluteUri
if ($localPage) {
    Write-Host '已在本机默认浏览器打开 QQ 扫码页；请用手机 QQ 摄像头扫描。' -ForegroundColor Green
}
else {
    Write-Host '已在默认浏览器打开远程 QQ 扫码页。' -ForegroundColor Green
}
$recoveryUrl = $null
