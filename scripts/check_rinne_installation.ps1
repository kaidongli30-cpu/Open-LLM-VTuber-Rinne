param(
    [switch]$SkipVoice
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$script:failed = $false

function Show-Check {
    param(
        [bool]$Ok,
        [string]$Name,
        [string]$Detail
    )

    if ($Ok) {
        Write-Host "[OK]   $Name - $Detail" -ForegroundColor Green
    }
    else {
        Write-Host "[MISS] $Name - $Detail" -ForegroundColor Red
        $script:failed = $true
    }
}

function Get-CommandVersion {
    param(
        [string]$Command,
        [string[]]$Arguments = @("--version")
    )

    $item = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $item) {
        return $null
    }

    try {
        return ((& $Command @Arguments 2>&1 | Select-Object -First 1) -join " ").Trim()
    }
    catch {
        return $item.Source
    }
}

Write-Host "Rinne Windows installation preflight" -ForegroundColor Cyan
Write-Host "Project: $projectRoot"
Write-Host "This script checks local programs and required files. It never prints API Key values.`n"

$os64 = [Environment]::Is64BitOperatingSystem
Show-Check $os64 "64-bit Windows" ([Environment]::OSVersion.VersionString)

$gitVersion = Get-CommandVersion "git"
Show-Check ($null -ne $gitVersion) "Git" ($(if ($gitVersion) { $gitVersion } else { "install Git for Windows" }))

$uvVersion = Get-CommandVersion "uv"
Show-Check ($null -ne $uvVersion) "uv" ($(if ($uvVersion) { $uvVersion } else { "install uv, then reopen PowerShell" }))

$ffmpegVersion = Get-CommandVersion "ffmpeg" @("-version")
Show-Check ($null -ne $ffmpegVersion) "FFmpeg" ($(if ($ffmpegVersion) { $ffmpegVersion } else { "install FFmpeg and add it to PATH" }))

$ollamaVersion = Get-CommandVersion "ollama"
if ($ollamaVersion) {
    Show-Check $true "Ollama" $ollamaVersion
}
else {
    Write-Host "[INFO] Ollama is optional only when conf.yaml sets translate_audio: False." -ForegroundColor Yellow
}

$requiredProjectFiles = @(
    "conf.yaml",
    "run_server.py",
    "pyproject.toml",
    "frontend",
    "live2d-models\rinne\rinne.model3.json",
    "avatars\rinne.jpg",
    "Rinne_model\rinne_voice_runtime_bundle\emotion_references\rinne_default.wav"
)

foreach ($relativePath in $requiredProjectFiles) {
    $fullPath = Join-Path $projectRoot $relativePath
    Show-Check (Test-Path -LiteralPath $fullPath) $relativePath $fullPath
}

if (-not $SkipVoice) {
    $defaultVoiceRoot = Join-Path $projectRoot "GPT-SoVITS-v2pro-20250604"
    $nestedVoiceRoot = Join-Path $defaultVoiceRoot "GPT-SoVITS-v2pro-20250604"
    if (-not (Test-Path -LiteralPath (Join-Path $defaultVoiceRoot "runtime\python.exe"))) {
        $defaultVoiceRoot = $nestedVoiceRoot
    }
    $voiceRoot = [Environment]::GetEnvironmentVariable("RINNE_GPT_SOVITS_ROOT", "Process")
    if ([string]::IsNullOrWhiteSpace($voiceRoot)) {
        $voiceRoot = $defaultVoiceRoot
    }

    $voiceFiles = @(
        "runtime\python.exe",
        "api_v2.py",
        "GPT_weights_v2\rinne_e15.ckpt",
        "SoVITS_weights_v2\rinne_e8_s456.pth"
    )

    foreach ($relativePath in $voiceFiles) {
        $fullPath = Join-Path $voiceRoot $relativePath
        Show-Check (Test-Path -LiteralPath $fullPath) "GPT-SoVITS $relativePath" $fullPath
    }
}

Write-Host ""
if ($script:failed) {
    Write-Host "Preflight found missing required items. Follow README.md, then rerun this script." -ForegroundColor Red
    exit 1
}

Write-Host "Preflight passed. You can start TTS and the backend." -ForegroundColor Green
exit 0
