$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

foreach ($name in @("RINNE_LLM_BASE_URL", "RINNE_LLM_API_KEY", "RINNE_LLM_MODEL")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
        throw "$name is not set in this PowerShell session. Follow the API section in README.md first."
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not available. Install uv and reopen PowerShell."
}

Set-Location -LiteralPath $projectRoot
& uv run run_server.py --verbose
