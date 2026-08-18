# Copy this file to local_config\start_rinne_private.ps1 before editing it.
# local_config is ignored by Git, so your private key and local paths stay local.

$env:RINNE_LLM_BASE_URL = "https://your-provider.example/v1"
$env:RINNE_LLM_API_KEY = "replace-with-your-private-api-key"
$env:RINNE_LLM_MODEL = "replace-with-your-model-name"

# Diary, weekly and monthly memory generation use a complete chat-completions URL.
$env:DIARY_LLM_API_URL = "https://your-provider.example/v1/chat/completions"
$env:DIARY_LLM_API_KEY = $env:RINNE_LLM_API_KEY
$env:DIARY_LLM_MODEL = $env:RINNE_LLM_MODEL

# Optional: use an existing GPT-SoVITS integrated package outside this repository.
# Remove this line when GPT-SoVITS is installed at the default path documented in README.md.
$env:RINNE_GPT_SOVITS_ROOT = "D:\path\to\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"

# CPU fallback (much slower): uncomment both lines when CUDA is unavailable.
# $env:RINNE_TTS_DEVICE = "cpu"
# $env:RINNE_TTS_HALF = "false"

& (Join-Path $PSScriptRoot "..\scripts\start_rinne_backend.ps1")
