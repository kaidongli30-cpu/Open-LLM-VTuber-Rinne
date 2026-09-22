param(
    [string]$ModelId = "Qwen/Qwen3-ASR-0.6B",
    [string]$ModelDirectory = "models/Qwen3-ASR-0.6B",
    [string]$PythonIndexUrl = "https://pypi.org/simple",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [switch]$BypassProxy
)

$ErrorActionPreference = "Stop"
if ($BypassProxy) {
    # Apply only to this installer process when a stale system proxy blocks downloads.
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourcePython = Join-Path $projectRoot ".venv/Scripts/python.exe"
$workerEnvironment = Join-Path $projectRoot ".venv-qwen3-asr"
$workerPython = Join-Path $workerEnvironment "Scripts/python.exe"
$modelPath = Join-Path $projectRoot $ModelDirectory

if (-not (Test-Path -LiteralPath $sourcePython -PathType Leaf)) {
    throw "Main project Python was not found at $sourcePython"
}

if (-not (Test-Path -LiteralPath $workerPython -PathType Leaf)) {
    Write-Host "Creating isolated Qwen3-ASR Python environment..."
    & $sourcePython -m venv $workerEnvironment
}

Write-Host "Installing the official local Qwen3-ASR runtime..."
& $workerPython -m pip install --index-url $PythonIndexUrl --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update pip from the configured package mirror."
}
& $workerPython -m pip install --index-url $PythonIndexUrl --no-deps "qwen-asr==0.0.6"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Qwen3-ASR package."
}
# qwen-asr declares Gradio and Flask for its optional demo applications. Rinne
# only needs the Transformers inference path, so install that smaller runtime.
& $workerPython -m pip install --index-url $PythonIndexUrl `
    "transformers==4.57.6" `
    "accelerate==1.12.0" `
    "qwen-omni-utils==0.0.9" `
    "nagisa==0.2.11" `
    "soynlp==0.0.493" `
    "librosa==0.11.0" `
    "soundfile==0.14.0" `
    "sox==1.5.0" `
    "pytz" `
    "modelscope>=1.31.0"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Qwen3-ASR runtime."
}

& $workerPython -c "import torch; assert '+cu126' in torch.__version__ and torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the CUDA 12.6 build of PyTorch..."
    & $workerPython -m pip install --index-url $TorchIndexUrl `
        --no-deps --force-reinstall "torch==2.14.0"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install CUDA-enabled PyTorch."
    }
}
& $workerPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print('CUDA ready:', torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installed, but it cannot access the NVIDIA GPU."
}

$modelWeights = Join-Path $modelPath "model.safetensors"
if (-not (Test-Path -LiteralPath $modelWeights -PathType Leaf)) {
    Write-Host "Downloading $ModelId from ModelScope to $modelPath ..."
    $modelscope = Join-Path $workerEnvironment "Scripts/modelscope.exe"
    & $modelscope download --model $ModelId --local_dir $modelPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download the Qwen3-ASR model from ModelScope."
    }
} else {
    Write-Host "Model directory already exists; leaving it unchanged: $modelPath"
}

Write-Host "Qwen3-ASR local runtime is ready."
Write-Host "Python: $workerPython"
Write-Host "Model:  $modelPath"
