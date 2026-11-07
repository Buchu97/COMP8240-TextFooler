param(
    [int]$PythonMinor = 11
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements-modern.txt'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    py "-3.$PythonMinor" -m venv $VenvPath
}

& $VenvPython -m pip install --upgrade pip wheel 'setuptools==80.9.0'
& $VenvPython -m pip install 'torch==2.7.1' --index-url 'https://download.pytorch.org/whl/cu128'
& $VenvPython -m pip install -r $Requirements

$env:TA_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\textattack'
$env:HF_HOME = Join-Path $ProjectRoot 'data\cache\huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot 'data\cache\torch'
$env:NLTK_DATA = Join-Path $ProjectRoot 'data\cache\nltk'
$env:TFHUB_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\tensorflow_hub'

@(
    $env:TA_CACHE_DIR,
    $env:HF_HOME,
    $env:TORCH_HOME,
    $env:NLTK_DATA,
    $env:TFHUB_CACHE_DIR,
    (Join-Path $ProjectRoot 'tmp')
) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

$nltkCache = $env:NLTK_DATA.Replace("'", "''")
& $VenvPython -c "import nltk; resources=('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','stopwords','omw','omw-1.4','universal_tagset','wordnet','punkt'); [nltk.download(x, download_dir=r'$nltkCache') for x in resources]"

& $VenvPython -c "import torch, tensorflow as tf, transformers, datasets, textattack; from textattack.attack_recipes import TextFoolerJin2019; print('TextFooler imports: OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('TensorFlow:', tf.__version__); print('Transformers:', transformers.__version__); print('Datasets:', datasets.__version__)"

Write-Host 'Setup complete.'
