param(
    [ValidateRange(1, 1000)]
    [int]$NumExamples = 1000,

    [int]$RandomSeed = 8240,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunName = 'core_replication_bert_mr_1000'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$AnalysisScript = Join-Path $ProjectRoot 'scripts\analyze_core_replication.py'
$UseModel = Join-Path $ProjectRoot 'models\use-large-v3\model'
$OutputDirectory = Join-Path $ProjectRoot "outputs\$RunName"

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'The isolated environment is missing. Run .\setup.ps1 first.'
}

if (-not (Test-Path -LiteralPath $AnalysisScript)) {
    throw 'The post-processing script is missing: scripts\analyze_core_replication.py'
}

if (-not (Test-Path -LiteralPath (Join-Path $UseModel 'saved_model.pb'))) {
    throw 'USE-Large v3 is missing. Run .\scripts\download_use_large_v3.ps1 first.'
}

if ((Test-Path -LiteralPath $OutputDirectory) -and
    (Get-ChildItem -LiteralPath $OutputDirectory -Force | Select-Object -First 1)) {
    throw "Output folder already contains files: $OutputDirectory. Choose a new -RunName."
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TA_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\textattack'
$env:HF_HOME = Join-Path $ProjectRoot 'data\cache\huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot 'data\cache\torch'
$env:NLTK_DATA = Join-Path $ProjectRoot 'data\cache\nltk'
$env:TFHUB_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\tensorflow_hub'
$env:TF_CPP_MIN_LOG_LEVEL = '3'

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

& $Python -m textattack attack `
    --recipe textfooler `
    --model bert-base-uncased-mr `
    --num-examples $NumExamples `
    --shuffle `
    --random-seed $RandomSeed `
    --checkpoint-interval 50 `
    --checkpoint-dir (Join-Path $OutputDirectory 'checkpoints') `
    --log-to-txt (Join-Path $OutputDirectory 'attack_results.txt') `
    --log-to-csv (Join-Path $OutputDirectory 'attack_results.csv') `
    --log-summary-to-json (Join-Path $OutputDirectory 'attack_summary.json')

if ($LASTEXITCODE -ne 0) {
    throw "Core replication attack failed with exit code $LASTEXITCODE. Check the saved checkpoints."
}

& $Python $AnalysisScript `
    (Join-Path $OutputDirectory 'attack_results.csv') `
    (Join-Path $OutputDirectory 'semantic_similarity.json') `
    --model-path $UseModel

if ($LASTEXITCODE -ne 0) {
    throw "Semantic-similarity analysis failed with exit code $LASTEXITCODE."
}

Write-Host "Core replication completed. Results: $OutputDirectory"
