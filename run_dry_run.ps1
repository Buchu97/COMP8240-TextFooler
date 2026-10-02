param(
    [ValidateRange(1, 1000)]
    [int]$NumExamples = 5,

    [int]$RandomSeed = 8240
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'The isolated environment is missing. Run .\setup.ps1 first.'
}

$env:TA_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\textattack'
$env:HF_HOME = Join-Path $ProjectRoot 'data\cache\huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot 'data\cache\torch'
$env:NLTK_DATA = Join-Path $ProjectRoot 'data\cache\nltk'
$env:TFHUB_CACHE_DIR = Join-Path $ProjectRoot 'data\cache\tensorflow_hub'
$env:TF_CPP_MIN_LOG_LEVEL = '2'

$OutputDirectory = Join-Path $ProjectRoot 'outputs\dry_run'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

& $Python -m textattack attack `
    --recipe textfooler `
    --model bert-base-uncased-mr `
    --num-examples $NumExamples `
    --random-seed $RandomSeed `
    --log-to-txt (Join-Path $OutputDirectory 'attack_results.txt') `
    --log-to-csv (Join-Path $OutputDirectory 'attack_results.csv') `
    --log-summary-to-json (Join-Path $OutputDirectory 'attack_summary.json')

if ($LASTEXITCODE -ne 0) {
    throw "TextFooler dry run failed with exit code $LASTEXITCODE."
}

Write-Host "Dry run completed. Results: $OutputDirectory"
