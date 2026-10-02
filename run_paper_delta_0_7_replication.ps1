param(
    [ValidateRange(1, 1000)]
    [int]$NumExamples = 1000,

    [int]$RandomSeed = 8240,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunName = 'core_replication_PAPER_DELTA_0_7_bert_mr_1000'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$AttackFile = Join-Path $ProjectRoot 'scripts\textfooler_paper_delta_0_7.py'
$AnalysisScript = Join-Path $ProjectRoot 'scripts\analyze_core_replication.py'
$UseModel = Join-Path $ProjectRoot 'models\use-large-v3\model'
$OutputDirectory = Join-Path $ProjectRoot "outputs\$RunName"

foreach ($RequiredFile in @($Python, $AttackFile, $AnalysisScript)) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Required file is missing: $RequiredFile"
    }
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

$Metadata = [ordered]@{
    experiment_tag = 'PAPER_STATED_WORD_DELTA_0_7'
    purpose = 'Controlled comparison with the earlier maintained-recipe delta=0.5 run'
    model = 'bert-base-uncased-mr'
    dataset = 'rotten_tomatoes test split selected by the model shortcut'
    num_examples = $NumExamples
    random_seed = $RandomSeed
    shuffled = $true
    max_word_candidates_N = 50
    minimum_word_embedding_cosine_delta = 0.7
    comparison_baseline = 'outputs/core_replication_bert_mr_1000 (MAINTAINED_TEXTATTACK_DELTA_0_5)'
    changed_from_baseline = 'WordEmbeddingDistance min_cos_sim only: 0.5 -> 0.7'
}
$Metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'EXPERIMENT_METADATA.json') -Encoding utf8
Set-Content -LiteralPath (Join-Path $OutputDirectory 'TAG_PAPER_DELTA_0_7.txt') `
    -Value 'PAPER_STATED_WORD_DELTA_0_7: N=50, minimum word-embedding cosine delta=0.7.' `
    -Encoding utf8

& $Python -m textattack attack `
    --attack-from-file "$AttackFile^Attack" `
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
    throw "Paper-delta replication failed with exit code $LASTEXITCODE. Check the saved checkpoints."
}

& $Python $AnalysisScript `
    (Join-Path $OutputDirectory 'attack_results.csv') `
    (Join-Path $OutputDirectory 'semantic_similarity.json') `
    --model-path $UseModel

if ($LASTEXITCODE -ne 0) {
    throw "Semantic-similarity analysis failed with exit code $LASTEXITCODE."
}

Write-Host "Paper-delta replication completed. Results: $OutputDirectory"
