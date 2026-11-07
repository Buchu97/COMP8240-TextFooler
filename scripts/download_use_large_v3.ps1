$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BaseDirectory = Join-Path $ProjectRoot 'models\use-large-v3'
$ModelDirectory = Join-Path $BaseDirectory 'model'
$Archive = Join-Path $BaseDirectory 'universal-sentence-encoder-tensorflow1-large-v3.tar.gz'
$PartialArchive = "$Archive.part"
$ExpectedMd5 = 'EBE7986BD072D690571463B7220AAC4F'
$DownloadUrl = 'https://www.kaggle.com/api/v1/models/google/universal-sentence-encoder/TensorFlow1/large/3/download'

New-Item -ItemType Directory -Path $BaseDirectory -Force | Out-Null

if (-not (Test-Path -LiteralPath $Archive)) {
    curl.exe -L --fail --retry 3 --output $PartialArchive $DownloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "USE-Large v3 download failed with exit code $LASTEXITCODE."
    }
    Move-Item -LiteralPath $PartialArchive -Destination $Archive
}

$ActualMd5 = (Get-FileHash -LiteralPath $Archive -Algorithm MD5).Hash
if ($ActualMd5 -ne $ExpectedMd5) {
    throw "USE-Large v3 checksum mismatch. Expected $ExpectedMd5, received $ActualMd5."
}

if (-not (Test-Path -LiteralPath (Join-Path $ModelDirectory 'saved_model.pb'))) {
    if ((Test-Path -LiteralPath $ModelDirectory) -and
        (Get-ChildItem -LiteralPath $ModelDirectory -Force | Select-Object -First 1)) {
        throw "Model destination is not empty: $ModelDirectory"
    }
    New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null
    tar.exe -xzf $Archive -C $ModelDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "USE-Large v3 extraction failed with exit code $LASTEXITCODE."
    }
}

Write-Host "USE-Large v3 is ready: $ModelDirectory"
