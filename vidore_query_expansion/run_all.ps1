param(
    [string]$Python = 'python',
    [string]$DataDir = "$PSScriptRoot/.cache/data",
    [string]$ModelDir = "$PSScriptRoot/.cache/model",
    [string]$VectorsDir = "$PSScriptRoot/.cache/page_vectors",
    [string]$CacheDir = "$PSScriptRoot/.cache/experiment"
)
$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '4'
$env:DSPY_CACHEDIR = "$CacheDir/dspy_cache"
if (!(Test-Path -LiteralPath "$VectorsDir/query.npy")) {
    & $Python "$PSScriptRoot/benchmark_core.py" download --data $DataDir --model $ModelDir
    if ($LASTEXITCODE -ne 0) { throw 'Data/model download failed.' }
    & $Python "$PSScriptRoot/benchmark_core.py" encode --data $DataDir --model $ModelDir --cache $VectorsDir
    if ($LASTEXITCODE -ne 0) { throw 'Page/query embedding failed.' }
}
foreach ($stage in @('prepare','baselines','optimize','generate_dev','embed','tune','generate_all','embed','evaluate')) {
    & $Python "$PSScriptRoot/experiment.py" $stage --data $DataDir --model $ModelDir --vectors $VectorsDir --cache $CacheDir
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage" }
}
& $Python "$PSScriptRoot/audit_results.py"
if ($LASTEXITCODE -ne 0) { throw 'Independent audit failed.' }
& $Python "$PSScriptRoot/check_encoder.py" --data $DataDir --model $ModelDir --vectors $VectorsDir --cache $CacheDir
if ($LASTEXITCODE -ne 0) { throw 'Encoder spot check failed.' }
& $Python "$PSScriptRoot/supplement.py" --data $DataDir --model $ModelDir --vectors $VectorsDir --cache $CacheDir
if ($LASTEXITCODE -ne 0) { throw 'Supplementary analysis failed.' }
