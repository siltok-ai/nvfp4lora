param(
    [string]$Python = 'C:\workspace\envs\T-20261027-nvfp4-lora\train\Scripts\python.exe',
    [string]$Bundle = 'C:\workspace\models\bundles\qwen3.6-nvfp4-lora\1355db6a052410cf-train-v2',
    [string]$Source = 'C:\workspace\models\inputs\qwen3.6-35b-a3b-nvfp4\1355db6a052410cfd62085d94b58866fd0f2c3c5',
    [string]$Corpus = 'C:\workspace\data\datasets\nvfp4lora\canary-replay-v1',
    [string]$TaskRoot = 'C:\workspace\tasks\T-20261027-nvfp4-lora',
    [string]$RunsRoot = 'C:\workspace\runs\T-20261027-nvfp4-lora',
    [switch]$ValidateOnly
)

# 顺序执行真实100步/50->100验证；任一子进程失败即停止，不覆盖或重试旧run。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$gitExecutable = 'C:\Program Files\Git\cmd\git.exe'
$codeCommit = & $gitExecutable rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve source commit' }
foreach ($path in @($Python, $Bundle, $Source, $Corpus, $TaskRoot, $RunsRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required input missing: $path" }
}
$trainingData = Join-Path $Corpus 'train.jsonl'
$dataHash = (Get-FileHash -LiteralPath $trainingData -Algorithm SHA256).Hash.ToLowerInvariant()
$origins = @{}
foreach ($variant in @('c3', 'expert')) {
    $pointer = Join-Path $TaskRoot "real-$variant-baseline-run.txt"
    $origins[$variant] = (Get-Content -LiteralPath $pointer -Raw).Trim()
    $results = Join-Path $origins[$variant] 'results'
    $summary = Get-Content -LiteralPath (Join-Path $results 'summary.json') -Raw | ConvertFrom-Json
    $config = Get-Content -LiteralPath (Join-Path $results 'train_config.json') -Raw | ConvertFrom-Json
    $expectedExpertRank = if ($variant -eq 'c3') { 0 } else { 2 }
    if ($summary.status -ne 'PASS' -or $summary.current_step -ne 4 -or $config.max_steps -ne 100 -or
        $config.dataset.file_sha256 -ne $dataHash -or $config.accumulation_steps -ne 8 -or
        $config.build.expert_rank -ne $expectedExpertRank) {
        throw "Origin is not the accepted step4 / logical100 baseline: $variant"
    }
}
if ($ValidateOnly) {
    [ordered]@{status='VALIDATED_INPUTS_ONLY'; code_commit=$codeCommit; origins=$origins;
        planned='C3 then expert-r2: 4->100, new process 50->100, full E2 comparison, release deltaW, effect measurements, FP8 ranges'} | ConvertTo-Json -Depth 4
    return
}

function Invoke-Recorded {
    param([string]$Name, [string]$Pointer, [string[]]$CommandArguments)
    $run = Join-Path $RunsRoot ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + $Name)
    $run | Set-Content -LiteralPath (Join-Path $TaskRoot $Pointer) -Encoding UTF8
    & $Python -B tools/run_recorded.py --output $run -- @CommandArguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Stage failed; inspect $run. Subsequent stages were not started." }
    return $run
}

foreach ($variant in @('c3', 'expert')) {
    $purpose = if ($variant -eq 'c3') { 'real-c3' } else { 'real-expert-r2' }
    $oldResults = Join-Path $origins[$variant] 'results'
    $common = @('--bundle', $Bundle, '--data', $trainingData, '--output', '{run}\results', '--code-commit', $codeCommit)
    $full = Invoke-Recorded -Name "$purpose-functional100" -Pointer "real-$variant-functional100-run.txt" -CommandArguments (
        @('tools/train_lora.py') + $common + @('--config', (Join-Path $oldResults 'train_config.json'),
            '--resume', (Join-Path $oldResults 'checkpoints\step-000004'), '--stop-at', '100',
            '--checkpoint-steps', '20', '50', '100', '--evaluation-steps', '10', '20', '50', '100',
            '--eval-corpus', $Corpus, '--snapshot-first-micro'))
    $fullResults = Join-Path $full 'results'
    $resumed = Invoke-Recorded -Name "$purpose-resume50" -Pointer "real-$variant-resume50-run.txt" -CommandArguments (
        @('tools/train_lora.py') + $common + @('--config', (Join-Path $fullResults 'train_config.json'),
            '--resume', (Join-Path $fullResults 'checkpoints\step-000050'), '--stop-at', '100',
            '--checkpoint-steps', '100', '--evaluation-steps', '100', '--eval-corpus', $Corpus))
    $null = Invoke-Recorded -Name "$purpose-resume50-comparison" -Pointer "real-$variant-resume50-comparison-run.txt" -CommandArguments @(
        'tools/compare_real_resume.py', '--baseline', $fullResults, '--resumed', (Join-Path $resumed 'results'), '--output', '{run}\comparison.json')
    $null = Invoke-Recorded -Name "$purpose-functional-release" -Pointer "real-$variant-functional-release-run.txt" -CommandArguments @(
        'tools/verify_training_release.py', '--bundle', $Bundle, '--training-results', $fullResults, '--output', '{run}\results')
    $adapter = Join-Path $fullResults 'publish'
    # 阈值未由用户确认时只测量；不把本脚本完成解释成效果已正式验收。
    $null = Invoke-Recorded -Name "$purpose-functional-evaluation" -Pointer "real-$variant-functional-evaluation-run.txt" -CommandArguments @(
        'tools/evaluate_training_adapter.py', '--bundle', $Bundle, '--adapter', $adapter, '--corpus', $Corpus,
        '--baseline-evaluation', (Join-Path $oldResults 'evaluation-step000000-disabled.json'), '--output', '{run}\results')
    $null = Invoke-Recorded -Name "$purpose-functional-fp8" -Pointer "real-$variant-functional-fp8-run.txt" -CommandArguments @(
        'tools/audit_fp8_input_ranges.py', '--bundle', $Bundle, '--source', $Source, '--adapter', $adapter,
        '--data', $trainingData, '--output', '{run}\results')
}
Write-Output 'All campaign processes exited successfully. Review raw metrics, effect thresholds, FP8 risks and remaining acceptance gates separately.'
