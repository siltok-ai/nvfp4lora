# Dense NVFP4 LoRA training CLI

Qwen3.8 actual100/resume/rebuild and Qwen3.6 actual10/rebuild passed. The
commands below reproduce those workflows in a new recorded output directory.
Actual commands, source snapshots and exit codes remain authoritative in the manifests.

Use the independent native Windows environment registered for this task. Keep
requirements/windows-cu130.lock and all39 hashed packages unchanged. Do not
redirect the original MoE environment's editable checkout.

The base is the fixed deployment NVFP4 source; never re-quantize or replace it
with a BF16 checkpoint. Dense uses family=qwen3_5_dense, raw streamed NVFP4/FP8
layers, ordinary CPU embedding, D-ALL r16 and FP32 LoRA/AdamW weight_decay=0.
Non-thinking text prompts use the checkpoint's own tokenizer/template.

Every command starts a new recorded run from the repository root. Long jobs
run in the background and are observed through their manifest/log. An
observation timeout is not a process failure. After a real interruption, resume
the latest complete optimizer checkpoint instead of repeating finished steps.

```powershell
$py = 'C:/workspace/envs/T-20261029-nvfp4-dense/train/Scripts/python.exe'
$bundle = 'C:/workspace/models/bundles/qwen3.8-27b-nvfp4-lora/319f741cce68d791-train-v1'
$corpus = 'C:/workspace/data/datasets/nvfp4lora/canary-replay-qwen38-27b-v1'
$runs = 'C:/workspace/runs/T-20261029-nvfp4-dense'
$commit = (& git rev-parse HEAD).Trim()
$run = Join-Path $runs ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-functional100')
& $py -B tools/run_recorded.py --output $run -- tools/train_lora.py `
  --bundle $bundle --data "$corpus/train.jsonl" --config configs/dense27b-all-r16.json `
  --checkpoint-steps 10 20 30 40 50 60 70 80 90 100 --evaluation-steps 10 20 50 100 `
  --eval-corpus $corpus --eval-batch-size 32 --code-commit $commit --output '{run}/results'
```

The owner approved holdout>=45/50 and replay NLL increase<=0.01 on2026-10-30.
Preserve the actual quote/provenance in effect-threshold-owner-decision.json.
Short qualification/stress runs and their step1/2 publications do not substitute
for the formal100-step adapter.

Use the saved logical configuration and identical data content for resume.
The data path may change; its content/tokenizer/template identity may not.

```powershell
$formal = 'C:\workspace\runs\T-20261029-nvfp4-dense\20261030T020753Z-d6-functional100\results'
$resume = Join-Path $runs ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-resume50')
& $py -B tools/run_recorded.py --output $resume -- tools/train_lora.py `
  --bundle $bundle --data "$corpus/train.jsonl" `
  --config "$formal/checkpoints/step-000050/train_config.json" `
  --resume "$formal/checkpoints/step-000050" --stop-at 100 --checkpoint-steps 100 `
  --evaluation-steps 100 --eval-corpus $corpus --eval-batch-size 32 `
  --code-commit $commit --output '{run}/results'
$compare = Join-Path $runs ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-resume50-comparison')
& $py -B tools/run_recorded.py --output $compare -- tools/compare_real_resume.py `
  --baseline $formal --resumed "$resume/results" --output '{run}/comparison.json'
```

The comparison checks all ten state/history/probe/DeltaW groups E2. Preserve
failed raw evidence. Changing max_steps, data or optimizer semantics creates a
different training identity and cannot be described as exact resume.

Publication contains adapter_model.safetensors, adapter_config.json and a
manifest with layout nvfp4-lora-dense-v1. The legacy MoE reader must reject it.
Publication does not contain optimizer/RNG/sampler state; use checkpoints to
resume. Reconstruct the model from only the original preprocessed bundle and
publish directory:

```powershell
$reload = Join-Path $runs ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-release-rebuild')
& $py -B tools/run_recorded.py --output $reload -- tools/verify_training_release.py `
  --bundle $bundle --training-results $formal --output '{run}/results'
$effect = Join-Path $runs ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-published-effects')
& $py -B tools/run_recorded.py --output $effect -- tools/evaluate_training_adapter.py `
  --bundle $bundle --adapter "$formal/publish" --corpus $corpus `
  --baseline-evaluation "$formal/evaluation-step000000-disabled.json" `
  --minimum-canary-accuracy 0.9 --max-replay-nll-increase 0.01 --output '{run}/results'
```

FP8 ranges compare actual activations with448 times the original input_scale.
An OBSERVED result is a completed risk observation, not a no-clipping or W8A8
equivalence certificate. Native FreeToken inference is qualified independently
on its own fork/venv, with the final adapter and the same corpus; training NLL
and native exact answer counts have different meanings.

Qwen3.6 uses its own fixed revision, verified source shards, bundle, tokenizer
and regenerated replay corpus. Its minimum campaign is10 real steps after the
Qwen3.8 DA6.6 gate; an optional full100-step campaign requires the owner's choice.


Qwen3.6本轮原始发布位于`C:\workspace\runs\T-20261029-nvfp4-dense\20261030T050306Z-qwen36-short10\results\publish`，使用自己的bundle
C:/workspace/models/bundles/qwen3.6-27b-nvfp4-lora/0893e1606ff3d5f9-train-v1
与canary-replay-qwen36-27b-v1。命令沿用上述train_lora.py，使用
`--stop-at 10 --checkpoint-steps 1 5 10 --evaluation-steps 10`，并把bundle/data/
eval-corpus都指向Qwen3.6自己的路径。
其logical max100保持不变，保存1/5/10；只有未来明确要求额外100及门槛后
再从step10继续，不能从max1资格探针静默延长。

完整制品目标`C:\workspace\artifacts\nvfp4lora-dense\0.0.1\20261030T054636Z`复制固定语料后路径可变，内容/模板/
分词器身份不可变；从制品checkpoint的train_config.json恢复。仅publish目录
足以重建权重和probe，但不能恢复Adam/RNG。实际source snapshots/exit codes
与全部精确命令见对应evidence manifest。
