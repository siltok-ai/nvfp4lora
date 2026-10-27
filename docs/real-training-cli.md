# 真实单 GPU 训练入口

`tools/train_lora.py` 使用固定原始 NVFP4 基座的预处理结果、JSONL 对话数据和 JSON 逻辑配置。非专家 PEFT 与自有专家参数共同进入标准 AdamW；每个 optimizer step 按总监督 token 数归一化，累积8、非重入检查点、分块CE和异步两槽为当前基线。

```powershell
python -B tools/run_recorded.py --output <新的run目录> -- tools/train_lora.py `
  --bundle <真实预处理目录> --data <语料目录>/train.jsonl `
  --config configs/real-c3.json --output '{run}/results' `
  --stop-at 4 --checkpoint-steps 1 2 4 --evaluation-steps 4 `
  --eval-corpus <语料目录> --snapshot-first-micro --code-commit <完整提交>
```

全专家版本选 `configs/real-expert-r2.json`。`max_steps` 是逻辑上限；`--stop-at` 只控制本次执行到哪个 optimizer step，不改变逻辑配置，可从保存边界继续。输出、保存/评测时机属于执行信息。数据、基座、数值设置和版本进入恢复身份；续训仍须匹配同一逻辑配置。

```powershell
python -B tools/run_recorded.py --output <另一个新run目录> -- tools/train_lora.py `
  --bundle <相同基座> --data <相同train.jsonl> --config configs/real-c3.json `
  --resume <基线run>/results/checkpoints/step-000002 --stop-at 4 `
  --checkpoint-steps 4 --evaluation-steps 4 --eval-corpus <相同语料目录> `
  --output '{run}/results' --code-commit <完整提交>
```

每步在更新前检查所有未命中专家的原始梯度为零；更新后检查从未命中专家的参数和Adam矩仍未变化。曾命中、本步未命中的行允许因动量变化并记录数量。packed参数的Adam step是Tensor级标量；不能把它误当成每个专家的独立step。ever-routed历史随trainer state保存；恢复后以已验证checkpoint的未命中行作为基线，继续归纳检查。

每个执行停止点写完整optimizer边界检查点和 `publish`。`delta-w.json` 是训练进程按固定分块几何计算的全部FP32增量矩阵SHA256；只保存指纹，不写全量矩阵或合并基座。新进程可用相同几何逐块重建比较。`--skip-delta-fingerprint` 可用于不需要该项证据的实验，正式P8验收必须保留。

`rng_probe=true` 只用于恢复验证，每步实际消耗Python、NumPy、Torch CPU/CUDA四类RNG并记录；LoRA dropout仍为0。`metrics.jsonl` 记录损失、监督数、样本ID、LR、范数、路由检查、搬运字节及内存。出现非有限值或不变量失败时保存当步batch并终止。

当前功能语料由 `tools/prepare_canary_replay.py` 固定生成：单个人工关联的300条训练问题、50条未用过模板的改写问题、300条真实基座生成的算术回放。它是功能金丝雀，不能推广为领域或通用能力评测。评测的NLL按全部监督token加权；短语料可批量左padding，其他长语料应降低 `--eval-batch-size` 控制显存。

本入口和恢复协议正在真实多步验证；已完成项与未完成项以 `progress.md` 为准。
