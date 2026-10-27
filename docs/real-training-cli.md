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

`tools/compare_real_resume.py` 比较基线与恢复执行的逐步数值、完整命名检查点、发布探针及训练内ΔW指纹。`tools/verify_training_release.py` 在新进程只用bundle/publish重建模型，核对固定loss、全部参数字节和全部ΔW。当前C3和专家r2的2→4步对照通过；它是短程证据，不能替代v2的50→100步验收。

长序列压力数据用 `tools/prepare_sequence_stress.py` 生成，8条记录保留完整1035-token答案，在2048/4096上只左截断提示。使用同一训练入口、相应长度配置、累积8、至少2个optimizer step并保留首micro快照，分别观察Adam首次分配和已有Adam的稳态。压力语料的适配器独立保存，不混入功能训练。压力运行可用 `--skip-delta-fingerprint`，因此不能充作P8证据。

`tools/summarize_training_session.py --results <run>/results --output <新文件>` 从已结束会话提取阶段内存、吞吐、H2D/等待和保存点A/B分组L2范数。其 `MEASURED` 状态只表示已汇总原始观测，不宣称金丝雀或完整阶段验收通过。

`tools/check_sequence_training.py` 进一步核对压力数据身份、完整8条答案、每步精确输入/监督数量、两步累积8、首步后Adam矩、冻结hash及首micro快照的解码生命周期。实际已执行的长度和配置仍以 `progress.md` 为准。

本机当前路径如下，权重和数据均不在Git仓库中：

| 用途 | 路径 |
| --- | --- |
| 解释器 | `C:/workspace/envs/T-20261027-nvfp4-lora/train/Scripts/python.exe` |
| 预处理基座 | `C:/workspace/models/bundles/qwen3.6-nvfp4-lora/1355db6a052410cf-train-v2` |
| 功能语料 | `C:/workspace/data/datasets/nvfp4lora/canary-replay-v1` |
| 长序列压力语料 | `C:/workspace/data/datasets/nvfp4lora/long-sequence-v1` |
| 完整执行证据 | `C:/workspace/runs/T-20261027-nvfp4-lora` |
| 活动进程和最新指针 | `C:/workspace/tasks/T-20261027-nvfp4-lora/README.md` |

每次运行先确认上一项GPU作业已结束，在仓库根调用入口。中断后从最后一个 `checkpoint_manifest.json` 标记complete且hash校验通过的optimizer边界恢复，使用原数据、逻辑配置和版本；新输出使用新目录。`publish` 不包含Adam/数据/RNG状态，续训需要 `checkpoints/step-NNNNNN`。

发布目录可由核心库直接载入，以下为Python调用示例，`adapter` 替换为已选定run的 `results/publish`：

```python
from transformers import AutoTokenizer
from nvfp4lora.export.artifacts import build_from_adapter
from nvfp4lora.model.generate import greedy_batch
from nvfp4lora.train.evaluate import evaluation_mode
from nvfp4lora.train.loop import configure_training

configure_training(20261028)
bundle = "C:/workspace/models/bundles/qwen3.6-nvfp4-lora/1355db6a052410cf-train-v2"
adapter = "<已验证功能训练run>/results/publish"
tokenizer = AutoTokenizer.from_pretrained(bundle, local_files_only=True)
built = build_from_adapter(bundle, adapter)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请计算2+2，仅输出数字。"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
with evaluation_mode(built):
    reply = greedy_batch(built, [prompt], eos_token_ids={248044, 248046},
                         pad_token_id=248044, max_new_tokens=32)
print(tokenizer.decode(reply["tokens"][0], skip_special_tokens=True))
built.streamer.close()
```

该调用使用训练核心的无cache验证路径；性能测量需注明每个生成token都重跑完整前向。专家适配器由 `build_from_adapter` 连同PEFT参数一起加载。Speed/FreeToken的产品加载器、融合段映射、CUDA Graph及缓存隔离属于后续G2/A′，当前发布目录尚不能直接交给其原有加载器使用。
