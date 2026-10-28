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

`tools/compare_real_resume.py` 比较基线与恢复执行的逐步数值、完整命名检查点、发布探针及训练内ΔW指纹。`tools/verify_training_release.py` 在新进程只用bundle/publish重建模型，核对固定loss、全部参数字节和全部ΔW。C3已完成50→100步的完整十项E2比较和250份ΔW发布重建；专家100步也已完成，正在新进程执行50→100对照。先前两套2→4步的结果保留为短程证据，不能替代专家尚未完成的完整续训和GPU发布验收。

长序列压力数据用 `tools/prepare_sequence_stress.py` 生成，8条记录保留完整1035-token答案，在2048/4096上只左截断提示。使用同一训练入口、相应长度配置、累积8、至少2个optimizer step并保留首micro快照，分别观察Adam首次分配和已有Adam的稳态。压力语料的适配器独立保存，不混入功能训练。压力运行可用 `--skip-delta-fingerprint`，因此不能充作P8证据。

`tools/summarize_training_session.py --results <run>/results --output <新文件>` 从已结束会话提取阶段内存、吞吐、H2D/等待和保存点A/B分组L2范数。其 `MEASURED` 状态只表示已汇总原始观测，不宣称金丝雀或完整阶段验收通过。

`tools/check_sequence_training.py` 进一步核对压力数据身份、完整8条答案、每步精确输入/监督数量、两步累积8、首步后Adam矩、冻结hash及首micro快照的解码生命周期。实际已执行的长度和配置仍以 `progress.md` 为准。

`tools/profile_recompute_memory.py --bundle <基座> --training-results <压力run>/results --output <新目录>` 从压力step2恢复LoRA/Adam，独立测一个micro的40层重算峰值，不更新参数。`--control` 再去掉采集hook核对loss和全部梯度E2。它在每次reset之前收集合并旧峰值，always_call后hook覆盖非重入重算提前停止；不依赖有上限的快照事件环来推断CUDA峰值。`history_complete`单列事件环的完整性。

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

本机完整真实验收使用 `tools/run_real_acceptance.ps1` 顺序执行C3、专家r2：从已验证step4继续到100（保存20/50/100，测量10/20/50/100），再新进程50→100、完整命名状态/逐步E2对照、发布ΔW重建、功能测量和FP8范围观测。各项使用新run并写独立task指针，首个失败立即停止。失败后按已有完成证据选择性继续，不重新启动整个脚本覆盖既有结果。`-ValidateOnly` 只校验现有路径、基线step4/逻辑100、专家rank和数据SHA，不启动GPU。该脚本尚未完成实际100步执行时，不表示完整验收已通过。

串行campaign结束后，分别对C3和专家run补采v2 6.5的周期诊断：

```powershell
python -B tools/run_recorded.py --output <新的诊断run> -- tools/inspect_training_checkpoints.py `
  --bundle <相同基座> --training-results <功能100步run>/results `
  --corpus <canary-replay语料目录> --steps 20 50 100 --output '{run}/results'
```

它只读取完整且所有文件hash通过的检查点，逐名字/形状载入参数，测量50条canary留出NLL、A/B分组范数和固定前8条训练样本的40层路由。路由观测包括rank0，并排除padding；基座统一禁用作对照。输出MEASURED，不是续训或领域效果验收。当前已运行的旧版父campaign不会动态增加该步骤，应在其结束后显式执行，避免与训练争用GPU。

`tools/audit_m2.py --task <任务目录> --bundle <基座> --output <新目录>` 逐项读取原始run证据，并对缺少或仍在运行的项保留PENDING。它检查严格50→100对照、两套2048/4096压力/40层重算/非零B解码生命周期、完整参数集合、训练与发布后评测一致、周期诊断及FP8风险。原A5.2无条件1%失败与调整后的判据分列，还核对新夹具和CPU参考已存在于测量前的Git提交。

最终门槛添加 `--require-complete`；任何PENDING或FAIL都会返回非零。效果门槛需显式传入 `--minimum-canary-accuracy`、`--max-replay-nll-increase` 和真实的 `--threshold-provenance`（用户决定的来源），工具不能代替用户确认。最终 `--artifact-dir` 应包含经hash验证、与已通过的原始run逐文件一致的publish及完整step50/100检查点；仅有名为manifest的空文件不满足交付检查。当前部分审核结果是INCOMPLETE，不能据此宣称M2通过。

所有技术项及两套后验诊断完成后，再以新预审生成制品：

```powershell
python -B tools/run_recorded.py --output <新的归档run> -- tools/archive_training_delivery.py `
  --task <任务目录> --bundle <相同基座> --corpus <功能语料目录> `
  --pre-audit <新M2预审>/m2-audit.json --output <artifacts/nvfp4lora/0.0.1/新的UTC目录>
```

`--validate-only` 只校验计划，不复制。它拒绝未完成的技术门槛，允许真实用户效果阈值尚未提供但相关观测已完成的情形；归档本身不能批准阈值。实际归档要求源码已提交、输出不存在且不在输入树内。制品保持每套results的相对目录，包含20/50/100完整检查点和固定数据，可从50重现到100；最后才写完整manifest。随后用本制品目录再执行M2严格审核。当前工具仅通过CPU复制和拒绝未完成输入的检查，尚未生成真实制品。

数据与基座身份按内容绑定，执行路径记录在`execution.json`，不会把数据的绝对目录作为逻辑配置身份。已用真实600条数据的新路径副本核对两套step50采样器到100的全部400个样本及token数量，并验证改动原文件内容会被拒绝，见[CPU重定位证据](evidence/delivery-data-portability.json)。最终制品仍须完成实际复制及全文件hash核验，不能仅凭此CPU核查宣称GPU续训或制品已经验收。
