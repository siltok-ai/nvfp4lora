# 训练、检查点和发布

`train/loop.py` 直接调用注入 PEFT 后的 backbone，使用非重入逐层检查点；`ops/chunked_ce.py` 只对有效答案位置生成分块 logits，并在 CE 块上重算。labels 与 input_ids 同位置，损失统一执行 causal shift。一个 optimizer step 使用所有 microbatch 的 CE 总和除以总监督 token 数。

LoRA 全为 FP32；标准 AdamW 设置 `weight_decay=0, foreach=False, fused=False`。没有命中的专家本步原始梯度为零，过去的 Adam 动量仍可更新参数。训练入口固定 RNG、关闭 TF32/BF16 低精度归约并要求确定性算法；不永久修改本机环境。

`train/checkpoint.py` 只允许 optimizer step 后 `zero_grad(set_to_none=True)` 的边界保存。所有状态写入新目录，最后写完整 manifest。Adam 状态按 canonical 参数名恢复，不依赖位置；包括 FP32 矩、逐参数 step、scheduler、数据游标和 Python/NumPy/Torch CPU/CUDA RNG。仅恢复本项目产生且校验完整的可信训练检查点。

`export/artifacts.py` 分开保存标准 PEFT 的非专家 LoRA 和 `expert_lora.safetensors`。manifest 绑定预处理基座内容、rank/alpha/scaling、canonical 名映射、参数形状和文件摘要。发布不含 optimizer 或 RNG pickle。`build_from_adapter` 只依赖预处理基座和发布目录；不合并 NVFP4 权重。

P4 入口示例（环境、tiny 来源与预处理步骤见当前进度）：

```powershell
python -B tools/run_recorded.py --output <新的运行目录> -- tools/check_tiny_training.py --bundle <tiny预处理目录> --output '{run}' --steps 100 --commit <完整提交>
```

它分别启动三个 Python 进程：生产/独立稠密参考的 100 optimizer 步（每步累积2）、第50步续训、发布重载。合成样本 ID 只用于 tiny 词表，不能代替真实 tokenizer 的模板/标签黄金样本。`--steps 2` 仅为短程检查，不能作为 P4 验收。

同步 P4 当时的31项回归、正式100步及两个新进程验收已通过；后续异步及真实训练新增检查以 `progress.md` 为准。这些tiny结果不代替真实模型验收。

`train/template.py` 用真实模板渲染非思考纯文本对话，检查完整输入保留 Speed 提示的精确 token 前缀，再只标记最后答案和 im_end。超长提示从左截断；无法保留完整答案及一个前置预测位置的样本丢弃。200 个真实 tokenizer 黄金样本直接执行 Speed `_prompt_tokens` 的源码 AST，对照提示、移位后的每个标签、强制截断与丢弃。保留与 Speed 相同的 tokenizer 加载选项，没有擅自修改 regex。

`train/data.py` 的 `ChatJSONLDataset` 读取每行 `{"messages": [...]}`，预编码真实tokenizer样本；原文件、tokenizer backend、模板和截断/丢弃记录共同组成数据身份。确定性采样器复用相同的epoch/position协议。训练核心提供可选的microbatch回调，供第一步快照和诊断使用，默认数值路径不变。

真实首步机制入口：

```powershell
python -B tools/run_recorded.py --output <新运行目录> -- tools/check_real_training.py --bundle <真实bundle> --output '{run}/results' --expert-rank 0 --commit <完整提交>
python -B tools/run_recorded.py --output <另一新运行目录> -- tools/check_real_training.py --bundle <真实bundle> --output '{run}/results' --expert-rank 2 --commit <完整提交>
```

该工具仅做1个optimizer step×累积8、B非零后的两层C3解析梯度、冻结不变量及发布；真实数据只有8条短合成问答。`tools/check_real_adapter_reload.py` 在另一个进程只用bundle+publish建模，固定batch与GPU参数原字节均已通过E2。不能把这个探针的 `max_steps=1` 配置静默延长成多步验收。

后续真实多步入口见 [真实训练CLI](real-training-cli.md)。独立配置预设逻辑上限100步；C3及专家r2的第2→4步新进程续训、训练内ΔW与发布重建全部E2，以及2048/4096长序列两步压力验证均已通过。两套配置现均完成完整100步；C3已通过50→100十项E2和仅用bundle/publish的新进程全部参数/固定probe/250份ΔW重建，专家正在执行完整50→100对照。专家GPU发布重建、两套后验诊断、效果门槛和最终交付仍待完成，最新实测记录以 [进度](progress.md) 为准。
