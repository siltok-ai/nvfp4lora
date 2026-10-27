# 实施决策

## 2026-10-27：启动与既有决定

- 本次用户授权实施；交接材料的旧会话“只分析”状态不再适用。
- 保留 v2 的 A→A′ 交付路线、原生 Windows、独立环境、原专家格式与完整梯度协议。
- 继承同版本传递依赖授权：accelerate==1.15.0 是 PEFT 必需依赖，只安装，不用其训练循环。
- GDN 先实测 Transformers 5.16.1 中现有纯 Torch chunk 实现。FLA 未安装，不因方案误称“只有逐 token”而先重写或新增依赖。
- M1 可下载/校验/预处理真实分片和执行真实单层验证；M1 通过前不执行完整 35B 模型训练。该解释消解 v2 G0.8/P1/P2 与“M1 后碰真实模型”的矛盾，遵循交接说明 6.2。
- 同步 store→P2→同步 tiny→异步 streamer→受影响项回归，M1 仍须包含异步验收。
- 优先标准 AdamW(weight_decay=0, foreach=False)，接受历史动量更新本步未命中专家。
- BF16 GEMM 输出及舍入、非专家解码、FP32/TF32 和 BF16 reduction 策略显式记录。单靠同 checkpoint 不声称与 FreeToken 整体 E2 等价。

## 2026-10-27：P0 实测与格式层准备

- GDN 的现成纯 Torch chunk 实现已通过独立短序列递推对照、真实 T4096 全输入梯度，以及完整单 GDN+PEFT 模块的两种检查点。先选这一已验收候选，不新增未安装 FLA 依赖；未测试 FLA 不等于 FLA 不可用。
- E2 比较采用 contiguous 原始 uint8 字节，避免浮点 torch.equal 把正负零视为相同。
- 专家库同时保留原始 `input_scale`：index 中共有 30720 个；按 F32 共 122880 字节。四个字段全部保留，预计专家 payload 与 v2 的 18119639040 字节吻合，最终以真实 header 为准。
- 冻结非专家采用 FP32 解码再舍入 BF16；lm_head 按行分块。这是明确的训练基座数值策略，不声称与运行时 W8A8/FP16 global 完整 E2 等价。
- G0.8 下载期间准备格式解码/预处理代码及小格式夹具，不运行未校验的真实权重，也不提前宣布进入 P1 验收。真实预处理 CLI 要求完整三分片 PASS 收据，并重做来源哈希检查。

## 2026-10-27：真实 P1 首轮过滤修复

三个原始分片的 SHA256 已全部通过。首次真实预处理在布局检查阶段失败：范围过宽的 `.mlp.experts.` 选择同时计入 `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`，使源专家集合比文本主干多两个名字。已用含 MTP 的小夹具复现；将专家集合限定为 `model.language_model.layers.*`。MTP 张量仍由原始分片保留，不加载为文本训练专家。失败 run 和空输出 `*-train-v1` 保留，修复后使用新的 `*-train-v2` 输出。

## 2026-10-27：P2 与物化构建

- 先采用逐专家 GEMM 的纯 Torch 数值基线；grouped_mm 资格已通过，但暂不叠加实现复杂度。真实目标形状前向、路由及六个 A/B 梯度与独立参考通过原阈值。BF16 dX 相对误差约 0.005084；FP32 为 7.3e-8，未调整容差。
- 生产按 route 唯一位置写回，再按 top-k 的原槽顺序用 FP32 累加。局部 A/B 叶子只覆盖本块，不用 allow_unused 隐藏 None，不直接写 `.grad`。只支持一阶训练梯度。
- CPU bank 的逻辑字节不是锁页 allocator 的物理分配量：真实构建时约 18.12GB 逻辑数据对应约 21.47GB pinned allocated，进程 peak working set 约 27.04GB。仍在本机主存预算内；后续按实测账本计算。
- meta 模型先转非专家 BF16，再替换专家并分配。`to(BF16)` 也会影响 meta RoPE buffer 的 dtype；实际加载后用正常 CPU 构造重新注册 FP32 inv_freq/original_inv_freq，不能只 copy 到 BF16 buffer。
- 专家 A 使用每个专家自己的 1/sqrt(fan_in) 范围，禁止对整个 [E,r,in] 套三维 Kaiming。PEFT 包装后显式恢复自有 A/B 的可训练性，并与预期 canonical 集合核对。
- 仅 tiny 参考模型允许常驻解码 W。真实 A1.4 只做构建和参数读回，没有运行完整模型前向或训练；M1 仍等待 P4 与 P3。
