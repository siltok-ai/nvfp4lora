# 专家算子与模型构建

## 算子协议

`NVFP4ExpertsFn.apply(x, indices, weights, Ag, Bg, Au, Bu, Ad, Bd, layer_id, cfg)` 的六个 A/B 都是顶层 Tensor 参数。C3 只训练非专家时使用 `NVFP4ExpertsBaseFn`，仍返回 x 和浮点路由权重的梯度。两者只支持一阶训练梯度；未验收 double backward、JVP 或 torch.compile。

| 投影 | A | B |
| --- | --- | --- |
| gate、up | [E,r,H] | [E,I,r] |
| down | [E,r,I] | [E,H,r] |

生产 BF16 策略：原 `LUT × block_scale` 精确解码到 BF16；`F.linear` 的 BF16 结果转 FP32，再乘原 FP32 global；LoRA、SwiGLU 和路由聚合用 FP32。down 基座使用舍入 BF16 的 h，down LoRA 使用 FP32 h。普通 BF16 GEMM 结果已经舍入，转 FP32 不恢复低位；这与把 global 预先折入 BF16 W 的路径只要求 E3。

FP32/FP64 模式用相同代数和显式 dtype；FP64 gradcheck 没有内部 BF16 强转。配置内关闭外部 autocast 的影响。TF32 与 BF16 reduced precision reduction 由训练/探针入口显式关闭。

## 生命周期与聚合

稳定排序保留原 token/top-k 槽位。每个 route 写唯一输出位置，最后按原 top-k 顺序求和；dX 也先恢复每条 route，再用 FP32/FP64 汇总。不用浮点原子 index_add 做生产路由聚合。

ctx 只保存输入、整数路由、浮点路由权重和可训练 A/B。前向及反向各自取得层句柄；反向每块重新解码，A/B 作为块内局部叶子，经 `autograd.grad` 返回梯度，不改 `.grad`，不保留图。不命中的专家得到精确零梯度。B=0 时仍执行 LoRA 分支，以便首步 B 得到梯度。

独立参考位于 `validation/experts_reference.py`：小形状逐 token/槽计算，真实目标形状逐专家选择 token/槽，均不调用生产排序、chunk forward 或 VJP。格式解码已单独通过原字节与 FreeToken 验证。

## 实测资格

[P2 摘要](evidence/p2-experts-summary.json) 记录真实第 0 层、T4096/H2048/I512/E256/top8/r2。FP32 前向、路由与 A/B 梯度和参考相同，dX 相对 L2 约 7.3e-8；BF16 对应 dX 约 0.005084，其他比较为零，符合原阈值。BF16 分支/路由累加舍入差异是该 dX 差异的解释，未放宽容差。

块大小 8/32/256 的 allocated 峰值为 1512809472/1966404608/6193164800 字节；三者反向后的 allocated 都回到测量前。保存的 Python 分配栈确实包含解码分配，快照末尾没有存活解码块。Windows 的 C++ 栈记录警告保留在日志中，Python 栈与实际内存数值已另行核对。

## 构建和开关

`model.build.build_model` 在 meta 构造，先把专家替换为 `PackedExperts`，再 `to_empty`。非专家逐文件实际加载；A/B 在真实设备重新初始化，RoPE 用普通 CPU 构造重新计算并注册为 FP32。C3 使用精确正则选择全注意力、GDN 三投影和共享专家；路由器、embedding、lm_head 保持冻结。PEFT 包装后再次核对自有专家 A/B 和实际可训练集合。

`model.adapters.disable_adapters` 同时管理 PEFT 和自有专家开关。runtime 名映射为 canonical 名，拒绝冲突。真实 r16 C3 有 19169280 个参数，全部专家 r2 有 157286400 个参数；共 740 个可训练 Tensor。完整构建/加载峰值 allocated 为 6517991424 字节；该值不含训练梯度、Adam 状态和训练激活。

`DenseReferenceExperts` 只接受带 `tiny-generated` 身份的来源。它的冻结浮点 W 用于 tiny 参考训练，不用于真实模型常驻专家。持久 tiny 来源和预处理产物在仓库外，100 步/重载/续训的 P4 资格仍须单独完成。
