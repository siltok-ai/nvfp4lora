# SDPA调用回退与DA2.3参考判据的历史

这两项发现均发生在2026-10-29正式训练之前：D0检查运行后端，D2检查线性算子精度。原设计文本、失败run和原判据结果均保留；本文件解释它们的来源、诊断和最终限定，不覆盖历史证据。

## SDPA：默认GQA调用选到math，改用显式KV展开的efficient路径

原设计DA0.7要求在本机原生Windows、固定Torch/transformers版本、B=1、T=4096、24个query头/4个KV头、head_dim256下，有无padding mask都使用融合SDPA后端。它关系到16GB预算：普通math实现需要显式中间注意力矩阵。

默认HF无mask路径使用enable_gqa=True。实测实际算子为aten::_scaled_dot_product_attention_math，独立注意力探针峰值allocated6,979,321,856字节，原DA0.7失败。此值是探针前后向峰值，不是完整模型显存。

“该构建没有Flash”具体指本机torch2.11.0+cu130 Windows wheel的内置SDPA Flash后端未编译，不是HF的Qwen模型结构缺少attention实现。2026-10-30只读核对得到is_flash_attention_available=False，同时flash_sdp_enabled=True；开关开启不能补上未编译的kernel。核对过程未初始化CUDA。HF的sdpa_attention_forward确实调用torch.nn.functional.scaled_dot_product_attention，由PyTorch分派后端；原生Windows并不因此总是使用math，本任务已经实测使用efficient。它也不同于HF另外集成的外部flash_attention_2路径，本任务未引入该额外依赖。

随后诊断12种组合，定位到该锁定Windows构建的后端限制：Flash未编译；确定性模式下cuDNN SDPA被拒绝；现有efficient实现要求q/k/v头数一致。于是把4个KV头各重复6份，成为与24个query头对齐的输入，再调用已有HF SDPA。重复的是同一份K/V，因此保留GQA的共享关系；它增加少量KV临时存储，换取融合后端避免显式大注意力矩阵。

实现仅在build_dense.py注册新的attention和mask键：显式repeat_kv后，以num_key_value_groups=1的局部代理调用原sdpa_attention_forward。全局sdpa键、原MoE配置、确定性设置和依赖版本保持原值。这里的“回退”是调用方式的备选；投入dense训练的是efficient SDPA。

证据包括：

- 有mask/无mask均实际调用efficient前后向；峰值allocated分别659,292,160/525,074,432字节。
- 重复前后向E2；相对math参考，输出L2约0.214%–0.215%，q梯度约0.327%–0.328%，全部输出/梯度均小于原1%控制门槛。
- D4 tiny100完成两个参考、续训和发布重建；D5真实整模前向实际16次efficient、零次math，重复hidden字节E2。
- D5真实CPU FP32无条件前四层各自误差0.291637%/0.294074%/0.406958%/0.339739%，原1%判据不变；D6两种长度压力与重算也已通过。

因此原默认路径仍记FAIL，替代路径单独记FALLBACK_PASS并完成后续整模资格。不能写成默认HF路径原本通过，也不能据此声称任何Windows/Torch版本均有同一限制。

来源：[原设计DA0.7](../reference/dense27b-design-v1.md)、[决策日志](../decisions.md)、[D0证据](../evidence/dense/d0-summary.json)、[D5整模证据](../evidence/dense/d5-summary.json)。实现为src/nvfp4lora/model/build_dense.py的dense_sdpa_forward。

## 旧MoE A5.2：离散选路与连续数值路径分开验收

这是2026-10-27的另一件历史，区别于下面dense DA2.3的大矩阵舍入控制。旧MoE首次无条件CPU FP32前四层误差为0.90%、2.05%、1.77%、2.46%，原1%检查失败。固定GPU实际路由、仍用CPU FP32端到端计算时，误差降为0.33%、0.40%、0.42%、0.54%。五组控制还发现，即使CPU改用相同BF16策略，自由选路仍可产生最高2.74%的端到端差异。

依据是top-k在第8名的cutoff附近不连续：很小的投影舍入或相等分数的tie次序可以换掉一个专家，使后续执行不同的数值图。当时决定分别核对固定实际route/权重的连续FP32路径（每层仍<=1%），以及独立CPU router投影、GPU所选集合的top-k正确性和softmax/归一化权重（相应误差仍<=1%）。同时保留自由路由误差、选路变化和cutoff margin，并要求在三条新的128-token提示上验证；没有修改生产路由或把原无条件FAIL改成PASS。

dense没有专家选路，因此本轮DA5.2继续要求无条件CPU FP32前四层各<=1%，没有继承上述固定route条件，实测0.291637%/0.294074%/0.406958%/0.339739%。[原MoE决策及五组控制结论](../decisions.md#2026-10-27a52-的离散路由与参考判据)与[本轮D5证据](../evidence/dense/d5-summary.json)分别保留。

## DA2.3：实现正确性与FP32库舍入分开核对

原设计DA2.3要求FP32模式相对FP64参考的相对L2<=1e-6。它是高精度测试模式的判据，不是把生产训练从一种精度改为另一种精度；生产仍执行设计3.3的BF16码值GEMM、FP32输出乘global、转回BF16，LoRA旁路为FP32。

首次真实大形状down_proj的归约长度K=17408、输入17行：FP32前向误差1.0246188661e-6，略超1e-6，原检查FAIL。dx为4.4421924e-7。没有直接把失败改成通过，而是先建立普通Torch autograd的独立物化权重参考。

两类控制说明了问题来源：

1. 同一大矩阵用普通F.linear和autograd计算，FP32前向/dx与PackedLinear逐字节一致。自定义算子没有产生超出普通FP32计算路径的差异。
2. 保留原17行，仅补零改变矩阵M，前向误差在17/32/128/4096行约为1.0246e-6/8.6455e-7/1.3309e-6；4096行dx约1.2220e-6。数学上原行的乘积不变，但矩阵几何会影响cuBLAS算法及FP32累加顺序。对照结果支持有限精度GEMM舍入这一解释。

调整判据之前，又固定三个新seed，对down/gate/o三种长归约形状和17/128/257/4096行进行了36组控制。两种精度模式均与普通autograd E2；相对FP64的最坏L2约1.3195891e-6。小形状NVFP4/FP8仍满足原1e-6，并通过FP64 gradcheck。TF32始终关闭。

最终将DA2.3拆为：

| 范围 | 现在的要求 |
| --- | --- |
| 小形状 | 保留原FP64相对L2<=1e-6及gradcheck |
| 真实大形状 | 与独立物化普通autograd E2，且相对FP64<=2e-6，两项都必须满足 |

2e-6是在上述控制范围内留出的有限裕量，不是任意矩阵FP32误差的通用数学上界。它也不替代实现的逐字节核对。每个结果同时保存原1e-6是否通过，汇总标为QUALIFIED_WITH_DOCUMENTED_REFERENCE_CONTROL；原大形状失败仍保留FAIL。

这次修改针对DA2.3测试的解释和判据，没有改变PackedLinearFn运算顺序、解码精度、BF16生产路径或LoRA初始化。其他E2要求继续保留：B=0开关loss、精确续训、发布重建和MoE回归；真实CPU FP32前四层原1%判据也不变。2026-10-30正式100步已完成；真实续训、发布重建和D7仍需各自的实际运行证据，不能用先前资格代替。

来源：[原设计DA2.3](../reference/dense27b-design-v1.md)、[决策日志](../decisions.md)、[D2完整证据及原失败](../evidence/dense/d2-summary.json)。对照入口为tools/check_dense_linear.py及tests/dense_fixture.py，不需要新增依赖或性能kernel。

## 本轮功能门槛确认

2026-10-30用户明确回复同意推荐门槛：留出>=45/50、回放NLL增量<=0.01，并要求继续执行。该确认允许启动正式100步；完整原话、基线和记录时间见[门槛确认](../evidence/dense/effect-threshold-owner-decision.json)。SDPA和DA2.3的历史事实仍按上面的范围报告。
