"""发布和 bundle 使用的固定数值策略标识。"""

NUMERICAL_POLICY = "nonexpert-fp32-decode-bf16-v1"
LINEAR_POLICY = "dense-packed-linear-bf16code-fp32scale-v1"

EXPERT_POLICY = "torch-linear-bf16-result-fp32-global-lora-swiglu-route-v1"
