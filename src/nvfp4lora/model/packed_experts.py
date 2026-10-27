"""HF 专家调用签名适配；只注册专家 LoRA，冻结权重由外部 store 管理。"""

from dataclasses import replace
import math

import torch
from torch import nn

from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.ops.experts_fn import NVFP4ExpertsFn, NVFP4ExpertsBaseFn
from nvfp4lora.validation.experts_reference import experts_reference


class ExpertAdapter(nn.Module):
    """单投影全部专家的浮点 A/B；每个专家独立使用线性层 fan-in 初始化。"""

    def __init__(self, experts, in_dim, out_dim, rank, alpha, *, device="meta"):
        super().__init__()
        self.rank, self.alpha, self.scaling, self.in_dim = rank, alpha, alpha / rank, in_dim
        self.lora_A = nn.Parameter(torch.empty(experts, rank, in_dim, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(experts, out_dim, rank, device=device, dtype=torch.float32))
        if self.lora_A.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self):
        """必须在 to_empty 后执行；不能依赖 meta 初始化保留任何实际值。"""
        with torch.no_grad():
            self.lora_A.uniform_(-1 / math.sqrt(self.in_dim), 1 / math.sqrt(self.in_dim))
            self.lora_B.zero_()


class PackedExperts(nn.Module):
    """与 HF experts 相同的三参数 forward，adapter 禁用使用独立 base Function。"""

    def __init__(self, config, layer, op_config, *, rank=0, alpha=None, device="meta"):
        super().__init__()
        self.layer, self.rank, self.enabled = layer, rank, True
        self.op_config = op_config
        if rank:
            alpha = float(2 * rank if alpha is None else alpha)
            e, h, i = config.num_experts, config.hidden_size, config.moe_intermediate_size
            self.gate_proj = ExpertAdapter(e, h, i, rank, alpha, device=device)
            self.up_proj = ExpertAdapter(e, h, i, rank, alpha, device=device)
            self.down_proj = ExpertAdapter(e, i, h, rank, alpha, device=device)
            self.op_config = replace(op_config, scaling=(alpha / rank,) * 3)

    def adapter_parameters(self):
        """固定 gate-A/B、up-A/B、down-A/B 的公开协议顺序。"""
        if not self.rank:
            return ()
        return tuple(parameter for name in ["gate_proj", "up_proj", "down_proj"]
                     for parameter in [getattr(self, name).lora_A, getattr(self, name).lora_B])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """启用但 B=0 仍执行 LoRA，确保首步 B 能获得梯度。"""
        if self.rank and self.enabled:
            return NVFP4ExpertsFn.apply(hidden_states, top_k_index, top_k_weights,
                                       *self.adapter_parameters(), self.layer, self.op_config)
        return NVFP4ExpertsBaseFn.apply(hidden_states, top_k_index, top_k_weights, self.layer, self.op_config)


class DenseReferenceExperts(PackedExperts):
    """仅供 tiny 的完整浮点 W 参考，沿用独立稠密 autograd 数学实现。"""

    def materialize_reference(self, *, device):
        """只允许 tiny 来源，避免误为真实模型分配全量浮点专家。"""
        store = self.op_config.streamer.store
        if store.source.get("kind") != "tiny-generated":
            raise ValueError("Dense reference experts are restricted to explicitly generated tiny checkpoints")
        packed = store.tensors(self.layer)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            raw = dequantize(packed[name + ".weight"], packed[name + ".weight_scale"],
                             dtype=self.op_config.compute_dtype).to(device)
            global_scale = packed[name + ".weight_scale_2"].to(device=device, dtype=self.op_config.accumulation_dtype)
            self.register_buffer(name + "_reference_weight", raw, persistent=False)
            self.register_buffer(name + "_reference_scale", global_scale, persistent=False)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """不复用生产 chunk/VJP/聚合，用同一显式精度的独立参考。"""
        names = ["gate_proj", "up_proj", "down_proj"]
        matrices = [getattr(self, name + "_reference_weight") for name in names]
        scales = [getattr(self, name + "_reference_scale") for name in names]
        parameters = self.adapter_parameters() if self.enabled else ()
        return experts_reference(hidden_states, top_k_index, top_k_weights, matrices, scales,
                                 parameters, scaling=self.op_config.scaling)
