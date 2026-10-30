"""冻结线性层、FP32 LoRA 旁路和保留在 CPU 的词嵌入表。"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nvfp4lora.model.adapters import ToggleableAdapter

from nvfp4lora.ops.packed_linear_fn import PackedLinearFn, decode_codes


class PackedLinear(nn.Module):
    """不持有Parameter或buffer；冻结基座的全部字节由外部store提供。"""

    def __init__(self, layer, name, kind, in_features, out_features, op_config):
        super().__init__()
        if kind not in ["nvfp4", "fp8"] or min(in_features, out_features) < 1:
            raise ValueError("Unsupported packed linear kind or geometry")
        self.layer, self.name, self.kind = layer, name, kind
        self.in_features, self.out_features = in_features, out_features
        self.op_config = op_config

    def forward(self, x):
        return PackedLinearFn.apply(x, self.layer, self.name, self.kind, self.op_config)


class LoRAPackedLinear(nn.Module, ToggleableAdapter):
    """PEFT兼容参数命名，数值为FP32增量加到BF16基座结果后再舍入。"""

    def __init__(self, base_layer, rank, alpha=None, *, device=None):
        super().__init__()
        alpha = 2 * rank if alpha is None else alpha
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive ordinary LoRA rank/alpha required")
        self.base_layer = base_layer
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.ModuleDict({"default": nn.Linear(self.in_features, rank, bias=False, dtype=torch.float32, device=device)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, self.out_features, bias=False, dtype=torch.float32, device=device)})
        self.scaling = {"default": alpha / rank}
        self.enabled = True
        self.reset_parameters()

    def reset_parameters(self):
        """to_empty后仍须调用；A按fan_in初始化，B严格为零。"""
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        base = self.base_layer(x)
        if not self.enabled:
            return base
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = self.lora_B["default"](self.lora_A["default"](x.float())) * self.scaling["default"]
            return (base.float() + delta).to(base.dtype)


class CPUEmbedding(nn.Module):
    """表是普通CPU属性；model.to/to_empty不会搬运，参数迭代也不会见到它。"""

    def __init__(self, table=None, *, num_embeddings=None, embedding_dim=None):
        super().__init__()
        if table is not None:
            num_embeddings, embedding_dim = table.shape
        if num_embeddings is None or embedding_dim is None:
            raise ValueError("CPUEmbedding needs a table or explicit geometry")
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.table = None
        if table is not None:
            self.set_table(table)

    def set_table(self, table):
        """只接受冻结BF16 CPU表，避免模型参数或buffer注册。"""
        if table.device.type != "cpu" or table.dtype != torch.bfloat16 or list(table.shape) != [self.num_embeddings, self.embedding_dim]:
            raise ValueError("Expected matching frozen BF16 CPU embedding table")
        self.table = table.detach().contiguous()

    def forward(self, ids):
        if self.table is None:
            raise RuntimeError("CPU embedding table has not been loaded")
        return F.embedding(ids.cpu(), self.table).to(ids.device)


class ReferencePackedLinear(PackedLinear):
    """仅tiny可物化的普通autograd参考；forward不调用Function或streamer。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codes = None
        self.global_scale = None

    @torch.no_grad()
    def materialize_reference(self):
        if self.op_config.streamer.store.source.get("kind") != "tiny-generated":
            raise ValueError("Resident decoded reference weights are restricted to tiny-generated sources")
        with self.op_config.streamer.acquire(self.layer) as handle:
            self.codes, global_scale = decode_codes(handle.tensors(), self.name, self.kind, self.op_config)
            self.global_scale = global_scale.clone()

    def forward(self, x):
        if self.codes is None:
            raise RuntimeError("Tiny reference has not been materialized")
        cfg = self.op_config
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (F.linear(x, self.codes).to(cfg.accumulation_dtype) * self.global_scale).to(x.dtype)
