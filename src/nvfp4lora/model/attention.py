"""显式展开 KV 的 SDPA 调用及 Transformers 注册。"""

from types import SimpleNamespace
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ATTENTION_IMPLEMENTATION = "nvfp4lora_dense_sdpa_repeat_kv_v1"


def dense_sdpa_forward(module, query, key, value, attention_mask, **kwargs):
    """Windows确定性efficient后端：只在dense显式展开KV，不改HF/MoE默认值。"""
    groups = module.num_key_value_groups
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    proxy = SimpleNamespace(num_key_value_groups=1, is_causal=getattr(module, "is_causal", True))
    return sdpa_attention_forward(proxy, query, key, value, attention_mask, **kwargs)


def register_dense_attention():
    """新增两个registry键；已有sdpa callable和mask callable保持原对象。"""
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, dense_sdpa_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
