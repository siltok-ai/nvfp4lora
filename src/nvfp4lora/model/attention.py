"""显式展开 KV 的 SDPA 调用及 Transformers 注册。"""

from types import SimpleNamespace
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ATTENTION_IMPLEMENTATION = "nvfp4lora_sdpa_repeat_kv_v1"


def sdpa_forward(module, query, key, value, attention_mask, **kwargs):
    """显式展开分组 KV，让 Windows 确定性 SDPA 可选择高效后端；输入和输出遵循 Transformers 协议。"""
    groups = module.num_key_value_groups
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    proxy = SimpleNamespace(num_key_value_groups=1, is_causal=getattr(module, "is_causal", True))
    return sdpa_attention_forward(proxy, query, key, value, attention_mask, **kwargs)


def register_attention():
    """注册项目自己的注意力和掩码入口，保留 Transformers 默认入口。"""
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, sdpa_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
