"""统一管理标准 PEFT 与自定义专家 adapter 的开关。"""

from contextlib import contextmanager, nullcontext

from .packed_experts import PackedExperts
from .packed_linear import LoRAPackedLinear


@contextmanager
def disable_adapters(model):
    """两套 adapter 同时禁用；退出时恢复原状态，不能留下假基座基线。"""
    experts = [module for module in model.modules() if isinstance(module, (PackedExperts, LoRAPackedLinear))]
    previous = [module.enabled for module in experts]
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    try:
        for module in experts:
            module.enabled = False
        with context:
            yield
    finally:
        for module, enabled in zip(experts, previous):
            module.enabled = enabled
