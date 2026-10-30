"""统一管理标准 PEFT 与自定义专家 adapter 的开关。"""

from contextlib import contextmanager, nullcontext

class ToggleableAdapter:
    """标记具有 enabled 开关的 adapter；不注册状态或改变初始化。"""

    enabled: bool


@contextmanager
def disable_adapters(model):
    """两套 adapter 同时禁用；退出时恢复原状态，不能留下假基座基线。"""
    adapters = [module for module in model.modules() if isinstance(module, ToggleableAdapter)]
    previous = [module.enabled for module in adapters]
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    try:
        for module in adapters:
            module.enabled = False
        with context:
            yield
    finally:
        for module, enabled in zip(adapters, previous):
            module.enabled = enabled
