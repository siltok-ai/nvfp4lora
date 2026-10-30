"""完整 GDN 训练路径：因果卷积、投影、门控 norm、PEFT 和两种检查点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from nvfp4lora.testing.numerics import configure_numerics
from nvfp4lora.testing.numerics import compare


def frozen_hashes(model):
    """逐冻结参数校验，不将完整副本保留在 GPU 或 Python 列表。"""
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters() if not p.requires_grad}


def evaluate(model, x, dy, trainable, mode):
    """通过 autograd 正常返回输入和全部 A/B 梯度，检查点保持相同参数对象。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    value = x.detach().requires_grad_()
    torch.cuda.synchronize()
    start = time.perf_counter()
    if mode == "direct":
        output = model(value)
    else:
        output = checkpoint(model, value, use_reentrant=(mode == "reentrant"))
    # 重入检查点不支持 autograd.grad，因此三条路径统一使用普通 backward。
    output.backward(dy)
    torch.cuda.synchronize()
    result = {"seconds": time.perf_counter() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    gradients = {"x": value.grad}
    gradients.update({name: p.grad for name, p in trainable.items()})
    for name, gradient in gradients.items():
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.count_nonzero() > 0, name
    result["gradients"] = {name: gradient.detach().cpu() for name, gradient in gradients.items()}
    result["output"] = output.detach().cpu()
    model.zero_grad(set_to_none=True)
    return result


def main():
    """只构造单个 GDN 模块，不分配真实模型专家或执行完整 35B 模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", "started_at": time.time(), "shape": [1, 4096, 2048],
              "dtype": "bfloat16", "r": 16, "alpha": 32, "dropout": 0, "cudnn_deterministic": True}
    code = 0
    try:
        config = Qwen3_5MoeTextConfig()
        model = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                               target_modules=["in_proj_qkv", "in_proj_z", "out_proj"]))
        model.train()
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert len(trainable) == 6
        result["trainable"] = {name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in trainable.items()}
        initial_hashes = frozen_hashes(model)
        x = torch.randn(1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel() ** 0.5
        with torch.no_grad():
            enabled = model(x)
            with model.disable_adapter():
                disabled = model(x)
        result["zero_adapter"] = compare(enabled.cpu(), disabled.cpu())
        assert result["zero_adapter"]["bitwise_equal"]
        del enabled, disabled
        with torch.no_grad():
            for name, p in trainable.items():
                if "lora_B" in name:
                    p.normal_(std=0.005)
        direct = evaluate(model, x, dy, trainable, "direct")
        result["direct"] = {k: v for k, v in direct.items() if k not in ["output", "gradients"]}
        for mode in ["non-reentrant", "reentrant"]:
            replay = evaluate(model, x, dy, trainable, mode)
            comparisons = {"output": compare(replay["output"], direct["output"])}
            comparisons.update({name: compare(replay["gradients"][name], gradient)
                                for name, gradient in direct["gradients"].items()})
            result[mode] = {**{k: v for k, v in replay.items() if k not in ["output", "gradients"]},
                            "comparisons": comparisons}
            assert all(m["bitwise_equal"] for m in comparisons.values()), comparisons
            print(mode, "PASS", flush=True)
        assert initial_hashes == frozen_hashes(model)
        result["frozen_unchanged"] = True
        result["status"] = "PASS"
    except Exception:
        code = 1
        result["status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    result.update(ended_at=time.time(), exit_code=code)
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
