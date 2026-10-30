"""检查锁定环境的 CUDA 运算、模型结构及普通线性层 LoRA。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nvfp4lora.testing.numerics import configure_numerics

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling




def bf16_check():
    """在实际设备执行 BF16 GEMM，并与独立 FP32 输入梯度对照。"""
    x = torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16) / 32
    y = x @ w.T
    dy = torch.randn_like(y)
    dx, = torch.autograd.grad(y, x, dy)
    ref = dy.float() @ w.float()
    error = float((dx.float() - ref).norm() / ref.norm())
    assert torch.isfinite(y).all() and torch.isfinite(dx).all()
    assert error <= 0.01, error
    return {"status": "PASS", "shape": [128, 2048, 512], "dx_relative_l2": error}


def meta_check():
    """只在 meta 构建完整 35B 架构，核对真实模块接口和融合维度。"""
    config = Qwen3_5MoeTextConfig()
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
    layers = model.model.layers
    expert = layers[0].mlp.experts
    assert all(p.is_meta for p in model.parameters())
    assert list(expert.gate_up_proj.shape) == [256, 1024, 2048]
    assert list(expert.down_proj.shape) == [256, 2048, 512]
    assert list(layers[3].self_attn.q_proj.weight.shape) == [8192, 2048]
    assert list(layers[0].linear_attn.in_proj_qkv.weight.shape) == [8192, 2048]
    expected_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    assert config.layer_types == expected_types
    result = {"status": "PASS", "parameter_count": sum(p.numel() for p in model.parameters()),
              "layers": len(layers), "layer_types": config.layer_types,
              "expert_forward": str(inspect.signature(expert.forward)),
              "gate_up_shape": list(expert.gate_up_proj.shape),
              "down_shape": list(expert.down_proj.shape),
              "q_proj_shape": list(layers[3].self_attn.q_proj.weight.shape),
              "tied_embeddings": config.tie_word_embeddings,
              "source_file": inspect.getfile(modeling),
              "source_sha256": hashlib.sha256(Path(inspect.getfile(modeling)).read_bytes()).hexdigest(),
              "source_review": {"q_proj": "view by head then split query/gate; interleaved by head",
                                "gate_up_proj": "split gate then up on final dimension"}}
    del model
    return result


def peft_check():
    """在实际 HF 模型中注入 PEFT，验证零初始化、梯度和一次更新。"""
    config = Qwen3_5MoeTextConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_experts=4, num_experts_per_tok=2, layer_types=["full_attention"],
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2]},
        tie_word_embeddings=False, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    base = Qwen3_5MoeForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                          r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "o_proj"]))
    model.train()
    ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    with torch.no_grad():
        enabled = model(input_ids=ids, labels=ids).loss
        with model.disable_adapter():
            disabled = model(input_ids=ids, labels=ids).loss
    assert torch.equal(enabled, disabled), (enabled, disabled)
    trainable = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01, weight_decay=0, foreach=False)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    for name, parameter in trainable.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if "lora_B" in name:
            assert parameter.grad.count_nonzero() > 0, name
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    with torch.no_grad():
        after = model(input_ids=ids, labels=ids).loss
    assert after != enabled
    return {"status": "PASS", "zero_adapter_bitwise_equal": True,
            "initial_loss": float(enabled), "after_step_loss": float(after),
            "trainable": {n: {"shape": list(p.shape), "dtype": str(p.dtype)} for n, p in trainable.items()}}


def main():
    """每个运行目录只写一次资格结果，异常和未运行项显式保留。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-closure", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"started_at": time.time(), "python": sys.executable, "python_version": sys.version,
              "platform": platform.platform(), "packages": {}, "gates": {},
              "numerics": {"tf32": False, "bf16_reduced_precision_reduction": False},
              "cpu_start": psutil.Process().memory_info()._asdict()}
    failed = False
    try:
        configure_numerics()
        versions = json.loads(args.baseline_closure.read_text(encoding="utf-8"))["versions"]
        for name, expected in versions.items():
            actual = metadata.version(name)
            result["packages"][name] = actual
            assert actual == expected, (name, expected, actual)
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
        result["device"] = {"name": torch.cuda.get_device_name(),
                            "capability": list(torch.cuda.get_device_capability()),
                            "total_bytes": torch.cuda.get_device_properties(0).total_memory}
        torch.cuda.reset_peak_memory_stats()
        for gate, function in [("G0.1", bf16_check), ("G0.4", meta_check), ("G0.5", peft_check)]:
            start = time.perf_counter()
            result["gates"][gate] = {"status": "RUNNING"}
            try:
                result["gates"][gate] = {**function(), "seconds": time.perf_counter() - start}
            except Exception:
                result["gates"][gate] = {"status": "FAIL", "traceback": traceback.format_exc()}
                failed = True
                break
            print(gate, result["gates"][gate]["status"], flush=True)
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except Exception:
        failed = True
        result["setup_error"] = traceback.format_exc()
    result.update(ended_at=time.time(), exit_code=int(failed), cpu_end=psutil.Process().memory_info()._asdict())
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "exit_code": int(failed)}), flush=True)
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
