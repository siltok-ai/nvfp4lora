"""A1.4：真实模型只构建/加载，逐参数读回，不运行完整模型前向或训练。"""

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from safetensors import safe_open

from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model

from check_environment import configure_numerics


def main():
    """验证真实非专家实际值、全部 buffer、两套 LoRA 集合与内存。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    built = build_model(args.bundle, expert_rank=2, c3_rank=16, c3_alpha=32)
    base = causal_model(built.model)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    checked = []
    with torch.no_grad():
        for entry in manifest["nonexpert_files"]:
            with safe_open(args.bundle / entry["file"], framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    actual = base.get_parameter(name).detach().cpu()
                    expected = reader.get_tensor(name)
                    assert actual.dtype == expected.dtype == torch.bfloat16, name
                    assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                       expected.contiguous().reshape(-1).view(torch.uint8)), name
                    assert torch.isfinite(actual).all(), name
                    checked.append(name)
                    del actual, expected
            print("loaded values checked:", entry["file"], flush=True)
        for name, parameter in built.trainable.items():
            assert parameter.dtype == torch.float32 and torch.isfinite(parameter).all(), name
            if ".lora_B" in name:
                assert parameter.count_nonzero() == 0, name
    fresh = type(base.model.rotary_emb)(base.config, device="cpu")
    for name, expected in fresh.named_buffers():
        actual = getattr(base.model.rotary_emb, name).cpu()
        assert actual.dtype == expected.dtype == torch.float32
        assert torch.equal(actual, expected), name
    assert all(not value.is_meta for value in [*base.parameters(), *base.buffers()])
    c3 = sum(p.numel() for name, p in built.trainable.items() if ".mlp.experts." not in name)
    expert = sum(p.numel() for name, p in built.trainable.items() if ".mlp.experts." in name)
    assert (c3, expert) == (19169280, 157286400), (c3, expert)
    torch.cuda.synchronize()
    result = {"status": "PASS", "scope": "A1.4 materialization only; no whole-model forward/training",
              "seconds": time.perf_counter() - start, "nonexpert_values_checked": len(checked),
              "all_nonexpert_bytes_equal": True, "initialized_buffers": built.initialized_buffers,
              "C3_numel": c3, "expert_lora_numel": expert, "canonical_trainable_tensors": len(built.trainable),
              "zero_B": True, "no_meta": True, "logical_cpu_bank_bytes": built.store.nbytes,
              "pinned_allocator": torch.cuda.memory.host_memory_stats(),
              "gpu_allocated_bytes": torch.cuda.memory_allocated(),
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "cpu_memory": psutil.Process().memory_info()._asdict()}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "nonexpert_values_checked", "C3_numel", "expert_lora_numel", "gpu_peak_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
