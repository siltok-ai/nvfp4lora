"""D1独立读回：源到CPU store全量E1、两次加载哈希及非流式文件集合。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open


def freetoken_samples(args, manifest):
    """在独立推理环境只读调用既有FreeToken核，raw F32 global与运行时F16分列。"""
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4
    from tools.validation.check_freetoken_dequant import ulp_distance
    from nvfp4lora.format.nvfp4 import dequantize
    weight_map = json.loads((args.source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    layers = [0, 3] if manifest["geometry"]["layers"] == 4 else [0, 3, 31, 63]
    cases = []
    for layer in layers:
        for projection in ["gate", "up", "down"]:
            prefix = f"model.language_model.layers.{layer}.mlp.{projection}_proj"
            names = [prefix + "." + field for field in ["weight", "weight_scale", "weight_scale_2"]]
            raw = {}
            for filename in sorted({weight_map[name] for name in names}):
                with safe_open(args.source / filename, framework="pt", device="cpu") as reader:
                    for name in names:
                        if weight_map[name] == filename:
                            raw[name] = reader.get_tensor(name).clone()
            weight, scale, global_scale = (raw[name] for name in names)
            for start in [0, weight.shape[0] // 2, weight.shape[0] - 64]:
                w, s = weight[start:start + 64], scale[start:start + 64]
                g = global_scale.reshape(()).expand(64).contiguous()
                expected = dequantize(w, s, global_scale, dtype=torch.bfloat16)
                actual = _dequant_nvfp4(w, s, g)
                runtime = _dequant_nvfp4(w, s, g.to(torch.float16))
                error = ulp_distance(actual, expected)
                row = {"layer": layer, "projection": projection, "row_start": start, "shape": list(actual.shape),
                       "raw_f32_global_max_ulp": error, "runtime_f16_vs_raw_f32_max_ulp": ulp_distance(runtime, actual)}
                cases.append(row)
                assert error <= 1, row
            del raw, weight, scale, global_scale, expected, actual, runtime
    source = Path(inspect.getfile(_dequant_nvfp4))
    result = {"status": "PASS", "scope": "DA1.3 real MLP weight sampling; no model forward/training",
              "source_identity": manifest["source"], "cases": cases,
              "raw_f32_global_worst_ulp": max(row["raw_f32_global_max_ulp"] for row in cases),
              "runtime_f16_vs_raw_f32_worst_ulp": max(row["runtime_f16_vs_raw_f32_max_ulp"] for row in cases),
              "freetoken_source": str(source), "freetoken_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freetoken-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["layout_version"] == "nvfp4-stream-dense-v1"
    if args.freetoken_only:
        freetoken_samples(args, manifest)
        return
    import psutil
    from nvfp4lora.format.bundle import file_hash, load_group
    from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT, fields
    from nvfp4lora.store.layer_store import PackedLayerStore
    for entry in manifest["source"]["shards"] + manifest["source"].get("metadata", []):
        assert file_hash(args.source / entry["filename"]) == entry["sha256"]
    weight_map = json.loads((args.source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    store = PackedLayerStore(args.bundle / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
    hashes = store.hashes()
    e1 = []
    for layer in range(manifest["geometry"]["layers"]):
        modules = manifest["geometry"]["modules"][str(layer)]
        prefix = f"model.language_model.layers.{layer}."
        names = [f"{prefix}{name}.{field}" for name, spec in modules.items() for field in fields(spec)]
        raw = load_group(args.source, weight_map, names)
        packed = store.tensors(layer)
        for name, spec in modules.items():
            for field in fields(spec):
                expected = raw[f"{prefix}{name}.{field}"].reshape(-1).view(torch.uint8)
                actual = packed[f"{name}.{field}"].reshape(-1).view(torch.uint8)
                assert torch.equal(actual, expected), (layer, name, field)
        e1.append({"layer": layer, "tensors": len(names), "bytes": store.banks[layer].numel(), "status": "PASS"})
        del raw, packed
        print(f"store E1 layer {layer}: PASS", flush=True)
    expected_keys = set(manifest["geometry"]["nonstream_parameters"]) | {"lm_head.weight"}
    observed = set()
    for entry in manifest["nonstream_files"]:
        assert file_hash(args.bundle / entry["file"]) == entry["sha256"]
        with safe_open(args.bundle / entry["file"], framework="pt", device="cpu") as reader:
            assert set(reader.keys()) == set(entry["keys"])
            for key in reader.keys():
                assert key not in observed
                observed.add(key)
                tensor = reader.get_tensor(key)
                shape = ([manifest["geometry"]["vocab_size"], manifest["geometry"]["hidden"]] if key == "lm_head.weight"
                         else manifest["geometry"]["nonstream_parameters"][key])
                assert list(tensor.shape) == shape and tensor.dtype == torch.bfloat16
                del tensor
    assert observed == expected_keys
    for filename, expected in manifest["output_files_sha256"].items():
        assert file_hash(args.bundle / filename) == expected
    payload = store.nbytes
    del store
    gc.collect()
    reloaded = PackedLayerStore(args.bundle / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
    assert hashes == reloaded.hashes()
    result = {"status": "PASS", "scope": "D1 independent pageable-store validation; not preprocessing peak or training",
              "source": manifest["source"], "full_e1": e1, "store_hashes": hashes, "store_reload_E2": True,
              "layer_payload_bytes": payload, "maximum_layer_bytes": max(item["bytes"] for item in e1),
              "nonstream_keys": sorted(observed), "nonstream_payload_bytes": manifest["nonstream_payload_bytes"],
              "fp8_code_exact_elements": sum(item["fp8_exact_elements"] for item in manifest["e1_full"]),
              "preprocessing_peak_working_set_bytes": manifest["peak_working_set_bytes"],
              "validation_process_memory": psutil.Process().memory_info()._asdict(),
              "optional_freetoken_nvfp4_comparison": "NOT RUN"}
    assert result["maximum_layer_bytes"] < 256 * 1024 ** 2 and manifest["peak_working_set_bytes"] <= 16_000_000_000
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS: DA1.1/DA1.2/DA1.4/DA1.5/DA1.6 and exact FP8 code conversion", flush=True)


if __name__ == "__main__":
    main()
