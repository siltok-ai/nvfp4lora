"""从完整检查点重建周期性诊断：A/B范数、金丝雀留出NLL、固定批次路由。"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
from nvfp4lora.train.checkpoint import configuration_hash
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.evaluate import evaluate_nll, evaluation_mode, padded_batch
from nvfp4lora.train.loop import configure_training, frozen_fingerprint, memory_observation
from nvfp4lora.train.session import logical_config
from nvfp4lora.validation.route_probe import MaskedRouteProbe


def load_parameters(built, checkpoint, config):
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["layout_version"] == "nvfp4-training-checkpoint-v1" and manifest["status"] == "complete"
    assert manifest["config_sha256"] == configuration_hash(config)
    assert all(file_hash(checkpoint / name) == digest for name, digest in manifest["files"].items())
    values = load_file(checkpoint / "lora_params.safetensors")
    assert values.keys() == built.trainable.keys() == manifest["parameters"].keys()
    squared = {}
    with torch.no_grad():
        for name, value in values.items():
            parameter = built.trainable[name]
            assert value.dtype == parameter.dtype == torch.float32 and value.shape == parameter.shape
            assert list(value.shape) == manifest["parameters"][name]["shape"]
            assert str(value.dtype) == manifest["parameters"][name]["dtype"] and torch.isfinite(value).all()
            parameter.copy_(value)
            assert ".lora_A" in name or ".lora_B" in name
            group = ("expert" if ".mlp.experts." in name else "nonexpert") + ("_A" if ".lora_A" in name else "_B")
            squared[group] = squared.get(group, 0.) + float(value.double().square().sum())
    return {"checkpoint_manifest_sha256": file_hash(checkpoint / "checkpoint_manifest.json"),
            "lora_file_sha256": file_hash(checkpoint / "lora_params.safetensors"),
            "group_l2_norms": {key: math.sqrt(value) for key, value in squared.items()}}


def route_probe(built, samples, *, disabled=False):
    base = causal_model(built.model)
    device = next(base.parameters()).device
    ids, labels, mask, positions = padded_batch(samples, device, pad_token_id=248044)
    observer = MaskedRouteProbe(built.model, mask, experts=base.config.num_experts)
    try:
        with evaluation_mode(built, disabled=disabled):
            hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
            assert torch.isfinite(hidden).all()
        result = observer.result()
    finally:
        observer.close()
    assert len(result) == base.config.num_hidden_layers
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.training_results / "train_config.json").read_text(encoding="utf-8"))
    corpus = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    assert corpus["base"] == base_identity(args.bundle)
    assert all(file_hash(args.corpus / name) == digest for name, digest in corpus["files"].items())
    configure_training(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    data = ChatJSONLDataset(args.corpus / "train.jsonl", tokenizer, max_length=config["max_length"])
    heldout = ChatJSONLDataset(args.corpus / "canary-holdout.jsonl", tokenizer, max_length=config["max_length"])
    assert logical_config(config, args.bundle, data) == config
    checkpoints = [args.training_results / "checkpoints" / f"step-{step:06d}" for step in args.steps]
    assert all((path / "checkpoint_manifest.json").is_file() for path in checkpoints)
    built = build_model(args.bundle, **config["build"])
    frozen = frozen_fingerprint(built)
    samples = [data[index] for index in range(min(8, len(data)))]
    result = {"status": "MEASURING", "scope": "Offline reconstructions from saved parameters; not original-time probe capture or optimizer resume",
              "probe_sample_ids": [row["sample_id"] for row in samples], "dataset": data.identity,
              "heldout_identity": heldout.identity, "checkpoints": {}}

    def measure(disabled=False):
        torch.cuda.reset_peak_memory_stats()
        routes = route_probe(built, samples, disabled=disabled)
        nll = evaluate_nll(built, heldout, ce_chunk_size=config["ce_chunk_size"], disabled=disabled)
        return {"routes": routes, "canary_holdout_nll": nll, "memory": memory_observation()}

    result["base_disabled"] = measure(disabled=True)
    for step, path in zip(args.steps, checkpoints):
        metadata = load_parameters(built, path, config)
        result["checkpoints"][str(step)] = {**metadata, **measure()}
        print(f"checkpoint {step}: canary holdout NLL={result['checkpoints'][str(step)]['canary_holdout_nll']['nll_per_token']:.8f}", flush=True)
    result["frozen_hashes_unchanged"] = frozen_fingerprint(built) == frozen
    assert result["frozen_hashes_unchanged"]
    result["status"] = "MEASURED"
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    built.streamer.close()


if __name__ == "__main__":
    main()
