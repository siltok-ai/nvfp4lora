"""Observe original FP8 activation ranges without changing training math."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from nvfp4lora.export.artifacts import build_from_adapter
from nvfp4lora.families import family_of_bundle
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.model.common import causal_model
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.evaluate import evaluation_mode, padded_batch
from nvfp4lora.train.loop import configure_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    family = family_of_bundle(manifest)
    for shard in manifest["source"]["shards"]:
        if file_hash(args.source / shard["filename"]) != shard["sha256"]:
            raise ValueError(f"FP8 source checksum mismatch: {shard['filename']}")
    configure_training(family.default_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(args.data, tokenizer, max_length=2048)
    built = build_from_adapter(args.bundle, args.adapter) if args.adapter else family.base_build(args.bundle)
    audit = family.fp8_audit(built, args.bundle, args.source)
    base = causal_model(built.model)
    device = next(base.parameters()).device
    _, pad = family.replay_stops(args.bundle, tokenizer)
    try:
        for disabled, phase in [(True, "base_disabled"), *([(False, "adapter_enabled")] if args.adapter else [])]:
            audit.begin(phase)
            with evaluation_mode(built, disabled=disabled):
                for start in range(0, len(dataset), args.batch_size):
                    samples = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
                    ids, labels, mask, positions = padded_batch(samples, device, pad_token_id=pad)
                    audit.mask = mask.bool()
                    hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
                    if not torch.isfinite(hidden).all():
                        raise FloatingPointError("Nonfinite FP8 range observation forward")
                    print(f"{phase}: {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
        result = {**audit.finish(), "dataset": dataset.identity, "family": family.name}
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        audit.close()
        built.streamer.close()


if __name__ == "__main__":
    main()
