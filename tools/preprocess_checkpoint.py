"""Preprocess a verified checkpoint, selecting its family from source config."""
import argparse
import json
from pathlib import Path
import torch
from nvfp4lora.families import family_of_source, family_named, family_names
from nvfp4lora.format.bundle import identity_from_receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--family", choices=family_names(), help="Assert the family detected from source config")
    args = parser.parse_args()
    torch.set_num_threads(8)
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    # Reject incomplete receipts before touching output or accepting a model config.
    identity = identity_from_receipt(receipt, source_kind="verified-checkpoint")
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    family = family_of_source(config)
    if args.family is not None and family_named(args.family).name != family.name:
        raise ValueError("Requested family differs from source config")
    if identity["kind"] != "tiny-generated":
        identity["kind"] = family.source_kind
    manifest = family.preprocess(args.model_dir, args.output, identity)
    print(json.dumps({"status": manifest["status"], "family": family.name,
                      "geometry": manifest["geometry"], "peak_working_set_bytes": manifest["peak_working_set_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
