"""发布适配器的金丝雀/回放/提示logits变化；阈值由调用者显式提供。"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import base_identity, build_from_adapter
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.families import family_of_bundle
from nvfp4lora.model.common import causal_model
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.evaluate import evaluate_canary, evaluate_nll, evaluation_mode
from nvfp4lora.train.loop import configure_training, memory_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-canary-accuracy", type=float)
    parser.add_argument("--max-replay-nll-increase", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(family_of_bundle(args.bundle).default_seed)
    corpus = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    assert corpus["base"] == base_identity(args.bundle)
    assert all(file_hash(args.corpus / name) == digest for name, digest in corpus["files"].items())
    baseline = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    identity = json.loads((args.baseline_evaluation.parent / "evaluation_identity.json").read_text(encoding="utf-8"))
    assert identity["replay"]["file_sha256"] == file_hash(args.corpus / "replay.jsonl")
    assert identity["canary_file_sha256"] == file_hash(args.corpus / "canary-holdout.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    built = build_from_adapter(args.bundle, args.adapter)
    replay = ChatJSONLDataset(args.corpus / "replay.jsonl", tokenizer, max_length=2048)
    result = {"status": "MEASURING", "base_canary_correct": baseline["canary"]["correct"],
              "base_replay_nll": baseline["replay"]["nll_per_token"]}
    result["replay"] = evaluate_nll(built, replay, batch_size=args.batch_size,
        progress=lambda done, total: print(f"replay {done}/{total}", flush=True))
    result["canary"] = evaluate_canary(built, tokenizer, args.corpus / "canary-holdout.jsonl", batch_size=args.batch_size,
        progress=lambda done, total: print(f"canary {done}/{total}", flush=True))
    row = json.loads((args.corpus / "canary-holdout.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ids = tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=True,
                                        enable_thinking=False, return_dict=False)
    device = next(built.model.parameters()).device
    ids = torch.tensor([ids], device=device)
    base = causal_model(built.model)
    def logits(disabled):
        with evaluation_mode(built, disabled=disabled):
            hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            return base.lm_head(hidden[:, -1]).float()
    enabled, disabled = logits(False), logits(True)
    difference = enabled - disabled
    result["prompt_logits_change"] = {"l2": float(difference.norm()), "max_abs": float(difference.abs().max()),
                                     "changed_vocabulary_entries": int(difference.count_nonzero()),
                                     "enabled_top1": int(enabled.argmax()), "disabled_top1": int(disabled.argmax())}
    result["replay_nll_increase"] = result["replay"]["nll_per_token"] - result["base_replay_nll"]
    criteria = {}
    if args.minimum_canary_accuracy is not None:
        criteria["canary"] = result["canary"]["accuracy"] >= args.minimum_canary_accuracy and result["canary"]["correct"] > result["base_canary_correct"]
    if args.max_replay_nll_increase is not None:
        criteria["replay"] = result["replay_nll_increase"] <= args.max_replay_nll_increase
    criteria["adapter_prompt_logits_nonzero"] = bool(difference.count_nonzero())
    result.update(status="MEASURED" if len(criteria) < 3 else ("PASS" if all(criteria.values()) else "FAIL"),
                  criteria=criteria, thresholds={"minimum_canary_accuracy": args.minimum_canary_accuracy,
                  "max_replay_nll_increase": args.max_replay_nll_increase}, memory=memory_observation())
    (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    built.streamer.close()
    print(json.dumps({key: value for key, value in result.items() if key != "canary"}, indent=2))
    if len(criteria) == 3:
        assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
