"""新进程只从真实预处理基座＋发布产物建模，按固定数据/数值记录核对E2。"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import _artifact_tensors, build_from_adapter
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.loop import configure_training, loss_sum, memory_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    # 原训练目录仅读取固定样本、逻辑配置和保存前数值记录；建模只使用publish。
    config = json.loads((args.source_results / "train_config.json").read_text(encoding="utf-8"))
    expected = json.loads((args.source_results / "results.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(args.source_results / "smoke.jsonl", tokenizer,
                               max_length=config["dataset"]["max_length"])
    assert dataset.identity == config["dataset"]
    built = build_from_adapter(args.bundle, args.source_results / "publish")
    built.model.eval()
    built.streamer.phase = "FORWARD"
    batch = dataset[0]
    with torch.no_grad():
        actual = loss_sum(built, batch, ce_chunk_size=config["ce_chunk_size"]) / supervised_tokens(batch["labels"])
        with disable_adapters(built.model):
            disabled = loss_sum(built, batch, ce_chunk_size=config["ce_chunk_size"]) / supervised_tokens(batch["labels"])
    expected_loss = torch.tensor(expected["loss_probe"]["after"], dtype=torch.float32)
    expected_disabled = torch.tensor(expected["loss_probe"]["disabled_after"], dtype=torch.float32)
    checks = {"fixed_batch_loss_E2": torch.equal(actual.cpu().view(torch.int32), expected_loss.view(torch.int32)),
              "disabled_baseline_loss_E2": torch.equal(disabled.cpu().view(torch.int32), expected_disabled.view(torch.int32))}
    published, _ = _artifact_tensors(args.source_results / "publish")
    checks["all_gpu_parameter_bytes_equal_to_publish"] = all(
        torch.equal(parameter.detach().cpu().contiguous().view(torch.uint8), published[name].contiguous().view(torch.uint8))
        for name, parameter in built.trainable.items())
    checks["canonical_set_equal"] = published.keys() == built.trainable.keys() == built.expected_trainable.keys()
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "loss": float(actual), "disabled_loss": float(disabled),
              "trainable_tensors": len(built.trainable), "trainable_numel": sum(p.numel() for p in built.trainable.values()),
              "memory": memory_observation(), "base": str(args.bundle),
              "adapter": str(args.source_results / "publish"),
              "scope": "Actual step1 adapter new-process reload; no resumed optimizer step or training-memory deltaW claim"}
    built.streamer.close()
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
