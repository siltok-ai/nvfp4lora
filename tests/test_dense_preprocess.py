"""Dense预处理的过滤、原字节、非流式分离与错误来源拒绝。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_tiny_dense import make_tiny_dense
from quantize_tiny_dense import quantize_tiny_dense
from nvfp4lora.format.inspect import inspect_checkpoint
from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT, fields, preprocess_dense, validate_layout
from nvfp4lora.store.expert_store import ExpertStore


class DensePreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        make_tiny_dense(cls.root / "bf16")
        cls.identity = quantize_tiny_dense(cls.root / "bf16", cls.root / "source")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_round_trip_filter_and_separate_nonstream(self):
        source = self.root / "source"
        bundle = self.root / "processed"
        result = preprocess_dense(source, bundle, self.identity)
        original = load_file(source / "model.safetensors")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["geometry"]["excluded"]), 8)
        self.assertTrue(any(name.startswith("mtp.") for name in result["geometry"]["excluded"]))
        self.assertTrue(any(name.startswith("model.visual.") for name in result["geometry"]["excluded"]))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            ExpertStore(bundle / "layers", pin_memory=False)
        store = ExpertStore(bundle / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
        for layer in range(4):
            tensors = store.tensors(layer)
            for name, spec in result["geometry"]["modules"][str(layer)].items():
                for field in fields(spec):
                    actual = tensors[f"{name}.{field}"].reshape(-1).view(torch.uint8)
                    expected = original[f"model.language_model.layers.{layer}.{name}.{field}"].reshape(-1).view(torch.uint8)
                    self.assertTrue(torch.equal(actual, expected), (layer, name, field))
        state = {}
        for entry in result["nonstream_files"]:
            state.update(load_file(bundle / entry["file"]))
        self.assertEqual(set(state), set(result["geometry"]["nonstream_parameters"]) | {"lm_head.weight"})
        self.assertFalse(any("mlp." in name or "mtp." in name or "visual." in name for name in state))
        self.assertTrue(torch.equal(state["model.embed_tokens.weight"].view(torch.uint8),
                                    original["model.language_model.embed_tokens.weight"].view(torch.uint8)))
        self.assertEqual(result["nonstream_numerical_policy"], "nonexpert-fp32-decode-bf16-v1")

    def test_source_checksum_failure_leaves_no_output(self):
        wrong = {**self.identity, "shards": [{"filename": "model.safetensors", "sha256": "0" * 64}]}
        output = self.root / "wrong-checksum"
        with self.assertRaisesRegex(ValueError, "checksum"):
            preprocess_dense(self.root / "source", output, wrong)
        self.assertFalse(output.exists())

    def test_extra_text_expert_cannot_hide_as_excluded_tensor(self):
        report = inspect_checkpoint(self.root / "source")
        report["tensors"]["model.language_model.layers.0.mlp.experts.0.weight"] = {"shape": [1], "dtype": "U8", "data_offsets": [0, 1]}
        config = json.loads((self.root / "source/config.json").read_text())["text_config"]
        with self.assertRaisesRegex(ValueError, "Unexpected dense text tensor"):
            validate_layout(report, config, tiny=True)

    def test_real_cli_rejects_partial_receipt(self):
        receipt = self.root / "partial.json"
        receipt.write_text(json.dumps({"status": "SHARD PASS; FULL G0.8 NOT RUN", "requested_weights": True}))
        output = self.root / "wrong-receipt"
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "tools/preprocess_dense_checkpoint.py"),
                   "--source-receipt", str(receipt), "--model-dir", str(self.root / "source"), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("three-shard", completed.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
