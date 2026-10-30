"""小格式夹具测试打包、输入校验与流式输出；不替代 P4 的真实 HF tiny 模型。"""

import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nvfp4lora.format.nvfp4 import dequantize, quantize
from nvfp4lora.format.bundle import FIELDS, file_hash
from nvfp4lora.format.preprocess_moe import PROJECTIONS, preprocess


def make_fixture(root):
    """每个专家/投影具有不同内容，便于发现编号或 gate/up/down 对调。"""
    torch.manual_seed(61)
    tensors = {"model.language_model.embed_tokens.weight": torch.randn(64, 32).bfloat16(),
               "model.language_model.norm.weight": torch.randn(32).bfloat16(),
               "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 32, 32).bfloat16(),
               "mtp.layers.0.mlp.experts.down_proj": torch.randn(2, 32, 16).bfloat16()}
    for expert in range(2):
        for projection in PROJECTIONS:
            shape = (32, 16) if projection == "down_proj" else (16, 32)
            part = quantize(torch.randn(*shape) * (expert + 1))
            for field, value in part.items():
                if field == "weight_scale":
                    value = value.view(torch.float8_e4m3fn)
                tensors[f"model.language_model.layers.0.mlp.experts.{expert}.{projection}.{field}"] = value
    head = quantize(torch.randn(64, 32))
    for field, value in head.items():
        tensors[f"lm_head.{field}"] = value.view(torch.float8_e4m3fn) if field == "weight_scale" else value
    for name, shape in [("in_proj_qkv", (32, 32)), ("in_proj_z", (16, 32)), ("out_proj", (32, 16))]:
        prefix = f"model.language_model.layers.0.linear_attn.{name}"
        tensors[prefix + ".weight"] = torch.randn(*shape).to(torch.float8_e4m3fn)
        tensors[prefix + ".weight_scale"] = torch.tensor(0.5)
        tensors[prefix + ".input_scale"] = torch.tensor(0.25)
    for name in PROJECTIONS:
        shape = (32, 16) if name == "down_proj" else (16, 32)
        for field, value in quantize(torch.randn(*shape)).items():
            if field == "weight_scale":
                value = value.view(torch.float8_e4m3fn)
            tensors[f"model.language_model.layers.0.mlp.shared_expert.{name}.{field}"] = value
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(x.numel() * x.element_size() for x in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    (root / "config.json").write_text(json.dumps({"text_config": {
        "hidden_size": 32, "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
        "num_experts": 2, "num_hidden_layers": 1, "vocab_size": 64,
        "layer_types": ["linear_attention"], "linear_num_key_heads": 1, "linear_key_head_dim": 8,
        "linear_num_value_heads": 2, "linear_value_head_dim": 8}}))
    identity = {"kind": "tiny-format-fixture", "shards": [
        {"filename": "model.safetensors", "sha256": file_hash(root / "model.safetensors")}]}
    return tensors, identity


class PreprocessTests(unittest.TestCase):
    def test_real_cli_rejects_single_shard_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"status": "SHARD PASS; FULL G0.8 NOT RUN", "requested_weights": True}))
            script = Path(__file__).resolve().parents[1] / "tools/preprocess_checkpoint.py"
            completed = subprocess.run([sys.executable, "-B", str(script), "--model-dir", str(root),
                                        "--output", str(root / "processed"), "--source-receipt", str(receipt)],
                                       capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((root / "processed").exists())

    def test_packed_bytes_canonical_names_and_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            original, identity = make_fixture(source)
            result = preprocess(source, root / "processed", identity)
            self.assertEqual(result["status"], "PASS")
            layer = load_file(root / "processed/experts/layer-00.safetensors")
            for expert in range(2):
                for projection in PROJECTIONS:
                    for field in FIELDS:
                        actual = layer[f"{projection}.{field}"][expert].reshape(-1).view(torch.uint8)
                        expected = original[f"model.language_model.layers.0.mlp.experts.{expert}.{projection}.{field}"].reshape(-1).view(torch.uint8)
                        self.assertTrue(torch.equal(actual, expected), (expert, projection, field))
            state = {}
            for file in result["nonexpert_files"]:
                state.update(load_file(root / "processed" / file["file"]))
            expected_keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
            expected_keys.update(f"model.layers.0.linear_attn.{name}.weight" for name in ["in_proj_qkv", "in_proj_z", "out_proj"])
            expected_keys.update(f"model.layers.0.mlp.shared_expert.{name}.weight" for name in PROJECTIONS)
            self.assertEqual(set(state), expected_keys)
            self.assertTrue(torch.equal(state["model.embed_tokens.weight"], original["model.language_model.embed_tokens.weight"]))
            expected_head = dequantize(original["lm_head.weight"], original["lm_head.weight_scale"],
                                       original["lm_head.weight_scale_2"], dtype=torch.bfloat16)
            self.assertTrue(torch.equal(state["lm_head.weight"], expected_head))

    def test_changed_source_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            _, identity = make_fixture(source)
            with (source / "model.safetensors").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                preprocess(source, root / "processed", identity)
            self.assertFalse((root / "processed").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
