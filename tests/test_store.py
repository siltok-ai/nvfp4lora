"""真实文件读入、槽占用与实际 CPU/CUDA 传输的同步基线验证。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from nvfp4lora.format.inspect import read_header
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.store.layer_store import ExpertStore
from nvfp4lora.store.streamer import SyncLayerStreamer


def make_banks(root):
    """三层不同字节的合法预处理 bank，独立保留期待张量以发现错误层命中。"""
    torch.manual_seed(152)
    entries, expected = [], {}
    for layer in range(3):
        values = {}
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            out_dim, in_dim = (32, 16) if projection == "down_proj" else (16, 32)
            values[projection + ".weight"] = torch.randint(0, 256, (2, out_dim, in_dim // 2), dtype=torch.uint8)
            values[projection + ".weight_scale"] = torch.randint(0, 127, (2, out_dim, in_dim // 16), dtype=torch.uint8)
            values[projection + ".weight_scale_2"] = torch.rand(2)
            values[projection + ".input_scale"] = torch.full((2,), float(layer + 1))
        filename = f"layer-{layer:02d}.safetensors"
        save_file(values, root / filename)
        header = read_header(root / filename)
        entries.append({"layer": layer, "file": filename, "sha256": file_hash(root / filename),
                        **{k: header[k] for k in ["header_bytes", "payload_bytes", "tensors"]}})
        expected[layer] = values
    (root / "index.json").write_text(json.dumps({"layout_version": "packed-nvfp4-experts-v1",
        "geometry": {"layers": 3}, "source": {"kind": "transport-fixture"}, "layers": entries}))
    return expected


class StoreTests(unittest.TestCase):
    def test_payload_views_and_frozen_hash_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = make_banks(root)
            store = ExpertStore(root, pin_memory=False)
            for layer, values in expected.items():
                for name, tensor in store.tensors(layer).items():
                    self.assertTrue(torch.equal(tensor, values[name]), (layer, name))
            before = store.hashes()
            store.tensors(0)["gate_proj.input_scale"][0] = 99
            self.assertNotEqual(store.hashes()[0], before[0])

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_banks(root)
            path = root / "layer-00.safetensors"
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                ExpertStore(root, pin_memory=False)

    def test_cpu_slot_pins_generation_and_actual_data(self):
        self._exercise_slots("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real H2D qualification")
    def test_cuda_slot_pins_generation_and_actual_data(self):
        self._exercise_slots("cuda")

    def _exercise_slots(self, device):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = make_banks(root)
            store = ExpertStore(root, pin_memory=(device == "cuda"))
            before = store.hashes()
            streamer = SyncLayerStreamer(store, device=device, slots=2)
            first = streamer.acquire(0)
            same = streamer.acquire(0)
            self.assertEqual(streamer.stats["copies"], 1)
            self.assertEqual(streamer.stats["cache_hits"], 1)
            other = streamer.acquire(1)
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                streamer.acquire(2)
            streamer.release(other)
            with streamer.acquire(2) as handle:
                for name, tensor in handle.tensors().items():
                    self.assertTrue(torch.equal(tensor.cpu(), expected[2][name]), name)
            streamer.release(same)
            streamer.release(first)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                first.tensors()
            for layer in [1, 2, 0, 2, 1, 0]:
                with streamer.acquire(layer) as handle:
                    for name, tensor in handle.tensors().items():
                        self.assertTrue(torch.equal(tensor.cpu(), expected[layer][name]), (layer, name))
            self.assertEqual(store.hashes(), before)
            self.assertGreater(streamer.stats["copies"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
