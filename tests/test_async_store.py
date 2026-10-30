"""真实 CUDA 流、事件、pin 与 generation，不用模拟事件代替 GPU 竞争。"""

import random
import tempfile
import unittest
from pathlib import Path

import torch

from nvfp4lora.store.async_streamer import AsyncLayerStreamer
from nvfp4lora.store.layer_store import PackedLayerStore
from test_store import make_banks


@unittest.skipUnless(torch.cuda.is_available(), "actual asynchronous CUDA transfer")
class AsyncTests(unittest.TestCase):
    def test_pins_generation_stream_and_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_banks(root)
            store = PackedLayerStore(root, layout_version="packed-nvfp4-experts-v1")
            streamer = AsyncLayerStreamer(store)
            first, second = streamer.acquire(0), streamer.acquire(1)
            self.assertFalse(streamer.prefetch(2))
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                streamer.acquire(2)
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                streamer.close()
            streamer.release(first)
            with streamer.acquire(2) as third:
                self.assertEqual(third.layer, 2)
                self.assertGreater(third.generation, first.generation)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                first.tensors()
            with self.assertRaisesRegex(RuntimeError, "Invalid"):
                streamer.release(first)
            alternate = torch.cuda.Stream()
            with torch.cuda.stream(alternate):
                with self.assertRaisesRegex(RuntimeError, "one CUDA compute"):
                    streamer.acquire(1)
            streamer.release(second)
            streamer.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                streamer.acquire(0)

    def test_thousand_random_delayed_readers_and_wrong_prefetch_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_banks(root)
            store = PackedLayerStore(root, layout_version="packed-nvfp4-experts-v1")
            before = store.hashes()
            streamer = AsyncLayerStreamer(store)
            rng = random.Random(42)
            layers = [rng.randrange(3) for _ in range(1000)]
            expected = torch.stack([store.banks[layer] for layer in layers])
            actual = torch.empty(expected.shape, dtype=torch.uint8, device="cuda")
            for index, layer in enumerate(layers):
                with streamer.acquire(layer) as handle:
                    streamer.prefetch((layer + 1) % 3)
                    # release 早于 GPU 读到槽；漏掉 last_reader 事件就会被下一次 copy 覆盖。
                    torch.cuda._sleep(100000)
                    actual[index].copy_(handle.slot.buffer[:store.banks[layer].numel()])
            streamer.collect_timings(synchronize=True)
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertEqual(store.hashes(), before)
            self.assertGreater(streamer.stats["prefetch_copies"], 0)
            self.assertGreater(streamer.stats["copy_gpu_ms"], 0)
            self.assertGreater(streamer.stats["acquire_wait_gpu_ms"], 0)
            streamer.close()


if __name__ == "__main__":
    unittest.main()
