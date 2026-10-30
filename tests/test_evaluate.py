"""真实HF tiny上的左padding、跨样本CE加权，以及评测后的训练状态恢复。"""

import gc
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_tiny_model import make_tiny_model
from quantize_tiny import quantize_tiny
from nvfp4lora.format.preprocess_moe import preprocess
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.train.evaluate import evaluate_nll


class EvaluationTests(unittest.TestCase):
    def test_real_fp32_tiny_batch_and_single_nll_match_and_restore_training(self):
        torch.set_num_threads(8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_tiny_model(root / "bf16")
            identity = quantize_tiny(root / "bf16", root / "raw")
            preprocess(root / "raw", root / "bundle", identity)
            built = build_model(root / "bundle", device="cpu", c3_rank=4, c3_alpha=8, expert_rank=2, reference=True)
            built.model.float().train()
            built.streamer.phase = "BACKWARD"
            samples = []
            for length, targets in [(13, 3), (17, 5)]:
                ids = torch.randint(3, 2048, (1, length))
                labels = ids.clone()
                labels[:, :-targets] = -100
                samples.append({"input_ids": ids, "labels": labels})
            single = evaluate_nll(built, samples, batch_size=1, ce_chunk_size=3, pad_token_id=2)
            batch = evaluate_nll(built, samples, batch_size=2, ce_chunk_size=3, pad_token_id=2)
            self.assertEqual(single["supervised_tokens"], 8)
            self.assertEqual(batch["supervised_tokens"], 8)
            self.assertAlmostEqual(single["nll_per_token"], batch["nll_per_token"], places=5)
            self.assertTrue(built.model.training)
            self.assertEqual(built.streamer.phase, "BACKWARD")
            del built
            gc.collect()


if __name__ == "__main__":
    unittest.main()
