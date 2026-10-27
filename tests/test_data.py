"""跨 epoch 游标与局部随机数据的恢复，不触碰全局 RNG。"""

import unittest

import torch

from nvfp4lora.train.data import DeterministicSampleStream, SyntheticTinyDataset


class DataTests(unittest.TestCase):
    def test_cross_epoch_resume_and_global_rng_unchanged(self):
        dataset = SyntheticTinyDataset(size=7)
        stream = DeterministicSampleStream(dataset)
        rng = torch.get_rng_state().clone()
        stream.next_group(9)
        state = stream.state_dict()
        expected = stream.next_group(11)
        resumed = DeterministicSampleStream(dataset)
        resumed.load_state_dict(state)
        actual = resumed.next_group(11)
        self.assertEqual(stream.state_dict(), resumed.state_dict())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for a, b in zip(actual, expected):
            self.assertEqual(a["sample_id"], b["sample_id"])
            self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))
            self.assertTrue(torch.equal(a["labels"], b["labels"]))
            self.assertGreaterEqual(int(a["input_ids"].min()), 1)
            self.assertLess(int(a["input_ids"].max()), 2048)
            self.assertEqual(int(a["labels"][0, -1]), 2)
        with self.assertRaisesRegex(ValueError, "identity"):
            DeterministicSampleStream(SyntheticTinyDataset(size=8)).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
