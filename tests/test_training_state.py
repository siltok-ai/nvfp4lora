"""真正的 CE/LoRA 梯度和 AdamW 更新续接验证，不依赖 mock。"""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from nvfp4lora.ops.chunked_ce import chunked_ce_sum, supervised_tokens
from nvfp4lora.train.checkpoint import load_checkpoint, save_checkpoint


class ChunkedCETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(911)

    def test_shift_mask_block_boundaries_and_head_lora_gradients(self):
        head = get_peft_model(torch.nn.Sequential(torch.nn.Linear(8, 17, bias=False)),
                              LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["0"]))
        with torch.no_grad():
            head.base_model.model[0].lora_B["default"].weight.normal_(std=0.05)
        params = tuple(p for p in head.parameters() if p.requires_grad)
        hidden = torch.randn(2, 11, 8).requires_grad_()
        labels = torch.randint(0, 17, (2, 11))
        labels[0, :4] = -100
        labels[1, :7] = -100
        labels[1, -1] = -100
        actual = chunked_ce_sum(hidden, labels, head, chunk_size=3)
        expected = F.cross_entropy(head(hidden[:, :-1]).float().reshape(-1, 17), labels[:, 1:].reshape(-1), reduction="sum")
        a = torch.autograd.grad(actual, (hidden, *params))
        b = torch.autograd.grad(expected, (hidden, *params))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
        for ga, gb in zip(a, b):
            torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-6)
        self.assertEqual(supervised_tokens(labels), 10)

    def test_optimizer_step_token_denominator_is_not_micro_mean(self):
        head = torch.nn.Linear(5, 13, bias=False)
        hidden = [torch.randn(1, 7, 5), torch.randn(1, 11, 5)]
        labels = [torch.randint(0, 13, (1, 7)), torch.randint(0, 13, (1, 11))]
        labels[0][:, :3], labels[1][:, :9] = -100, -100
        tokens = sum(supervised_tokens(x) for x in labels)
        sums = [chunked_ce_sum(h, y, head, chunk_size=2) for h, y in zip(hidden, labels)]
        actual = sum(sums) / tokens
        logits = torch.cat([head(h[:, :-1])[y[:, 1:] != -100] for h, y in zip(hidden, labels)])
        targets = torch.cat([y[:, 1:][y[:, 1:] != -100] for y in labels])
        reference = F.cross_entropy(logits, targets)
        ga, = torch.autograd.grad(actual, head.weight, retain_graph=True)
        gb, = torch.autograd.grad(reference, head.weight)
        torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-6)
        wrong = sum(loss / supervised_tokens(y) for loss, y in zip(sums, labels)) / 2
        self.assertGreater(float((actual - wrong).abs()), 1e-3)

    def test_no_supervised_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supervised"):
            chunked_ce_sum(torch.randn(1, 4, 3), torch.full((1, 4), -100), torch.nn.Linear(3, 7))


def make_optimizer(parameters, reverse=False):
    values = list(parameters.values())
    optimizer = torch.optim.AdamW(list(reversed(values)) if reverse else values, lr=0.003, weight_decay=0, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 4))
    return optimizer, scheduler


def update(parameters, optimizer, scheduler):
    a, b = parameters["left"], parameters["right"]
    loss = (a.sin() * torch.tensor([1., 2., 4.]) + b.square() * torch.tensor([3., 1., 2.])).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def draws():
    return {"python": random.random(), "numpy": float(np.random.rand()), "cpu": torch.rand(4),
            "cuda": torch.rand(4, device="cuda").cpu() if torch.cuda.is_available() else None}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(515)
        random.seed(516)
        np.random.seed(517)
        self.parameters = {"left": torch.nn.Parameter(torch.tensor([.2, -.3, .6])),
                           "right": torch.nn.Parameter(torch.tensor([.8, .1, -.4]))}
        self.optimizer, self.scheduler = make_optimizer(self.parameters)
        self.config = {"policy": "named-AdamW-unit", "max_steps": 10}

    def test_parameter_reordering_future_update_and_all_rng_streams(self):
        for _ in range(2):
            update(self.parameters, self.optimizer, self.scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            save_checkpoint(path, self.parameters, self.optimizer, self.scheduler,
                            {"epoch": 1, "position": 3}, {"global_step": 2}, self.config)
            expected_rng = draws()
            update(self.parameters, self.optimizer, self.scheduler)
            expected = {name: parameter.detach().clone() for name, parameter in self.parameters.items()}
            reconstructed = {name: torch.nn.Parameter(torch.randn(3)) for name in self.parameters}
            optimizer, scheduler = make_optimizer(reconstructed, reverse=True)
            random.random(), np.random.rand(), torch.rand(10)
            data, trainer = load_checkpoint(path, reconstructed, optimizer, scheduler, self.config)
            actual_rng = draws()
            self.assertEqual(actual_rng["python"], expected_rng["python"])
            self.assertEqual(actual_rng["numpy"], expected_rng["numpy"])
            for key in ["cpu", "cuda"]:
                if expected_rng[key] is not None:
                    self.assertTrue(torch.equal(actual_rng[key], expected_rng[key]), key)
            update(reconstructed, optimizer, scheduler)
            for name, value in reconstructed.items():
                self.assertTrue(torch.equal(value, expected[name]), name)
                for key in ["step", "exp_avg", "exp_avg_sq"]:
                    self.assertTrue(torch.equal(optimizer.state[value][key], self.optimizer.state[self.parameters[name]][key]))
            self.assertEqual(optimizer.param_groups[0]["lr"], self.optimizer.param_groups[0]["lr"])
            self.assertEqual(data, {"epoch": 1, "position": 3})
            self.assertEqual(trainer["global_step"], 2)

    def test_mid_accumulation_or_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.parameters["left"].sum().backward()
            with self.assertRaisesRegex(ValueError, "boundary"):
                save_checkpoint(root / "bad", self.parameters, self.optimizer, self.scheduler, {}, {"global_step": 0}, self.config)
            self.assertFalse((root / "bad").exists())
            self.optimizer.zero_grad(set_to_none=True)
            save_checkpoint(root / "good", self.parameters, self.optimizer, self.scheduler, {}, {"global_step": 0}, self.config)
            with (root / "good/optimizer.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(root / "good", self.parameters, self.optimizer, self.scheduler, self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
