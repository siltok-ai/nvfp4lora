"""单进程确定性样本流；tiny 使用合法合成词表 ID，独立于真实模板验收。"""

import random

import torch

from .checkpoint import configuration_hash


class SyntheticTinyDataset:
    """每个样本独立的局部 Generator；不消耗训练的任何全局 RNG。"""

    def __init__(self, *, size=37, vocab_size=2048, seed=20261028):
        self.size, self.vocab_size, self.seed = size, vocab_size, seed
        self.identity = {"recipe": "synthetic-causal-v1", "size": size, "vocab_size": vocab_size, "seed": seed}
        self.fingerprint = configuration_hash(self.identity)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        length, prompt = 24 + index % 25, 7 + index % 5
        generator = torch.Generator().manual_seed(self.seed + 104729 * index)
        ids = torch.randint(3, self.vocab_size, (1, length), generator=generator)
        ids[:, 0], ids[:, -1] = 1, 2
        labels = ids.clone()
        labels[:, :prompt] = -100
        return {"input_ids": ids, "labels": labels, "sample_id": index}


class DeterministicSampleStream:
    """保存 epoch 和下一样本位置；恢复不依赖 DataLoader worker 或全局随机数。"""

    def __init__(self, dataset, *, seed=113):
        self.dataset, self.seed = dataset, seed
        self.epoch = self.position = self.consumed = 0

    def _order(self):
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def next_group(self, count):
        if count < 1:
            raise ValueError("Positive accumulation count required")
        batches = []
        for _ in range(count):
            if self.position == len(self.dataset):
                self.epoch += 1
                self.position = 0
            index = self._order()[self.position]
            batches.append(self.dataset[index])
            self.position += 1
            self.consumed += 1
        return batches

    def state_dict(self):
        return {"dataset_sha256": self.dataset.fingerprint, "seed": self.seed, "epoch": self.epoch,
                "position": self.position, "consumed": self.consumed}

    def load_state_dict(self, state):
        if state["dataset_sha256"] != self.dataset.fingerprint or state["seed"] != self.seed:
            raise ValueError("Dataset/sampler identity mismatch")
        if not (0 <= state["position"] <= len(self.dataset) and state["epoch"] >= 0):
            raise ValueError("Invalid data cursor")
        if state["consumed"] != state["epoch"] * len(self.dataset) + state["position"]:
            raise ValueError("Inconsistent data cursor")
        self.epoch, self.position, self.consumed = (state[key] for key in ["epoch", "position", "consumed"])
