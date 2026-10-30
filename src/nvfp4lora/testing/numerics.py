"""验证器共享的数值比较、固定随机设置和状态观测。"""
import json
import struct
import random
from pathlib import Path
import numpy as np
import torch

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def rng_probe():
    """只用于续训验收：每步实际消耗四类 RNG，避免只比较未使用的随机状态。"""
    return {"python": random.random(), "numpy": float(np.random.random()),
            "torch_cpu": float(torch.rand(())), "torch_cuda": float(torch.rand((), device="cuda"))}


def compare_vectors(actual, expected):
    """FP64 计算比较指标；零向量单独检查，不使用 eps 掩盖差异。"""
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    an, bn = float(a.norm()), float(b.norm())
    if bn == 0 or an == 0:
        return {"relative_l2": 0.0 if an == bn else float("inf"), "cosine": 1.0 if an == bn else 0.0}
    return {"relative_l2": float((a - b).norm()) / bn, "cosine": float(torch.dot(a, b)) / (an * bn)}


def configure_numerics():
    """固定精度选项，防止不同运行共享不可见的舍入策略。"""
    torch.manual_seed(20261027)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False


def compare(actual, expected):
    """每个张量单独统计，零参考不使用无定义的相对误差。"""
    a, b = actual.double(), expected.double()
    norm = float(b.norm())
    absolute = float((a - b).abs().max())
    relative = float((a - b).norm() / b.norm()) if norm else (0.0 if absolute == 0 else float("inf"))
    bitwise_equal = (actual.dtype == expected.dtype and actual.shape == expected.shape and
                     torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                 expected.contiguous().reshape(-1).view(torch.uint8)))
    return {"relative_l2": relative, "max_absolute": absolute,
            "reference_norm": norm, "bitwise_equal": bitwise_equal}


def equal_tree(actual, expected):
    """按类型及原字节比较状态树，包含浮点正负零、数组 dtype 与张量形状。"""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, torch.Tensor):
        return actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(
            actual.detach().contiguous().reshape(-1).view(torch.uint8), expected.detach().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(actual, np.ndarray):
        return actual.shape == expected.shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    if isinstance(actual, float):
        return struct.pack("!d", actual) == struct.pack("!d", expected)
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(equal_tree(value, expected[key]) for key, value in actual.items())
    if isinstance(actual, (tuple, list)):
        return len(actual) == len(expected) and all(equal_tree(a, b) for a, b in zip(actual, expected))
    return actual == expected
