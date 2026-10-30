"""共享 bundle 字节读取、哈希和非流式矩阵解码。"""

import hashlib
from contextlib import ExitStack
from pathlib import Path
import torch
from safetensors import safe_open
from . import nvfp4

FIELDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def file_hash(path):
    """分块哈希校验输入和输出，不创建整文件字节副本。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_group(directory, weight_map, names):
    """一组只打开各分片一次；返回独立拷贝，关闭本组 mmap 避免累计驻留。"""
    with ExitStack() as stack:
        readers = {filename: stack.enter_context(safe_open(Path(directory) / filename, framework="pt", device="cpu"))
                   for filename in sorted({weight_map[name] for name in names})}
        return {name: readers[weight_map[name]].get_tensor(name).clone() for name in names}


def _decode_nvfp4_rows(raw, prefix, rows_per_chunk=2048):
    """限制 lm_head 解码临时空间；只为最终 BF16 权重分配完整矩阵。"""
    weight, scale = raw[prefix + ".weight"], raw[prefix + ".weight_scale"]
    global_scale = raw[prefix + ".weight_scale_2"]
    decoded = torch.empty(weight.shape[0], weight.shape[1] * 2, dtype=torch.bfloat16)
    for start in range(0, weight.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, weight.shape[0])
        decoded[start:stop].copy_(nvfp4.dequantize(weight[start:stop], scale[start:stop], global_scale,
                                                dtype=torch.bfloat16))
    return decoded
