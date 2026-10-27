"""不注册为 Parameter/buffer 的冻结 CPU 专家 bank。"""

import hashlib
import json
from pathlib import Path

import torch

from nvfp4lora.format.inspect import read_header


class ExpertStore:
    """从已校验的预处理文件直接读入 CPU uint8 bank，保留全部缩放字节。"""

    def __init__(self, directory, *, pin_memory=True):
        self.directory = Path(directory)
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if index["layout_version"] != "packed-nvfp4-experts-v1":
            raise ValueError("Unsupported expert store layout")
        self.geometry, self.source = index["geometry"], index["source"]
        self.entries, self.banks = {}, {}
        self.pin_memory = pin_memory
        for entry in index["layers"]:
            layer = entry["layer"]
            if layer in self.entries:
                raise ValueError(f"Duplicate expert layer: {layer}")
            path = self.directory / entry["file"]
            header = read_header(path)
            for field in ["header_bytes", "payload_bytes", "tensors"]:
                if header[field] != entry[field]:
                    raise ValueError(f"Expert index/header mismatch: layer={layer}, field={field}")
            data = torch.empty(entry["payload_bytes"], dtype=torch.uint8, pin_memory=pin_memory)
            view = memoryview(data.numpy())
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                digest.update(stream.read(entry["header_bytes"]))
                if stream.readinto(view) != entry["payload_bytes"]:
                    raise ValueError(f"Incomplete expert payload: {path}")
            digest.update(view)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Expert file checksum mismatch: {path}")
            self.entries[layer], self.banks[layer] = entry, data
        if set(self.entries) != set(range(self.geometry["layers"])):
            raise ValueError("Expert store does not cover every layer")
        self.nbytes = sum(x.numel() for x in self.banks.values())

    def hashes(self):
        """对实际训练使用的 CPU payload 做哈希，供训练前后冻结不变量检查。"""
        return {layer: hashlib.sha256(memoryview(bank.numpy())).hexdigest() for layer, bank in self.banks.items()}

    def tensors(self, layer):
        """从 CPU bank 取得零拷贝视图；调用者必须保持冻结。"""
        return tensor_views(self.banks[layer], self.entries[layer]["tensors"])


def tensor_views(buffer, specs):
    """按 safetensors payload 相对偏移解释 U8/F32，不改变原始字节。"""
    dtypes = {"U8": torch.uint8, "F32": torch.float32}
    views = {}
    for name, spec in specs.items():
        start, stop = spec["data_offsets"]
        views[name] = buffer[start:stop].view(dtypes[spec["dtype"]]).reshape(spec["shape"])
    return views
