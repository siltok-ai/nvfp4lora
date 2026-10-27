"""只读 safetensors 文件头，核对 dtype、几何、偏移及索引对应关系。"""

import json
import math
import struct
from pathlib import Path

DTYPE_BYTES = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def read_header(path):
    """文件头不加载权重；同时拒绝越界、重叠和未覆盖的 payload。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        if length > size - 8:
            raise ValueError(f"Header exceeds file: {path}")
        raw = json.loads(stream.read(length))
    tensors = {name: value for name, value in raw.items() if name != "__metadata__"}
    end = 0
    for name, value in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
        begin, stop = value["data_offsets"]
        expected = math.prod(value["shape"]) * DTYPE_BYTES[value["dtype"]]
        if begin != end or stop - begin != expected:
            raise ValueError(f"Invalid payload coverage at {name} in {path}")
        end = stop
    if end != size - 8 - length:
        raise ValueError(f"Payload size mismatch in {path}")
    return {"file": str(path), "header_bytes": length + 8, "payload_bytes": end,
            "tensors": tensors, "metadata": raw.get("__metadata__", {})}


def inspect_checkpoint(directory):
    """把固定源索引与分片实际头部逐键核对，不能用索引代替真实文件。"""
    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    actual, shards = {}, []
    for filename in sorted(set(index["weight_map"].values())):
        header = read_header(directory / filename)
        for name, value in header["tensors"].items():
            if name in actual or index["weight_map"].get(name) != filename:
                raise ValueError(f"Index/filename conflict for {name}")
            actual[name] = {**value, "file": filename}
        shards.append({k: v for k, v in header.items() if k != "tensors"})
    if actual.keys() != index["weight_map"].keys():
        raise ValueError("Index tensor names do not equal shard header names")
    payload = sum(x["payload_bytes"] for x in shards)
    if payload != index["metadata"]["total_size"]:
        raise ValueError("Index total_size differs from actual shard payload")
    return {"tensor_count": len(actual), "payload_bytes": payload, "shards": shards, "tensors": actual}
