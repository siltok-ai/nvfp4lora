"""逐位比较 tiny 的训练数值和参数，耗时及资源观测单独报告。"""

import argparse
import json
from pathlib import Path
import struct

import torch
from safetensors.torch import load_file

OBSERVATIONS = {'backward_seconds', 'optimizer_seconds', 'step_wall_seconds',
                'streamer_stats', 'transfers', 'memory'}


def equal_bits(actual, expected):
    """比较 JSON 数值的精确表示，以及容器中的键、次序和元素。"""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, float):
        return struct.pack('!d', actual) == struct.pack('!d', expected)
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(equal_bits(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(equal_bits(a, b) for a, b in zip(actual, expected))
    return actual == expected


def compare_runs(baseline, actual):
    """读取 paired 输出，比较全部非观测指标及中间、最终检查点的参数字节。"""
    baseline, actual = Path(baseline), Path(actual)
    histories = [[{key: value for key, value in json.loads(line).items() if key not in OBSERVATIONS}
                  for line in (root / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                 for root in [baseline, actual]]
    differences = []
    if not equal_bits(histories[1], histories[0]):
        differences.append('numerical_metrics')
    counts = {}
    for boundary in ['checkpoint-middle', 'checkpoint-final']:
        expected, found = [load_file(root / boundary / 'lora_params.safetensors') for root in [baseline, actual]]
        counts[boundary] = len(expected)
        if expected.keys() != found.keys():
            differences.append(boundary + '.parameter_names')
        for name in expected.keys() & found.keys():
            a, b = found[name], expected[name]
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)):
                differences.append(boundary + '.' + name)
    return {'status': 'PASS' if not differences else 'FAIL', 'steps': len(histories[0]),
            'numerical_metrics_equal_bits': equal_bits(histories[1], histories[0]),
            'parameter_counts': counts, 'differences': differences,
            'separately_reported_observation_fields': sorted(OBSERVATIONS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--actual', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare_runs(args.baseline, args.actual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    assert result['status'] == 'PASS', result['differences']


if __name__ == '__main__':
    main()
