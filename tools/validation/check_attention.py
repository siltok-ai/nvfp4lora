"""对真实模型的注意力几何比较高效实现、数学实现及确定性。"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nvfp4lora.families import family_of_bundle
from nvfp4lora.model.attention import sdpa_forward
from nvfp4lora.train.loop import configure_training


def same_bytes(actual, expected):
    return actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8))


def relative_l2(actual, expected):
    actual, expected = actual.double(), expected.double()
    norm = float(expected.norm())
    return float((actual - expected).norm()) / norm if norm else (0. if actual.count_nonzero() == 0 else float('inf'))


def evaluate(mode, inputs, mask, dy, *, groups, profile=False):
    """同一输入求输出和三个输入梯度，计时区间不包含结果拷回 CPU。"""
    gc.collect()
    torch.cuda.empty_cache()
    values = [value.detach().clone(memory_format=torch.preserve_format).requires_grad_() for value in inputs]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    observer = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    backend = sdpa_kernel(SDPBackend.MATH) if mode == 'math' else nullcontext()
    with observer, backend:
        if mode == 'math':
            output = F.scaled_dot_product_attention(*values, attn_mask=mask, dropout_p=0.,
                is_causal=mask is None, scale=values[0].shape[-1] ** -.5, enable_gqa=True).transpose(1, 2)
        else:
            proxy = SimpleNamespace(num_key_value_groups=groups, is_causal=True)
            output, _ = sdpa_forward(proxy, *values, mask, dropout=0., scaling=values[0].shape[-1] ** -.5)
        gradients = torch.autograd.grad(output, values, dy)
    torch.cuda.synchronize()
    observation = {'wall_seconds': time.perf_counter() - began,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    if profile:
        observation['actual_SDPA_operators'] = {row.key: row.count for row in observer.key_averages()
                                                if '_scaled_dot_product' in row.key}
    result = [output.detach().cpu(), *[value.detach().cpu() for value in gradients]]
    assert all(torch.isfinite(value).all() for value in result)
    return result, observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261030)
    report = {'status': 'RUNNING', 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'tf32': torch.backends.cuda.matmul.allow_tf32, 'cases': [],
              'scope': 'Real attention geometry only; full-model memory and step throughput are separate measurements',
              'timing': 'One full warmup per backend and case, followed by measurements; profiler adds overhead to the profiled repetition'}
    seen = set()
    path = args.output / 'results.json'
    def save(): path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        for bundle in args.bundle:
            family = family_of_bundle(bundle)
            if family.name in seen:
                raise ValueError('Duplicate family: ' + family.name)
            seen.add(family.name)
            config = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
            heads, kv_heads, width = (config[key] for key in ['num_attention_heads', 'num_key_value_heads', 'head_dim'])
            for length in [2048, 4096]:
                for padding in [False, True]:
                    generator = torch.Generator(device='cuda').manual_seed(412 + length + int(padding))
                    # B,T,H,D storage, transposed to B,H,T,D, also exercises noncontiguous head/sequence strides.
                    inputs = [torch.randn(1, length, count, width, dtype=torch.bfloat16, device='cuda', generator=generator).transpose(1, 2)
                              for count in [heads, kv_heads, kv_heads]]
                    dy = torch.randn(1, length, heads, width, dtype=torch.bfloat16, device='cuda', generator=generator)
                    mask = None
                    if padding:
                        mask = torch.ones(length, length, dtype=torch.bool, device='cuda').tril()
                        mask[:, :7] = False
                        mask = mask[None, None]
                    groups = heads // kv_heads
                    for mode in ['math', 'efficient']:
                        evaluate(mode, inputs, mask, dy, groups=groups)
                    math_result, math_observation = evaluate('math', inputs, mask, dy, groups=groups, profile=True)
                    efficient, efficient_observation = evaluate('efficient', inputs, mask, dy, groups=groups)
                    repeated, repeated_observation = evaluate('efficient', inputs, mask, dy, groups=groups, profile=True)
                    errors = {name: relative_l2(actual, expected) for name, actual, expected in
                              zip(['output', 'dq', 'dk', 'dv'], efficient, math_result)}
                    deterministic = {name: same_bytes(actual, expected) for name, actual, expected in
                                     zip(['output', 'dq', 'dk', 'dv'], efficient, repeated)}
                    operators = repeated_observation['actual_SDPA_operators']
                    backend_correct = (operators.get('aten::_scaled_dot_product_efficient_attention') == 1
                                       and operators.get('aten::_scaled_dot_product_efficient_attention_backward') == 1
                                       and not any('attention_math' in key for key in operators))
                    row = {'family': family.name, 'length': length, 'padding_mask': padding,
                           'query_heads': heads, 'kv_heads': kv_heads, 'head_dim': width,
                           'input_strides': [list(value.stride()) for value in inputs],
                           'relative_l2_vs_math': errors, 'repeated_E2': deterministic,
                           'math': math_observation, 'efficient': efficient_observation,
                           'efficient_profiled_repeat': repeated_observation,
                           'status': 'PASS' if max(errors.values()) < .01 and all(deterministic.values()) and backend_correct else 'FAIL'}
                    report['cases'].append(row)
                    save()
                    print(f"{family.short_name} {length} mask={padding}: {row['status']}, max error={max(errors.values()):.6g}", flush=True)
                    if row['status'] != 'PASS':
                        raise AssertionError(row)
                    del inputs, dy, mask, math_result, efficient, repeated
        report['status'] = 'PASS'
        save()
    except Exception:
        report.update(status='FAIL', traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
