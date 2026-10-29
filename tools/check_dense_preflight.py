"""D0 dense architecture, native Torch GDN/SDPA, GEMM/decode and pinned-memory gates."""

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace

import psutil
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from check_gdn import compare, make_inputs, recurrent_reference
from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.train.loop import configure_training, memory_observation


def observe():
    """记录Torch峰值、整卡余量及主存，保持不同统计口径。"""
    torch.cuda.synchronize()
    return {**memory_observation(), 'system_available': psutil.virtual_memory().available,
            'host': torch.cuda.memory.host_memory_stats(),
            'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                                                  '--format=csv'], text=True).strip()}


def meta_gate(config_path, emit):
    """在meta核对真实架构，并以无Parameter线性层/CPU查表验证HF接口。"""
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    config = Qwen3_5TextConfig.from_dict(raw.get('text_config', raw))
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    assert all(p.is_meta for p in model.parameters())
    assert len(model.model.layers) == 64
    shapes = {}
    for layer_id, layer in enumerate(model.model.layers):
        names = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        if layer_id % 4 == 3:
            names += ['self_attn.' + p + '_proj' for p in ['q', 'k', 'v', 'o']]
        else:
            names += ['linear_attn.' + p for p in ['in_proj_qkv', 'in_proj_z', 'out_proj']]
        for name in names:
            shapes[f'{layer_id}.{name}'] = list(layer.get_submodule(name).weight.shape)
    assert shapes['0.mlp.gate_proj'] == [17408, 5120]
    assert shapes['0.mlp.down_proj'] == [5120, 17408]
    assert shapes['0.linear_attn.in_proj_qkv'] == [10240, 5120]
    assert shapes['0.linear_attn.in_proj_z'] == [6144, 5120]
    assert shapes['3.self_attn.q_proj'] == [12288, 5120]
    assert shapes['3.self_attn.k_proj'] == [1024, 5120]
    assert shapes['3.self_attn.o_proj'] == [5120, 6144]
    report = {'gate': 'DA0.5', 'meta_parameter_count': sum(p.numel() for p in model.parameters()),
              'layers': 64, 'linear_shapes': shapes}
    del model
    tiny_config = Qwen3_5TextConfig(vocab_size=2048, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, layer_types=['linear_attention'] * 3 + ['full_attention'],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        tie_word_embeddings=False, use_cache=False,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .25, 'mrope_section': [2, 3, 3]})
    tiny_config._attn_implementation = 'sdpa'
    tiny = Qwen3_5ForCausalLM(tiny_config).to(device='cuda', dtype=torch.bfloat16)

    class InterfaceLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.values = weight.detach()
        def forward(self, x):
            return F.linear(x, self.values)

    class InterfaceEmbedding(torch.nn.Module):
        def __init__(self, table):
            super().__init__()
            self.table = table.detach().cpu()
        def forward(self, ids):
            return F.embedding(ids.cpu(), self.table).to(ids.device)

    for layer in tiny.model.layers:
        for name, module in list(layer.named_modules()):
            if isinstance(module, torch.nn.Linear) and not name.endswith(('in_proj_a', 'in_proj_b')):
                parent, attr = name.rsplit('.', 1)
                setattr(layer.get_submodule(parent), attr, InterfaceLinear(module.weight))
    tiny.model.embed_tokens = InterfaceEmbedding(tiny.model.embed_tokens.weight)
    assert not list(tiny.model.embed_tokens.parameters()) and not list(tiny.model.embed_tokens.buffers())
    with torch.no_grad():
        output = tiny(input_ids=torch.randint(0, 2048, (1, 17), device='cuda')).logits
    assert torch.isfinite(output).all() and list(output.shape) == [1, 17, 2048]
    report.update(tiny_interface_forward='PASS', tiny_scope='HF replacement interface only; not PackedLinear qualification',
                  memory=observe())
    emit(report)


def evaluate_gdn(function, inputs, dy, mode):
    """统一用backward核对全部输入梯度，兼容重入检查点。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    values = [x.detach().requires_grad_() for x in inputs]
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function(*values) if mode == 'direct' else checkpoint(function, *values, use_reentrant=mode == 'reentrant')
    output.backward(dy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for gradient in [x.grad for x in values]:
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
    return ([output.detach().cpu()] + [x.grad.detach().cpu() for x in values],
            {'mode': mode, 'seconds': elapsed, 'memory': observe()})


def gdn_gate(config_path, emit):
    """先独立递推短序列E3，再测2048/4096真实头数及两种重算E2。"""
    def candidate(q, k, v, g, beta):
        q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
        k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
        return modeling.torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64,
                     use_qk_l2norm_in_kernel=True, output_final_state=False)[0]
    for length, dtype in [(17, torch.float32), (129, torch.bfloat16)]:
        inputs = make_inputs(length, 16, 48, 128, dtype)
        dy = torch.randn_like(inputs[2]) / inputs[2].numel() ** .5
        actual, metrics = evaluate_gdn(candidate, inputs, dy, 'direct')
        expected, reference_metrics = evaluate_gdn(lambda *xs: recurrent_reference(*xs).to(dtype), inputs, dy, 'direct')
        comparisons = {name: compare(a, b) for name, a, b in zip(['output', 'q', 'k', 'v', 'g', 'beta'], actual, expected)}
        emit({'gate': 'DA0.6', 'scope': 'independent short recurrence', 'T': length, 'dtype': str(dtype),
              'comparisons': comparisons, 'direct': metrics, 'reference': reference_metrics})
        assert all(row['relative_l2'] <= .01 for row in comparisons.values()), comparisons
        del inputs, dy, actual, expected
    for length in [2048, 4096]:
        inputs = make_inputs(length, 16, 48, 128, torch.bfloat16)
        dy = torch.randn_like(inputs[2]) / inputs[2].numel() ** .5
        actual, metrics = evaluate_gdn(candidate, inputs, dy, 'direct')
        emit({'gate': 'DA0.6', 'T': length, 'heads': [16, 48], 'head_dim': 128, **metrics})
        for mode in ['non-reentrant', 'reentrant']:
            expected, replay_metrics = evaluate_gdn(candidate, inputs, dy, mode)
            comparisons = {name: compare(a, b) for name, a, b in zip(['output', 'q', 'k', 'v', 'g', 'beta'], expected, actual)}
            emit({'gate': 'DA0.6', 'T': length, **replay_metrics, 'comparisons': comparisons})
            assert all(row['bitwise_equal'] for row in comparisons.values()), comparisons
            del expected
        del inputs, dy, actual


def gdn_module_gate(config_path, emit):
    """补测实际bf16 beta及完整HF GDN层，区分kernel与层级峰值。"""
    inputs = make_inputs(129, 16, 48, 128, torch.bfloat16)
    inputs[4] = inputs[4].to(torch.bfloat16)
    dy = torch.randn_like(inputs[2]) / inputs[2].numel() ** .5
    def candidate(q, k, v, g, beta):
        return modeling.torch_chunk_gated_delta_rule(q.repeat_interleave(3, dim=2), k.repeat_interleave(3, dim=2),
                    v, g, beta, chunk_size=64, use_qk_l2norm_in_kernel=True, output_final_state=False)[0]
    actual, metrics = evaluate_gdn(candidate, inputs, dy, 'direct')
    expected, reference_metrics = evaluate_gdn(lambda *xs: recurrent_reference(*xs).to(torch.bfloat16), inputs, dy, 'direct')
    comparisons = {name: compare(a, b) for name, a, b in zip(['output', 'q', 'k', 'v', 'g', 'beta'], actual, expected)}
    emit({'gate': 'DA0.6', 'scope': 'independent recurrence with production bf16 beta', 'T': 129,
          'comparisons': comparisons, 'direct': metrics, 'reference': reference_metrics})
    assert all(row['relative_l2'] <= .01 for row in comparisons.values()), comparisons
    del inputs, dy, actual, expected
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    config = Qwen3_5TextConfig.from_dict(raw.get('text_config', raw))
    module = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device='cuda', dtype=torch.bfloat16)
    module.requires_grad_(False)
    for length in [2048, 4096]:
        inputs = [torch.randn(1, length, config.hidden_size, device='cuda', dtype=torch.bfloat16)]
        dy = torch.randn_like(inputs[0]) / inputs[0].numel() ** .5
        actual, metrics = evaluate_gdn(module, inputs, dy, 'direct')
        emit({'gate': 'DA0.6', 'scope': 'full HF GDN module; frozen synthetic bf16 projections; no LoRA', 'T': length, **metrics})
        for mode in ['non-reentrant', 'reentrant']:
            expected, replay_metrics = evaluate_gdn(module, inputs, dy, mode)
            comparisons = {name: compare(a, b) for name, a, b in zip(['output', 'x'], expected, actual)}
            emit({'gate': 'DA0.6', 'scope': 'full HF GDN module', 'T': length, **replay_metrics, 'comparisons': comparisons})
            assert all(row['bitwise_equal'] for row in comparisons.values()), comparisons
            del expected
        del inputs, dy, actual


def sdpa_gate(config_path, emit, explicit_repeat=False):
    """以HF实际GQA分派测试有无padding mask，记录真实backend算子。"""
    module = SimpleNamespace(num_key_value_groups=6, is_causal=True)
    proxy = SimpleNamespace(num_key_value_groups=1, is_causal=True)
    failures = []
    for padding in [False, True]:
        gc.collect()
        torch.cuda.empty_cache()
        q = torch.randn(1, 4096, 24, 256, device='cuda', dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        k = torch.randn(1, 4096, 4, 256, device='cuda', dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        v = torch.randn_like(k).requires_grad_()
        mask = None
        if padding:
            mask = torch.ones(4096, 4096, dtype=torch.bool, device='cuda').tril()[None, None]
            mask[..., :17] = False
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        def candidate():
            if explicit_repeat:
                return sdpa_attention_forward(proxy, q, repeat_kv(k, 6), repeat_kv(v, 6), mask, scaling=256 ** -.5)[0]
            return sdpa_attention_forward(module, q, k, v, mask, scaling=256 ** -.5)[0]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
            y = candidate()
            dy = torch.randn_like(y) / y.numel() ** .5
            y.backward(dy)
        assert torch.isfinite(y).all() and all(torch.isfinite(x.grad).all() for x in [q, k, v])
        names = sorted({event.key for event in trace.key_averages() if 'scaled_dot_product' in event.key or 'attention' in event.key})
        row = {'gate': 'DA0.7', 'padding_mask': padding, 'explicit_repeat_kv': explicit_repeat, 'shape': [1, 24, 4, 4096, 256],
               'operators': names, 'seconds': time.perf_counter() - began, 'memory': observe()}
        emit(row)
        if any('attention_math' in name for name in names):
            failures.append(row)
        if explicit_repeat:
            actual = [y.detach().cpu()] + [x.grad.detach().cpu() for x in [q, k, v]]
            for x in [q, k, v]:
                x.grad = None
            replay = candidate()
            replay.backward(dy)
            repeated = [replay.detach().cpu()] + [x.grad.detach().cpu() for x in [q, k, v]]
            same_path = {name: compare(a, b) for name, a, b in zip(['output', 'q', 'k', 'v'], actual, repeated)}
            assert all(value['bitwise_equal'] for value in same_path.values()), same_path
            del replay, repeated
            for x in [q, k, v]:
                x.grad = None
            with sdpa_kernel(SDPBackend.MATH):
                reference = sdpa_attention_forward(module, q, k, v, mask, scaling=256 ** -.5)[0]
                reference.backward(dy)
            expected = [reference.detach().cpu()] + [x.grad.detach().cpu() for x in [q, k, v]]
            cross_backend = {name: compare(a, b) for name, a, b in zip(['output', 'q', 'k', 'v'], actual, expected)}
            emit({'gate': 'DA0.7', 'padding_mask': padding, 'scope': 'explicit-repeat fallback qualification',
                  'same_path_E2': same_path, 'math_reference_E3': cross_backend})
            assert all(value['relative_l2'] <= .01 for value in cross_backend.values()), cross_backend
            del actual, expected, reference
        del q, k, v, y, mask, dy
    assert not failures, failures


def cuda_time(function, repeats=5):
    """预热后用CUDA事件测同一运算的GPU时间，wall另行记录。"""
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before.record()
    for _ in range(repeats):
        function()
    after.record()
    after.synchronize()
    return {'gpu_ms': before.elapsed_time(after) / repeats, 'wall_ms': (time.perf_counter() - started) * 1000 / repeats}


def gemm_gate(config_path, emit):
    """在真实形状测前向和dx；用有效合成原字节单独测分块解码。"""
    for length in [2048, 4096]:
        for in_dim, out_dim in [(5120, 17408), (17408, 5120), (5120, 10240)]:
            x = torch.randn(length, in_dim, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(out_dim, in_dim, device='cuda', dtype=torch.bfloat16)
            dy = torch.randn(length, out_dim, device='cuda', dtype=torch.bfloat16)
            for direction, function in [('forward', lambda: F.linear(x, w)), ('dx', lambda: dy.matmul(w))]:
                metrics = cuda_time(function)
                emit({'gate': 'DA0.8', 'kind': 'gemm', 'T': length, 'in': in_dim, 'out': out_dim,
                      'direction': direction, **metrics, 'TFLOPS': 2 * length * in_dim * out_dim / (metrics['gpu_ms'] * 1e9)})
            del x, w, dy
    for kind, out_dim, in_dim in [('nvfp4', 17408, 5120), ('fp8', 10240, 5120)]:
        packed = torch.full((out_dim, in_dim // 2 if kind == 'nvfp4' else in_dim), 0x23 if kind == 'nvfp4' else 0x38,
                            dtype=torch.uint8, device='cuda')
        scale = torch.ones(out_dim, in_dim // 16, device='cuda').to(torch.float8_e4m3fn) if kind == 'nvfp4' else None
        decoded = torch.empty(out_dim, in_dim, device='cuda', dtype=torch.bfloat16)
        def decode():
            if kind == 'fp8':
                decoded.copy_(packed.view(torch.float8_e4m3fn))
                return
            elements = 8 * 1024 * 1024
            for start in range(0, decoded.numel(), elements):
                stop = min(start + elements, decoded.numel())
                part = dequantize(packed.flatten()[start // 2:stop // 2].reshape(1, -1),
                                  scale.flatten()[start // 16:stop // 16].reshape(1, -1), dtype=torch.bfloat16)
                decoded.flatten()[start:stop].copy_(part.flatten())
        torch.cuda.reset_peak_memory_stats()
        metrics = cuda_time(decode)
        emit({'gate': 'DA0.8', 'kind': 'decode-' + kind, 'shape': [out_dim, in_dim],
              'source': 'synthetic valid packed bytes; performance qualification only', 'chunk_elements': 8 * 1024 * 1024,
              **metrics, 'G_elements_per_second': decoded.numel() / (metrics['gpu_ms'] * 1e6), 'memory': observe()})
        del packed, scale, decoded


def pinned_gate(config_path, emit):
    """按真实payload分配64块，记录allocator取整并测全遍异步H2D。"""
    before = torch.cuda.memory.host_memory_stats()
    sizes = [255262776 if i % 4 == 3 else 265748528 for i in range(64)]
    started = time.perf_counter()
    banks = [torch.empty(size, dtype=torch.uint8, pin_memory=True).fill_(i) for i, size in enumerate(sizes)]
    after = torch.cuda.memory.host_memory_stats()
    physical = after['allocated_bytes.current'] - before['allocated_bytes.current']
    row = {'gate': 'DA0.9', 'logical_bytes': sum(sizes), 'matrix_only_bytes_in_design': 16840130560,
           'scalar_bytes': 3200, 'allocation_seconds': time.perf_counter() - started,
           'physical_pinned_bytes': physical, 'per_layer_requested': sizes, 'host_before': before, 'host_after': after}
    emit(row)
    assert physical < 32 * 1024 ** 3, 'STOP: pinned allocation rounded to 512 MiB per layer'
    buffers = [torch.empty(max(sizes), dtype=torch.uint8, device='cuda') for _ in range(2)]
    def copy_pass():
        for layer, bank in enumerate(banks):
            buffers[layer % 2][:bank.numel()].copy_(bank, non_blocking=True)
    metrics = cuda_time(copy_pass, repeats=3)
    for layer in [62, 63]:
        assert torch.equal(buffers[layer % 2][:sizes[layer]].cpu(), banks[layer])
    emit({'gate': 'DA0.9', 'scope': 'sequential full-bank pinned H2D; not overlap proof', **metrics,
          'GB_per_second': sum(sizes) / (metrics['gpu_ms'] * 1e6), 'memory': observe()})


def main():
    """一个独立运行只测一个关卡，失败结果和真实退出码保留。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', choices=['meta', 'gdn', 'gdn_module', 'sdpa', 'gemm', 'pinned'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--explicit-repeat-kv', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    torch.cuda.init()
    source = Path(inspect.getfile(modeling))
    result = {'status': 'RUNNING', 'gate': args.gate, 'cases': [], 'started_at': time.time(),
              'modeling_source': str(source), 'modeling_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'precision': 'deterministic; TF32 off; BF16 reduced precision reduction off'}
    def emit(row):
        result['cases'].append(row)
        (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    try:
        if args.gate == 'sdpa':
            sdpa_gate(args.config, emit, explicit_repeat=args.explicit_repeat_kv)
        else:
            globals()[args.gate + '_gate'](args.config, emit)
        result['status'] = 'PASS'
    except Exception:
        result.update(status='FAIL', traceback=traceback.format_exc())
        raise
    finally:
        result['ended_at'] = time.time()
        (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
