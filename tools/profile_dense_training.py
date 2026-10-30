"""Dense restored-state accumulation timing; no optimizer update, with gradient E2 control."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoTokenizer

from check_real_forward import CudaTimings
from nvfp4lora.model.build_dense import build_dense_model
import nvfp4lora.ops.packed_linear_fn as packed_ops
from nvfp4lora.train.checkpoint import load_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream, SyntheticTinyDataset
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, memory_observation
from nvfp4lora.train.session import logical_config


class MatrixTimings(TorchDispatchMode):
    def __init__(self, timing):
        super().__init__()
        self.timing = timing

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(func)
        if name in ['aten.mm.default', 'aten.addmm.default', 'aten.bmm.default']:
            with self.timing.span(self.timing.scope + '_gemm_' + name):
                return func(*args, **kwargs)
        if '_scaled_dot_product' in name or '_efficient_attention' in name:
            with self.timing.span(self.timing.scope + '_sdpa_' + name):
                return func(*args, **kwargs)
        return func(*args, **kwargs)


class TrainingTimings(CudaTimings):
    def __init__(self, built, *, event_factory=None):
        super().__init__()
        self.built, self.scope = built, 'other'
        self.make_event = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self.backward_pending, self.tensor_hooks = {}, []
        self.modules = 0

    @contextmanager
    def span(self, name):
        start, end = self.make_event(), self.make_event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.records.append((name, start, end))

    def finish_micro(self):
        if self.pending or self.backward_pending:
            raise RuntimeError('Incomplete dense module timing interval')
        for hook in self.tensor_hooks:
            hook.remove()
        self.tensor_hooks.clear()

    @contextmanager
    def instrument(self):
        original = packed_ops.decode_codes
        hooks = []
        def decode(*args, **kwargs):
            with self.span(self.scope + '_decode'):
                return original(*args, **kwargs)
        packed_ops.decode_codes = decode
        for block in self.built.model.model.layers:
            for name, module in [('gdn', getattr(block, 'linear_attn', None)),
                                 ('attention', getattr(block, 'self_attn', None)), ('mlp', block.mlp)]:
                if module is None:
                    continue
                self.modules += 1
                def begin(current, inputs, key=name):
                    phase = 'forward' if self.built.streamer.phase == 'FORWARD' else 'recompute'
                    start = self.make_event()
                    start.record()
                    self.pending[id(current)] = (self.scope, key + '_' + phase, start)
                    self.scope = key + '_' + phase
                def end(current, inputs, output, key=name):
                    previous, label, start = self.pending.pop(id(current))
                    finish = self.make_event()
                    finish.record()
                    self.records.append((label + '_inclusive', start, finish))
                    self.scope = previous
                    if label.endswith('_forward') and isinstance(output, torch.Tensor) and output.requires_grad:
                        identity = id(current)
                        def begin_backward(grad):
                            start = self.make_event()
                            start.record()
                            self.backward_pending[identity] = (self.scope, start)
                            self.scope = key + '_backward'
                        def end_backward(gradients):
                            previous, start = self.backward_pending.pop(identity)
                            finish = self.make_event()
                            finish.record()
                            self.records.append((key + '_backward_inclusive', start, finish))
                            self.scope = previous
                        tensors = [value for value in inputs if isinstance(value, torch.Tensor) and value.requires_grad]
                        tensors += [parameter for parameter in current.parameters() if parameter.requires_grad]
                        self.tensor_hooks.append(output.register_hook(begin_backward))
                        self.tensor_hooks.append(torch.autograd.graph.register_multi_grad_hook(tuple(tensors), end_backward))
                hooks.extend([module.register_forward_pre_hook(begin), module.register_forward_hook(end, always_call=True)])
        try:
            with MatrixTimings(self):
                yield
        finally:
            packed_ops.decode_codes = original
            for hook in hooks:
                hook.remove()
            for hook in self.tensor_hooks:
                hook.remove()
            self.tensor_hooks.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--training-results', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tiny-synthetic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.training_results / 'train_config.json').read_text(encoding='utf-8'))
    configure_training(settings['seed'])
    if args.tiny_synthetic:
        dataset, config = SyntheticTinyDataset(), settings
        checkpoint = args.checkpoint or args.training_results / 'checkpoint-final'
    else:
        execution = json.loads((args.training_results / 'execution.json').read_text(encoding='utf-8'))
        tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
        dataset = ChatJSONLDataset(execution['data'], tokenizer, max_length=settings['max_length'])
        config = logical_config(settings, args.bundle, dataset)
        if config != settings:
            raise ValueError('Dense profile logical configuration differs')
        checkpoint = args.checkpoint or args.training_results / 'checkpoints/step-000100'
    built = build_dense_model(args.bundle, **config['build'])
    trainer = LoRATrainer(built, config)
    data, trainer.state = load_checkpoint(checkpoint, built.trainable, trainer.optimizer, trainer.scheduler, config)
    stream = DeterministicSampleStream(dataset, seed=config['sampler_seed'])
    stream.load_state_dict(data)
    batches = stream.next_group(config['accumulation_steps'])
    frozen = frozen_fingerprint(built)
    starting_step = trainer.state['global_step']
    try:
        timing = TrainingTimings(built)
        built.streamer.collect_timings(synchronize=True)
        before = dict(built.streamer.stats)
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        with timing.instrument(), timing.span('whole_accumulation'):
            measured = trainer.backward_group(batches, on_microbatch=lambda index, current: timing.finish_micro())
        groups = timing.result()
        built.streamer.collect_timings(synchronize=True)
        instrumented_seconds = time.perf_counter() - began
        transfers = {key: value - before[key] for key, value in built.streamer.stats.items()}
        memory = memory_observation()
        gradients = {name: parameter.grad.detach().cpu().contiguous() for name, parameter in built.trainable.items()}
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.state['accumulation_position'] = 0
        began = time.perf_counter()
        control = trainer.backward_group(batches)
        torch.cuda.synchronize()
        control_seconds = time.perf_counter() - began
        byte_equal = all(torch.equal(parameter.grad.detach().cpu().contiguous().view(torch.uint8),
                                    gradients[name].view(torch.uint8)) for name, parameter in built.trainable.items())
        sdpa_calls = {}
        for name, value in groups.items():
            if '_sdpa_' in name:
                operation = name.rsplit('_sdpa_', 1)[-1]
                sdpa_calls[operation] = sdpa_calls.get(operation, 0) + value['intervals']
        attention_layers = sum(hasattr(block, 'self_attn') for block in built.model.model.layers)
        expected_backward = attention_layers * config['accumulation_steps']
        expected_forward = expected_backward * (2 if config['gradient_checkpointing'] else 1)
        efficient_forward = sum(count for name, count in sdpa_calls.items()
                                if 'efficient_attention' in name and 'backward' not in name)
        efficient_backward = sum(count for name, count in sdpa_calls.items()
                                 if 'efficient_attention' in name and 'backward' in name)
        checks = {'loss_and_samples_E2': all(measured[key] == control[key] for key in ['loss', 'supervised_tokens', 'sample_ids']),
                  'all_gradients_E2': byte_equal, 'frozen_unchanged': frozen_fingerprint(built) == frozen,
                  'optimizer_step_unchanged': trainer.state['global_step'] == starting_step,
                  'all_module_backward_intervals': sum(value['intervals'] for name, value in groups.items()
                      if name.endswith('_backward_inclusive')) == timing.modules * config['accumulation_steps'],
                  'efficient_sdpa_observed': any('_sdpa_' in name and 'efficient_attention' in name for name in groups),
                  'all_attention_forward_recompute_and_backward_efficient': (
                      efficient_forward == expected_forward and efficient_backward == expected_backward),
                  'math_sdpa_absent': not any('_sdpa_' in name and 'attention_math' in name for name in groups)}
        breakdown = {'H2D_ms': transfers['copy_gpu_ms'],
                     'decode_ms': sum(value['cuda_interval_ms'] for name, value in groups.items() if name.endswith('_decode')),
                     'all_GEMM_ms': sum(value['cuda_interval_ms'] for name, value in groups.items() if '_gemm_' in name),
                     'SDPA_forward_and_backward_ms': sum(value['cuda_interval_ms'] for name, value in groups.items() if '_sdpa_' in name)}
        for module in ['gdn', 'attention', 'mlp']:
            breakdown[module + '_forward_and_backward_inclusive_ms'] = sum(groups.get(module + phase + '_inclusive', {}).get('cuda_interval_ms', 0.)
                    for phase in ['_forward', '_backward'])
            breakdown[module + '_recompute_subset_ms'] = groups.get(module + '_recompute_inclusive', {}).get('cuda_interval_ms', 0.)
        result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'scope': 'One restored-state accumulation group, no optimizer update; not an additional training step',
            'restored_optimizer_step': starting_step, 'accumulation_steps': config['accumulation_steps'],
            'sequence_tokens': sum(batch['input_ids'].numel() for batch in batches), 'supervised_tokens': measured['supervised_tokens'],
            'instrumented_seconds': instrumented_seconds, 'uninstrumented_control_seconds': control_seconds,
            'SDPA_operator_call_counts': sdpa_calls,
            'expected_efficient_calls': {'forward_including_recompute': expected_forward, 'backward': expected_backward},
            'cuda_intervals': groups, 'breakdown': breakdown, 'transfers': transfers, 'memory': memory,
            'timing_limits': 'CUDA intervals include CPU enqueue gaps. H2D may overlap compute. Module backward includes checkpoint recompute; recompute is a subset. GEMM/SDPA/decode are subsets of module intervals, so these numbers must not be summed.'}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'checks': checks, 'breakdown': breakdown}, indent=2), flush=True)
        assert result['status'] == 'PASS', checks
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        built.streamer.close()


if __name__ == '__main__':
    main()
