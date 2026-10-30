"""Dense DA audit: preserve original failures, require actual full runs and owner thresholds."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess

from audit_m2 import Audit, Pending, STATE_CHECKS
from nvfp4lora.format.preprocess import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--freetoken', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'docs/evidence/dense'
    audit = Audit(args.task)
    verified_summaries = set()

    def summary(filename):
        value = audit.read(evidence / filename)
        if filename not in verified_summaries:
            for record in value.get('runs', {}).values():
                if not isinstance(record, dict) or 'run' not in record:
                    continue
                path = Path(record['run']) / 'manifest.json'
                manifest = audit.read(path)
                expected = record.get('execution_status', record.get('status'))
                if expected in ['PASS', 'FAIL']:
                    assert manifest['status'] == expected
                if 'manifest_sha256' in record:
                    assert file_hash(path) == record['manifest_sha256']
            verified_summaries.add(filename)
        return value

    def historical(filename, gate):
        value = summary(filename)
        item = value['gates'][gate]
        status = item['status'] if isinstance(item, dict) else item
        if gate == 'DA0.7':
            assert status == 'FALLBACK_PASS' and item['original_default_HF_status'] == 'FAIL'
            real = summary('d5-summary.json')['real_reference_and_resources']['case']
            assert real['actual_SDPA_operators'] == {'aten::_scaled_dot_product_efficient_attention': 16}
        elif gate == 'DA2.3':
            assert status == 'QUALIFIED_WITH_DOCUMENTED_REFERENCE_CONTROL'
            control = value['DA2.3']
            assert control['original_failure_preserved']
            assert control['held_out_max_relative_l2'] <= 2e-6
            assert control['original_all_real_shape_1e6'] == 'FAIL'
        else:
            assert status == 'PASS', (gate, status)
        return {'evidence': filename, 'original_state': status}

    for stage, count in [(0, 9), (1, 6), (2, 7), (3, 5), (4, 6), (5, 6)]:
        filename = f'd{stage}-summary.json'
        for number in range(1, count + 1):
            gate = f'DA{stage}.{number}'
            audit.gate(gate, lambda f=filename, g=gate: historical(f, g),
                       success='PASS_WITH_DOCUMENTED_CHANGE' if gate in ['DA0.7', 'DA2.3'] else 'PASS')
    for number in range(3):
        gate = f'DA6.{number}'
        audit.gate(gate, lambda g=gate: historical('d6-preflight-summary.json', g))

    def approval():
        decision = audit.read(evidence / 'effect-threshold-owner-decision.json')
        assert decision['status'] == 'APPROVED' and decision['gate'] == 'DA6.3'
        assert decision['source'].startswith('Actual user reply') and '同意采用推荐选项' in decision['quoted_user_message']
        assert decision['minimum_canary_correct'] == 45 and decision['canary_total'] == 50
        assert decision['minimum_canary_accuracy'] == .9 and decision['max_replay_nll_increase'] == .01
        assert not decision['hard_constraints_changed']
        return decision
    audit.gate('DA6.3', approval)

    def functional():
        run, report = audit.evidence('d6-functional100-run.txt', 'results/summary.json')
        assert report['status'] == 'PASS' and report['current_step'] == 100 and report['starting_step'] == 0
        assert report['max_steps_reached'] and report['logical_max_steps'] == 100 and report['frozen_hashes_unchanged']
        assert report['trainable_tensors'] == 800 and report['trainable_numel'] == 108789760
        config = audit.read(run / 'results/train_config.json')
        assert config['family'] == 'qwen3_5_dense' and config['max_steps'] == 100
        assert config['max_length'] == 2048 and config['accumulation_steps'] == 8
        assert config['build']['lora_rank'] == 16 and config['build']['lora_scope'] == 'D-ALL'
        path = run / 'results/metrics.jsonl'
        audit.sources[str(path)] = file_hash(path)
        metrics = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert [row['global_step'] for row in metrics] == list(range(1, 101))
        assert all(len(row['sample_ids']) == 8 and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in metrics)
        for step in [10, 20, 50, 100]:
            result = audit.read(run / f'results/evaluation-step{step:06d}.json')
            assert result['step'] == step and not result['disabled']
            assert result['canary']['samples'] == 50 and result['replay']['samples'] == 300
        for step in [20, 50, 100]:
            checkpoint = run / f'results/checkpoints/step-{step:06d}'
            manifest = audit.read(checkpoint / 'checkpoint_manifest.json')
            assert manifest['status'] == 'complete'
            assert all(file_hash(checkpoint / name) == digest for name, digest in manifest['files'].items())
        delta = audit.read(run / 'results/delta-w.json')
        assert len(delta['tensors']) == 400
        return {'run': str(run), 'steps': 100, 'eval_steps': [10, 20, 50, 100], 'checkpoint_steps': [20, 50, 100]}
    audit.gate('DA6.4', functional)

    def resumed():
        run, report = audit.evidence('d6-resume50-comparison-run.txt', 'comparison.json')
        assert report['status'] == 'PASS' and not report['differences']
        assert STATE_CHECKS <= report['checks'].keys() and all(report['checks'][key] for key in STATE_CHECKS)
        assert report['starting_step'] == 50 and report['endpoint'] == 100 and report['compared_steps'] == 50
        return {'run': str(run), 'all_ten_groups_E2': True}
    audit.gate('DA6.5', resumed)

    def release():
        run, report = audit.evidence('d6-functional-release-run.txt')
        assert report['status'] == 'PASS' and all(report['checks'].values())
        assert report['delta_W_fingerprint_count'] == 400
        assert {'fixed_batch_loss_E2', 'disabled_loss_E2', 'all_parameter_bytes_equal', 'all_delta_W_bytes_E2'} <= report['checks'].keys()
        return {'run': str(run), 'delta_W_count': 400}
    audit.gate('DA6.6', release)

    def effects():
        decision = approval()
        run, report = audit.evidence('d6-functional-evaluation-run.txt')
        assert report['status'] == 'PASS' and all(report['criteria'].values())
        assert report['thresholds']['minimum_canary_accuracy'] == decision['minimum_canary_accuracy']
        assert report['thresholds']['max_replay_nll_increase'] == decision['max_replay_nll_increase']
        assert report['base_canary_correct'] == 0 and report['canary']['samples'] == 50
        assert report['canary']['correct'] >= decision['minimum_canary_correct']
        assert report['replay_nll_increase'] <= decision['max_replay_nll_increase']
        assert report['prompt_logits_change']['changed_vocabulary_entries'] > 0
        return {'run': str(run), 'canary_correct': report['canary']['correct'], 'replay_nll_increase': report['replay_nll_increase']}
    audit.gate('DA6.7', effects)

    def ranges():
        run, report = audit.evidence('d6-functional-fp8-run.txt')
        assert report['status'] == 'OBSERVED' and report['family'] == 'qwen3_5_dense' and report['fp8_modules'] == 208
        assert set(report['phases']) == {'base_disabled', 'adapter_enabled'}
        assert all(len(rows) == 208 and all(row['calls'] > 0 and math.isfinite(row['ratio']) for row in rows.values()) for rows in report['phases'].values())
        return {'run': str(run), 'scope': 'Range observation, not a no-clipping certificate',
                'above_range_counts': {name: len(values) for name, values in report['above_original_range'].items()}}
    audit.gate('DA6.8', ranges)

    def timings():
        run, report = audit.evidence('d6-training-timing-run.txt')
        assert report['status'] == 'PASS' and all(report['checks'].values())
        assert report['restored_optimizer_step'] == 100 and report['accumulation_steps'] == 8
        assert {'H2D_ms', 'decode_ms', 'all_GEMM_ms', 'SDPA_forward_and_backward_ms',
                'gdn_forward_and_backward_inclusive_ms'} <= report['breakdown'].keys()
        functional()
        return {'run': str(run), 'instrumentation_control_E2': True, 'scope': report['scope']}
    audit.gate('DA6.9', timings)

    def future_summary(filename, required):
        report = summary(filename)
        assert report['status'] == 'PASS' and all(report['gates'].get(key) == 'PASS' for key in required)
        return {'evidence': filename}
    audit.gate('Qwen3.6-short', lambda: future_summary('qwen36-summary.json',
        ['DA0.4', 'D1', 'D5', 'DA6.0', 'DA6.1', 'short10', 'release_rebuild']))
    audit.gate('DA7.1', lambda: future_summary('d7-summary.json', ['DA7.1']))

    def documentation():
        for name in ['progress.md', 'results.md', 'training-cli.md']:
            path = root / 'docs/dense' / name
            if not path.is_file():
                raise Pending(f'Missing dense documentation: {name}')
            audit.sources[str(path)] = file_hash(path)
        readme = root / 'README.md'
        assert 'docs/dense/' in readme.read_text(encoding='utf-8')
        audit.sources[str(readme)] = file_hash(readme)
        return {'documents_present': True}
    audit.gate('DA7.2', documentation)

    git = shutil.which('git') or 'C:/Program Files/Git/cmd/git.exe'
    def publication():
        head = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        output = subprocess.check_output([git, 'ls-remote', 'origin', 'refs/heads/feat/dense27b-lora'], cwd=root, text=True).strip()
        if not output or output.split()[0] != head:
            raise Pending('Final feature branch HEAD is not yet published')
        assert not subprocess.check_output([git, 'status', '--porcelain'], cwd=root, text=True).strip()
        return {'branch': 'feat/dense27b-lora', 'head': head}
    audit.gate('DA7.3', publication)

    def inference():
        report = audit.read(args.freetoken / 'docs/evidence/dense/final-dense-summary.json')
        assert report['status'] == 'PASS' and report['final_training_adapter'] is True
        assert report['qwen38_complete_model'] and report['qwen36_complete_model']
        assert report['zero_B_controls'] and report['fresh_process_reload_E2']
        return {'evidence': 'FreeToken final-dense-summary.json'}
    audit.gate('FreeToken-final', inference)

    counts = Counter(gate['status'] for gate in audit.gates.values())
    result = {'status': 'FAIL' if counts['FAIL'] else ('INCOMPLETE' if counts['PENDING'] else 'QUALIFIED_WITH_DOCUMENTED_ADJUSTMENTS'),
              'gates': audit.gates, 'counts': dict(counts), 'source_files_sha256': audit.sources,
              'scope': 'Original DA0.7 and large-shape DA2.3 failures remain recorded. Qualified replacement controls are explicit. No partial run substitutes for real100 or final inference.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'counts': result['counts']}, indent=2), flush=True)
    assert not counts['FAIL'], [(name, value) for name, value in audit.gates.items() if value['status'] == 'FAIL']
    if args.require_complete:
        assert not counts['PENDING'], 'Required dense acceptance is still incomplete'


if __name__ == '__main__':
    main()
