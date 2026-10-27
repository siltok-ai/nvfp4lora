"""P5～P8及横向交付的证据审核；运行中/缺失项保持PENDING，不用短程代替100步。"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from safetensors import safe_open

from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.train.checkpoint import configuration_hash


class Pending(Exception):
    pass


class Audit:
    def __init__(self, task):
        self.task, self.gates, self.sources = task, {}, {}

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            raise Pending(f"Missing evidence: {path}")
        raw = path.read_bytes()
        self.sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    def recorded(self, pointer, *, allow_failed=False):
        path = self.task / pointer
        if not path.is_file():
            raise Pending(f"Stage not yet recorded: {pointer}")
        run = Path(path.read_text(encoding="utf-8-sig").strip())
        record = self.read(run / "manifest.json")
        if record["status"] == "RUNNING":
            raise Pending(f"Recorded stage still running: {run}")
        assert allow_failed or (record["status"] == "PASS" and record["exit_code"] == 0), f"Failed stage: {run}"
        return run

    def evidence(self, pointer, relative="results/results.json", **kwargs):
        run = self.recorded(pointer, **kwargs)
        return run, self.read(run / relative)

    def gate(self, name, check, *, success="PASS"):
        try:
            facts = check()
            self.gates[name] = {"status": success, "facts": facts}
        except Pending as error:
            self.gates[name] = {"status": "PENDING", "reason": str(error)}
        except (AssertionError, KeyError, ValueError) as error:
            self.gates[name] = {"status": "FAIL", "reason": f"{type(error).__name__}: {error}"}


STATE_CHECKS = {"parameters", "optimizer.pt", "scheduler.pt", "rng.pt", "data_state.json", "trainer_state.json", "train_config.json",
                "per_step_E2", "published_probe_E2", "training_memory_delta_W_E2"}
PRESSURE_CHECKS = {"fresh_two_optimizer_steps", "fixture_identity", "all_eight_samples_retained", "accumulation_eight",
                   "checkpointing_and_chunked_ce", "exact_sequence_and_supervision_counts", "adam_present_after_first_step",
                   "frozen_hashes_unchanged", "decoded_allocations_released_after_backward", "snapshot_identity"}


def comparison(audit, pointer, start, end):
    run, result = audit.evidence(pointer, "comparison.json")
    assert result["status"] == "PASS" and not result["differences"]
    assert STATE_CHECKS <= result["checks"].keys() and all(result["checks"][key] for key in STATE_CHECKS)
    assert result["starting_step"] == start and result["endpoint"] == end and result["compared_steps"] == end - start
    assert file_hash(run / "source/tools/compare_real_resume.py") == file_hash(Path(__file__).with_name("compare_real_resume.py"))
    return {"run": str(run), "start": start, "end": end, "all_ten_state_and_step_checks": True}


def functional(audit, variant):
    run, summary = audit.evidence(f"real-{variant}-functional100-run.txt", "results/summary.json")
    config = audit.read(run / "results/train_config.json")
    rows = [json.loads(line) for line in (run / "results/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    audit.sources[str(run / "results/metrics.jsonl")] = file_hash(run / "results/metrics.jsonl")
    assert summary["status"] == "PASS" and summary["current_step"] == config["max_steps"] == 100
    assert summary["starting_step"] < 50 and summary["max_steps_reached"]
    assert [row["global_step"] for row in rows] == list(range(summary["starting_step"] + 1, 101))
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) for row in rows)
    assert config["accumulation_steps"] == 8 and config["gradient_checkpointing"] and config["build"]["expert_rank"] == (0 if variant == "c3" else 2)
    assert config["build"]["c3_rank"] == 16 and config["build"]["c3_alpha"] == 32
    assert variant == "c3" or config["build"]["expert_alpha"] == 4
    assert 1e-4 <= config["learning_rate"] <= 2e-4 and config["warmup_steps"] == 10 and config["max_grad_norm"] == 1
    assert config["optimizer"] == "AdamW(weight_decay=0,foreach=False,fused=False)"
    assert summary["frozen_hashes_unchanged"]
    checkpoint = run / "results/checkpoints/step-000050"
    manifest = audit.read(checkpoint / "checkpoint_manifest.json")
    assert manifest["status"] == "complete" and manifest["config_sha256"] == configuration_hash(config)
    state = audit.read(checkpoint / "trainer_state.json")
    assert state["global_step"] == 50 and state["accumulation_position"] == 0
    assert "evaluation_step_100" in summary
    return run, summary, config, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-canary-accuracy", type=float)
    parser.add_argument("--max-replay-nll-increase", type=float)
    parser.add_argument("--threshold-provenance", help="Caller must supply truthful owner decision provenance; this tool cannot approve a threshold")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = Audit(args.task)
    root = Path(__file__).resolve().parents[1]

    def m1():
        run, data = audit.evidence("m1-audit-run.txt", "results/m1-audit.json")
        assert data["status"] == "PASS" and len(data["required_gates"]) == 32
        assert all(item["status"] == "PASS" for item in data["required_gates"].values()) and all(data["auxiliary"].values())
        target = audit.recorded("expert-target-run.txt")
        for relative in ["src/nvfp4lora/ops/experts_fn.py", "src/nvfp4lora/validation/experts_reference.py", "src/nvfp4lora/format/nvfp4.py"]:
            assert file_hash(root / relative) == file_hash(target / "source" / relative), relative
        return {"run": str(run), "required_gates": 32, "original_P2_math_unchanged": True}
    audit.gate("M1 prerequisite", m1)

    def p5_memory():
        run, data = audit.evidence("real-forward-run.txt", allow_failed=True)
        memory = data["memory_load"]
        assert memory["inventory"]["nonexpert_frozen_gpu"] == 4896711936 and memory["inventory"]["expert_cpu_payload"] == 18119639040
        assert memory["gpu_peak_allocated"] > 0 and memory["cpu_working_set"] > 0
        assert data["case"]["tokens"] == 128 and data["case"]["timed_gpu_forward_seconds"] > 0
        assert data["frozen_expert_bank_unchanged"]
        return {"run": str(run), "loaded_inventory": memory["inventory"], "original_reference_status": data["status"]}
    audit.gate("A5.1 materialization and memory", p5_memory)

    def p5_reference():
        original_run, original = audit.evidence("real-forward-run.txt", allow_failed=True)
        run, data = audit.evidence("real-routed-reference-run.txt")
        record = audit.read(run / "manifest.json")
        git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
        decision = subprocess.check_output([git, "rev-parse", "69a8cbd"], cwd=root, text=True).strip()
        assert subprocess.run([git, "merge-base", "--is-ancestor", decision, record["git_head"]], cwd=root, capture_output=True).returncode == 0
        for relative in ["tools/check_routed_reference.py", "src/nvfp4lora/validation/cpu_layers.py"]:
            committed = subprocess.check_output([git, "show", f"{decision}:{relative}"], cwd=root)
            captured = (run / "source" / relative).read_bytes()
            assert committed.replace(b"\r\n", b"\n") == captured.replace(b"\r\n", b"\n"), relative
            audit.sources[str(run / "source" / relative)] = hashlib.sha256(captured).hexdigest()
        assert original["status"] == "FAIL" and any(row["relative_l2"] > .01 for row in original["case"]["layers"])
        assert data["status"] == "PASS" and data["criteria_fixed_before_new_fixtures"] and len(data["cases"]) == 3
        for case in data["cases"]:
            assert case["status"] == "PASS" and len(case["layers"]) == 4
            for layer in case["layers"]:
                assert layer["fp32_actual_routes_relative_l2"] <= .01 and layer["router_same_input_fp32_logit_relative_l2"] <= .01
                assert layer["gpu_selection_is_topk"] and layer["gpu_selection_has_no_duplicates"]
                assert layer["router_weight_relative_l2"] == 0 and layer["changes_outside_2epsilon_margin"] == 0
        audit.sources[str(root / "docs/decisions.md")] = file_hash(root / "docs/decisions.md")
        return {"original_run": str(original_run), "original_unconditioned_1percent": "FAIL retained",
                "qualified_run": str(run), "decision_and_fixture_commit": decision,
                "policy": "Pre-recorded route-conditioned FP32 and independent route checks; see decisions.md"}
    audit.gate("A5.2 adjusted reference", p5_reference, success="PASS_WITH_DOCUMENTED_CHANGE")

    def p5_greedy():
        run, answers = audit.evidence("real-greedy-run.txt", "results/answers.json")
        assert [row["answer"].strip() for row in answers] == ["2", "北京", "Hello"] and all(row["stopped"] for row in answers)
        return {"run": str(run), "answers": [row["answer"] for row in answers]}
    audit.gate("A5.4 short greedy semantics", p5_greedy)

    def p5_timings():
        run, data = audit.evidence("real-forward-run.txt", allow_failed=True)
        case = data["case"]
        assert {"gdn", "attention", "expert_decode", "expert_compute"} <= case["cuda_timings"].keys()
        assert case["transfers"]["copy_bytes"] == 18119639040 and case["transfers"]["copy_gpu_ms"] > 0
        assert case["transfers"]["acquire_wait_gpu_ms"] >= 0
        return {"run": str(run), "cuda_intervals_may_overlap": True, "wall_seconds": case["timed_gpu_forward_seconds"]}
    audit.gate("A5.5 forward timing decomposition", p5_timings)

    for variant in ["c3", "expert"]:
        prefix = "A6" if variant == "c3" else "A7.3/A6"

        for length in [2048, 4096]:
            def capacity(variant=variant, length=length):
                run, data = audit.evidence(f"real-{variant}-sequence{length}-audit-run.txt", "audit.json")
                assert data["status"] == "PASS" and PRESSURE_CHECKS <= data["checks"].keys()
                assert all(data["checks"][key] for key in PRESSURE_CHECKS)
                observation = data["observation"]
                assert observation["max_length"] == length and observation["build"]["expert_rank"] == (0 if variant == "c3" else 2)
                assert [row["global_step"] for row in observation["observations"]] == [1, 2]
                assert all(row["sequence_tokens"] == 8 * length and row["supervised_tokens"] == 8280 for row in observation["observations"])
                profile_run, profile = audit.evidence(f"real-{variant}-recompute{length}-run.txt")
                assert profile["status"] == "PASS" and all(profile["checks"][key] for key in
                    ["all_40_layers_recomputed", "adam_present", "frozen_hashes_unchanged"])
                assert profile["input_tokens"] == length and profile["supervised_tokens"] == 1035
                assert sorted(row["layer"] for row in profile["memory"]["recomputations"]) == list(range(40))
                assert profile["history_complete"] == all(size < profile["history_limit"] for size in profile["history_trace_lengths"])
                if length == 2048:
                    assert profile["checks"]["uninstrumented_loss_E2"] and profile["checks"]["all_uninstrumented_gradients_E2"]
                _, snapshot = audit.evidence(f"real-{variant}-recompute{length}-snapshot-run.txt", "snapshot-audit.json")
                assert snapshot["status"] == "PASS" and snapshot["decode_allocation_count"] > 0 and snapshot["active_decode_allocation_count"] == 0
                assert file_hash(profile_run / "results/micro-backward.pickle") == snapshot["snapshot_sha256"]
                return {"run": str(run), "recompute_run": str(profile_run), "length": length,
                        "peak_allocated": max(row["memory"]["gpu_peak_allocated"] for row in observation["observations"]),
                        "history_complete": profile["history_complete"], "live_decoded_blocks": 0}
            audit.gate(f"Capacity {variant} T{length}", capacity)

        def zero(variant=variant):
            run, data = audit.evidence(f"real-{variant}-smoke-run.txt")
            assert data["status"] == "PASS" and data["zero_adapter_loss_E2"] and data["disabled_loss_unchanged_E2"]
            return {"run": str(run), "scope": "Actual zero and nonzero-B unified-disable fixed probes"}
        audit.gate(f"{prefix}.1 zero adapters E2", zero)

        def linear_gradients(variant=variant):
            run, data = audit.evidence(f"real-{variant}-smoke-run.txt")
            comparisons = data["linear_gradient_comparison"]
            assert data["status"] == "PASS" and len(comparisons) == 26
            assert {int(name.split(".")[2]) for name in comparisons} == {0, 3}
            assert all(item["status"] == "PASS" and item["relative_l2"] <= .02 and item["cosine"] >= .999 and item["backward_edges"] == 8 for item in comparisons.values())
            return {"run": str(run), "A_B_tensors": len(comparisons), "max_relative_l2": max(item["relative_l2"] for item in comparisons.values())}
        audit.gate(f"{prefix}.2 two-layer C3 gradients", linear_gradients)

        def resume100(variant=variant):
            functional(audit, variant)
            return comparison(audit, f"real-{variant}-resume50-comparison-run.txt", 50, 100)
        audit.gate(f"{prefix}.3 real 50->100 resume", resume100)

        def effects(variant=variant):
            run, _, config, _ = functional(audit, variant)
            published_run, data = audit.evidence(f"real-{variant}-functional-evaluation-run.txt")
            trained = audit.read(run / "results/evaluation-step000100.json")
            assert data["canary"]["samples"] == 50 and data["replay"]["samples"] == 300
            assert data["canary"]["results"] == trained["canary"]["results"] and data["replay"] == trained["replay"]
            assert data["prompt_logits_change"]["changed_vocabulary_entries"] > 0
            assert config["base_fingerprint"] == configuration_hash(base_identity(args.bundle))
            if args.minimum_canary_accuracy is None or args.max_replay_nll_increase is None or not args.threshold_provenance:
                raise Pending(f"Effect observations exist at {published_run}; owner thresholds/provenance not supplied")
            assert data["canary"]["accuracy"] >= args.minimum_canary_accuracy and data["canary"]["correct"] > data["base_canary_correct"]
            assert data["replay_nll_increase"] <= args.max_replay_nll_increase
            return {"run": str(published_run), "canary": data["canary"]["correct"], "base_canary": data["base_canary_correct"],
                    "replay_nll_increase": data["replay_nll_increase"], "threshold_provenance": args.threshold_provenance}
        audit.gate(f"{prefix}.4 effects and reload agreement", effects)

        def frozen_and_sets(variant=variant):
            run, summary, config, rows = functional(audit, variant)
            count, numel = (500, 19169280) if variant == "c3" else (740, 176455680)
            manifest = audit.read(run / "results/publish/manifest.json")
            assert manifest["status"] == "complete" and manifest["layout_version"] == "nvfp4-lora-v1" and manifest["global_step"] == 100
            assert len(manifest["tensor_specifications"]) == summary["trainable_tensors"] == count
            assert sum(item["numel"] for item in manifest["tensor_specifications"].values()) == summary["trainable_numel"] == numel
            assert manifest["training_config_sha256"] == configuration_hash(config) and manifest["base_fingerprint"] == configuration_hash(base_identity(args.bundle))
            assert manifest["nonexpert"]["rank"] == 16 and manifest["nonexpert"]["alpha"] == 32 and manifest["nonexpert"]["scaling"] == 2 and not manifest["nonexpert"]["rsLoRA"]
            assert len(manifest["experts"]) == (0 if variant == "c3" else 40)
            peft = audit.read(run / "results/publish/adapter_config.json")
            assert peft["lora_dropout"] == 0
            assert all(file_hash(run / "results/publish" / name) == digest for name, digest in manifest["files"].items())
            names = set()
            for filename in ["adapter_model.safetensors", "expert_lora.safetensors"]:
                with safe_open(run / "results/publish" / filename, framework="pt", device="cpu") as reader:
                    names.update(reader.keys())
            assert len(names) == count and summary["frozen_hashes_unchanged"]
            assert all(row["route_audit"]["never_routed_parameters_unchanged"] and row["route_audit"]["never_routed_adam_moments_zero"] for row in rows)
            return {"run": str(run), "tensors": count, "numel": numel, "frozen": True}
        audit.gate(f"{prefix}.5 frozen weights and sets", frozen_and_sets)

        def training_memory(variant=variant):
            run, summary, _, rows = functional(audit, variant)
            resumed, resumed_summary = audit.evidence(f"real-{variant}-resume50-run.txt", "results/summary.json")
            assert resumed_summary["status"] == "PASS" and "load_or_restore" in resumed_summary["memory"]
            assert "save_50" in summary["memory"] and "save_100" in summary["memory"] and "publish_and_probe" in summary["memory"]
            assert all(row["step_wall_seconds"] > 0 and row["transfers"]["copy_bytes"] > 0 and row["transfers"]["copy_gpu_ms"] > 0 for row in rows)
            for length in [2048, 4096]:
                _, pressure = audit.evidence(f"real-{variant}-sequence{length}-audit-run.txt", "audit.json")
                assert pressure["status"] == "PASS" and all(pressure["checks"].values())
            return {"run": str(run), "restore_run": str(resumed), "long_lengths": [2048, 4096], "cpu_fields": "Working-set observations and process-lifetime peaks, not per-stage CPU interval peaks"}
        audit.gate(f"{prefix}.6 training stages and capacity", training_memory)

        def release(variant=variant):
            run, data = audit.evidence(f"real-{variant}-functional-release-run.txt")
            assert data["status"] == "PASS" and all(data["checks"][key] for key in
                ["fixed_batch_loss_E2", "disabled_loss_E2", "all_parameter_bytes_equal", "all_delta_W_bytes_E2"])
            assert data["delta_W_fingerprint_count"] == (250 if variant == "c3" else 4090)
            assert file_hash(run / "source/tools/verify_training_release.py") == file_hash(root / "tools/verify_training_release.py")
            return {"run": str(run), "all_delta_W_and_fixed_probe_E2": True}
        audit.gate(f"A8.1/A8.2 {variant} publish rebuild", release)
        audit.gate(f"A8.3 {variant} short real resume", lambda variant=variant: comparison(audit, f"real-{variant}-resume-comparison-run.txt", 2, 4))

        def diagnostics(variant=variant):
            run, data = audit.evidence(f"real-{variant}-checkpoint-diagnostics-run.txt")
            assert data["status"] == "MEASURED" and data["frozen_hashes_unchanged"] and set(data["checkpoints"]) == {"20", "50", "100"}
            assert len(data["probe_sample_ids"]) == 8
            for point in [data["base_disabled"], *data["checkpoints"].values()]:
                assert point["canary_holdout_nll"]["samples"] == 50 and math.isfinite(point["canary_holdout_nll"]["nll_per_token"])
                assert len(point["routes"]) == 40
                assert all(len(row["counts"]) == 256 and sum(row["counts"]) == row["selected_routes"] == 8 * row["valid_token_vectors"] for row in point["routes"].values())
            assert all(all(math.isfinite(value) and value > 0 for value in point["group_l2_norms"].values()) for point in data["checkpoints"].values())
            return {"run": str(run), "steps": [20, 50, 100], "scope": data["scope"]}
        audit.gate(f"6.5 {variant} periodic diagnostics", diagnostics)

        def fp8(variant=variant):
            run, data = audit.evidence(f"real-{variant}-functional-fp8-run.txt")
            assert data["status"] == "OBSERVED" and data["fp8_modules"] == 130
            assert set(data["phases"]) == {"base_disabled", "adapter_enabled"}
            for phase in data["phases"].values():
                assert len(phase) == 130 and all(row["calls"] > 0 and row["valid_token_vectors"] > 0 and math.isfinite(row["ratio"]) for row in phase.values())
            return {"run": str(run), "above_original_range": data["above_original_range"], "scope": data["scope"]}
        audit.gate(f"7 {variant} FP8 risk observations", fp8)

    def expert_vjp():
        run, data = audit.evidence("real-expert-vjp-run.txt")
        assert data["status"] == "PASS" and {row["layer"] for row in data["layers"]} == {0, 3}
        for row in data["layers"]:
            assert row["unused_raw_gradients_exact_zero"] and row["local_AB_matches_full_model_gradient_E2"] and row["all_AB_nonzero"]
            assert all(item["relative_l2"] <= .02 and item["cosine"] >= .999 for item in row["comparisons"].values())
        return {"run": str(run), "layers": [0, 3]}
    audit.gate("A7.1 expert actual-activation VJP", expert_vjp)

    def expert_routes():
        run, _, _, rows = functional(audit, "expert")
        for row in rows:
            report = row["route_audit"]
            assert report["unused_raw_gradients_exact_zero"] and report["never_routed_parameters_unchanged"] and report["never_routed_adam_moments_zero"]
            assert len(report["routing_counts"]) == 40
        original, _ = audit.evidence("real-expert-baseline-run.txt", "results/summary.json")
        old = [json.loads(line) for line in (original / "results/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any(row["route_audit"]["never_routed_expert_layer_pairs"] > 0 for row in old)
        assert any(row["route_audit"]["historical_inactive_changed_parameter_rows"] > 0 for row in old)
        return {"run": str(run), "never_and_historical_cases_nonvacuous_in_prefix": True}
    audit.gate("A7.2 raw unused gradients and Adam history", expert_routes)

    def delivery():
        if args.artifact_dir is None:
            raise Pending("Final artifact directory not supplied")
        manifest = audit.read(args.artifact_dir / "manifest.json")
        assert manifest["status"] == "verified" and manifest["files"]
        actual = {path.relative_to(args.artifact_dir).as_posix() for path in args.artifact_dir.rglob("*") if path.is_file() and path != args.artifact_dir / "manifest.json"}
        assert actual == set(manifest["files"])
        assert all(file_hash(args.artifact_dir / name) == digest for name, digest in manifest["files"].items())
        for variant, destination in [("c3", "c3"), ("expert", "expert-r2")]:
            run, _, _, _ = functional(audit, variant)
            for relative, manifest_name in [("publish", "manifest.json"),
                    ("checkpoints/step-000050", "checkpoint_manifest.json"), ("checkpoints/step-000100", "checkpoint_manifest.json")]:
                source = run / "results" / relative
                receipt = audit.read(source / manifest_name)
                for name in [*receipt["files"], manifest_name]:
                    target_name = f"{destination}/{relative}/{name}"
                    assert target_name in manifest["files"]
                    assert manifest["files"][target_name] == file_hash(source / name)
        return {"artifact_dir": str(args.artifact_dir)}
    audit.gate("Delivery reproducible code and resumable artifacts", delivery)

    counts = Counter(gate["status"] for gate in audit.gates.values())
    ready = not counts["FAIL"] and not counts["PENDING"]
    result = {"status": "QUALIFIED_WITH_DOCUMENTED_CHANGE" if ready else ("FAIL" if counts["FAIL"] else "INCOMPLETE"),
              "scope": "G1 training only; G2/Speed integration and G3 domain benefits remain separate",
              "gates": audit.gates, "counts": dict(counts), "source_files_sha256": audit.sources,
              "thresholds": {"minimum_canary_accuracy": args.minimum_canary_accuracy, "max_replay_nll_increase": args.max_replay_nll_increase,
                             "owner_decision_provenance": args.threshold_provenance},
              "optional_not_run": ["A5.3 FreeToken teacher-forcing comparison", "Expert rank4", "FLA; selected validated Torch chunk fallback"],
              "limitations": ["Original unconditional A5.2 1% FP32 reference FAIL is retained, not relabeled PASS",
                              "Canary/replay is a functional synthetic association/arithmetic corpus, not domain/general capability evidence",
                              "Threshold provenance is a caller statement and must reflect the actual owner decision"]}
    (args.output / "m2-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "pending": [key for key, value in audit.gates.items() if value["status"] == "PENDING"]}, indent=2))
    if args.require_complete:
        assert ready, "M2 completion remains unproven"


if __name__ == "__main__":
    main()
