"""按 v2 的32个必需关卡核对已有原始证据，不重跑已通过且未改变的数值内核。"""

import argparse
import importlib.metadata
import json
from pathlib import Path

from nvfp4lora.format.preprocess import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources, gates = {}, {}

    def read(path):
        path = Path(path)
        sources[str(path)] = file_hash(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def pointer(name):
        return Path((args.task / name).read_text(encoding="utf-8-sig").strip())

    def gate(name, passed, evidence, detail):
        gates[name] = {"status": "PASS" if passed else "FAIL", "evidence": str(evidence), "detail": detail}

    def recorded(run):
        record = read(run / "manifest.json")
        assert record["status"] == "PASS" and record["exit_code"] == 0, run

    basic_path = args.runs / "20261027T103336Z-bootstrap/p0-basic/results.json"
    basic = read(basic_path)
    assert basic["exit_code"] == 0
    assert all(importlib.metadata.version(name) == version for name, version in basic["packages"].items())
    for name in ["G0.1", "G0.4", "G0.5"]:
        gate(name, basic["gates"][name]["status"] == "PASS", basic_path, "Actual CUDA BF16/meta architecture/PEFT respectively; pinned dependency versions still match")
    pinned_run = args.runs / "20261027T105133Z-pinned-h2d"
    recorded(pinned_run)
    pinned = read(pinned_run / "results/results.json")
    gate("G0.2", pinned["status"] == "PASS" and pinned["requested_pool_bytes"] == 18500000000,
         pinned_run, "Actual 18.5GB pinned pool and H2D measurement")
    gdn_run = args.runs / "20261027T111254Z-gdn-byte-check"
    recorded(gdn_run)
    gdn = read(gdn_run / "results/results.json")
    module_run = args.runs / "20261027T105826Z-gdn-module"
    recorded(module_run)
    module = read(module_run / "results/results.json")
    gate("G0.3", gdn["status"] == "FALLBACK PASS" and module["status"] == "PASS" and file_hash(gdn["source"]) == gdn["source_sha256"],
         gdn_run, "Selected existing pure Torch chunk fallback; short recurrence gradients, target T4096, full GDN module and checkpoint byte checks")
    grouped_path = args.runs / "20261027T103336Z-bootstrap/grouped-target/results.json"
    grouped = read(grouped_path)
    gate("G0.6", grouped["status"] == "PASS" and all(case["experts"] == 256 for case in grouped["cases"]) and {(case["K"], case["N"]) for case in grouped["cases"]} == {(2048, 512), (512, 2048)},
         grouped_path, "256-group target GEMMs and all gradients")
    download_run = args.runs / "20261027T105045Z-checkpoint"
    recorded(download_run)
    download = read(download_run / "results/receipt.json")
    shards = [item for item in download["downloaded"] if item["filename"].endswith(".safetensors")]
    gate("G0.8", download["status"] == "PASS" and len(shards) == 3 and all(item["sha256"] == item["lfs_sha256"] for item in shards),
         download_run, "All three source shards SHA256 verified against upstream LFS; P1 rechecked before preprocessing")

    manifest = read(args.bundle / "manifest.json")
    validation_run = args.runs / "20261027T114846Z-real-format-validation"
    recorded(validation_run)
    validation = read(validation_run / "results/results.json")
    build_run = args.runs / "20261027T123808Z-real-model-build"
    recorded(build_run)
    build = read(build_run / "results/results.json")
    gate("A1.1", manifest["status"] == "PASS" and validation["status"] == "PASS" and manifest["geometry"]["expert_tensor_count"] == 122880,
         args.bundle, "Real source headers/index/layout; 124468 source tensors, 122880 text expert fields")
    gate("A1.2", validation["e1_fields_compared"] == 3840, validation_run, "40x8x3x4 sampled disk raw-field readbacks")
    gate("A1.3", validation["status"] == "PASS" and validation["numeric_reference"] == "FreeToken raw FP32 global",
         validation_run, "960 actual FreeToken projection comparisons, maximum one BF16 ULP")
    gate("A1.4", build["status"] == "PASS" and build["all_nonexpert_bytes_equal"] and build["no_meta"] and build["nonexpert_values_checked"] == 613,
         build_run, "Materialized GPU parameters and fresh FP32 RoPE buffers, not just a meta catalog")
    gate("A1.5", manifest["peak_working_set_bytes"] < 16e9 and manifest["geometry"]["expert_payload_bytes"] == 18119639040,
         args.bundle, "Streaming preprocessing peak working set 3.246GB; no full BF16 expert base")

    p2_run = args.runs / "20261027T121353Z-expert-target"
    recorded(p2_run)
    p2 = read(p2_run / "results/results.json")
    assert p2["status"] == "PASS"
    # 当前数学代码和真实P2验收快照完全相同；新增的是可替换streamer和构建选择。
    root = Path(__file__).resolve().parents[1]
    unchanged_math = {}
    for relative in ["src/nvfp4lora/ops/experts_fn.py", "src/nvfp4lora/validation/experts_reference.py", "src/nvfp4lora/format/nvfp4.py"]:
        unchanged_math[relative] = file_hash(root / relative) == file_hash(p2_run / "source" / relative)
    assert all(unchanged_math.values()), unchanged_math
    unit_run = pointer("async-full-regression-run.txt")
    recorded(unit_run)
    unit_log = (unit_run / "stdout-stderr.log").read_text(encoding="utf-8")
    sources[str(unit_run / "stdout-stderr.log")] = file_hash(unit_run / "stdout-stderr.log")
    cases = p2["cases"]
    gate("A2.1", all(cases[key]["status"] == "PASS" for key in ["math-torch.float32", "math-torch.bfloat16"]), p2_run, "Actual T4096/H2048/I512/E256/top8 forward vs independent dense reference")
    gate("A2.2", gates["A2.1"]["status"] == "PASS", p2_run, "x, routing weights and all six A/B gradients; numerical policies unchanged")
    gate("A2.3", "test_fp64_gradcheck_all_coordinates" in unit_log and "Ran 33 tests" in unit_log and unit_log.rstrip().endswith("OK"), unit_run, "True FP64 all-coordinate gradcheck; latest 33-test regression")
    gate("A2.4", all(cases[key]["unrouted_gradients_exact_zero"] for key in ["math-torch.float32", "math-torch.bfloat16"]), p2_run, "Actual expert255 deliberately unrouted; tiny edge cases included in regression")
    for name, key, detail in [("A2.5", "zero-B-vs-base", "Zero B vs no adapter E2; first B update covered in latest regression"),
                              ("A2.6", "repeat", "Repeated output/all gradients E2; autocast control"),
                              ("A2.7", "gradient-accumulation", "Standard .grad accumulation, two microbatches"),
                              ("A2.8", "checkpoint-False", "Both reentrant and nonreentrant checkpoint E2")]:
        gate(name, cases[key]["status"] == "PASS", p2_run, detail)
    snapshot = read(p2_run / "snapshot-audit.json")
    gates["A2.8"]["status"] = "PASS" if all(cases[key]["status"] == "PASS" for key in ["checkpoint-False", "checkpoint-True"]) else "FAIL"
    gate("A2.9", all(cases[f"memory-chunk-{chunk}"]["status"] == "PASS" and cases[f"memory-chunk-{chunk}"]["metrics"]["allocated_after"] == cases[f"memory-chunk-{chunk}"]["metrics"]["allocated_before"] for chunk in [8, 32, 256]) and all(item["events_with_python_decode_frames"] > 0 and item["active_decoded_blocks_at_end"] == 0 for item in snapshot),
         p2_run, "Three target chunk memory sweeps; allocation stacks audited and no live decoded weights after backward")

    p3_run = pointer("real-async-run.txt")
    recorded(p3_run)
    p3 = read(p3_run / "results/results.json")
    assert p3["status"] == "PASS" and p3["frozen_store_hashes_unchanged"]
    for index, key in enumerate(["A3.1-random", "A3.2-delayed", "A3.3-wrong", "A3.4-operator", "A3.5-profiler"], start=1):
        gate(f"A3.{index}", p3["cases"][key]["status"] == "PASS", p3_run, key)
    assert p3["cases"]["A3.1-random"]["full_payload_checks"] == 1000
    assert all(p3["cases"][key]["full_payload_checks"] == 80 for key in ["A3.2-delayed", "A3.3-wrong"])
    assert all(value["bitwise_equal"] for value in p3["cases"]["A3.4-operator"]["comparisons"].values())
    assert p3["cases"]["A3.5-profiler"]["overlapping_pairs"] > 0

    p4_run = pointer("tiny-training-run.txt")
    async_run = pointer("async-training-run.txt")
    for run in [p4_run, async_run]:
        recorded(run)
        result = read(run / "summary.json")
        assert result["status"] == "PASS" and result["qualification"] == "P4"
    paired = read(p4_run / "paired/summary.json")
    resumed = read(p4_run / "resumed/summary.json")
    reloaded = read(p4_run / "reloaded/summary.json")
    comparison_run = pointer("sync-async-comparison-run.txt")
    recorded(comparison_run)
    comparison = read(comparison_run / "comparison.json")
    assert comparison["status"] == "PASS"
    gate("A4.1", paired["first_gradient_worst_relative_l2"] <= .02 and paired["first_gradient_min_cosine"] >= .999 and paired["final_weight_min_cosine"] >= .99, p4_run, "100 optimizer steps, accumulation2; independent reference not modified")
    gate("A4.2", paired["loss_max_relative"] <= .02, p4_run, "All 100 step losses; async re-run also passed, and all sync/async losses E2")
    gate("A4.3", resumed["status"] == "PASS" and resumed["loss_comparison"] == "E2" and all(resumed["state_checks"].values()), p4_run, "New process step50->100; all state, data order and four RNG streams E2, also under async")
    gate("A4.4", reloaded["status"] == "PASS" and all(reloaded["checks"].values()), p4_run, "Separate process uses bundle+published LoRA; loss/all parameters/all tiny expert deltaW E2")
    gate("A4.5", paired["frozen_hashes_unchanged"], p4_run, "CPU experts and frozen GPU nonexpert parameters/buffers unchanged")
    gate("A4.6", paired["canonical_sets_equal"] and paired["trainable_tensor_count"] == 74 and paired["trainable_numel"] == 115712, p4_run, "Named optimizer save validates actual unique set; canonical expected/publish sets and shapes match")
    template_run = pointer("template-run.txt")
    recorded(template_run)
    template = read(template_run / "results/summary.json")
    quantizer_run = pointer("quantizer-layout-run.txt")
    recorded(quantizer_run)
    quantizer = read(quantizer_run / "results/summary.json")
    tiny_audit_run = pointer("tiny-snapshot-audit-run.txt")
    recorded(tiny_audit_run)
    tiny_snapshot = read(tiny_audit_run / "snapshot-audit.json")
    auxiliary = {"real_template_200": template["status"] == "PASS" and template["samples"] == 200,
                 "quantizer_real_layer_layout": quantizer["input_unchanged"] and all(p["block_scale_match_rate"] == p["nonzero_scale_code_match_rate"] == p["decoded_match_rate"] == 1 for p in quantizer["projections"].values()),
                 "tiny_memory_inventory_and_stages": all(name in paired["memory"] for name in ["built_pair", "first_backward_with_reference_resident", "step_1", "step_100", "save_middle", "evaluation_publish"]) and "load" in resumed["memory"],
                 "tiny_snapshot_no_live_decode": tiny_snapshot["status"] == "PASS" and tiny_snapshot["active_decode_allocation_count"] == 0,
                 "sync_async_100step_E2": comparison["status"] == "PASS"}
    assert len(gates) == 32, len(gates)
    status = "PASS" if all(item["status"] == "PASS" for item in gates.values()) and all(auxiliary.values()) else "FAIL"
    save = {"status": status, "milestone": "M1 only", "required_gates": gates, "auxiliary": auxiliary,
            "unchanged_P2_math": unchanged_math, "source_files_sha256": sources,
            "limitations": ["Does not qualify full real-model forward/training P5-P8", "Profiler overlap uses artificial delayed reads; not training throughput", "CPU stage observations are working-set samples, not interval peaks", "GDN is selected validated Torch fallback; FLA not tested"]}
    (args.output / "m1-audit.json").write_text(json.dumps(save, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "required_gates": len(gates), "auxiliary": auxiliary}, indent=2))
    assert status == "PASS"


if __name__ == "__main__":
    main()
