"""A7.1：真实训练激活/dY，非零A/B，两个专家层的全部VJP独立稠密对照。"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import build_from_adapter
from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.model.build import causal_model
from nvfp4lora.ops.experts_fn import NVFP4ExpertsFn
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.loop import configure_training, loss_sum, memory_observation
from nvfp4lora.validation.experts_reference import experts_reference


NAMES = ["output", "x", "routing_weights", "Ag", "Bg", "Au", "Bu", "Ad", "Bd"]


def compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    norm = float(b.norm())
    relative = float((a - b).norm()) / norm if norm else (0. if not a.count_nonzero() else float("inf"))
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm())) if a.norm() and b.norm() else (1. if torch.equal(a, b) else 0.)
    return {"relative_l2": relative, "cosine": cosine}


def dense_layer_one_expert_at_a_time(store, layer):
    """只在本层对照期间保存BF16 W；按专家CPU解码，避免整层GPU解码临时峰值。"""
    packed, matrices, scales = store.tensors(layer), [], []
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        weight = packed[projection + ".weight"]
        matrix = torch.empty(*weight.shape[:-1], weight.shape[-1] * 2, dtype=torch.bfloat16, device="cuda")
        for expert in range(weight.shape[0]):
            decoded = dequantize(weight[expert], packed[projection + ".weight_scale"][expert], dtype=torch.bfloat16)
            matrix[expert].copy_(decoded)
        matrices.append(matrix)
        scales.append(packed[projection + ".weight_scale_2"].to(device="cuda"))
    return matrices, scales


def qualify_layer(module, record, store):
    x_cpu, indices_cpu, weights_cpu, dy_cpu, output_cpu = record
    params = module.adapter_parameters()
    x, indices, weights, dy = (value.to("cuda") for value in [x_cpu, indices_cpu, weights_cpu, dy_cpu])
    x, weights = x.requires_grad_(), weights.requires_grad_()
    module.op_config.streamer.phase = "FORWARD"
    actual = NVFP4ExpertsFn.apply(x, indices, weights, *params, module.layer, module.op_config)
    assert torch.equal(actual.detach().cpu().view(torch.uint8), output_cpu.view(torch.uint8)), "Captured output changed"
    module.op_config.streamer.phase = "BACKWARD"
    gradients = torch.autograd.grad(actual, (x, weights, *params), grad_outputs=dy)
    production = [actual.detach().cpu(), *[gradient.detach().cpu() for gradient in gradients]]
    module_parameter_grads_E2 = all(torch.equal(parameter.grad.detach().cpu().view(torch.uint8), gradient.view(torch.uint8))
                                   for parameter, gradient in zip(params, production[3:]))
    assert module_parameter_grads_E2
    del actual, gradients
    matrices, scales = dense_layer_one_expert_at_a_time(store, module.layer)
    ref_x, ref_weights = x.detach().clone().requires_grad_(), weights.detach().clone().requires_grad_()
    ref_params = tuple(parameter.detach().clone().requires_grad_() for parameter in params)
    expected = experts_reference(ref_x, indices, ref_weights, matrices, scales, ref_params, scaling=module.op_config.scaling)
    reference_gradients = torch.autograd.grad(expected, (ref_x, ref_weights, *ref_params), grad_outputs=dy)
    reference = [expected.detach().cpu(), *[gradient.detach().cpu() for gradient in reference_gradients]]
    metrics = {name: compare(a, b) for name, a, b in zip(NAMES, production, reference)}
    active = torch.bincount(indices_cpu.flatten(), minlength=store.geometry["experts"]) != 0
    unused_zero = all(not gradient[~active].count_nonzero() for gradient in production[3:])
    status = "PASS" if all(value["relative_l2"] <= (.01 if name == "output" else .02) for name, value in metrics.items()) and unused_zero else "FAIL"
    return {"status": status, "layer": module.layer, "actual_tokens": x.shape[0],
            "active_experts": int(active.sum()), "unused_experts": int((~active).sum()),
            "unused_raw_gradients_exact_zero": bool(unused_zero), "local_AB_matches_full_model_gradient_E2": module_parameter_grads_E2,
            "all_AB_nonzero": all(bool(parameter.count_nonzero()) for parameter in params), "comparisons": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(args.source_results / "smoke.jsonl", tokenizer, max_length=128)
    built = build_from_adapter(args.bundle, args.source_results / "publish")
    base = causal_model(built.model)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    built.model.train()
    modules = {index: base.model.layers[index].mlp.experts for index in [0, 3]}
    records, hooks = {index: [] for index in modules}, []
    for index, module in modules.items():
        def capture(module, inputs, output, layer=index):
            if output.requires_grad:
                frozen_inputs, frozen_output = tuple(value.detach() for value in inputs), output.detach()
                def receive(dy):
                    records[layer].append((*[value.cpu() for value in frozen_inputs], dy.detach().cpu(), frozen_output.cpu()))
                output.register_hook(receive)
        hooks.append(module.register_forward_hook(capture))
    batch = dataset[0]
    built.streamer.phase = "FORWARD"
    loss = loss_sum(built, batch, ce_chunk_size=128) / supervised_tokens(batch["labels"])
    built.streamer.phase = "BACKWARD"
    loss.backward()
    for hook in hooks:
        hook.remove()
    torch.save(records, args.output / "actual-activation-dy.pt")
    result = {"status": "RUNNING", "loss": float(loss.detach()), "layers": []}
    for index, module in modules.items():
        assert len(records[index]) == 1, (index, len(records[index]))
        row = qualify_layer(module, records[index][0], built.store)
        result["layers"].append(row)
        print(json.dumps(row), flush=True)
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
    result.update(status="PASS" if all(row["status"] == "PASS" for row in result["layers"]) else "FAIL", memory=memory_observation())
    built.streamer.close()
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
