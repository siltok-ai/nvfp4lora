"""实际专家算子＋标准AdamW：新命中、从未命中、历史动量与恢复后的归纳检查。"""

import tempfile
from types import SimpleNamespace
import unittest

import torch

from expert_fixture import make_operator_store
from nvfp4lora.model.packed_experts import PackedExperts
from nvfp4lora.ops.experts_fn import ExpertOpConfig
from nvfp4lora.validation.route_audit import ExpertRouteAudit


@unittest.skipUnless(torch.cuda.is_available(), "actual packed expert update semantics")
class RouteAuditTests(unittest.TestCase):
    def test_never_routed_and_historical_momentum_across_audit_resume(self):
        torch.manual_seed(715)
        with tempfile.TemporaryDirectory() as temporary:
            store, streamer = make_operator_store(temporary, experts=4, hidden=32, intermediate=16, device="cuda")
            config = SimpleNamespace(num_experts=4, hidden_size=32, moe_intermediate_size=16)
            module = PackedExperts(config, 0, ExpertOpConfig(streamer, chunk_size=2), rank=2, device="cuda")
            built = SimpleNamespace(model=module, trainable=dict(module.named_parameters()))
            audit = ExpertRouteAudit(built)
            optimizer = torch.optim.AdamW(module.parameters(), lr=.002, weight_decay=0, foreach=False)
            x = torch.randn(3, 32, dtype=torch.bfloat16, device="cuda")
            weights = torch.ones(3, 1, device="cuda")
            dy = torch.randn_like(x)
            reports = []
            for expert in [0, 1]:
                audit.begin_step()
                module(x, torch.full((3, 1), expert, device="cuda", dtype=torch.long), weights).backward(dy)
                reports.append(audit.before_update())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                reports[-1].update(audit.after_update(optimizer))
            self.assertEqual(reports[-1]["never_routed_expert_layer_pairs"], 2)
            self.assertEqual(reports[-1]["historical_inactive_expert_layer_pairs"], 1)
            self.assertGreater(reports[-1]["historical_inactive_changed_parameter_rows"], 0)
            state = audit.state_dict()
            audit.close()
            resumed = ExpertRouteAudit(built, state)
            self.assertEqual(state, resumed.state_dict())
            resumed.begin_step()
            module(x, torch.ones(3, 1, device="cuda", dtype=torch.long), weights).backward(dy)
            resumed.before_update()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            after = resumed.after_update(optimizer)
            self.assertEqual(after["never_routed_expert_layer_pairs"], 2)
            resumed.begin_step()
            module(x, torch.ones(3, 1, device="cuda", dtype=torch.long), weights).backward(dy)
            module.gate_proj.lora_A.grad[3].fill_(1)
            with self.assertRaisesRegex(ValueError, "Unrouted"):
                resumed.before_update()
            resumed.close()


if __name__ == "__main__":
    unittest.main()
