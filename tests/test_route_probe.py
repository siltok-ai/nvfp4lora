"""C3 rank0的路由仍应记录；padding不能凭空增加专家命中。"""

import math
import unittest

import torch
from torch import nn

from nvfp4lora.model.packed_experts import PackedExperts
from nvfp4lora.validation.route_probe import MaskedRouteProbe


class ProbeExperts(PackedExperts):
    def __init__(self, layer):
        nn.Module.__init__(self)
        self.layer, self.rank = layer, 0

    def forward(self, hidden, indices, weights):
        return hidden


class RouteProbeTests(unittest.TestCase):
    def test_rank_zero_and_left_padding_are_handled(self):
        model = ProbeExperts(0)
        mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
        indices = torch.tensor([[3, 3], [0, 1], [1, 2], [3, 3], [3, 3], [2, 0]])
        probe = MaskedRouteProbe(model, mask, experts=4)
        model(torch.ones(6, 2), indices, torch.ones(6, 2))
        result = probe.result()["0"]
        self.assertEqual(result["counts"], [2, 2, 2, 0])
        self.assertEqual(result["valid_token_vectors"], 3)
        self.assertEqual(result["selected_routes"], 6)
        self.assertAlmostEqual(result["entropy_nats"], math.log(3))
        probe.close()
        model(torch.ones(6, 2), indices, torch.ones(6, 2))
        self.assertEqual(probe.result()["0"]["calls"], 1)

    def test_wrong_flattened_mask_is_rejected(self):
        model = ProbeExperts(0)
        probe = MaskedRouteProbe(model, torch.ones(2), experts=4)
        try:
            with self.assertRaisesRegex(ValueError, "padding mask"):
                model(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2))
        finally:
            probe.close()

    def test_missing_layer_is_rejected(self):
        model = nn.ModuleList([ProbeExperts(0), ProbeExperts(1)])
        probe = MaskedRouteProbe(model, torch.ones(2), experts=4)
        try:
            model[0](torch.ones(2, 2), torch.tensor([[0, 1], [1, 2]]), torch.ones(2, 2))
            with self.assertRaisesRegex(ValueError, "not observed"):
                probe.result()
        finally:
            probe.close()


if __name__ == "__main__":
    unittest.main()
