"""Regenerate the GLM router examples with Transformers 5.23.0 / PyTorch 2.15.0."""

from types import SimpleNamespace

import torch
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteTopkRouter

CASES = [
    ("grouped", [2., -1., .2, -2., 1., .1, -1., .4], [-.9, .4, 0., 0., 0., .3, .1, .2], 2, 2, 1, 2.45),
    ("top_one", [.3, -.5, 2., -.2], [0., 1., -1., .1], 1, 2, 2, 2.45),
    ("saturation", [-100., -90., 100., 88., -80., -85., -70., -60.], [.1, .2, 0., -.1, 0., .1, .2, .3], 1, 2, 1, 3.1),
]
for name, logits, bias, top_k, groups, selected, scale in CASES:
    config = SimpleNamespace(num_experts_per_tok=top_k, num_local_experts=len(logits),
                             hidden_size=len(logits), routed_scaling_factor=scale,
                             n_group=groups, topk_group=selected, norm_topk_prob=True)
    router = Glm4MoeLiteTopkRouter(config).eval()
    with torch.no_grad():
        router.weight.copy_(torch.eye(len(logits)))
        router.e_score_correction_bias.copy_(torch.tensor(bias))
        _, weights, indices = router(torch.tensor([logits]))
    print(name, indices.tolist(), weights.tolist())
