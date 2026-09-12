"""Lightweight Dynamic Fusion.

Adaptively combines the spatial ``F_spa`` and spectral ``F_spe`` features with
a small channel-gating mechanism (no heavy multi-stage attention). A compact
projection reads both branches, produces per-token mixing weights and a light
dynamic-coupling term, then re-combines.

Original  (B, N, 2D) concat -> gate -> (B, N, D)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LightweightDynamicFusion(nn.Module):
    def __init__(self, d_model=64, d=16):
        super().__init__()
        self.d_model = d_model
        self.gate_proj = nn.Linear(2 * d_model, 2, bias=True)
        self.dyn = nn.Linear(d_model, d_model)
        self.gate_act = nn.Softmax(dim=-1)

    def forward(self, spa, spe):
        """spa,spe: (B, N, D) -> fused (B, N, D)."""
        cat = torch.cat([spa, spe], dim=-1)          # (B,N,2D)
        w = self.gate_act(self.gate_proj(cat))        # (B,N,2); sum=1 per token
        fused = w[..., :1] * spa + w[..., 1:] * spe   # (B,N,D)
        d = torch.sigmoid(self.dyn(fused))            # (B,N,D) dynamic perturbation
        out = fused * (1.0 + 0.1 * d)
        return out