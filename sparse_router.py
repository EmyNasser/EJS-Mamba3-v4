"""Sparse Top-K Router.

A lightweight router that scores the ``num_experts`` spatial Mamba3 experts
from the pooled spatial tokens, then keeps only the Top-K experts. The router
returns an importance score per expert; the caller (spatial branch) executes
*only* the selected experts (grouped by routing decision) so sparsity yields a
real computational saving rather than a masked all-experts evaluation.

Design
------
* pool tokens -> router MLP -> ``num_experts`` scores
* ``top_k`` configurable (default select 2 of 4)
* scores->softmax->routing weights for the *selected* experts

Tensors
-------
* x (B, N, D)  ->  _router.forward(x) -> (B, num_experts) logits
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTopKRouter(nn.Module):
    def __init__(self, d_model=64, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts),
        )
        self._router_load = torch.zeros(num_experts)  # running load tracker
        self._routed = 0

    def score(self, x):
        """x: (B, N, D) -> (B, num_experts) logits."""
        pooled = x.mean(dim=1)                       # (B, D)
        return self.net(pooled)

    def select(self, x):
        """Return (logits, topk_mask). Top-k entries of mask are 1 (train & eval)."""
        logits = self.score(x)                       # (B, num_experts)
        k = self.top_k
        if k >= self.num_experts:
            mask = torch.ones_like(logits)
        else:
            sel = torch.topk(logits, k, dim=1).indices
            mask = torch.zeros_like(logits)
            mask = mask.scatter(1, sel, torch.ones_like(mask))
            self._router_load += mask.sum(0).detach().cpu().float()
            self._routed += mask.sum(0).detach().cpu().float().sum().item()
        return logits, mask

    def weight(self, logits, mask):
        """Softmax over masked logits -> weights summing to 1 over selected."""
        masked = logits.masked_fill(mask <= 0, -1e9)
        return F.softmax(masked, dim=-1)             # (B, num_experts)