"""Spatial Mamba3 Experts (selected experts).

Each expert is a single directional Mamba3 selective-SSM block processing a
fixed-length jump-scanned sequence of K tokens. The Sparse Top-K Router scores
the experts; the Mixture-Of-Experts execution is *grouped* per expert so that
a given expert only runs forward on the batch samples routed to it. The total
forward work therefore scales with (Top_K / num_experts) instead of executing
all experts and masking outputs.

Tensors
-------
* toks : (B, N=D=D, D)
* idx  : (B, num_dirs, K) jump positions to gather
* logits/mask from router -> g (B, D),  route_reg (B, num_dirs)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm_core import MambaCoreBlock


class SpatialMambaExperts(nn.Module):
    def __init__(self, d_model=64, num_dirs=4, top_k=2, d_state=16, expand=2, d_conv=4, **kw):
        super().__init__()
        self.num_dirs = num_dirs
        self.top_k = top_k
        self.experts = nn.ModuleList([
            MambaCoreBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv, **kw)
            for _ in range(num_dirs)
        ])

    def forward(self, toks, idx, logits, mask, weight):
        """toks:(B,N,D) idx:(B,dirs,K) -> weighted spatial feature (B,D) + sel mask."""
        B, N, D = toks.shape
        sel = torch.topk(logits, min(self.top_k, self.num_dirs), dim=1).indices  # (B,k)
        sel_mask = torch.zeros_like(logits)
        sel_mask = sel_mask.scatter(1, sel, torch.ones_like(sel_mask))
        sel_mask = sel_mask.bool()                     # (B,num_dirs)
        w = weight(logits, sel_mask.float())           # (B,num_dirs)
        g = torch.zeros(B, D, device=toks.device, dtype=toks.dtype)
        for e in range(self.num_dirs):
            rows = sel_mask[:, e].nonzero(as_tuple=False).view(-1)
            if rows.numel() == 0:
                continue
            s_idx = idx[rows, e]                       # (se, K)
            sub = toks[rows]                           # (se, N, D)
            seq = sub.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, D))  # (se,K,D)
            seq = self.experts[e](seq)                 # (se,K,D)
            fe = seq.mean(dim=1)                       # (se,D)
            g.index_add_(0, rows, w[rows, e].unsqueeze(-1) * fe)
        return g, sel.long()