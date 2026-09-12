"""EJS-Mamba3 - Efficient Jump-Scanning Mamba3 for HSI classification.

Proposed architecture:

    Input HSI (B, C, P, P)
      -> Mamba3 Stem                     -> (B, D, P, P) feature + (B, N, D) tokens
      -> Adaptive Jump Scanning Spatial  -> positional sequences + spatial path
      -> Sparse Top-K Router             -> (B num_dirs scores)
      -> Selected Mamba3 Experts         -> (B, D) pooled spatial feature
      -> Shared Spectral Mamba3 Path     -> (B, N, D) spectral features
     -> Lightweight Dynamic Fusion       -> (B, N, D)
      -> Residual + RMSNorm
      -> Mamba3 Core Blocks (n_blocks)
     -> Global Pooling -> Classification Head

The backbone is a selective state-space (Mamba) core. Only lightweight modules
of clear purpose are added on top (adaptive jump scan; sparse top-k routing;
shared spectral SSM; lightweight dynamic fusion), keeping the efficiency
advantage of Mamba.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3_stem import Mamba3Stem
from .adaptive_jump_scan import spatial_complexity, adaptive_jump_indices
from .sparse_router import SparseTopKRouter
from .spatial_experts import SpatialMambaExperts
from .spectral_mamba import SharedSpectralMamba3
from .lightweight_fusion import LightweightDynamicFusion
from .ssm_core import MambaCoreBlock, RMSNorm, count_parameters


class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


class EJMamba3(nn.Module):
    def __init__(
        self,
        n_bands,
        n_classes,
        patch=13,
        d_model=64,
        num_dirs=4,
        top_k=2,
        num_blocks=3,
        d_state=16,
        expand=2,
        d_conv=4,
        K_ratio=0.5,
        bidirectional_spectral=True,
        dropout=0.1,
        **kw,
    ):
        super().__init__()
        self.patch = patch
        self.d_model = d_model
        self.num_dirs = num_dirs
        self.top_k = top_k
        self.num_blocks = num_blocks
        self.N = patch * patch
        self.K = max(8, int(patch * patch * K_ratio))

        self.stem = Mamba3Stem(n_bands, d_model)
        self.router = SparseTopKRouter(d_model, num_experts=num_dirs, top_k=top_k)
        self.spatial_experts = SpatialMambaExperts(
            d_model, num_dirs=num_dirs, top_k=top_k,
            d_state=d_state, expand=expand, d_conv=d_conv)
        self.spectral = SharedSpectralMamba3(
            n_bands, d_model=d_model, d_state=max(4, d_state // 2),
            bidirectional=bidirectional_spectral,
            K_band=max(8, int(n_bands * kw.get("K_ratio", 0.5))))
        self.fusion = LightweightDynamicFusion(d_model)

        self.pre_norm = RMSNorm(d_model)
        self.core = nn.ModuleList([
            MambaCoreBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            for _ in range(num_blocks)
        ])
        self.head = ClassificationHead(d_model, n_classes)

    def forward(self, x, return_routes=False):
        """x: (B, C, P, P) -> logits (B, n_classes)."""
        B, C, P, P = x.shape
        mapped, toks = self.stem(x)                    # mapped (B,D,P,P); toks (B,N,D)
        n = self.N

        # ---- branching on complexity ----
        cmap = spatial_complexity(x)                 # (B,1,P,P)
        idx = adaptive_jump_indices(cmap, self.num_dirs, self.K)  # (B,num_dirs,K)

        logits, mask = self.router.select(toks)      # (B, dirs) scores + top-k mask
        g_spa, sel = self.spatial_experts(toks, idx, logits, mask, self.router.weight)
        f_spa = toks + g_spa.unsqueeze(1)            # (B,N,D)

        pix = x.permute(0, 2, 3, 1).reshape(B, n, C)
        f_spe = self.spectral(pix)                   # (B,N,D)

        fused = self.fusion(f_spa, f_spe)            # (B,N,D)
        fused = self.pre_norm(fused + toks)          # residual + RMSNorm

        for blk in self.core:
            fused = blk(fused)                       # (B,N,D)

        pooled = fused.mean(dim=1)
        logits = self.head(pooled)
        if return_routes:
            return logits, sel
        return logits

    @property
    def num_parameters(self):
        return count_parameters(self)


_model_cls = EJMamba3


def build_model(cfg, model_dict=None):
    m = EJMamba3(
        n_bands=cfg["n_bands"], n_classes=cfg["n_classes"], patch=cfg["patch"],
        d_model=cfg.get("d_model", 64), num_dirs=cfg.get("num_dirs", 4),
        top_k=cfg.get("top_k", 2), num_blocks=cfg.get("num_blocks", 3),
        d_state=cfg.get("d_state", 16), expand=cfg.get("expand", 2),
        d_conv=cfg.get("d_conv", 4),
    )
    return m