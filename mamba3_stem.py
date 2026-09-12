"""Mamba3 Stem.

Lightweight learned embedding that converts a hyperspectral patch
``x: (B, C, P, P)`` into a spatial feature map ``(B, D, P, P)``
(and flat token sequence ``(B, P*P, D)``) suitable for the spatial and
spectral branches.

Design goals
------------
* 3x3 conv (not patchify) to *preserve* local spectral-spatial structure.
* BatchNorm + GELU for stable, cheap training.
* Param count ~ O(C*D) - the only place the raw band count is touched.

Tensors
-------
* input   : (B, n_bands, P, P)
* conv    : (B, D, P, P)
* output  : (B, N=P*P, D) token sequence
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Mamba3Stem(nn.Module):
    def __init__(self, n_bands, d_model=64, conv_kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(n_bands, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(d_model)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

    def forward(self, x):
        """x: (B, C, P, P) -> (B, N=P*P, D) token features + (B, D, P, P) map."""
        mapped = self.act(self.bn(self.conv(x)))
        mapped = mapped + self.proj(mapped)
        B, D, H, W = mapped.shape
        seq = mapped.reshape(B, D, H * W).transpose(1, 2)  # (B, N, D)
        return mapped, seq