"""Shared Spectral Mamba3 Path.

Models spectral dependencies with a *single* shared (optionally bidirectional)
Mamba3 selective-SSM block rather than one expert per band-range. Every pixel
is treated as a small sequence over a *fixed-length, adaptively chosen* set of
spectral bands; the shared block maps those sequences into spectral features.

Adaptive band sampling (keeps the sequence length *fixed* for the batched SSM
while skipping low-information bands)::

    comp_c = mean_{|d|<=2} (x_c - x_{c+d})^2      # local spectral variation
    idx    = top-K(comp_c) per pixel              # K = K_band
    x_used = x[..., idx]                          # (B, N, K_band)

This mirrors adaptive jump band scanning: dense bands (high variation) are
kept, flat ones discarded, and every pixel uses the same sequence length so a
single batched SSM call suffices.

Embedding (cheap, no per-band network)::

    x[n, k, :] = x_used[n, k] * edge + band_bias[k]     (D-dimensional)

Memory is bounded by processing the pixel axis in non-overlapping chunks
(``chunk`` rows), each wrapped in ``torch.utils.checkpoint`` so the scan's
saved states are released per chunk instead of accumulating on the autograd
graph.

Tensors
-------
* pix  : (B, N=P*P, C)
* out  : (B, N, D)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .ssm_core import MambaCoreBlock


def spectral_variation(pix):
    """pix (B, N, C) -> (B, N, C) local spectral variation in [0,1]."""
    B, N, C = pix.shape
    csum = pix.new_zeros(B, N, C)
    for d in (-2, -1, 1, 2):
        shifted = torch.roll(pix, d, dims=-1)
        if d > 0:
            diff = pix[:, :, d:] - shifted[:, :, :-d]
            csum[:, :, d:] += diff * diff
        else:
            diff = pix[:, :, :d] - shifted[:, :, -d:]
            csum[:, :, :d] += diff * diff
    out = csum / 4.0
    denom = out.max(dim=-1, keepdim=True).values + 1e-8
    return (out / denom).clamp(0, 1)


class SharedSpectralMamba3(nn.Module):
    def __init__(self, n_bands, d_model=64, d_state=8, expand=1, d_conv=3,
                 bidirectional=True, K_band=None, chunk=1024, **kw):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.chunk = chunk
        self.K_band = min(K_band if K_band is not None else max(16, n_bands // 2),
                          n_bands)
        self.edge = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5))
        self.band_bias = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.mamba = MambaCoreBlock(d_model, d_state=d_state, expand=expand,
                                    d_conv=d_conv, **kw)
        self.out = nn.Linear(d_model, d_model)

    def _embed(self, pix_sel):
        """pix_sel (R, K) raw band values -> (R, K, D) embedding."""
        return pix_sel.unsqueeze(-1) * self.edge + self._bias

    def _process(self, pix_sel, bias):
        """(R, K), (R, K, D) -> (R, D) checkpointed shared SSM over bands."""
        x = pix_sel.unsqueeze(-1) * self.edge + bias        # (R,K,D)
        rows = self.norm(x)
        out = torch.utils.checkpoint.checkpoint(
            self.mamba, rows, use_reentrant=False)          # (R,K,D)
        return out.mean(dim=1)                               # (R,D)

    def forward(self, pix):
        """pix (B, N, C) -> (B, N, D)."""
        B, N, C = pix.shape
        K = self.K_band
        # adaptive per-pixel band sampling (fixed length K)
        comp = spectral_variation(pix.detach())              # (B,N,C)
        idx = torch.topk(comp, K, dim=-1).indices            # (B,N,K)
        pix_sel = torch.gather(pix, 2, idx)                  # (B,N,K)
        bias = torch.gather(self.band_bias.view(1, 1, C, -1).expand(B, N, C, -1),
                            2, idx.unsqueeze(-1).expand(B, N, K, self.d_model))  # (B,N,K,D)

        R = B * N
        flat = pix_sel.reshape(R, K)
        flat_bias = bias.reshape(R, K, self.d_model)
        outs = []
        for s in range(0, R, self.chunk):
            e = min(s + self.chunk, R)
            fwd = self._process(flat[s:e], flat_bias[s:e])
            if self.bidirectional:
                bwd = self._process(flat[s:e].flip(1), flat_bias[s:e].flip(1))
                fwd = 0.5 * (fwd + bwd)
            outs.append(fwd)
        return torch.cat(outs, dim=0).view(B, N, self.d_model)