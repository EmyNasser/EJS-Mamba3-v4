"""Shared selective-scan (SSM) core.

Pure-PyTorch implementation of the selective state-space recurrence used by
the Mamba family (fused Mamba1/2/3 kernels all reduce to this scan):

    h_{t} = dA_t * h_{t-1} + dB_t * x_t      dA = exp(dt*A), dB = dt*B
    y_{t} = (C_t * h_{t}).sum(N)

The scan is wrapped in one ``torch.autograd.Function`` with a closed-form
backward so the whole sequence is a single autograd node (much faster to
train than an unrolled graph of ``L`` nodes).

Using this single implementation for *every* model in the fair comparison
guarantees the SSM primitive is numerically identical across all five models,
so reported differences reflect architecture, not the scan implementation.

Dimension layout
----------------
* u   : (B, L, D)  input
* dtA : (B, L, D, N) = exp(dt * A)
* dB  : (B, L, D, N) = dt * B
* C   : (B, L, N)
* y   : (B, L, D)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = [
    "SsmScan",
    "ssm_scan",
    "ssm_reference_scan",
    "RMSNorm",
    "MambaCoreBlock",
]


class SsmScan(Function):
    @staticmethod
    def forward(ctx, u, dtA, dB, C):
        Bs, L, D = u.shape
        N = dtA.shape[-1]
        h = torch.zeros(Bs, D, N, device=u.device, dtype=u.dtype)
        hs = torch.empty(Bs, L, D, N, device=u.device, dtype=u.dtype)
        y = torch.empty(Bs, L, D, device=u.device, dtype=u.dtype)
        xU = u.unsqueeze(-1)  # (B,L,D,1)
        for t in range(L):
            h = dtA[:, t] * h + dB[:, t] * xU[:, t]
            hs[:, t] = h
            y[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)
        ctx.save_for_backward(u, dtA, dB, C, hs)
        return y

    @staticmethod
    def backward(ctx, gy):
        u, dtA, dB, C, hs = ctx.saved_tensors
        Bs, L, D = u.shape
        N = dtA.shape[-1]
        g_u = torch.zeros_like(u)
        g_A = torch.zeros_like(dtA)
        g_B = torch.zeros_like(dB)
        g_C = torch.zeros_like(C)
        err = torch.zeros(Bs, D, N, device=u.device, dtype=u.dtype)
        for t in reversed(range(L)):
            gy_t = gy[:, t].unsqueeze(-1)       # (B,D,1)
            C_t = C[:, t].unsqueeze(1)          # (B,1,N)
            err = err + gy_t * C_t
            g_C[:, t] = (gy[:, t].unsqueeze(-1) * hs[:, t]).sum(1)   # (B,N)
            h_prev = hs[:, t - 1] if t > 0 else 0.0
            g_A[:, t] = err * h_prev            # (B,D,N)
            g_B[:, t] = err * u[:, t].unsqueeze(-1)   # (B,D,N)
            g_u[:, t] = (err * dB[:, t]).sum(-1)      # (B,D)
            err = err * dtA[:, t]               # propagate to h_{t-1}
        return g_u, g_A, g_B, g_C


def ssm_scan(u, dtA, dB, C):
    return SsmScan.apply(u, dtA, dB, C)


def ssm_reference_scan(u, dtA, dB, C):
    """Reference scan building an autograd graph (used for gradient checks)."""
    Bs, L, D = u.shape
    N = dtA.shape[-1]
    h = torch.zeros(Bs, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dtA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=1)


def _dt_init(d_inner, dt_rank):
    dt_min, dt_max = 0.001, 0.1
    dt = torch.rand(d_inner, 1) * (dt_max - dt_min) + dt_min
    return dt


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out * self.weight


class MambaCoreBlock(nn.Module):
    """Gated selective-SSM mixing block (the Mamba3 core).

    x : (B, L, D) -> x + update   (residual inside)

        nx     = RMSNorm(x)
        u, v   = split(Linear(nx))               d -> expand*d each
        u      = depthwise causal Conv1d(u)
        dt, B, C = split(Linear(u))
        dt     = softplus(dt_proj(dt))          -> (B, L, expand*d)
        y      = scan(u, dt, A, B, C)
        out    = out_proj( y * silu(v) )
        return x + out
    """

    def __init__(self, d_model, d_state=16, expand=2, d_conv=3, dt_rank=-1,
                 bias=False, norm_epsilon=1e-6, **kwargs):
        super().__init__()
        d_inner = int(expand * d_model)
        dt_rank = d_inner if dt_rank == -1 else dt_rank
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.norm = RMSNorm(d_model, norm_epsilon)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                              padding=d_conv - 1, bias=not bias)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init = _dt_init(d_inner, dt_rank)
        self.dt_bias = nn.Parameter(dt_init.squeeze(-1).clone())
        A = torch.rand(d_inner, d_state) + 0.5        # positive decay rates
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x, dims=None, **kwargs):
        B_, L, _D = x.shape
        nx = self.norm(x)
        inp = self.in_proj(nx)                        # (B,L,2*d_inner)
        u, v = inp.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B,L,d_inner)
        xz = self.x_proj(u)                           # (B,L,dt_rank+2N)
        dt_x, B_par, C_par = torch.split(
            xz, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_x) + self.dt_bias)   # (B,L,d_inner)
        A_mat = -torch.exp(self.A_log)                # (d_inner,N)
        dtA = torch.exp(dt.unsqueeze(-1) * A_mat.unsqueeze(0).unsqueeze(0))  # (B,L,d_inner,N)
        dB = dt.unsqueeze(-1) * B_par.unsqueeze(2)    # (B,L,d_inner,N)
        y = ssm_scan(u, dtA, dB, C_par)               # (B,L,d_inner)
        y = y + u * (self.D.unsqueeze(0).unsqueeze(0))    # D term
        y = y * F.silu(v)
        out = self.out_proj(y)                        # (B,L,D)
        return x + out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)