"""Adaptive Jump Scanning (Spatial Path).

Contains both the (1) complexity-adaptive jump-index generator and a thin
list-of-directional-scan module used by the spatial branch.

Given a spatial feature map, we compute a per-pixel complexity map, derive an
adaptive jump stride (large in smooth regions, dense in textured regions) and
produce, per scanning direction, a *fixed-length* sequence ``K`` of positions
so the batched selective-scan SSM can run in a single pass.

Tensors
-------
* complexity : (B, 1, P, P) complexity map in [0, 1]
* indices    : (B, num_dirs, K) grid-linear positions to gather
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def spatial_complexity(x):
    """(B, C, H, W) -> (B, 1, H, W) smoothed [0,1] gradient complexity map."""
    B, C, H, W = x.shape
    dev = x.device
    kx = SOBEL_X.to(dev).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    ky = SOBEL_Y.to(dev).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    mag = (gx * gx + gy * gy + 1e-8).sqrt().mean(dim=1, keepdim=True)  # (B,1,H,W)
    mag = F.avg_pool2d(mag, kernel_size=3, stride=1, padding=1)
    denom = mag.view(B, -1).max(dim=1).values.view(B, 1, 1, 1) + 1e-8
    return (mag / denom).clamp(0, 1)


def direction_orders(size, num_dirs=4):
    """(num_dirs, size*size) flatten orders: row-major, rev, col-major, rev."""
    grid = torch.arange(size * size, dtype=torch.long).view(size, size)
    orders = [grid.reshape(-1), grid.reshape(-1).flip(0),
              grid.t().reshape(-1), grid.t().reshape(-1).flip(0)]
    return torch.stack(orders[:num_dirs], dim=0)


def _walk(stride):
    """stride: (B,L) -> visited (T,B) positions, -1 = terminated."""
    B = stride.shape[0]
    L = stride.shape[1]
    visited = torch.full((L, B), -1, dtype=torch.long, device=stride.device)
    pos = torch.zeros(B, dtype=torch.long, device=stride.device)
    active = torch.ones(B, dtype=torch.bool, device=stride.device)
    t = 0
    while active.any() and t < L:
        cur = pos.clamp(0, L - 1)
        visited[t] = torch.where(active, cur, torch.full_like(cur, -1))
        s = stride.gather(1, cur[:, None])[:, 0]
        nxt = cur + s
        keep = active & (nxt < L)
        pos = torch.where(keep, nxt, torch.zeros_like(nxt))
        active = keep
        t += 1
    return visited[:t]


def _compact(visited, comp_along, K):
    """Compact variable-length walks to exactly K positions per row.
    visited:(T,B); comp_along:(B,L) -> (B,K) linear positions in [0,L)."""
    B, L = comp_along.shape
    K = min(K, L)
    T = visited.shape[0]
    dev = visited.device
    if T < K:
        pad = torch.full((K - T, B), -1, dtype=visited.dtype, device=dev)
        visited = torch.cat([visited, pad], dim=0)
    cnt = (visited >= 0).sum(0).clamp(max=K)             # (B,)
    need = (K - cnt).clamp(min=0)
    vis_vals = visited[:K].clamp(0, L - 1).t()            # (B, K)
    vmask = visited >= 0
    seen_idx = torch.clamp(visited, 0, L - 1).t()
    seen = torch.zeros((B, L), dtype=torch.float, device=dev)
    seen.scatter_add_(1, seen_idx, vmask.float().t())
    scores = comp_along.masked_fill(seen > 0, -1e9)
    F_ = min(K, max(1, L))
    fill_cand = torch.topk(scores, F_, dim=1).indices      # (B, F_)
    cols = torch.arange(K, device=dev).view(1, K)
    take_b = cnt.view(-1, 1)
    assign_prefix = cols < take_b
    fill_pos = cols - take_b
    fill_state = (fill_pos >= 0) & (fill_pos < need.view(-1, 1))
    fill_val = torch.gather(fill_cand, 1, fill_pos.clamp(0, F_ - 1))
    seq = torch.where(assign_prefix, vis_vals,
                      torch.where(fill_state, fill_val, torch.zeros_like(vis_vals)))
    return seq


def adaptive_jump_indices(complexity, num_dirs=4, K=None, s_min=1, s_max=None, alpha=10.0):
    """(B,1,P,P) complexity -> (B, num_dirs, K) grid positions to gather.

    The returned positions index the *linearized* (row-major) feature map,
    so callers gather with ``seq = toks.gather(1, idx.unsqueeze(-1).expand(-1,-1,D))``.
    """
    B, _, H, W = complexity.shape
    L = H * W
    P = H
    K = K if K is not None else min(L // 2, L)
    P = H
    smax = s_max if s_max is not None else max(1, P // 2)
    orders = direction_orders(P, num_dirs).to(complexity.device)   # (D, L)
    cflat = complexity.view(B, -1)
    R = B * num_dirs
    order_rows = orders.unsqueeze(1).expand(num_dirs, B, L).reshape(R, L)
    co = torch.cat([cflat.gather(1, o.unsqueeze(0).expand(B, -1)) for o in orders], dim=0)  # (R,L)
    sig = torch.sigmoid(alpha * (co - 0.5))
    stride_f = s_min + (1 - sig) * (smax - s_min)
    stride = stride_f.to(torch.long).clamp(s_min, smax)
    visited = _walk(stride)
    idx = _compact(visited, co, K)                        # (R, K) linear positions in ordered coords
    # map ordered-coordinate positions back to actual grid-linear positions per row
    idx = order_rows.gather(1, idx)
    return idx.view(B, num_dirs, K)