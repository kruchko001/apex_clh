"""Original PT TronBot — 1-ply degree greedy only."""

from typing import Tuple

import torch
import torch.nn as nn


@torch.jit.script
def _head(ch: torch.Tensor) -> Tuple[int, int]:
    flat = ch.reshape(-1)
    idx = int(torch.argmax(flat).item())
    if flat[idx] <= 0.5:
        return -1, -1
    w = ch.size(1)
    return idx // w, idx % w


@torch.jit.script
def _internal_to_action(m: int) -> int:
    if m == 0:
        return 0
    if m == 1:
        return 2
    if m == 2:
        return 1
    return 3


@torch.jit.script
def _flat_get(flat: torch.Tensor, w: int, r: int, c: int) -> int:
    return int(flat[r * w + c].item())


@torch.jit.script
def _flat_set(flat: torch.Tensor, w: int, r: int, c: int, v: int):
    flat[r * w + c] = v


@torch.jit.script
def _next(r: int, c: int, m: int) -> Tuple[int, int]:
    if m == 0:
        return r - 1, c
    if m == 1:
        return r + 1, c
    if m == 2:
        return r, c + 1
    return r, c - 1


@torch.jit.script
def _degree(flat: torch.Tensor, h: int, w: int, r: int, c: int) -> int:
    d = 4
    if r <= 0 or _flat_get(flat, w, r - 1, c):
        d -= 1
    if r >= h - 1 or _flat_get(flat, w, r + 1, c):
        d -= 1
    if c <= 0 or _flat_get(flat, w, r, c - 1):
        d -= 1
    if c >= w - 1 or _flat_get(flat, w, r, c + 1):
        d -= 1
    return d


@torch.jit.script
def _greedy_move(flat: torch.Tensor, h: int, w: int, br: int, bc: int) -> int:
    bestm = 1
    bestv = -1
    m = 0
    while m < 4:
        nr, nc = _next(br, bc, m)
        if nr <= 0 or nc <= 0 or nr >= h - 1 or nc >= w - 1:
            m += 1
            continue
        if _flat_get(flat, w, nr, nc):
            m += 1
            continue
        sc = 4 - _degree(flat, h, w, nr, nc)
        if sc > bestv:
            bestv = sc
            bestm = m
        m += 1
    return bestm


@torch.jit.script
def obs_forward(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 4:
        x = obs[0]
    else:
        x = obs
    h = x.size(1)
    w = x.size(2)
    flat = ((x[0] + x[1] + x[2]) > 0.5).to(torch.int32).reshape(-1).clone()
    i = 0
    while i < w:
        _flat_set(flat, w, 0, i, 1)
        _flat_set(flat, w, h - 1, i, 1)
        i += 1
    j = 0
    while j < h:
        _flat_set(flat, w, j, 0, 1)
        _flat_set(flat, w, j, w - 1, 1)
        j += 1
    hy, hx = _head(x[3])
    oy, ox = _head(x[4])
    if hy < 0:
        return torch.zeros(4)
    if oy < 0:
        oy, ox = 1, 1
        while oy < h - 1 and ox < w - 1 and _flat_get(flat, w, oy, ox):
            ox += 1
    _flat_set(flat, w, hy, hx, 1)
    _flat_set(flat, w, oy, ox, 1)
    internal = _greedy_move(flat, h, w, hy, hx)
    action = _internal_to_action(internal)
    out = torch.full((4,), -1e6)
    out[action] = 1e6
    return out


class TronBotSubmitGreedy(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs_forward(obs)
