"""Fast TorchScript TronBot for Apex 0.1s limit — component floodfill + BFS spacefill."""

from typing import Tuple

import torch
import torch.nn as nn

MAX_SF_DEPTH = 2
NCOMP = 512


@torch.jit.script
def _pat(b: int) -> int:
    tbl = [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    return tbl[b]


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
def _color(r: int, c: int) -> int:
    return (r ^ c) & 1


@torch.jit.script
def _num_fillable(red: int, black: int, startcolor: int) -> int:
    if startcolor:
        return 2 * min(red - 1, black) + (1 if black >= red else 0)
    return 2 * min(red, black - 1) + (1 if red >= black else 0)


@torch.jit.script
def _degree_flat(flat: torch.Tensor, h: int, w: int, r: int, c: int) -> int:
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
def _neighbor_bits(flat: torch.Tensor, h: int, w: int, r: int, c: int) -> int:
    bits = 0
    if r <= 0 or c <= 0 or _flat_get(flat, w, r - 1, c - 1):
        bits |= 1
    if r <= 0 or _flat_get(flat, w, r - 1, c):
        bits |= 2
    if r <= 0 or c >= w - 1 or _flat_get(flat, w, r - 1, c + 1):
        bits |= 4
    if c >= w - 1 or _flat_get(flat, w, r, c + 1):
        bits |= 8
    if r >= h - 1 or c >= w - 1 or _flat_get(flat, w, r + 1, c + 1):
        bits |= 16
    if r >= h - 1 or _flat_get(flat, w, r + 1, c):
        bits |= 32
    if r >= h - 1 or c <= 0 or _flat_get(flat, w, r + 1, c - 1):
        bits |= 64
    if c <= 0 or _flat_get(flat, w, r, c - 1):
        bits |= 128
    return bits


@torch.jit.script
def _potential_articulation(flat: torch.Tensor, h: int, w: int, r: int, c: int) -> int:
    b = _neighbor_bits(flat, h, w, r, c)
    if b < 0 or b >= 256:
        return 0
    return _pat(b)


@torch.jit.script
def _recalc_components(
    flat: torch.Tensor,
    h: int,
    w: int,
    c: torch.Tensor,
    cedges: torch.Tensor,
    red: torch.Tensor,
    black: torch.Tensor,
) -> int:
    equiv = torch.zeros(512, dtype=torch.int32)
    c.zero_()
    cedges.zero_()
    red.zero_()
    black.zero_()
    nextclass = 1
    idx = w + 1
    mapbottom = w * (h - 1) - 1
    while idx < mapbottom:
        if _flat_get(flat, w, idx // w, idx % w):
            idx += 1
            continue
        cup = 0
        if idx >= w:
            cup = int(equiv[int(c[idx - w].item())].item())
        cleft = 0
        if idx % w != 0:
            cleft = int(equiv[int(c[idx - 1].item())].item())
        if cup == 0 and cleft == 0:
            equiv[nextclass] = nextclass
            c[idx] = nextclass
            nextclass += 1
        elif cup == cleft:
            c[idx] = cup
        else:
            if cleft == 0 or (cup != 0 and cup < cleft):
                c[idx] = cup
                if cleft != 0:
                    k = 0
                    while k < nextclass:
                        if int(equiv[k].item()) == cleft:
                            equiv[k] = cup
                        k += 1
            else:
                c[idx] = cleft
                if cup != 0:
                    k = 0
                    while k < nextclass:
                        if int(equiv[k].item()) == cup:
                            equiv[k] = cleft
                        k += 1
        idx += 1
        if idx % w == w - 1:
            idx += 2
    j = 1
    while j < h - 1:
        i = 1
        while i < w - 1:
            idx = j * w + i
            e = int(equiv[int(c[idx].item())].item())
            c[idx] = e
            cedges[e] += _degree_flat(flat, h, w, j, i)
            if _color(j, i):
                red[e] += 1
            else:
                black[e] += 1
            i += 1
        j += 1
    return nextclass


@torch.jit.script
def _comp_at(cgrid: torch.Tensor, w: int, r: int, col: int) -> int:
    return int(cgrid[r * w + col].item())


@torch.jit.script
def _fillablearea(cgrid: torch.Tensor, red: torch.Tensor, black: torch.Tensor, w: int, r: int, col: int) -> int:
    comp = _comp_at(cgrid, w, r, col)
    return _num_fillable(int(red[comp].item()), int(black[comp].item()), _color(r, col))


@torch.jit.script
def _connectedvalue(cgrid: torch.Tensor, cedges: torch.Tensor, w: int, r: int, col: int) -> int:
    comp = _comp_at(cgrid, w, r, col)
    return int(cedges[comp].item())


@torch.jit.script
def _score_cell(
    flat: torch.Tensor,
    h: int,
    w: int,
    cgrid: torch.Tensor,
    cedges: torch.Tensor,
    red: torch.Tensor,
    black: torch.Tensor,
    r: int,
    col: int,
) -> int:
    v = _connectedvalue(cgrid, cedges, w, r, col) + _fillablearea(cgrid, red, black, w, r, col)
    v -= 2 * _degree_flat(flat, h, w, r, col)
    v -= 4 * _potential_articulation(flat, h, w, r, col)
    return v


@torch.jit.script
def _pick_1ply_scored(
    flat: torch.Tensor,
    h: int,
    w: int,
    br: int,
    bc: int,
    cgrid: torch.Tensor,
    cedges: torch.Tensor,
    red: torch.Tensor,
    black: torch.Tensor,
) -> int:
    bestm = 1
    bestv = -1000000000
    m = 0
    while m < 4:
        nr, nc = _next(br, bc, m)
        if nr <= 0 or nc <= 0 or nr >= h - 1 or nc >= w - 1:
            m += 1
            continue
        if _flat_get(flat, w, nr, nc):
            m += 1
            continue
        sc = _score_cell(flat, h, w, cgrid, cedges, red, black, nr, nc)
        if sc > bestv:
            bestv = sc
            bestm = m
        m += 1
    return bestm


@torch.jit.script
def _bfs_reachable(flat: torch.Tensor, h: int, w: int, sr: int, sc: int, limit: int) -> int:
    seen = torch.zeros(h * w, dtype=torch.int32)
    q = torch.zeros(h * w, dtype=torch.int32)
    head = 0
    tail = 0
    idx = sr * w + sc
    seen[idx] = 1
    q[tail] = idx
    tail += 1
    count = 0
    while head < tail:
        cur = int(q[head].item())
        head += 1
        count += 1
        if count >= limit:
            return count
        cr = cur // w
        cc = cur % w
        m = 0
        while m < 4:
            nr, nc = _next(cr, cc, m)
            if nr <= 0 or nc <= 0 or nr >= h - 1 or nc >= w - 1:
                m += 1
                continue
            ni = nr * w + nc
            if seen[ni] or _flat_get(flat, w, nr, nc):
                m += 1
                continue
            seen[ni] = 1
            q[tail] = ni
            tail += 1
            m += 1
    return count


@torch.jit.script
def _pick_spacefill(flat: torch.Tensor, h: int, w: int, br: int, bc: int) -> int:
    bestm = 1
    bestv = 0
    m = 0
    while m < 4:
        nr, nc = _next(br, bc, m)
        if nr <= 0 or nc <= 0 or nr >= h - 1 or nc >= w - 1:
            m += 1
            continue
        if _flat_get(flat, w, nr, nc):
            m += 1
            continue
        _flat_set(flat, w, nr, nc, 1)
        v = _bfs_reachable(flat, h, w, nr, nc, 400)
        _flat_set(flat, w, nr, nc, 0)
        if v > bestv:
            bestv = v
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
    c = torch.zeros(h * w, dtype=torch.int32)
    cedges = torch.zeros(512, dtype=torch.int32)
    red = torch.zeros(512, dtype=torch.int32)
    black = torch.zeros(512, dtype=torch.int32)
    _recalc_components(flat, h, w, c, cedges, red, black)
    mycomp = _comp_at(c, w, hy, hx)
    oppcomp = _comp_at(c, w, oy, ox)
    if mycomp == oppcomp:
        internal = _pick_1ply_scored(flat, h, w, hy, hx, c, cedges, red, black)
    else:
        internal = _pick_spacefill(flat, h, w, hy, hx)
    action = _internal_to_action(internal)
    out = torch.full((4,), -1e6)
    out[action] = 1e6
    return out


class TronBotSubmit(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs_forward(obs)
