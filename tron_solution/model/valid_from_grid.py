import torch


@torch.jit.script
def valid_mask_from_grid(grid: torch.Tensor) -> torch.Tensor:
    b = grid.size(0)
    c = grid.size(1)
    h = grid.size(2)
    w = grid.size(3)
    n_frames = c // 4
    last = (n_frames - 1) * 4
    last_head = grid[:, last + 2]
    idx1 = last_head.view(b, -1).argmax(dim=1)
    r1 = idx1 // w
    c1 = idx1 % w
    if n_frames >= 2:
        prev = (n_frames - 2) * 4
        prev_head = grid[:, prev + 2]
        idx0 = prev_head.view(b, -1).argmax(dim=1)
        r0 = idx0 // w
        c0 = idx0 % w
    else:
        r0 = r1
        c0 = c1
    cur = torch.full((b,), 2, dtype=torch.long, device=grid.device)
    moved = (r0 != r1) | (c0 != c1)
    dr = r1 - r0
    dc = c1 - c0
    cur = torch.where(moved & (dr == -1) & (dc == 0), torch.zeros_like(cur), cur)
    cur = torch.where(moved & (dr == 0) & (dc == 1), torch.ones_like(cur), cur)
    cur = torch.where(moved & (dr == 1) & (dc == 0), torch.full_like(cur, 2), cur)
    cur = torch.where(moved & (dr == 0) & (dc == -1), torch.full_like(cur, 3), cur)
    blocked = grid[:, last] + grid[:, last + 1]
    ar = torch.arange(b, device=grid.device)

    nr0 = r1 - 1
    nc0 = c1
    ok0 = (nr0 >= 0) & (nr0 < h) & (nc0 >= 0) & (nc0 < w)
    ok0 = ok0 & (cur != 2)
    ok0 = ok0 & (blocked[ar, nr0, nc0] < 0.5)

    nr1 = r1
    nc1 = c1 + 1
    ok1 = (nr1 >= 0) & (nr1 < h) & (nc1 >= 0) & (nc1 < w)
    ok1 = ok1 & (cur != 3)
    ok1 = ok1 & (blocked[ar, nr1, nc1] < 0.5)

    nr2 = r1 + 1
    nc2 = c1
    ok2 = (nr2 >= 0) & (nr2 < h) & (nc2 >= 0) & (nc2 < w)
    ok2 = ok2 & (cur != 0)
    ok2 = ok2 & (blocked[ar, nr2, nc2] < 0.5)

    nr3 = r1
    nc3 = c1 - 1
    ok3 = (nr3 >= 0) & (nr3 < h) & (nc3 >= 0) & (nc3 < w)
    ok3 = ok3 & (cur != 1)
    ok3 = ok3 & (blocked[ar, nr3, nc3] < 0.5)

    mask = torch.stack([ok0.float(), ok1.float(), ok2.float(), ok3.float()], dim=1)
    none_ok = mask.sum(dim=1) < 0.5
    fallback = torch.zeros_like(mask)
    fallback.scatter_(1, cur.unsqueeze(1), 1.0)
    mask = torch.where(none_ok.unsqueeze(1), fallback, mask)
    return mask
