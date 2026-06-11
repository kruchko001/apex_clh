import torch
from typing import Tuple


@torch.jit.script
def _spread_front(front: torch.Tensor, blocked: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    free = (~blocked) & (~seen)
    up = torch.zeros_like(front)
    down = torch.zeros_like(front)
    left = torch.zeros_like(front)
    right = torch.zeros_like(front)
    up[1:, :] = front[:-1, :]
    down[:-1, :] = front[1:, :]
    left[:, 1:] = front[:, :-1]
    right[:, :-1] = front[:, 1:]
    return (up | down | left | right) & free


@torch.jit.script
def _head_rc(head: torch.Tensor) -> Tuple[int, int]:
    h, w = head.size(0), head.size(1)
    flat = head.view(-1).float()
    idx = flat.argmax()
    if flat[idx] <= 0.5:
        return -1, -1
    return int(idx // w), int(idx % w)


@torch.jit.script
def _seed_front(hy: int, hx: int, h: int, w: int, blocked: torch.Tensor) -> torch.Tensor:
    front = torch.zeros((h, w), dtype=torch.bool, device=blocked.device)
    if hy > 0 and not blocked[hy - 1, hx]:
        front[hy - 1, hx] = True
    if hy + 1 < h and not blocked[hy + 1, hx]:
        front[hy + 1, hx] = True
    if hx > 0 and not blocked[hy, hx - 1]:
        front[hy, hx - 1] = True
    if hx + 1 < w and not blocked[hy, hx + 1]:
        front[hy, hx + 1] = True
    return front


@torch.jit.script
def _spread_front_batched(front: torch.Tensor, blocked: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    free = (~blocked) & (~seen)
    up = torch.zeros_like(front)
    down = torch.zeros_like(front)
    left = torch.zeros_like(front)
    right = torch.zeros_like(front)
    up[:, 1:, :] = front[:, :-1, :]
    down[:, :-1, :] = front[:, 1:, :]
    left[:, :, 1:] = front[:, :, :-1]
    right[:, :, :-1] = front[:, :, 1:]
    return (up | down | left | right) & free


@torch.jit.script
def _bfs_reachable_batched(blocked: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    b, h, w = blocked.size(0), blocked.size(1), blocked.size(2)
    flat = heads.view(b, -1).float()
    idx = flat.argmax(dim=1)
    has = flat.gather(1, idx.unsqueeze(1)).squeeze(1) > 0.5
    hy = idx // w
    hx = idx % w

    reachable = torch.zeros((b, h, w), dtype=torch.bool, device=blocked.device)
    front = torch.zeros((b, h, w), dtype=torch.bool, device=blocked.device)
    for i in range(b):
        if has[i]:
            y = int(hy[i])
            x = int(hx[i])
            reachable[i, y, x] = True
            front[i] = _seed_front(y, x, h, w, blocked[i])

    for _ in range(h * w):
        if not front.any():
            break
        reachable = reachable | front
        front = _spread_front_batched(front, blocked, reachable)
    return reachable


@torch.jit.script
def _bfs_reachable(blocked: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    return _bfs_reachable_batched(blocked.unsqueeze(0), head.unsqueeze(0)).squeeze(0)


@torch.jit.script
def _blocked_from_obs(obs: torch.Tensor) -> torch.Tensor:
    blocked = obs[:, 0] > 0.5
    for c in range(1, 5):
        blocked = blocked | (obs[:, c] > 0.5)
    return blocked


@torch.jit.script
def phase_separated(obs: torch.Tensor) -> torch.Tensor:
    blocked = _blocked_from_obs(obs)[0]
    h, w = blocked.size(0), blocked.size(1)
    my_head = obs[0, 3] > 0.5
    opp_head = obs[0, 4] > 0.5
    my_y, my_x = _head_rc(my_head)
    opp_y, opp_x = _head_rc(opp_head)
    if my_y < 0 or opp_y < 0:
        return torch.zeros((), dtype=torch.float32, device=obs.device)

    seen_my = torch.zeros((h, w), dtype=torch.bool, device=obs.device)
    seen_opp = torch.zeros((h, w), dtype=torch.bool, device=obs.device)
    seen_my[my_y, my_x] = True
    seen_opp[opp_y, opp_x] = True
    my_front = _seed_front(my_y, my_x, h, w, blocked)
    opp_front = _seed_front(opp_y, opp_x, h, w, blocked)

    for _ in range(h * w):
        if not my_front.any() and not opp_front.any():
            break
        seen_my = seen_my | my_front
        seen_opp = seen_opp | opp_front
        my_front = _spread_front(my_front, blocked, seen_my)
        opp_front = _spread_front(opp_front, blocked, seen_opp)

    overlap = (seen_my & seen_opp).any()
    if overlap:
        return torch.zeros((), dtype=torch.float32, device=obs.device)
    return torch.ones((), dtype=torch.float32, device=obs.device)


@torch.jit.script
def mask_unreachable_as_walls(obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    blocked = _blocked_from_obs(obs)
    reachable = _bfs_reachable_batched(blocked, obs[:, 3] > 0.5)
    out[:, 0] = torch.where(reachable, out[:, 0], torch.ones_like(out[:, 0]))
    return out


@torch.jit.script
def crop_play_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, :, 1:31, 1:31]


@torch.jit.script
def extract_non_stationary_input(obs: torch.Tensor) -> torch.Tensor:
    return crop_play_obs(obs)


@torch.jit.script
def extract_stationary_input(obs: torch.Tensor) -> torch.Tensor:
    cropped = crop_play_obs(obs)
    masked = mask_unreachable_as_walls(cropped)
    unreachable = (masked[:, 0:1] - cropped[:, 0:1]).clamp(0.0, 1.0)
    head = cropped[:, 3:4]
    return torch.cat([unreachable, head], dim=1)


@torch.jit.script
def _opposite_action(action: int) -> int:
    if action == 0:
        return 2
    if action == 1:
        return 3
    if action == 2:
        return 0
    return 1


@torch.jit.script
def _infer_facing_single(trail: torch.Tensor, head: torch.Tensor) -> int:
    hy, hx = _head_rc(head)
    if hy < 0:
        return -1
    h, w = head.size(0), head.size(1)
    if hy > 0 and trail[hy - 1, hx] > 0.5:
        return 2
    if hy + 1 < h and trail[hy + 1, hx] > 0.5:
        return 0
    if hx > 0 and trail[hy, hx - 1] > 0.5:
        return 1
    if hx + 1 < w and trail[hy, hx + 1] > 0.5:
        return 3
    return -1


@torch.jit.script
def _neighbor_invalid_single(walls: torch.Tensor, hy: int, hx: int, action: int) -> bool:
    h, w = walls.size(0), walls.size(1)
    if action == 0:
        if hy <= 0:
            return True
        if walls[hy - 1, hx] > 0.5:
            return True
        return False
    if action == 1:
        if hx + 1 >= w:
            return True
        if walls[hy, hx + 1] > 0.5:
            return True
        return False
    if action == 2:
        if hy + 1 >= h:
            return True
        if walls[hy + 1, hx] > 0.5:
            return True
        return False
    if hx <= 0:
        return True
    if walls[hy, hx - 1] > 0.5:
        return True
    return False


@torch.jit.script
def _action_mask_single(walls: torch.Tensor, head: torch.Tensor, trail: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(4, dtype=torch.bool)
    hy, hx = _head_rc(head)
    if hy < 0:
        return mask
    facing = _infer_facing_single(trail, head)
    if facing >= 0:
        mask[_opposite_action(facing)] = False
    for a in range(4):
        if _neighbor_invalid_single(walls, hy, hx, a):
            mask[a] = False
    if mask.any():
        return mask
    if facing >= 0:
        out = torch.zeros(4, dtype=torch.bool)
        out[facing] = True
        return out
    out = torch.zeros(4, dtype=torch.bool)
    out[0] = True
    return out


@torch.jit.script
def compute_action_mask(obs: torch.Tensor, stationary: bool) -> torch.Tensor:
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    cropped = crop_play_obs(obs)
    b = cropped.size(0)
    trail = cropped[:, 1] > 0.5
    if stationary:
        prep = extract_stationary_input(obs)
        walls = (cropped[:, 0] > 0.5) | (prep[:, 0] > 0.5) | trail
        heads = prep[:, 1] > 0.5
    else:
        walls = (cropped[:, 0] > 0.5) | trail | (cropped[:, 2] > 0.5)
        heads = cropped[:, 3] > 0.5
    out = torch.zeros(b, 4, dtype=torch.bool, device=obs.device)
    for i in range(b):
        out[i] = _action_mask_single(walls[i], heads[i], trail[i])
    return out
