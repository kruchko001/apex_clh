import numpy as np
import torch

from tronbot.python.mytronbot import MyTronBot

DY = (-1, 0, 1, 0)
DX = (0, 1, 0, -1)


def _head_rc(ch: np.ndarray):
    flat = ch.reshape(-1)
    idx = int(flat.argmax())
    if flat[idx] <= 0.5:
        return -1, -1
    h, w = ch.shape
    return idx // w, idx % w


def obs_to_wall_grid(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 4:
        obs = obs[0]
    blocked = (obs[0] + obs[1] + obs[2]) > 0.5
    return blocked.astype(np.int8)


def obs_to_logits(obs, bot: MyTronBot = None) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        t = obs.detach().cpu().float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        obs_np = t.numpy()
    else:
        obs_np = obs
        if obs_np.ndim == 3:
            obs_np = obs_np[np.newaxis]

    bot = bot or MyTronBot()
    blocked = obs_to_wall_grid(obs_np)
    my = obs_np[0, 3]
    opp = obs_np[0, 4]
    hy, hx = _head_rc(my)
    oy, ox = _head_rc(opp)
    if hy < 0:
        return torch.zeros(4)

    if oy < 0:
        oy, ox = 1, 1
        while blocked[oy, ox] and oy < blocked.shape[0] - 2:
            ox += 1

    action = bot.choose_move(blocked, (hy, hx), (oy, ox))
    logits = torch.full((4,), -1e6, dtype=torch.float32)
    logits[int(action)] = 1e6
    return logits


def obs_to_action(obs, bot: MyTronBot = None) -> int:
    return int(obs_to_logits(obs, bot).argmax().item())
