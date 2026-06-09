import importlib.util
import os

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("tron_paper_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent

import random

import numpy as np
import torch

from competition.tron.tron import GameConfig, TronGame
from tron_paper.env.encode import GRID, encode_official


def launcher_select_action(model, state_tensor: torch.Tensor, valid_actions: list) -> int:
    with torch.no_grad():
        output = model(state_tensor)
    if output.dim() == 1:
        q = output
    elif output.dim() == 2:
        q = output.squeeze(0)
    else:
        q = output.flatten()[:4]
    q_np = q.cpu().numpy()[:4]
    masked = np.full(4, float("-inf"))
    for a in valid_actions:
        if 0 <= a < 4:
            masked[a] = q_np[a]
    best = int(np.argmax(masked))
    if masked[best] == float("-inf"):
        return valid_actions[0] if valid_actions else 0
    return best


def _outcome(game: TronGame) -> str:
    p0, p1 = game.get_player(0), game.get_player(1)
    if p0.alive and not p1.alive:
        return "W"
    if p1.alive and not p0.alive:
        return "L"
    return "D"


def test_paper_model(model_path: str, episodes: int = 5, seed: int = 0, grid: int = GRID):
    print(f"Loading {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()

    dummy = torch.zeros(1, 5, grid, grid)
    with torch.no_grad():
        out = model(dummy)
    if out.dim() == 1:
        assert out.shape == (4,), f"expected (4,), got {tuple(out.shape)}"
    elif out.dim() == 2:
        assert out.shape == (1, 4), f"expected (1,4), got {tuple(out.shape)}"
    print(f"I/O OK: in (1,5,{grid},{grid}) -> out {tuple(out.shape)}")

    rng = random.Random(seed)
    wins = losses = draws = 0
    rewards = []

    for ep in range(episodes):
        game = TronGame(GameConfig(width=grid, height=grid, max_steps=500, num_players=2))
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            valid0 = game.get_valid_actions(0)
            valid1 = game.get_valid_actions(1)
            obs = encode_official(game, 0)
            state = torch.from_numpy(obs).unsqueeze(0).float()
            a0 = launcher_select_action(model, state, valid0) if valid0 else 0
            a1 = rng.choice(valid1) if valid1 else int(game.get_player(1).direction)

            _, _, done, info = game.step({0: a0, 1: a1})
            steps += 1
            if not done:
                ep_reward -= 1.0
            elif _outcome(game) == "W":
                ep_reward += 100.0
            elif _outcome(game) == "L":
                ep_reward -= 100.0

        o = _outcome(game)
        if o == "W":
            wins += 1
        elif o == "L":
            losses += 1
        else:
            draws += 1
        rewards.append(ep_reward)
        print(f"Episode {ep + 1}: steps={steps} outcome={o} reward={ep_reward:.1f}")

    print(f"\nW/L/D={wins}/{losses}/{draws}  avg_reward={np.mean(rewards):.2f}  (vs random, launcher-style masking)")
    return {"wins": wins, "losses": losses, "draws": draws}
