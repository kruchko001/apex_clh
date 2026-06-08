"""
Test exported TorchScript model.
"""

import argparse
import os
import importlib.util

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise ImportError("Could not locate tron_solution package root")
    _d = _parent

import torch
import numpy as np

from tron_solution.env.tron_env import TronEnv
from tron_solution.model.obs import to_sandbox_obs_np


def _is_wrapped_sandbox(model) -> bool:
    try:
        with torch.no_grad():
            model(torch.randn(1, 5, 32, 32))
        return True
    except Exception:
        return False


def _is_raw_core(model) -> bool:
    try:
        with torch.no_grad():
            model(torch.randn(1, 16, 30, 30))
        return True
    except Exception:
        return False


def test_model(model_path: str, num_episodes: int = 5, render: bool = False):
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()
    wrapped = _is_wrapped_sandbox(model)
    raw_core = not wrapped and _is_raw_core(model)
    if wrapped:
        print("Model: sandbox wrapper (1,5,32,32) with internal 4-frame stack")
    elif raw_core:
        print("Model: raw core (1,16,30,30)")
    else:
        print("Model: unknown input shape")

    env = TronEnv(grid_size=32, max_steps=500, render_mode="rgb_array" if render else None)
    frame_stack = None
    if raw_core:
        from tron_solution.model.frame_stack import FrameStack
        frame_stack = FrameStack()

    total_rewards, total_steps, wins = [], [], 0

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        if raw_core:
            stacked = frame_stack.reset(obs)
        episode_reward, steps, done = 0.0, 0, False

        while not done:
            if wrapped:
                x = torch.from_numpy(to_sandbox_obs_np(obs, env.walls)).float().unsqueeze(0)
            elif raw_core:
                x = torch.from_numpy(stacked).float().unsqueeze(0)
            else:
                raise RuntimeError("Unsupported model format")
            with torch.no_grad():
                out = model(x)
                logits = out if out.dim() == 1 else out.squeeze(0)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            if raw_core:
                stacked = frame_stack.step(obs)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(steps)
        outcome = "Draw"
        if info.get("clean_kill") or info.get("opponent_self_destruct"):
            outcome, wins = "Win", wins + 1
        elif info.get("my_collision_type") and not info.get("opponent_collision_type"):
            outcome = "Loss"
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}, Outcome={outcome}")

    print("\n" + "=" * 50)
    print(f"Episodes: {num_episodes}  Wins: {wins}/{num_episodes}")
    print(f"Avg reward: {np.mean(total_rewards):.2f}  Avg steps: {np.mean(total_steps):.1f}")
    print("=" * 50)
    return {"wins": wins, "win_rate": wins / num_episodes}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    test_model(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
