import os
from datetime import datetime

import numpy as np

import tron_paper_BH  # noqa: F401
from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper_BH.tronbot_teacher import TronBotStationaryTeacher

DEFAULT_DATA = "./tron_paper_BH/data/stationary_bc.npz"


def collect_dataset(
    episodes: int = 500,
    seed: int = 0,
    output: str = DEFAULT_DATA,
    backend: str = "cpp",
    bot_path: str = None,
    move_timeout: float = 5.0,
    max_depth: int = 100,
    mode: str = "spacefill",
    verbose: int = 1,
):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    env = StationaryTronEnv()
    teacher = TronBotStationaryTeacher(
        backend=backend,
        bot_path=bot_path,
        move_timeout=move_timeout,
        max_depth=max_depth,
        mode=mode,
    )
    obs_buf, act_buf, mask_buf = [], [], []
    ep_lens = []

    for ep in range(episodes):
        for attempt in range(32):
            env.reset(seed=seed + ep * 32 + attempt)
            if env.action_masks().any():
                break
        teacher.reset_episode(env.grid, env.player.y, env.player.x)
        steps = 0
        while True:
            obs = env._obs()
            mask = env.action_masks()
            valid = [a for a in range(4) if mask[a]]
            if not valid:
                break
            action = teacher.action(
                env.grid, env.player.y, env.player.x, mask, int(env.player.direction)
            )
            obs_buf.append(obs)
            act_buf.append(action)
            mask_buf.append(mask.astype(np.uint8))
            _, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break
        ep_lens.append(steps)
        if verbose:
            print(f"  ep {ep + 1}/{episodes}  len={steps}  samples={len(obs_buf)}", flush=True)

    obs = np.stack(obs_buf).astype(np.float32)
    acts = np.array(act_buf, dtype=np.int64)
    masks = np.stack(mask_buf)
    np.savez_compressed(
        output,
        obs=obs,
        acts=acts,
        masks=masks,
        episodes=episodes,
        seed=seed,
        backend=backend,
        collected_at=datetime.now().isoformat(),
    )
    print(f"Saved {len(acts)} samples ({episodes} eps, mean_len={np.mean(ep_lens):.1f}) -> {output}")
    return output
