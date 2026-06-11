import os
from datetime import datetime

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import tron_paper_dqn  # noqa: F401
from tron_paper.env.duel_env import MRLDuelEnv
from tron_paper_dqn.training.policy import dqn_policy_kwargs
from tron_paper_dqn.training.train_stationary import _sb3_to_qnet, load_stationary_qnet


def train_non_stationary(
    total_timesteps: int = 500_000,
    save_dir: str = "./tron_paper_dqn_checkpoints",
    stationary_weights: str = None,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    batch_size: int = 64,
    exploration_fraction: float = 0.3,
    verbose: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    stationary_fn = None
    sw = stationary_weights or os.path.join(save_dir, "stationary_agent.pt")
    if os.path.isfile(sw):
        stationary_qnet = load_stationary_qnet(sw)
        stationary_fn = stationary_qnet.act_greedy

    env = MRLDuelEnv(stationary_policy_fn=stationary_fn)
    model_ref = {"dqn": None}

    def opponent_fn(obs):
        m = model_ref["dqn"]
        if m is None:
            return int(np.random.randint(0, 4))
        with th.no_grad():
            t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            q = m.q_net(t)
            return int(q.argmax(dim=-1).item())

    env.set_opponent_fn(opponent_fn)
    vec = DummyVecEnv([lambda: Monitor(env)])
    device = "cuda" if th.cuda.is_available() else "cpu"

    model = DQN(
        "CnnPolicy",
        vec,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=0.05,
        policy_kwargs=dqn_policy_kwargs(stationary=False),
        verbose=verbose,
        device=device,
        tensorboard_log=os.path.join(save_dir, "tb"),
    )
    model_ref["dqn"] = model

    model.learn(total_timesteps=total_timesteps, progress_bar=False, tb_log_name=f"non_stationary_{ts}")

    sb3_path = os.path.join(save_dir, f"non_stationary_{ts}")
    model.save(sb3_path)
    qnet = _sb3_to_qnet(model, stationary=False)
    pt_path = os.path.join(save_dir, "non_stationary_agent.pt")
    th.save(qnet.state_dict(), pt_path)
    vec.close()
    print(f"Saved {sb3_path}.zip and {pt_path}")
    return model, pt_path


def train_all(
    stationary_steps: int = 300_000,
    non_stationary_steps: int = 500_000,
    save_dir: str = "./tron_paper_dqn_checkpoints",
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    verbose: int = 1,
):
    from tron_paper_dqn.training.train_stationary import train_stationary

    _, stationary_pt = train_stationary(
        stationary_steps, save_dir, n_envs, learning_rate, verbose=verbose,
    )
    train_non_stationary(non_stationary_steps, save_dir, stationary_pt, learning_rate, verbose=verbose)
