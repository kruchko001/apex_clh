import os
from datetime import datetime

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tron_paper.env.duel_env import MRLDuelEnv
from tron_paper.training.policy import MRLPolicy
from tron_paper.training.train_stationary import _sb3_to_ac, load_stationary_ac


def train_non_stationary(
    total_timesteps: int = 500_000,
    save_dir: str = "./tron_paper_checkpoints",
    stationary_weights: str = None,
    learning_rate: float = 3e-4,
    verbose: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    stationary_fn = None
    sw = stationary_weights or os.path.join(save_dir, "stationary_agent.pt")
    if os.path.isfile(sw):
        stationary_ac = load_stationary_ac(sw)
        stationary_fn = stationary_ac.act_greedy

    env = MRLDuelEnv(stationary_policy_fn=stationary_fn)
    model_ref = {"ppo": None}

    def opponent_fn(obs):
        m = model_ref["ppo"]
        if m is None:
            return int(np.random.randint(0, 4))
        with th.no_grad():
            t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            feat = m.policy.extract_features(t, m.policy.pi_features_extractor)
            latent = m.policy.mlp_extractor.forward_actor(feat)
            logits = m.policy.action_net(latent)
            return int(logits.argmax(dim=-1).item())

    env.set_opponent_fn(opponent_fn)
    vec = DummyVecEnv([lambda: Monitor(env)])

    model = PPO(
        MRLPolicy,
        vec,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=verbose,
        policy_kwargs={"stationary": False},
        tensorboard_log=os.path.join(save_dir, "tb"),
    )
    model_ref["ppo"] = model

    model.learn(total_timesteps=total_timesteps, progress_bar=False, tb_log_name=f"non_stationary_{ts}")

    sb3_path = os.path.join(save_dir, f"non_stationary_{ts}")
    model.save(sb3_path)
    ac = _sb3_to_ac(model, stationary=False)
    pt_path = os.path.join(save_dir, "non_stationary_agent.pt")
    th.save(ac.state_dict(), pt_path)
    vec.close()
    print(f"Saved {sb3_path}.zip and {pt_path}")
    return model, pt_path


def train_all(
    stationary_steps: int = 300_000,
    non_stationary_steps: int = 500_000,
    save_dir: str = "./tron_paper_checkpoints",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1,
):
    from tron_paper.training.train_stationary import train_stationary

    _, stationary_pt = train_stationary(stationary_steps, save_dir, n_envs, learning_rate, verbose)
    train_non_stationary(non_stationary_steps, save_dir, stationary_pt, learning_rate, verbose)
