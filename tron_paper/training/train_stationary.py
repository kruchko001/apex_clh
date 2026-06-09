import os
from datetime import datetime

import tron_paper  # noqa: F401
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper.model.mrl_net import MRLActorCritic
from tron_paper.training.policy import MRLPolicy


def _make_env():
    return Monitor(StationaryTronEnv())


def _sb3_to_ac(model: PPO, stationary: bool) -> MRLActorCritic:
    ac = MRLActorCritic(stationary=stationary)
    ac.trunk.load_state_dict(model.policy.features_extractor.ac.trunk.state_dict())
    pi = model.policy.mlp_extractor.policy_net
    ac.pi1.weight.data = pi[0].weight.data.clone()
    ac.pi1.bias.data = pi[0].bias.data.clone()
    ac.pi2.weight.data = model.policy.action_net.weight.data.clone()
    ac.pi2.bias.data = model.policy.action_net.bias.data.clone()
    vf = model.policy.mlp_extractor.value_net
    ac.v1.weight.data = vf[0].weight.data.clone()
    ac.v1.bias.data = vf[0].bias.data.clone()
    ac.v2.weight.data = vf[2].weight.data.clone()
    ac.v2.bias.data = vf[2].bias.data.clone()
    ac.v3.weight.data = model.policy.value_net.weight.data.clone()
    ac.v3.bias.data = model.policy.value_net.bias.data.clone()
    return ac


def train_stationary(
    total_timesteps: int = 300_000,
    save_dir: str = "./tron_paper_checkpoints",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = SubprocVecEnv([_make_env for _ in range(n_envs)]) if n_envs > 1 else DummyVecEnv([_make_env])

    model = PPO(
        MRLPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=verbose,
        policy_kwargs={"stationary": True},
        tensorboard_log=os.path.join(save_dir, "tb"),
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False, tb_log_name=f"stationary_{ts}")

    sb3_path = os.path.join(save_dir, f"stationary_{ts}")
    model.save(sb3_path)

    ac = _sb3_to_ac(model, stationary=True)
    pt_path = os.path.join(save_dir, "stationary_agent.pt")
    th.save(ac.state_dict(), pt_path)

    env.close()
    print(f"Saved {sb3_path}.zip and {pt_path}")
    return model, pt_path


def load_stationary_ac(path: str) -> MRLActorCritic:
    ac = MRLActorCritic(stationary=True)
    ac.load_state_dict(th.load(path, map_location="cpu", weights_only=True))
    ac.eval()
    return ac
