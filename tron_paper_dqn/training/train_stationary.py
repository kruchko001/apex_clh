import os
from datetime import datetime

import tron_paper_dqn  # noqa: F401
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper_dqn.model.q_net import MRLQNet
from tron_paper_dqn.training.policy import dqn_policy_kwargs


def _make_env():
    return Monitor(StationaryTronEnv())


def _sb3_to_qnet(model: DQN, stationary: bool) -> MRLQNet:
    qnet = MRLQNet(stationary=stationary)
    qnet.trunk.load_state_dict({k: v.cpu() for k, v in model.q_net.features_extractor.qnet.trunk.state_dict().items()})
    mlp = model.q_net.q_net
    qnet.q1.weight.data = mlp[0].weight.data.cpu().clone()
    qnet.q1.bias.data = mlp[0].bias.data.cpu().clone()
    qnet.q2.weight.data = mlp[2].weight.data.cpu().clone()
    qnet.q2.bias.data = mlp[2].bias.data.cpu().clone()
    return qnet


def train_stationary(
    total_timesteps: int = 300_000,
    save_dir: str = "./tron_paper_dqn_checkpoints",
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    batch_size: int = 64,
    exploration_fraction: float = 0.3,
    verbose: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = SubprocVecEnv([_make_env for _ in range(n_envs)]) if n_envs > 1 else DummyVecEnv([_make_env])
    device = "cuda" if th.cuda.is_available() else "cpu"

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=0.05,
        policy_kwargs=dqn_policy_kwargs(stationary=True),
        verbose=verbose,
        device=device,
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False, tb_log_name=f"stationary_{ts}")

    sb3_path = os.path.join(save_dir, f"stationary_{ts}")
    model.save(sb3_path)

    qnet = _sb3_to_qnet(model, stationary=True)
    pt_path = os.path.join(save_dir, "stationary_agent.pt")
    th.save(qnet.state_dict(), pt_path)

    env.close()
    print(f"Saved {sb3_path}.zip and {pt_path}")
    return model, pt_path


def load_stationary_qnet(path: str) -> MRLQNet:
    qnet = MRLQNet(stationary=True)
    qnet.load_state_dict(th.load(path, map_location="cpu", weights_only=True))
    qnet.eval()
    return qnet
