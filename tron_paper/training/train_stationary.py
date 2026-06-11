import os
from datetime import datetime

import tron_paper  # noqa: F401
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper.model.mrl_net import MRLActorCritic
from tron_paper.training.policy import MRLPolicy


def _make_env():
    return Monitor(StationaryTronEnv())


def _sb3_to_ac(model: PPO, stationary: bool) -> MRLActorCritic:
    ac = MRLActorCritic(stationary=stationary)
    ac.trunk.load_state_dict({
        k: v.cpu() for k, v in model.policy.features_extractor.ac.trunk.state_dict().items()
    })
    pi = model.policy.mlp_extractor.policy_net
    ac.pi1.weight.data = pi[0].weight.data.cpu().clone()
    ac.pi1.bias.data = pi[0].bias.data.cpu().clone()
    ac.pi2.weight.data = model.policy.action_net.weight.data.cpu().clone()
    ac.pi2.bias.data = model.policy.action_net.bias.data.cpu().clone()
    vf = model.policy.mlp_extractor.value_net
    ac.v1.weight.data = vf[0].weight.data.cpu().clone()
    ac.v1.bias.data = vf[0].bias.data.cpu().clone()
    ac.v2.weight.data = vf[2].weight.data.cpu().clone()
    ac.v2.bias.data = vf[2].bias.data.cpu().clone()
    ac.v3.weight.data = model.policy.value_net.weight.data.cpu().clone()
    ac.v3.bias.data = model.policy.value_net.bias.data.cpu().clone()
    return ac


def _save_stationary(model: PPO, save_dir: str, ts: str):
    pt_path = os.path.join(save_dir, "stationary_agent.pt")
    th.save(_sb3_to_ac(model, stationary=True).state_dict(), pt_path)

    sb3_path = os.path.join(save_dir, f"stationary_{ts}")
    model.save(sb3_path, exclude=["env"])
    return sb3_path, pt_path


class StationaryPtCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            pt_path = os.path.join(self.save_dir, "stationary_agent.pt")
            th.save(_sb3_to_ac(self.model, stationary=True).state_dict(), pt_path)
            if self.verbose:
                print(f"Checkpoint {pt_path} @ {self.num_timesteps} steps")
        return True


def train_stationary(
    total_timesteps: int = 300_000,
    save_dir: str = "./tron_paper_checkpoints",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.02,
    checkpoint_freq: int = 50_000,
    verbose: int = 1,
    show_gui: bool = False,
    gui_delay_ms: int = 50,
    gui_every_updates: int = 1,
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
        ent_coef=ent_coef,
        verbose=verbose,
        policy_kwargs={"stationary": True},
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    callbacks = [StationaryPtCallback(save_dir, checkpoint_freq, verbose)]
    if show_gui:
        from tron_paper.training.gui_callback import TrainGUICallback
        callbacks.append(TrainGUICallback(gui_delay_ms, gui_every_updates, verbose))

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        tb_log_name=f"stationary_{ts}",
        callback=CallbackList(callbacks),
    )

    sb3_path, pt_path = _save_stationary(model, save_dir, ts)
    env.close()
    print(f"Saved {sb3_path}.zip and {pt_path}")
    return model, pt_path


def load_stationary_ac(path: str) -> MRLActorCritic:
    ac = MRLActorCritic(stationary=True)
    ac.load_state_dict(th.load(path, map_location="cpu", weights_only=True))
    ac.eval()
    return ac
