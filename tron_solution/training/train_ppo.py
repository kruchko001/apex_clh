"""
PPO Training Script for Tron Environment.

Usage:
    python -m tron_solution.training.train_ppo --timesteps 100000 --verbose
"""

import argparse
import os
import shutil
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from datetime import datetime
import multiprocessing as mp

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.frame_stack_wrapper import GridFrameStackWrapper
from tron_solution.env.opponents import DEFAULT_OPPONENT_TYPE, DEFAULT_MINIMAX_DEPTH
from tron_solution.model.obs import N_STACK, VALID_DIM, apply_action_mask, cnn_flat_size


class TronMaskedActorCriticPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic: bool = False):
        self._valid = obs["valid"]
        return super().forward(obs, deterministic=deterministic)

    def get_distribution(self, obs):
        self._valid = obs["valid"]
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def evaluate_actions(self, obs, actions):
        self._valid = obs["valid"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = apply_action_mask(self.action_net(latent_pi), self._valid)
        return self.action_dist.proba_distribution(action_logits=logits)


class TronFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        grid_space = observation_space["grid"]
        input_channels = grid_space.shape[0]
        spatial = grid_space.shape[1]
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.shared_fc = torch.nn.Linear(cnn_flat_size(spatial) + VALID_DIM, features_dim)

    def forward(self, obs) -> torch.Tensor:
        x = obs["grid"]
        valid = obs["valid"]
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, valid], dim=1)
        return torch.nn.functional.relu(self.shared_fc(x))


def outcome_from_info(info: dict) -> str:
    if info.get("clean_kill") or info.get("opponent_self_destruct"):
        return "W"
    if info.get("my_collision_type") and not info.get("opponent_collision_type"):
        return "L"
    return "D"


def count_wld(model, env, n_episodes, deterministic=True):
    wins = losses = draws = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]
        o = outcome_from_info(info)
        if o == "W":
            wins += 1
        elif o == "L":
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


class RewardLogCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.iteration = 0
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                o = outcome_from_info(info)
                if o == "W":
                    self.wins += 1
                elif o == "L":
                    self.losses += 1
                else:
                    self.draws += 1
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if self.episode_rewards:
            mean_r = float(np.mean(self.episode_rewards))
            n = len(self.episode_rewards)
            print(
                f"Iteration {self.iteration} | timesteps={self.num_timesteps} | "
                f"mean_reward={mean_r:.3f} | W/L/D={self.wins}/{self.losses}/{self.draws} | episodes={n}"
            )
        elif self.verbose:
            print(f"Iteration {self.iteration} | timesteps={self.num_timesteps} | no completed episodes")
        self.episode_rewards = []
        self.wins = self.losses = self.draws = 0


class TronEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        prev_n = len(self.evaluations_results)
        continue_training = super()._on_step()
        if len(self.evaluations_results) > prev_n:
            w, l, d = count_wld(self.model, self.eval_env, self.n_eval_episodes, self.deterministic)
            print(f"Eval W/L/D: {w}/{l}/{d} ({self.n_eval_episodes} episodes)")
        return continue_training


class EarlyStopEvalCallback(TronEvalCallback):
    def __init__(self, *args, patience=6, min_delta=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self._no_improve_count = 0
        self.stopped_early = False

    def _on_step(self) -> bool:
        prev_best = self.best_mean_reward
        prev_n = len(self.evaluations_results)
        continue_training = super()._on_step()
        if not continue_training:
            return False
        if len(self.evaluations_results) > prev_n:
            if self.best_mean_reward > prev_best + self.min_delta:
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
                if self.verbose >= 1:
                    print(
                        f"EarlyStop: no eval improvement "
                        f"({self._no_improve_count}/{self.patience}, "
                        f"best={self.best_mean_reward:.3f})"
                    )
                if self._no_improve_count >= self.patience:
                    print(
                        f"Early stopping at {self.num_timesteps} timesteps "
                        f"(best eval reward {self.best_mean_reward:.3f})"
                    )
                    self.stopped_early = True
                    return False
        return True


DEFAULT_CURRICULUM = [
    (4, 75000),
    (6, 150000),
    (10, 100000),
    (12, 100000),
    (14, 75000),
]
DEFAULT_CURRICULUM_TOTAL = sum(st for _, st in DEFAULT_CURRICULUM)


def _obs_signature(space):
    if hasattr(space, "spaces"):
        return {k: tuple(v.shape) for k, v in space.spaces.items()}
    return tuple(space.shape)


def _check_obs_compat(expected_space, actual_space, label="checkpoint"):
    if _obs_signature(expected_space) != _obs_signature(actual_space):
        raise ValueError(
            f"{label} observation {_obs_signature(expected_space)} != "
            f"env {_obs_signature(actual_space)}. Train from scratch without --resume."
        )


def _zip_path(path):
    if not path:
        return path
    return path if path.endswith(".zip") else path + ".zip"


def curriculum_stage_paths(save_dir, run_id, depth):
    model = os.path.join(save_dir, f"tron_ppo_final_{run_id}_d{depth}.zip")
    vec = os.path.join(save_dir, f"vec_normalize_final_{run_id}_d{depth}.pkl")
    return model, vec


def warmup_paths(save_dir, run_id):
    model = os.path.join(save_dir, f"tron_ppo_final_{run_id}_warmup.zip")
    vec = os.path.join(save_dir, f"vec_normalize_final_{run_id}_warmup.pkl")
    return model, vec


def find_curriculum_resume(save_dir, run_id, stages):
    resume_model, resume_vec, next_idx = None, None, 0
    for idx, (depth, _) in enumerate(stages):
        model, vec = curriculum_stage_paths(save_dir, run_id, depth)
        if os.path.isfile(model) and os.path.isfile(vec):
            resume_model, resume_vec = model, vec
            next_idx = idx + 1
        else:
            break
    return resume_model, resume_vec, next_idx


def parse_curriculum_stages(spec, total_timesteps=None):
    if spec:
        stages = []
        for part in spec.split(","):
            depth_s, steps_s = part.strip().split(":")
            stages.append((int(depth_s), int(steps_s)))
        return stages
    stages = list(DEFAULT_CURRICULUM)
    if total_timesteps is not None:
        budget = total_timesteps
        base = sum(st for _, st in stages)
        if base != budget:
            scale = budget / base
            stages = [(d, max(10000, int(st * scale))) for d, st in stages]
            diff = budget - sum(st for _, st in stages)
            if diff:
                stages[-1] = (stages[-1][0], stages[-1][1] + diff)
    return stages


def default_n_envs(cpu_count, opponent_type, minimax_depth, max_envs=None):
    reserve = 2
    available = max(1, (cpu_count or 4) - reserve)
    if opponent_type != "minimax":
        cap = 8
    else:
        depth = minimax_depth or DEFAULT_MINIMAX_DEPTH
        if depth >= 14:
            cap = 8
        elif depth >= 12:
            cap = 10
        elif depth >= 8:
            cap = 12
        else:
            cap = 8
    if max_envs is not None:
        cap = min(cap, max_envs)
    return min(available, cap)


def train(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    verbose: int = 1,
    save_dir: str = "./ppo_tron_checkpoints",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    opponent_type: str = None,
    minimax_depth: int = None,
    n_envs: int = None,
    max_envs: int = None,
    resume_path: str = None,
    vec_normalize_path: str = None,
    early_stop: bool = True,
    early_stop_patience: int = 6,
    early_stop_min_delta: float = 0.005,
    run_id: str = None,
    stage_label: str = None,
):
    """
    Train PPO agent on Tron environment.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs when updating
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: Clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        verbose: Verbosity level (0: silent, 1: info, 2: debug)
        save_dir: Directory to save checkpoints
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of episodes for evaluation
    """
    
    opponent_type = opponent_type or DEFAULT_OPPONENT_TYPE
    if opponent_type == "minimax" and minimax_depth is None:
        minimax_depth = DEFAULT_MINIMAX_DEPTH

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if n_envs is None:
        n_envs = default_n_envs(mp.cpu_count(), opponent_type, minimax_depth, max_envs)
    elif max_envs is not None:
        n_envs = min(n_envs, max_envs)
    print(
        f"Creating {n_envs} parallel environments "
        f"(cpus={mp.cpu_count()}, opponent={opponent_type}"
        + (f", depth={minimax_depth}" if opponent_type == "minimax" else "")
        + ")..."
    )

    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_suffix = stage_label or ""
    name_prefix = f"tron_ppo_{timestamp}{stage_suffix}"
    
    def make_env():
        return Monitor(GridFrameStackWrapper(TronEnv(
            grid_size=32, max_steps=500,
            opponent_type=opponent_type, minimax_depth=minimax_depth,
        ), n_stack=N_STACK))
    
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    print(f"Applying frame stacking with N={N_STACK}...")
    
    if vec_normalize_path and os.path.isfile(vec_normalize_path):
        import pickle
        with open(vec_normalize_path, "rb") as f:
            saved = pickle.load(f)
        _check_obs_compat(saved.observation_space, env.observation_space, "VecNormalize")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize from {vec_normalize_path}")
    else:
        env = VecNormalize(
            env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.,
            norm_obs_keys=["grid"],
        )
    
    os.makedirs(save_dir, exist_ok=True)
    
    callback_freq = max(eval_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq * 10 // n_envs, 1),
        save_path=save_dir,
        name_prefix=name_prefix,
        verbose=verbose,
    )
    
    # Create evaluation environment (single env for eval)
    eval_env = DummyVecEnv([lambda: Monitor(GridFrameStackWrapper(TronEnv(
        grid_size=32, max_steps=500,
        opponent_type=opponent_type, minimax_depth=minimax_depth,
    ), n_stack=N_STACK))])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False, norm_obs_keys=["grid"],
    )
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    eval_cls = EarlyStopEvalCallback if early_stop else TronEvalCallback
    eval_kw = dict(
        eval_env=eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=callback_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    if early_stop:
        eval_kw["patience"] = early_stop_patience
        eval_kw["min_delta"] = early_stop_min_delta
    eval_callback = eval_cls(**eval_kw)
    
    # Create or resume PPO model
    if resume_path and os.path.isfile(_zip_path(resume_path)):
        print(f"Resuming from {_zip_path(resume_path)}")
        probe = PPO.load(_zip_path(resume_path), device=device)
        _check_obs_compat(probe.observation_space, env.observation_space, "Model")
        model = PPO.load(_zip_path(resume_path), env=env, device=device)
        model.learning_rate = learning_rate
    else:
        model = PPO(
            TronMaskedActorCriticPolicy,
            env,
            policy_kwargs=dict(
                features_extractor_class=TronFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                normalize_images=False,
                net_arch=dict(pi=[], vf=[]),
            ),
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
            device=device,
        )
    
    reward_log_callback = RewardLogCallback(verbose=verbose)

    # Train
    print(f"Starting training for {total_timesteps} timesteps with {n_envs} parallel environments...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, reward_log_callback],
        tb_log_name="tron_ppo",
        reset_num_timesteps=True,
    )

    best_path = os.path.join(save_dir, "best_model.zip")
    if early_stop and eval_callback.stopped_early and os.path.isfile(best_path):
        print(f"Reloading best eval checkpoint from {best_path}")
        model = PPO.load(best_path, env=env, device=device)

    final_path = os.path.join(save_dir, f"tron_ppo_final_{timestamp}{stage_suffix}")
    model.save(final_path)
    final_zip = _zip_path(final_path)
    vec_path = os.path.join(save_dir, f"vec_normalize_final_{timestamp}{stage_suffix}.pkl")
    env.save(vec_path)

    if early_stop and eval_callback.stopped_early:
        print(f"Early stop complete! Best-eval model saved to {final_zip}")
    else:
        print(f"Training complete! Model saved to {final_zip}")
    
    env.close()
    eval_env.close()
    
    return model, final_zip, vec_path


def evaluate_vs_opponent(model, vec_path, opponent_type, minimax_depth=None, n_episodes=20, seed_base=0):
    def make_env():
        return Monitor(GridFrameStackWrapper(TronEnv(
            grid_size=32, max_steps=500,
            opponent_type=opponent_type, minimax_depth=minimax_depth,
        ), n_stack=N_STACK))

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    wins = losses = draws = 0
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        seed = seed_base + ep
        if hasattr(env, "venv") and hasattr(env.venv, "reset"):
            try:
                env.venv.reset(seed=seed)
            except TypeError:
                pass
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            ep_reward += float(r[0])
            steps += 1
            done = bool(dones[0])
            info = infos[0]
        rewards.append(ep_reward)
        lengths.append(steps)
        o = outcome_from_info(info)
        if o == "W":
            wins += 1
        elif o == "L":
            losses += 1
        else:
            draws += 1
    env.close()
    n = max(n_episodes, 1)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def evaluate_vs_minimax(model, vec_path, depth, n_episodes=20, seed_base=0):
    return evaluate_vs_opponent(
        model, vec_path, "minimax", minimax_depth=depth,
        n_episodes=n_episodes, seed_base=seed_base,
    )


def train_curriculum(
    curriculum_stages=None,
    total_timesteps=None,
    save_dir="./ppo_tron_checkpoints",
    eval_freq=20000,
    n_envs=None,
    max_envs=None,
    resume_path=None,
    vec_normalize_path=None,
    curriculum_run_id=None,
    early_stop=True,
    early_stop_patience=6,
    early_stop_min_delta=0.005,
    stage_gate_win_rate=0.25,
    stage_gate_episodes=20,
    stage_gate_extra_steps=100000,
    no_stage_gate=False,
    warmup_opponent="heuristic",
    warmup_timesteps=200000,
    **train_kw,
):
    import gc

    stages = parse_curriculum_stages(curriculum_stages, total_timesteps)
    run_id = curriculum_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    resume = _zip_path(resume_path) if resume_path else None
    vec_norm = vec_normalize_path
    last_model_path = None
    last_vec_path = None
    start_idx = 0

    if curriculum_run_id:
        found_model, found_vec, start_idx = find_curriculum_resume(save_dir, run_id, stages)
        if found_model:
            resume, vec_norm = found_model, found_vec
            print(f"Resuming curriculum run {run_id} after stage {start_idx}/{len(stages)}")
        elif start_idx == 0:
            w_model, w_vec = warmup_paths(save_dir, run_id)
            if os.path.isfile(w_model) and os.path.isfile(w_vec):
                resume, vec_norm = w_model, w_vec
                print(f"Resuming curriculum run {run_id} from completed warmup -> stage 1")
        if not resume and resume_path:
            print(f"Curriculum run {run_id}: no completed stages found, using --resume checkpoint")
        elif not resume:
            print(f"Curriculum run {run_id}: starting from scratch")
    elif resume_path and not os.path.isfile(_zip_path(resume_path)):
        print(f"Warning: resume checkpoint not found: {resume_path}")

    if start_idx >= len(stages):
        print(f"Curriculum run {run_id} already complete ({len(stages)} stages).")
        last_model_path, last_vec_path = resume, vec_norm
        model = PPO.load(last_model_path)
        curriculum_final = os.path.join(save_dir, f"tron_ppo_curriculum_final_{run_id}")
        model.save(curriculum_final)
        return model, curriculum_final, last_vec_path

    print("Curriculum plan:")
    for i, (depth, steps) in enumerate(stages, 1):
        marker = " (skip)" if i <= start_idx else ""
        print(f"  stage {i}: minimax depth {depth}, {steps} timesteps{marker}")
    print(f"  total: {sum(st for _, st in stages)} timesteps")
    print(f"  run_id: {run_id}")
    if not no_stage_gate:
        print(f"  stage gate: {stage_gate_win_rate:.0%} win rate over {stage_gate_episodes} eval games")
        if stage_gate_extra_steps:
            print(f"  gate retry: +{stage_gate_extra_steps} timesteps per failed gate")
    if warmup_opponent and warmup_timesteps > 0 and not resume and not curriculum_run_id:
        gate_note = f", {stage_gate_win_rate:.0%} win gate" if not no_stage_gate else ""
        print(f"  warmup: {warmup_timesteps} steps vs {warmup_opponent}{gate_note}")
    print()

    if warmup_opponent and warmup_timesteps > 0 and not resume and not curriculum_run_id:
        print(f"\n{'=' * 60}")
        print(f"Warmup: {warmup_opponent} opponent, {warmup_timesteps} timesteps")
        print(f"{'=' * 60}\n")
        w_n_envs = n_envs or default_n_envs(mp.cpu_count(), warmup_opponent, None, max_envs)
        w_steps = warmup_timesteps
        w_resume = None
        w_vec = None
        passed = False
        while not passed:
            model, final_path, vec_path = train(
                total_timesteps=w_steps,
                save_dir=save_dir,
                eval_freq=eval_freq,
                opponent_type=warmup_opponent,
                n_envs=w_n_envs,
                max_envs=max_envs,
                resume_path=w_resume,
                vec_normalize_path=w_vec,
                early_stop=False,
                run_id=run_id,
                stage_label="_warmup",
                **train_kw,
            )
            if no_stage_gate:
                passed = True
            else:
                stats = evaluate_vs_opponent(
                    model, vec_path, warmup_opponent,
                    n_episodes=stage_gate_episodes,
                )
                print(
                    f"\nWarmup gate ({warmup_opponent}): "
                    f"{stats['wins']}W {stats['losses']}L {stats['draws']}D / {stage_gate_episodes} "
                    f"win_rate={stats['win_rate']:.1%} mean_reward={stats['mean_reward']:.2f} "
                    f"mean_length={stats['mean_length']:.1f}"
                )
                if stats["win_rate"] >= stage_gate_win_rate:
                    passed = True
                elif stage_gate_extra_steps > 0:
                    print(
                        f"Warmup gate not met ({stats['win_rate']:.1%} < {stage_gate_win_rate:.0%}), "
                        f"training {stage_gate_extra_steps} more timesteps vs {warmup_opponent}..."
                    )
                    w_resume = final_path
                    w_vec = vec_path
                    w_steps = stage_gate_extra_steps
                else:
                    print(
                        f"\nWarmup stopped: did not reach {stage_gate_win_rate:.0%} win rate "
                        f"vs {warmup_opponent}."
                    )
                    print(f"Checkpoint: {final_path}")
                    print(f"VecNormalize: {vec_path}")
                    model = PPO.load(_zip_path(final_path))
                    return model, final_path, vec_path
            resume = final_path
            vec_norm = vec_path
            last_model_path = final_path
            last_vec_path = vec_path
            del model
            gc.collect()
        print(f"Warmup finished -> {final_path}\n")

    for i, (depth, steps) in enumerate(stages, 1):
        if i <= start_idx:
            continue
        stage_early_stop = early_stop and (i == len(stages))
        stage_steps = steps
        stage_resume = resume
        stage_vec = vec_norm
        passed = False

        while not passed:
            print(f"\n{'=' * 60}")
            print(f"Curriculum stage {i}/{len(stages)}: depth={depth}, timesteps={stage_steps}")
            if not stage_early_stop and early_stop:
                print("Early stop disabled for intermediate curriculum stages")
            print(f"{'=' * 60}\n")
            stage_n_envs = n_envs
            if stage_n_envs is None:
                stage_n_envs = default_n_envs(mp.cpu_count(), "minimax", depth, max_envs)
            try:
                model, final_path, vec_path = train(
                    total_timesteps=stage_steps,
                    save_dir=save_dir,
                    eval_freq=eval_freq,
                    opponent_type="minimax",
                    minimax_depth=depth,
                    n_envs=stage_n_envs,
                    max_envs=max_envs,
                    resume_path=stage_resume,
                    vec_normalize_path=stage_vec,
                    early_stop=stage_early_stop,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    run_id=run_id,
                    stage_label=f"_d{depth}",
                    **train_kw,
                )
            except Exception:
                print(f"\nCurriculum failed at stage {i}/{len(stages)} (depth={depth}).")
                print(f"Resume with: --curriculum-run-id {run_id}")
                if last_model_path:
                    print(f"Last good checkpoint: {last_model_path}")
                raise

            if no_stage_gate:
                passed = True
            else:
                stats = evaluate_vs_minimax(
                    model, vec_path, depth, n_episodes=stage_gate_episodes,
                )
                print(
                    f"\nStage {i} gate (depth {depth}): "
                    f"{stats['wins']}W {stats['losses']}L {stats['draws']}D / {stage_gate_episodes} "
                    f"win_rate={stats['win_rate']:.1%} mean_reward={stats['mean_reward']:.2f} "
                    f"mean_length={stats['mean_length']:.1f}"
                )
                if stats["win_rate"] >= stage_gate_win_rate:
                    passed = True
                elif stage_gate_extra_steps > 0:
                    print(
                        f"Gate not met ({stats['win_rate']:.1%} < {stage_gate_win_rate:.0%}), "
                        f"training {stage_gate_extra_steps} more timesteps at depth {depth}..."
                    )
                    stage_resume = final_path
                    stage_vec = vec_path
                    stage_steps = stage_gate_extra_steps
                else:
                    print(
                        f"\nCurriculum stopped: model did not reach {stage_gate_win_rate:.0%} win rate "
                        f"vs minimax depth {depth}."
                    )
                    print(f"Checkpoint: {final_path}")
                    print(f"VecNormalize: {vec_path}")
                    print(
                        f"Continue same stage:\n"
                        f"  python main.py train --opponent minimax --minimax-depth {depth} "
                        f"--resume {final_path} --vec-normalize {vec_path} --timesteps {steps}"
                    )
                    return model, final_path, vec_path

            resume = final_path
            vec_norm = vec_path
            last_model_path = final_path
            last_vec_path = vec_path
            del model
            gc.collect()
            print(f"Stage {i}/{len(stages)} finished -> {final_path}")

    curriculum_final = os.path.join(save_dir, f"tron_ppo_curriculum_final_{run_id}")
    shutil.copy2(_zip_path(last_model_path), _zip_path(curriculum_final))
    shutil.copy2(
        last_vec_path,
        os.path.join(save_dir, f"vec_normalize_curriculum_final_{run_id}.pkl"),
    )
    model = PPO.load(_zip_path(last_model_path))
    print(f"\nCurriculum complete! Final model: {curriculum_final}.zip")
    print(f"VecNormalize: vec_normalize_curriculum_final_{run_id}.pkl")
    return model, curriculum_final, last_vec_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Tron environment")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps (curriculum default: 500000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=512, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--save-dir", type=str, default="./ppo_tron_checkpoints", help="Save directory")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--opponent", type=str, default=DEFAULT_OPPONENT_TYPE,
                       choices=["random", "heuristic", "lookahead", "minimax", "tronbot"],
                       help="Opponent type to train against")
    parser.add_argument("--minimax-depth", type=int, default=DEFAULT_MINIMAX_DEPTH,
                       help="Minimax search depth when --opponent minimax")
    parser.add_argument("--n-envs", type=int, default=None,
                       help="Parallel env count (default: auto from CPU and minimax depth)")
    parser.add_argument("--max-envs", type=int, default=None,
                       help="Cap auto-detected parallel env count")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to SB3 .zip checkpoint to resume training")
    parser.add_argument("--vec-normalize", type=str, default=None,
                       help="VecNormalize .pkl to load (recommended when resuming)")
    parser.add_argument("--no-early-stop", action="store_true",
                       help="Disable eval early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=6,
                       help="Stop after this many evals without improvement (default: 6)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005,
                       help="Min eval reward gain to reset patience (default: 0.005)")
    parser.add_argument("--curriculum", action="store_true",
                       help="Train minimax depth curriculum 4->6->10->12->14")
    parser.add_argument("--curriculum-stages", type=str, default=None,
                       help='Custom stages as "depth:steps,..." e.g. "6:100000,10:150000,14:200000"')
    parser.add_argument("--curriculum-run-id", type=str, default=None,
                       help="Resume interrupted curriculum by run id (e.g. 20260608_104639)")
    parser.add_argument("--stage-gate-win-rate", type=float, default=0.25,
                       help="Min win rate vs current depth to advance curriculum (default 0.25)")
    parser.add_argument("--stage-gate-episodes", type=int, default=20,
                       help="Eval games for stage gate (default 20)")
    parser.add_argument("--stage-gate-extra-steps", type=int, default=100000,
                       help="Extra timesteps on same stage if 25%% gate fails (default 100000, 0=stop)")
    parser.add_argument("--no-stage-gate", action="store_true",
                       help="Advance curriculum regardless of win rate")
    parser.add_argument("--warmup-opponent", type=str, default="heuristic",
                       choices=["random", "heuristic", "lookahead"],
                       help="Opponent for warmup phase before minimax curriculum")
    parser.add_argument("--warmup-timesteps", type=int, default=200000,
                       help="Heuristic warmup timesteps before minimax (default 200000)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip heuristic warmup")
    
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None:
        timesteps = DEFAULT_CURRICULUM_TOTAL if (args.curriculum or args.curriculum_stages) else 100000

    train_kw = dict(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=args.verbose,
        save_dir=args.save_dir,
        n_eval_episodes=5,
    )

    if args.curriculum or args.curriculum_stages:
        train_curriculum(
            curriculum_stages=args.curriculum_stages,
            total_timesteps=timesteps,
            eval_freq=args.eval_freq,
            n_envs=args.n_envs,
            max_envs=args.max_envs,
            resume_path=args.resume,
            vec_normalize_path=args.vec_normalize,
            curriculum_run_id=args.curriculum_run_id,
            early_stop=not args.no_early_stop,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            stage_gate_win_rate=args.stage_gate_win_rate,
            stage_gate_episodes=args.stage_gate_episodes,
            stage_gate_extra_steps=args.stage_gate_extra_steps,
            no_stage_gate=args.no_stage_gate,
            warmup_opponent=None if args.no_warmup else args.warmup_opponent,
            warmup_timesteps=0 if args.no_warmup else args.warmup_timesteps,
            **train_kw,
        )
        return
    
    train(
        total_timesteps=timesteps,
        eval_freq=args.eval_freq,
        opponent_type=args.opponent,
        minimax_depth=args.minimax_depth,
        n_envs=args.n_envs,
        max_envs=args.max_envs,
        resume_path=args.resume,
        vec_normalize_path=args.vec_normalize,
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        **train_kw,
    )


if __name__ == "__main__":
    main()
