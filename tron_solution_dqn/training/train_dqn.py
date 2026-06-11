import os
import shutil
import importlib.util
from datetime import datetime
import multiprocessing as mp

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
        raise ImportError("Could not locate tron_solution_dqn package root")
    _d = _parent

import torch
import tron_solution_dqn  # noqa: F401
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.frame_stack_wrapper import GridFrameStackWrapper
from tron_solution.env.opponents import DEFAULT_OPPONENT_TYPE, DEFAULT_MINIMAX_DEPTH
from tron_solution.model.obs import N_STACK
from tron_solution.training.train_ppo import (
    DEFAULT_CURRICULUM,
    DEFAULT_CURRICULUM_TOTAL,
    EarlyStopEvalCallback,
    TronEvalCallback,
    _check_obs_compat,
    _zip_path,
    default_n_envs,
    evaluate_vs_minimax,
    evaluate_vs_opponent,
    parse_curriculum_stages,
)
from tron_solution_dqn.training.policy import MaskedDQN, TronMaskedDQNPolicy, dqn_policy_kwargs


def curriculum_stage_paths(save_dir, run_id, depth):
    model = os.path.join(save_dir, f"tron_dqn_final_{run_id}_d{depth}.zip")
    vec = os.path.join(save_dir, f"vec_normalize_final_{run_id}_d{depth}.pkl")
    return model, vec


def warmup_paths(save_dir, run_id):
    model = os.path.join(save_dir, f"tron_dqn_final_{run_id}_warmup.zip")
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


def train(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    target_update_interval: int = 1000,
    verbose: int = 1,
    save_dir: str = "./dqn_tron_checkpoints",
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
    opponent_type = opponent_type or DEFAULT_OPPONENT_TYPE
    if opponent_type == "minimax" and minimax_depth is None:
        minimax_depth = DEFAULT_MINIMAX_DEPTH

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
    name_prefix = f"tron_dqn_{timestamp}{stage_suffix}"

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

    if resume_path and os.path.isfile(_zip_path(resume_path)):
        print(f"Resuming from {_zip_path(resume_path)}")
        probe = MaskedDQN.load(_zip_path(resume_path), device=device)
        _check_obs_compat(probe.observation_space, env.observation_space, "Model")
        model = MaskedDQN.load(_zip_path(resume_path), env=env, device=device)
        model.learning_rate = learning_rate
    else:
        model = MaskedDQN(
            TronMaskedDQNPolicy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=4,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=0.05,
            policy_kwargs=dqn_policy_kwargs(),
            verbose=verbose,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
            device=device,
        )

    print(f"Starting DQN training for {total_timesteps} timesteps with {n_envs} parallel environments...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="tron_dqn",
        reset_num_timesteps=True,
    )

    best_path = os.path.join(save_dir, "best_model.zip")
    if early_stop and eval_callback.stopped_early and os.path.isfile(best_path):
        print(f"Reloading best eval checkpoint from {best_path}")
        model = MaskedDQN.load(best_path, env=env, device=device)

    final_path = os.path.join(save_dir, f"tron_dqn_final_{timestamp}{stage_suffix}")
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


def train_curriculum(
    curriculum_stages=None,
    total_timesteps=None,
    save_dir="./dqn_tron_checkpoints",
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

    if start_idx >= len(stages):
        print(f"Curriculum run {run_id} already complete ({len(stages)} stages).")
        last_model_path, last_vec_path = resume, vec_norm
        model = MaskedDQN.load(last_model_path)
        curriculum_final = os.path.join(save_dir, f"tron_dqn_curriculum_final_{run_id}")
        model.save(curriculum_final)
        return model, curriculum_final, last_vec_path

    print("Curriculum plan:")
    for i, (depth, steps) in enumerate(stages, 1):
        marker = " (skip)" if i <= start_idx else ""
        print(f"  stage {i}: minimax depth {depth}, {steps} timesteps{marker}")
    print(f"  total: {sum(st for _, st in stages)} timesteps")
    print(f"  run_id: {run_id}")
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
                    f"win_rate={stats['win_rate']:.1%}"
                )
                if stats["win_rate"] >= stage_gate_win_rate:
                    passed = True
                elif stage_gate_extra_steps > 0:
                    w_resume = final_path
                    w_vec = vec_path
                    w_steps = stage_gate_extra_steps
                else:
                    return model, final_path, vec_path
            resume = final_path
            vec_norm = vec_path
            last_model_path = final_path
            last_vec_path = vec_path
            del model
            gc.collect()

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
            print(f"{'=' * 60}\n")
            stage_n_envs = n_envs
            if stage_n_envs is None:
                stage_n_envs = default_n_envs(mp.cpu_count(), "minimax", depth, max_envs)
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

            if no_stage_gate:
                passed = True
            else:
                stats = evaluate_vs_minimax(
                    model, vec_path, depth, n_episodes=stage_gate_episodes,
                )
                print(
                    f"\nStage {i} gate (depth {depth}): "
                    f"{stats['wins']}W {stats['losses']}L {stats['draws']}D / {stage_gate_episodes} "
                    f"win_rate={stats['win_rate']:.1%}"
                )
                if stats["win_rate"] >= stage_gate_win_rate:
                    passed = True
                elif stage_gate_extra_steps > 0:
                    stage_resume = final_path
                    stage_vec = vec_path
                    stage_steps = stage_gate_extra_steps
                else:
                    return model, final_path, vec_path

            resume = final_path
            vec_norm = vec_path
            last_model_path = final_path
            last_vec_path = vec_path
            del model
            gc.collect()

    curriculum_final = os.path.join(save_dir, f"tron_dqn_curriculum_final_{run_id}")
    shutil.copy2(_zip_path(last_model_path), _zip_path(curriculum_final))
    shutil.copy2(
        last_vec_path,
        os.path.join(save_dir, f"vec_normalize_curriculum_final_{run_id}.pkl"),
    )
    model = MaskedDQN.load(_zip_path(last_model_path))
    print(f"\nCurriculum complete! Final model: {curriculum_final}.zip")
    return model, curriculum_final, last_vec_path
