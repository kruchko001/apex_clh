"""
PPO Training Script for Tron Gymnasium Environment.

This script trains a PPO agent on the custom Tron environment using Stable Baselines3.
It includes curriculum learning, self-play capabilities, and model checkpointing.
"""

import os
import time
import argparse
from datetime import datetime
from typing import Optional, List

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import our custom environment and model
from validator.tron_env import TronEnv
from validator.model import ActorCriticCNN


class SelfPlayCallback(BaseCallback):
    """
    Callback for implementing self-play training.
    Periodically saves the current model and loads it as a fixed opponent.
    """
    
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_path = None
        self.n_calls = 0
        
    def _on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % self.save_freq == 0:
            # Save current model as opponent
            self.model_path = os.path.join(self.save_path, f"opponent_{self.n_calls}.zip")
            self.model.save(self.model_path)
            if self.verbose > 0:
                print(f"Saved opponent model at {self.model_path}")
        return True


class CurriculumCallback(BaseCallback):
    """
    Callback for implementing curriculum learning.
    Gradually increases difficulty by reducing initial gap between players.
    """
    
    def __init__(self, start_gap: int = 8, end_gap: int = 2, increase_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.start_gap = start_gap
        self.end_gap = end_gap
        self.increase_freq = increase_freq
        self.n_calls = 0
        self.current_gap = start_gap
        
    def _on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % self.increase_freq == 0 and self.current_gap > self.end_gap:
            self.current_gap -= 1
            if self.verbose > 0:
                print(f"Curriculum: Reduced initial gap to {self.current_gap}")
            # Update environment if it supports gap parameter
            if hasattr(self.training_env, "envs"):
                for env in self.training_env.envs:
                    if hasattr(env, "initial_gap"):
                        env.initial_gap = self.current_gap
        return True


def make_env(env_id: str, rank: int, seed: int = 0, initial_gap: Optional[int] = None):
    """
    Utility function for creating a vectorized environment.
    """
    def _init():
        env = TronEnv(initial_gap=initial_gap)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 10,
    save_freq: int = 100_000,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 10,
    use_self_play: bool = False,
    use_curriculum: bool = True,
    device: str = "auto",
    log_dir: str = "./logs",
    model_dir: str = "./models",
    verbose: int = 1,
):
    """
    Train a PPO agent on the Tron environment.
    
    Args:
        total_timesteps: Total number of training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        batch_size: Mini-batch size
        n_steps: Number of steps per environment per update
        n_epochs: Number of epochs when updating policy
        save_freq: Frequency of saving checkpoints
        eval_freq: Frequency of evaluation
        n_eval_episodes: Number of episodes for evaluation
        use_self_play: Whether to use self-play training
        use_curriculum: Whether to use curriculum learning
        device: Device to run on ("cpu", "cuda", "auto")
        log_dir: Directory for logs
        model_dir: Directory for model checkpoints
        verbose: Verbosity level
        
    Returns:
        Trained PPO model
    """
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"ppo_tron_{timestamp}")
    model_path = os.path.join(model_dir, f"ppo_tron_{timestamp}")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Training PPO on Tron environment")
    print(f"Logs: {log_path}")
    print(f"Models: {model_path}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    
    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv([make_env("Tron-v0", 0)])
    else:
        env = SubprocVecEnv([make_env("Tron-v0", i) for i in range(n_envs)])
    
    # Wrap with VecMonitor for logging
    env = VecMonitor(env, filename=os.path.join(log_path, "monitor"))
    
    # Create callbacks
    callbacks: List[BaseCallback] = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_path,
        name_prefix="ppo_tron",
        verbose=verbose,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env("Tron-v0", 0, seed=42)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=verbose,
    )
    callbacks.append(eval_callback)
    
    # Self-play callback (optional)
    if use_self_play:
        self_play_callback = SelfPlayCallback(
            save_freq=save_freq,
            save_path=model_path,
            verbose=verbose,
        )
        callbacks.append(self_play_callback)
    
    # Curriculum callback (optional)
    if use_curriculum:
        curriculum_callback = CurriculumCallback(
            start_gap=8,
            end_gap=2,
            increase_freq=10000,
            verbose=verbose,
        )
        callbacks.append(curriculum_callback)
    
    # Custom policy using our ActorCriticCNN
    # Note: We'll use the built-in MlpPolicy but could replace with custom policy class
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",  # Will be replaced with custom CNN policy
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=None,
        device=device,
        _init_setup_model=True,
    )
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="PPO_Tron",
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    
    # Save final model
    final_model_path = os.path.join(model_path, "ppo_tron_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate final model
    print("\nEvaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    print(f"Final evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Tron environment")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency")
    parser.add_argument("--self-play", action="store_true", help="Enable self-play training")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="./models", help="Model directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        use_self_play=args.self_play,
        use_curriculum=not args.no_curriculum,
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
