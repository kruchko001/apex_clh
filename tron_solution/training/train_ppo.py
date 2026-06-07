"""
PPO Training Script for Tron Environment.

Usage:
    python -m tron_solution.training.train_ppo --timesteps 100000 --verbose
"""

import argparse
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
import multiprocessing as mp

from tron_solution.env.tron_env import TronEnv
from tron_solution.model.tron_cnn import TronCNN


class TronFeaturesExtractor(torch.nn.Module):
    """Custom feature extractor for SB3."""
    
    def __init__(self, observation_space):
        super().__init__()
        
        # Same architecture as TronCNN
        self.conv1 = torch.nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        self.flat_size = 32 * 8 * 8
        
        self.shared_fc = torch.nn.Linear(self.flat_size, 128)
        
    def forward(self, x) -> torch.Tensor:
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.shared_fc(x))
        return x


def train(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
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
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create parallel environments using SubprocVecEnv
    num_envs = min(mp.cpu_count(), 8)  # Use up to 8 parallel environments
    print(f"Creating {num_envs} parallel environments...")
    
    def make_env():
        return TronEnv(grid_size=32, max_steps=500)
    
    # Use SubprocVecEnv for parallel execution
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix=f"tron_ppo_{timestamp}",
        verbose=verbose,
    )
    
    # Create evaluation environment (single env for eval)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False, norm_reward=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    
    # Create PPO model with custom policy
    model = PPO(
        "CnnPolicy",
        env,
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
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps with {num_envs} parallel environments...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="tron_ppo",
    )
    
    # Save final model and vec_normalize
    final_path = os.path.join(save_dir, f"tron_ppo_final_{timestamp}")
    model.save(final_path)
    env.save(os.path.join(save_dir, f"vec_normalize_final_{timestamp}.pkl"))
    
    print(f"Training complete! Model saved to {final_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model, env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Tron environment")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--save-dir", type=str, default="./ppo_tron_checkpoints", help="Save directory")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=args.verbose,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
