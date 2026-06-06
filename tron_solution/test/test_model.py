"""
Test exported TorchScript model.

Usage:
    python -m tron_solution.test.test_model --model_path tron_model.pt --episodes 5
"""

import argparse
import torch
import numpy as np
from typing import Optional

from tron_solution.env.tron_env import TronEnv


def test_model(model_path: str, num_episodes: int = 5, render: bool = False):
    """
    Test exported TorchScript model.
    
    Args:
        model_path: Path to .pt model file
        num_episodes: Number of episodes to run
        render: Whether to render episodes
    """
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()
    
    env = TronEnv(grid_size=32, max_steps=500, render_mode="rgb_array" if render else None)
    
    total_rewards = []
    total_steps = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Prepare observation
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                logits, value = model(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # Determine outcome
        outcome = "Draw"
        if info.get("clean_kill") or info.get("opponent_self_destruct"):
            outcome = "Win"
            wins += 1
        elif info.get("my_collision_type") and not info.get("opponent_collision_type"):
            outcome = "Loss"
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}, Outcome={outcome}")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Wins: {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print("="*50)
    
    return {
        "wins": wins,
        "win_rate": wins / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Test exported TorchScript model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
