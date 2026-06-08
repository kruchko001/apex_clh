"""
Export Trained PPO Model to TorchScript.

Usage:
    # Train and export in one step
    python -m tron_solution.export.export_model --train_and_export --timesteps 100000
    
    # Export existing trained model
    python -m tron_solution.export.export_model --model_path ./ppo_tron_checkpoints/tron_ppo_final_*.zip
    
    # Just export (model already trained)
    python -m tron_solution.export.export_model --export_only --model_path path/to/model.zip
"""

import argparse
import os
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

import glob
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tron_solution.env.opponents import DEFAULT_OPPONENT_TYPE, DEFAULT_MINIMAX_DEPTH
from tron_solution.model.tron_cnn import TronCNN
from tron_solution.model.obs import INPUT_CHANNELS, PLAY_SIZE
from tron_solution.model.stacked_policy import export_stacked_policy


def extract_actor_weights(sb3_model: PPO) -> dict:
    """Extract actor network weights from SB3 PPO model."""
    
    # Get the policy network
    policy = sb3_model.policy
    
    # Extract weights
    state_dict = {}
    
    # Conv layers
    if hasattr(policy, 'features_extractor'):
        feat_ext = policy.features_extractor
        
        # Conv1
        if hasattr(feat_ext, 'conv1'):
            state_dict['conv1.weight'] = feat_ext.conv1.weight.data
            state_dict['conv1.bias'] = feat_ext.conv1.bias.data
        
        # Conv2
        if hasattr(feat_ext, 'conv2'):
            state_dict['conv2.weight'] = feat_ext.conv2.weight.data
            state_dict['conv2.bias'] = feat_ext.conv2.bias.data
        
        # Shared FC
        if hasattr(feat_ext, 'shared_fc'):
            state_dict['shared_fc.weight'] = feat_ext.shared_fc.weight.data
            state_dict['shared_fc.bias'] = feat_ext.shared_fc.bias.data
    
    # Actor head
    if hasattr(policy, 'action_net'):
        state_dict['actor.weight'] = policy.action_net.weight.data
        state_dict['actor.bias'] = policy.action_net.bias.data
    
    # Critic head (value_net in SB3)
    if hasattr(policy, 'value_net'):
        # SB3 value_net might be a sequential or direct linear
        if isinstance(policy.value_net, torch.nn.Sequential):
            for i, module in enumerate(policy.value_net):
                if isinstance(module, torch.nn.Linear):
                    state_dict[f'critic.weight'] = module.weight.data
                    state_dict[f'critic.bias'] = module.bias.data
                    break
        elif isinstance(policy.value_net, torch.nn.Linear):
            state_dict['critic.weight'] = policy.value_net.weight.data
            state_dict['critic.bias'] = policy.value_net.bias.data
    
    return state_dict


def export_model(
    model_path: str = None,
    output_path: str = "tron_model.pt",
    train_first: bool = False,
    timesteps: int = 100000,
):
    """
    Export PPO model to TorchScript format.
    
    Args:
        model_path: Path to trained SB3 model (.zip file)
        output_path: Output path for TorchScript model (.pt file)
        train_first: If True, train a new model before exporting
        timesteps: Training timesteps if training first
    """
    
    # Train if requested
    if train_first:
        print(f"Training new model for {timesteps} timesteps...")
        from tron_solution.training.train_ppo import train
        model, env = train(
            total_timesteps=timesteps,
            verbose=1,
            opponent_type=DEFAULT_OPPONENT_TYPE,
            minimax_depth=DEFAULT_MINIMAX_DEPTH,
        )
        
        # Find the saved model
        checkpoint_dir = "./ppo_tron_checkpoints"
        model_files = glob.glob(os.path.join(checkpoint_dir, "tron_ppo_final_*.zip"))
        if model_files:
            model_path = sorted(model_files)[-1]  # Get latest
            print(f"Using trained model: {model_path}")
        else:
            raise RuntimeError("Training completed but no model file found!")
    
    # Load model if path provided
    if model_path is None and not train_first:
        # Try to find latest model
        checkpoint_dir = "./ppo_tron_checkpoints"
        model_files = glob.glob(os.path.join(checkpoint_dir, "tron_ppo_final_*.zip"))
        if model_files:
            model_path = sorted(model_files)[-1]
            print(f"Found existing model: {model_path}")
        else:
            raise ValueError("No model path provided and no trained models found.")
    
    print(f"Loading model from {model_path}...")
    sb3_model = PPO.load(model_path)
    
    # Create our custom model
    custom_model = TronCNN(input_channels=INPUT_CHANNELS, spatial=PLAY_SIZE)
    
    # Extract and load weights
    weights = extract_actor_weights(sb3_model)
    
    if weights:
        # Load available weights
        missing_keys = []
        for key, value in weights.items():
            if key in custom_model.state_dict():
                if custom_model.state_dict()[key].shape == value.shape:
                    custom_model.state_dict()[key].copy_(value)
                    print(f"Loaded {key}")
                else:
                    missing_keys.append(f"{key} (shape mismatch)")
            else:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Warning: Could not load some weights: {missing_keys}")
    else:
        print("Warning: No weights extracted, using random initialization")
    
    # Export to TorchScript
    custom_model.eval()
    export_stacked_policy(custom_model, output_path)
    
    print("\nVerifying exported model (sandbox 1x5x32x32 -> crop+stack -> 4 logits)...")
    loaded_model = torch.jit.load(output_path)
    
    test_input = torch.randn(1, 5, 32, 32)
    with torch.no_grad():
        logits = loaded_model(test_input)
    
    print(f"Inference successful!")
    print(f"  - Output logits shape: {logits.shape}")
    
    import time
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            loaded_model(test_input)
    elapsed = time.time() - start
    avg_time = (elapsed / iterations) * 1000
    
    print(f"  - Average inference time: {avg_time:.2f}ms")
    print(f"  - Meets <50ms requirement: {'Yes' if avg_time < 50 else 'No'}")
    
    print(f"\nModel exported successfully to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PPO model to TorchScript")
    parser.add_argument("--model_path", type=str, help="Path to trained SB3 model (.zip)")
    parser.add_argument("--output", type=str, default="tron_model.pt", help="Output path for .pt file")
    parser.add_argument("--train_and_export", action="store_true", help="Train then export")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model_path,
        output_path=args.output,
        train_first=args.train_and_export,
        timesteps=args.timesteps,
    )


if __name__ == "__main__":
    main()
