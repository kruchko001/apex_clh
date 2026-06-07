"""
Lightweight CNN Actor-Critic Model for Tron.

Input: (batch, 20, 32, 32) observation tensor (5 channels × 4 stacked frames)
Output: 
  - Actor: (batch, 4) action logits
  - Critic: (batch, 1) value estimate

Architecture:
  - Conv2d(20, 16, 3x3) + ReLU + MaxPool
  - Conv2d(16, 32, 3x3) + ReLU + MaxPool
  - Flatten -> Linear(8192, 128) + ReLU (shared)
  - Actor head: Linear(128, 4)
  - Critic head: Linear(128, 1)

TorchScript compatible for deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TronCNN(nn.Module):
    """Lightweight CNN for Tron game."""
    
    def __init__(self, input_channels: int = 20):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after convolutions
        # Input: (input_channels, 32, 32)
        # After conv1 + pool: (16, 16, 16)
        # After conv2 + pool: (32, 8, 8)
        self.flat_size = 32 * 8 * 8  # 2048
        
        # Shared fully connected layer
        self.shared_fc = nn.Linear(self.flat_size, 128)
        
        # Actor head (outputs action logits)
        self.actor = nn.Linear(128, 4)
        
        # Critic head (outputs value estimate)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 20, 32, 32)
            
        Returns:
            Tuple of (action_logits, value)
                - action_logits: (batch, 4)
                - value: (batch, 1)
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared FC layer
        x = F.relu(self.shared_fc(x))
        
        # Actor and critic heads
        action_logits = self.actor(x)
        value = self.critic(x)
        
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample or select action from policy.
        
        Args:
            obs: Observation tensor of shape (batch, 20, 32, 32) or (20, 32, 32)
            deterministic: If True, select argmax action; otherwise sample
            
        Returns:
            Action tensor of shape (batch,) or scalar
        """
        # Ensure batch dimension
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        action_logits, _ = self.forward(obs)
        
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            return dist.sample()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            obs: Observation tensor of shape (batch, 20, 32, 32)
            actions: Action tensor of shape (batch,)
            
        Returns:
            Tuple of (log_probs, entropy, values)
                - log_probs: (batch,) log probability of taken actions
                - entropy: (batch,) entropy of policy
                - values: (batch, 1) value estimates
        """
        action_logits, values = self.forward(obs)
        
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values


def create_model() -> TronCNN:
    """Create a new TronCNN model."""
    return TronCNN()


def export_to_torchscript(model: TronCNN, path: str) -> None:
    """
    Export model to TorchScript format.
    
    Args:
        model: TronCNN model
        path: Output path for .pt file
    """
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f"Model exported to {path}")


def load_from_torchscript(path: str) -> torch.jit.ScriptModule:
    """
    Load model from TorchScript format.
    
    Args:
        path: Path to .pt file
        
    Returns:
        Loaded ScriptModule
    """
    return torch.jit.load(path)
