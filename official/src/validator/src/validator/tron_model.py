"""
Lightweight Actor-Critic CNN for Tron game.
Designed for CPU inference < 0.05s with input (1, 5, 32, 32) and output (4,) logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TronActorCritic(nn.Module):
    """
    Shallow CNN architecture for Tron game.
    Input: (batch, 5, 32, 32) observation tensor
    Output: actor_logits (4,) and critic_value (1,)
    """
    
    def __init__(self, num_actions: int = 4):
        super(TronActorCritic, self).__init__()
        
        # Feature extraction: shallow CNN (2 conv layers + pooling)
        # Input: (5, 32, 32)
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)  # -> (16, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # -> (32, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (32, 16, 16)
        
        # After pooling: 32 * 16 * 16 = 8192 features
        self.fc_shared = nn.Linear(32 * 16 * 16, 128)
        
        # Actor head: outputs raw logits for 4 actions
        self.actor = nn.Linear(128, num_actions)
        
        # Critic head: outputs value estimate
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for TorchScript compatibility.
        
        Args:
            x: Input tensor of shape (batch, 5, 32, 32)
            
        Returns:
            actor_logits: Raw action logits of shape (batch, 4)
            critic_value: Value estimate of shape (batch, 1)
        """
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layer
        x = F.relu(self.fc_shared(x))
        
        # Actor and critic outputs
        actor_logits = self.actor(x)
        critic_value = self.critic(x)
        
        return actor_logits, critic_value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample or select action from policy.
        
        Args:
            x: Input tensor of shape (batch, 5, 32, 32)
            deterministic: If True, select argmax action; otherwise sample
            
        Returns:
            action: Action tensor of shape (batch,)
        """
        actor_logits, _ = self.forward(x)
        
        if deterministic:
            return torch.argmax(actor_logits, dim=-1)
        else:
            probs = F.softmax(actor_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return dist.sample()
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate log probabilities and entropy for PPO.
        
        Args:
            x: Input tensor of shape (batch, 5, 32, 32)
            actions: Actions tensor of shape (batch,)
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of the policy
            values: Value estimates
        """
        actor_logits, critic_value = self.forward(x)
        
        probs = F.softmax(actor_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, critic_value


def export_to_torchscript(model: TronActorCritic, path: str):
    """
    Export model to TorchScript format.
    
    Args:
        model: Trained TronActorCritic model
        path: Output path for .pt file
    """
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(path)
    print(f"Model exported to {path}")


def load_from_torchscript(path: str) -> TronActorCritic:
    """
    Load model from TorchScript format.
    
    Args:
        path: Path to .pt file
        
    Returns:
        Loaded TronActorCritic model
    """
    return torch.jit.load(path)


if __name__ == "__main__":
    # Test the model
    model = TronActorCritic()
    print(f"Model created: {model}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 5, 32, 32)
    actor_logits, critic_value = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Actor logits shape: {actor_logits.shape}")
    print(f"Critic value shape: {critic_value.shape}")
    
    # Test action sampling
    action = model.get_action(dummy_input, deterministic=False)
    print(f"Sampled action: {action}")
    
    # Test deterministic action
    det_action = model.get_action(dummy_input, deterministic=True)
    print(f"Deterministic action: {det_action}")
    
    # Test evaluation
    log_probs, entropy, values = model.evaluate_actions(dummy_input, action)
    print(f"Log probs: {log_probs}, Entropy: {entropy}, Values: {values}")
    
    # Test TorchScript export
    export_to_torchscript(model, "test_tron_model.pt")
    
    # Test loading
    loaded_model = load_from_torchscript("test_tron_model.pt")
    actor_logits2, critic_value2 = loaded_model(dummy_input)
    print(f"Loaded model output matches: {torch.allclose(actor_logits, actor_logits2)}")
    
    # Clean up test file
    import os
    os.remove("test_tron_model.pt")
    print("All tests passed!")
