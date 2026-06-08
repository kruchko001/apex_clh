import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from tron_solution.model.obs import INPUT_CHANNELS, PLAY_SIZE, VALID_DIM, cnn_flat_size
from tron_solution.model.valid_from_grid import valid_mask_from_grid


class TronCNN(nn.Module):
    def __init__(self, input_channels: int = INPUT_CHANNELS, spatial: int = PLAY_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flat_size = cnn_flat_size(spatial)
        self.shared_fc = nn.Linear(self.flat_size + VALID_DIM, 128)
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = valid_mask_from_grid(grid)
        x = self.pool(F.relu(self.conv1(grid)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, valid], dim=1)
        x = F.relu(self.shared_fc(x))
        logits = self.actor(x)
        logits = logits.masked_fill(valid < 0.5, -1e8)
        return logits, self.critic(x)


def create_model() -> TronCNN:
    return TronCNN()


def export_to_torchscript(model: TronCNN, path: str) -> None:
    model.eval()
    torch.jit.script(model).save(path)


def load_from_torchscript(path: str) -> torch.jit.ScriptModule:
    return torch.jit.load(path)
