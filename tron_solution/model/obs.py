import numpy as np
import torch

FULL_GRID = 32
PLAY_SIZE = 30
MARGIN = 1
GRID_CHANNELS = 4
VALID_DIM = 4
OBS_CHANNELS = GRID_CHANNELS
OBS_CHANNELS_FULL = 5
N_STACK = 4
INPUT_CHANNELS = GRID_CHANNELS * N_STACK


def valid_mask_from_actions(valid_actions) -> np.ndarray:
    mask = np.zeros(4, dtype=np.float32)
    for a in valid_actions:
        if 0 <= a < 4:
            mask[a] = 1.0
    return mask


def apply_action_mask(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.dim() == 1:
        valid = valid.unsqueeze(0)
    if valid.shape[0] != logits.shape[0]:
        valid = valid.expand(logits.shape[0], -1)
    return logits.masked_fill(valid < 0.5, -1e8)


def crop_sandbox_obs_np(obs: np.ndarray) -> np.ndarray:
    return obs[1:, MARGIN : MARGIN + PLAY_SIZE, MARGIN : MARGIN + PLAY_SIZE].astype(np.float32)


def crop_sandbox_obs(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:5, MARGIN : MARGIN + PLAY_SIZE, MARGIN : MARGIN + PLAY_SIZE]


def to_sandbox_obs_np(obs_grid: np.ndarray, walls: np.ndarray) -> np.ndarray:
    full = np.zeros((OBS_CHANNELS_FULL, FULL_GRID, FULL_GRID), dtype=np.float32)
    full[0] = walls.astype(np.float32)
    full[1:5, MARGIN : MARGIN + PLAY_SIZE, MARGIN : MARGIN + PLAY_SIZE] = obs_grid[:GRID_CHANNELS]
    return full


def cnn_flat_size(spatial: int, filters: int = 32) -> int:
    s = spatial // 4
    return filters * s * s
