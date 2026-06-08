import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tron_solution.model.obs import GRID_CHANNELS, N_STACK, PLAY_SIZE, VALID_DIM


class GridFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_stack: int = N_STACK):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = []
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                0.0, 1.0,
                (GRID_CHANNELS * n_stack, PLAY_SIZE, PLAY_SIZE),
                dtype=np.float32,
            ),
            "valid": spaces.Box(0.0, 1.0, (VALID_DIM,), dtype=np.float32),
        })

    def _stack(self, grid: np.ndarray, valid: np.ndarray):
        z = np.zeros((GRID_CHANNELS, PLAY_SIZE, PLAY_SIZE), dtype=np.float32)
        if not self.frames:
            self.frames = [z.copy() for _ in range(self.n_stack - 1)]
        self.frames.append(grid.astype(np.float32).copy())
        if len(self.frames) > self.n_stack:
            self.frames.pop(0)
        return {
            "grid": np.concatenate(self.frames, axis=0),
            "valid": valid.astype(np.float32),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = []
        valid = self.env.valid_actions().astype(np.float32)
        return self._stack(obs, valid), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        valid = self.env.valid_actions().astype(np.float32)
        return self._stack(obs, valid), reward, terminated, truncated, info
