import numpy as np
from tron_solution.model.obs import GRID_CHANNELS, N_STACK


class FrameStack:
    def __init__(self, n_stack: int = N_STACK):
        self.n_stack = n_stack
        self.frames = []

    def reset(self, grid: np.ndarray) -> np.ndarray:
        z = np.zeros((GRID_CHANNELS, grid.shape[1], grid.shape[2]), dtype=np.float32)
        self.frames = [z.copy() for _ in range(self.n_stack - 1)] + [grid.astype(np.float32).copy()]
        return np.concatenate(self.frames, axis=0)

    def step(self, grid: np.ndarray) -> np.ndarray:
        grid = grid.astype(np.float32)
        self.frames.pop(0)
        self.frames.append(grid.copy())
        return np.concatenate(self.frames, axis=0)
