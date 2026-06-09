import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encode import CHANNELS, GRID, encode_stationary
from .stationary_gen import generate_stationary_map
from competition.tron.tron import EMPTY, PLAYER_TRAIL_START, WALL, Direction, Player, TronGame, GameConfig

STEP_REWARD = -1.0
CRASH_REWARD = -100.0


class StationaryTronEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, width: int = GRID, height: int = GRID, max_steps: int = 500):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.observation_space = spaces.Box(0.0, 1.0, (CHANNELS, height, width), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.grid = None
        self.player = None
        self.step_count = 0
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self.grid = generate_stationary_map(self.width, self.height, self._rng)
        sy = self._rng.randint(1, self.height - 2)
        sx = self._rng.randint(1, self.width - 2)
        while self.grid[sy, sx] != EMPTY:
            sy = self._rng.randint(1, self.height - 2)
            sx = self._rng.randint(1, self.width - 2)
        direction = Direction(self._rng.randint(0, 3))
        self.player = Player(0, sy, sx, direction)
        self.grid[sy, sx] = PLAYER_TRAIL_START
        self.step_count = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        game = TronGame(GameConfig(width=self.width, height=self.height, max_steps=self.max_steps))
        game.grid = self.grid.copy()
        game.players = [self.player]
        game.game_over = False
        game.step_count = self.step_count
        return encode_stationary(game, 0)

    def step(self, action: int):
        self.step_count += 1
        self.player.apply_absolute_action(int(action))
        dy, dx = Direction(int(action)).delta
        ny, nx = self.player.y + dy, self.player.x + dx
        terminated = False
        truncated = False
        reward = STEP_REWARD

        if ny <= 0 or nx <= 0 or ny >= self.height - 1 or nx >= self.width - 1:
            terminated = True
            reward = CRASH_REWARD
        elif self.grid[ny, nx] == WALL or self.grid[ny, nx] >= PLAYER_TRAIL_START:
            terminated = True
            reward = CRASH_REWARD
        else:
            self.grid[self.player.y, self.player.x] = PLAYER_TRAIL_START
            self.player.move(self.step_count)
            self.player.y, self.player.x = ny, nx
            self.grid[ny, nx] = PLAYER_TRAIL_START

        if self.step_count >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, {}

    def action_masks(self) -> np.ndarray:
        valid = set()
        for a in range(4):
            d = Direction(a)
            if d == self.player.direction.opposite:
                continue
            dy, dx = d.delta
            ny, nx = self.player.y + dy, self.player.x + dx
            if 1 <= ny < self.height - 1 and 1 <= nx < self.width - 1:
                if self.grid[ny, nx] == EMPTY:
                    valid.add(a)
        return np.array([a in valid for a in range(4)], dtype=bool)
