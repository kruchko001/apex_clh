import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .encode import CHANNELS, GRID, PLAY_SIZE, encode_stationary
from .stationary_gen import generate_stationary_map
from competition.tron.tron import EMPTY, PLAYER_TRAIL_START, WALL, Direction, Player, TronGame, GameConfig
from tron_paper.model.phase_torch import extract_stationary_input
from tron_paper.env.tronbot_reward import tronbot_stationary_score

STEP_REWARD = 0.5
CRASH_PENALTY = -30.0
AVOIDABLE_CRASH_PENALTY = -30.0
SUCCESS_BONUS = 200.0
TB_SHAPING_COEF = 50.0
TB_TERMINAL_COEF = 150.0
TB_SCORE_NORM = 500.0
DEADEND_FILL_THRESHOLD = 0.95
DEADEND_FILL_BONUS = 150.0
DEADEND_FILL_OVER_SCALE = 5.0
USE_TB_REWARD = True


def deadend_fill_reward(fill: float) -> float:
    if fill < DEADEND_FILL_THRESHOLD:
        return 0.0
    bonus = DEADEND_FILL_BONUS
    pct = fill * 100.0
    if pct > DEADEND_FILL_THRESHOLD * 100.0:
        bonus += (pct - DEADEND_FILL_THRESHOLD * 100.0) * DEADEND_FILL_OVER_SCALE
    return bonus


def stationary_input_fill_ratio(obs: np.ndarray) -> float:
    """Fraction of 30x30 cells set in model ch0 (unreachable) + ch1 (head)."""
    t = torch.from_numpy(obs).unsqueeze(0).float()
    inp = extract_stationary_input(t)[0]
    filled = (inp[0] + inp[1]).clamp(0.0, 1.0)
    return float(filled.sum().item() / (PLAY_SIZE * PLAY_SIZE))


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
        safe_actions = [a for a in range(4) if self.action_masks()[a]]
        pre_obs = self._obs()
        pre_tb = tronbot_stationary_score(self.grid, self.player.y, self.player.x) if USE_TB_REWARD else 0.0
        self.step_count += 1
        self.player.apply_absolute_action(int(action))
        dy, dx = Direction(int(action)).delta
        ny, nx = self.player.y + dy, self.player.x + dx
        terminated = False
        truncated = False
        reward = STEP_REWARD
        info = {}

        if ny <= 0 or nx <= 0 or ny >= self.height - 1 or nx >= self.width - 1:
            terminated = True
            reward += CRASH_PENALTY
        elif self.grid[ny, nx] == WALL or self.grid[ny, nx] >= PLAYER_TRAIL_START:
            terminated = True
            reward += CRASH_PENALTY
        else:
            self.grid[self.player.y, self.player.x] = PLAYER_TRAIL_START
            self.player.move(self.step_count)
            self.player.y, self.player.x = ny, nx
            self.grid[ny, nx] = PLAYER_TRAIL_START
            if USE_TB_REWARD:
                post_tb = tronbot_stationary_score(self.grid, self.player.y, self.player.x)
                reward += TB_SHAPING_COEF * (post_tb - pre_tb) / TB_SCORE_NORM
                info["tb_score"] = post_tb
                info["tb_delta"] = post_tb - pre_tb

        if terminated and safe_actions:
            reward += AVOIDABLE_CRASH_PENALTY
            info["avoidable_crash"] = True
        elif terminated and not safe_actions:
            fill = stationary_input_fill_ratio(pre_obs)
            info["model_fill_ratio"] = fill
            if USE_TB_REWARD:
                tb_final = tronbot_stationary_score(self.grid, self.player.y, self.player.x)
                reward = STEP_REWARD + TB_TERMINAL_COEF * tb_final / TB_SCORE_NORM
                info["tb_terminal_score"] = tb_final
            elif fill >= DEADEND_FILL_THRESHOLD:
                fill_bonus = deadend_fill_reward(fill)
                reward = STEP_REWARD + fill_bonus
                info["deadend_fill_reward"] = True
                info["deadend_fill_bonus"] = fill_bonus

        if self.step_count >= self.max_steps:
            truncated = True
            reward += SUCCESS_BONUS

        return self._obs(), reward, terminated, truncated, info

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
