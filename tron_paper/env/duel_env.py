import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encode import CHANNELS, GRID, encode_non_stationary, encode_stationary
from .phase import agents_separated, approximate_survival_steps
from competition.tron.tron import GameConfig, TronGame

STEP_REWARD = -1.0
CRASH_REWARD = -100.0
WIN_CRASH_REWARD = 100.0
SEPARATION_SCALE = 1.0


class MRLDuelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int = GRID,
        height: int = GRID,
        max_steps: int = 500,
        opponent_fn: Optional[Callable[[np.ndarray], int]] = None,
        stationary_policy_fn: Optional[Callable[[np.ndarray], int]] = None,
        separation_scale: float = SEPARATION_SCALE,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.opponent_fn = opponent_fn
        self.stationary_policy_fn = stationary_policy_fn
        self.separation_scale = separation_scale
        self.observation_space = spaces.Box(0.0, 1.0, (CHANNELS, height, width), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.game: Optional[TronGame] = None
        self._rng = random.Random()

    def set_opponent_fn(self, fn: Callable[[np.ndarray], int]):
        self.opponent_fn = fn

    def set_stationary_policy_fn(self, fn: Callable[[np.ndarray], int]):
        self.stationary_policy_fn = fn

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        cfg = GameConfig(width=self.width, height=self.height, max_steps=self.max_steps, num_players=2, spawn_mode="random")
        self.game = TronGame(cfg)
        p0 = self.game.get_player(0)
        p1 = self.game.get_player(1)
        sy = self.height - 1 - p0.y
        sx = self.width - 1 - p0.x
        if 1 <= sy < self.height - 1 and 1 <= sx < self.width - 1:
            self.game.grid[p1.y, p1.x] = 0
            p1.y, p1.x = sy, sx
            self.game.grid[sy, sx] = 3
        return encode_non_stationary(self.game, 0), {}

    def step(self, action: int):
        me = self.game.get_player(0)
        opp = self.game.get_player(1)
        pos0 = (me.y, me.x)
        pos1 = (opp.y, opp.x)

        if agents_separated(self.game.grid, pos0, pos1):
            reward = self._separation_reward()
            obs = encode_non_stationary(self.game, 0)
            return obs, reward, True, False, {"phase": "stationary", "separated": True}

        opp_action = 0
        if self.opponent_fn is not None:
            opp_obs = encode_non_stationary(self.game, 1)
            opp_action = int(self.opponent_fn(opp_obs))
        else:
            valid = self.game.get_valid_actions(1)
            opp_action = self._rng.choice(valid) if valid else int(opp.direction)

        _, _, done, info = self.game.step({0: int(action), 1: opp_action})
        me = self.game.get_player(0)
        opp = self.game.get_player(1)

        if done:
            reward = self._terminal_reward(me.alive, opp.alive, info)
            terminated = not info.get("truncated", False)
            truncated = bool(info.get("truncated", False))
        else:
            if agents_separated(self.game.grid, (me.y, me.x), (opp.y, opp.x)):
                reward = self._separation_reward()
                done = True
                terminated = True
                truncated = False
                info = {**info, "phase": "stationary", "separated": True}
            else:
                reward = STEP_REWARD
                terminated = False
                truncated = False

        obs = encode_non_stationary(self.game, 0)
        return obs, reward, terminated, truncated, info

    def _terminal_reward(self, me_alive: bool, opp_alive: bool, info: dict) -> float:
        deaths = set(info.get("deaths_this_step", []))
        if 0 in deaths and 1 in deaths:
            return 0.0
        if 0 in deaths:
            return CRASH_REWARD
        if 1 in deaths:
            return WIN_CRASH_REWARD
        if me_alive and not opp_alive:
            return WIN_CRASH_REWARD
        if opp_alive and not me_alive:
            return CRASH_REWARD
        return 0.0

    def _separation_reward(self) -> float:
        if self.stationary_policy_fn is None:
            return 0.0
        my_obs = encode_stationary(self.game, 0)
        opp_obs = encode_stationary(self.game, 1)
        my_steps = approximate_survival_steps(self.stationary_policy_fn, my_obs)
        opp_steps = approximate_survival_steps(self.stationary_policy_fn, opp_obs)
        return (my_steps - opp_steps) * self.separation_scale

    def action_masks(self) -> np.ndarray:
        valid = set(self.game.get_valid_actions(0))
        return np.array([a in valid for a in range(4)], dtype=bool)
