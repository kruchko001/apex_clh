"""
Custom Gymnasium Environment for Tron Light Cycles Game.

Observation Space: (4, 30, 30) — trails + heads

Action Space: Discrete(4)
  - 0: UP
  - 1: RIGHT
  - 2: DOWN
  - 3: LEFT

Training reward:
  - Per-step: -0.01 + space shaping | Win: +15 | Loss: -10 | Draw: -8
  - info["official_score"] still uses competition cascade for eval
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .opponents import get_opponent, DEFAULT_OPPONENT_TYPE
from ..model.obs import GRID_CHANNELS, PLAY_SIZE, crop_sandbox_obs_np


class TronEnv(gym.Env):
    """Tron Light Cycles environment."""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    # Action constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    SPAWN_P0 = (1, 1)
    SPAWN_P1 = (30, 30)
    SPAWN_P0_DIR = DOWN
    SPAWN_P1_DIR = UP
    OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    STEP_REWARD = -0.01
    WIN_REWARD = 15.0
    LOSS_REWARD = -10.0
    DRAW_REWARD = -8.0
    SPACE_COEF = 0.02
    SPACE_NORM = PLAY_SIZE * PLAY_SIZE
    
    # Direction vectors
    DIRECTIONS = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1),
    }
    
    def __init__(self, grid_size: int = 32, max_steps: int = 500, render_mode: Optional[str] = None,
                 opponent_type: str = None, minimax_depth: int = None,
                 render_my_label: Optional[str] = None, render_opp_label: Optional[str] = None,
                 play_as: int = 0):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_my_label = render_my_label
        self.render_opp_label = render_opp_label
        self.play_as = play_as
        
        self.opponent_type = opponent_type or DEFAULT_OPPONENT_TYPE
        self.minimax_depth = minimax_depth
        self.opponent = get_opponent(self.opponent_type, self.minimax_depth)

        self.observation_space = spaces.Box(
            0.0, 1.0, (GRID_CHANNELS, PLAY_SIZE, PLAY_SIZE), dtype=np.float32,
        )
        
        # Action space: Discrete(4)
        self.action_space = spaces.Discrete(4)
        
        # State variables
        self.my_trail = None
        self.opponent_trail = None
        self.my_head = None
        self.opponent_head = None
        self.current_direction = None
        self.opponent_direction = None
        self.step_count = 0
        self.walls = None
        
        # For rendering
        self.window = None
        self.clock = None
        self.cell_size = 20
        self.render_message = None

    def set_minimax_depth(self, depth: int):
        self.minimax_depth = depth
        if hasattr(self.opponent, "max_depth"):
            self.opponent.max_depth = depth
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        play_as = opts.get("play_as", self.play_as)

        self.my_trail = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.opponent_trail = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.walls[0, :] = True
        self.walls[-1, :] = True
        self.walls[:, 0] = True
        self.walls[:, -1] = True

        if play_as == 0:
            self.my_head = self.SPAWN_P0
            self.opponent_head = self.SPAWN_P1
            self.current_direction = self.SPAWN_P0_DIR
            self.opponent_direction = self.SPAWN_P1_DIR
        else:
            self.my_head = self.SPAWN_P1
            self.opponent_head = self.SPAWN_P0
            self.current_direction = self.SPAWN_P1_DIR
            self.opponent_direction = self.SPAWN_P0_DIR

        self.my_trail[self.my_head] = True
        self.opponent_trail[self.opponent_head] = True
        self.step_count = 0
        if hasattr(self.opponent, "reset"):
            self.opponent.reset()

        return self._get_observation(), {"step_count": 0, "play_as": play_as}

    @property
    def grid(self) -> np.ndarray:
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        g[self.walls] = -1
        g[self.my_trail] = 1
        g[self.opponent_trail] = 2
        return g

    @grid.setter
    def grid(self, value: np.ndarray):
        self.walls = value == -1
        self.my_trail = value == 1
        self.opponent_trail = value == 2

    @property
    def p1_pos(self):
        return self.my_head

    @p1_pos.setter
    def p1_pos(self, value):
        self.my_head = value

    @property
    def p2_pos(self):
        return self.opponent_head

    @p2_pos.setter
    def p2_pos(self, value):
        self.opponent_head = value

    @property
    def p1_dir(self):
        return self.current_direction

    @p1_dir.setter
    def p1_dir(self, value):
        self.current_direction = value

    @property
    def p2_dir(self):
        return self.opponent_direction

    @p2_dir.setter
    def p2_dir(self, value):
        self.opponent_direction = value
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if action != self.OPPOSITE[self.current_direction]:
            self.current_direction = action
        self._move_opponent()
        return self._step_after_directions()

    def step_dual(self, my_action: int, opp_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if my_action != self.OPPOSITE[self.current_direction]:
            self.current_direction = my_action
        if opp_action != self.OPPOSITE[self.opponent_direction]:
            self.opponent_direction = opp_action
        return self._step_after_directions()

    def _step_after_directions(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        dr, dc = self.DIRECTIONS[self.current_direction]
        new_my_head = (self.my_head[0] + dr, self.my_head[1] + dc)
        dr_opp, dc_opp = self.DIRECTIONS[self.opponent_direction]
        new_opponent_head = (self.opponent_head[0] + dr_opp, self.opponent_head[1] + dc_opp)

        my_collision, my_collision_type = self._check_collision(new_my_head)
        opp_collision, opp_collision_type = self._check_collision(new_opponent_head)
        if not my_collision and not opp_collision and new_my_head == new_opponent_head:
            my_collision = opp_collision = True
            my_collision_type = opp_collision_type = "head_on"

        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "my_collision_type": my_collision_type if my_collision else "",
            "opponent_collision_type": opp_collision_type if opp_collision else "",
            "clean_kill": False,
            "opponent_self_destruct": False,
            "mutual_destruction": False,
            "timeout": False,
            "truncated": False,
            "official_score": 0.0,
            "log_step": self.step_count,
        }

        if not my_collision and not opp_collision:
            self.my_head = new_my_head
            self.opponent_head = new_opponent_head
            self.my_trail[self.my_head] = True
            self.opponent_trail[self.opponent_head] = True
            my_space = self._count_space(self.my_head)
            opp_space = self._count_space(self.opponent_head)
            reward = self.STEP_REWARD + self.SPACE_COEF * (my_space - opp_space) / self.SPACE_NORM
            if self.step_count >= self.max_steps:
                truncated = True
                info["timeout"] = True
                reward = self.LOSS_REWARD * 0.5
                info["official_score"] = 0.25
        elif my_collision and opp_collision:
            terminated = True
            info["mutual_destruction"] = True
            if opp_collision_type == "my_trail":
                reward = self.WIN_REWARD * 0.4
                info["official_score"] = 0.40
                info["clean_kill"] = True
            elif my_collision_type == "opponent_trail":
                reward = self.LOSS_REWARD * 0.5
                info["official_score"] = 0.10
            else:
                reward = self.DRAW_REWARD
                info["official_score"] = 0.40
        elif my_collision:
            terminated = True
            reward = self.LOSS_REWARD
            info["official_score"] = 0.0
        else:
            terminated = True
            if opp_collision_type == "my_trail":
                reward = self.WIN_REWARD
                info["official_score"] = 1.0
                info["clean_kill"] = True
            elif opp_collision_type in ("wall", "my_trail"):
                reward = self.WIN_REWARD * 0.8
                info["official_score"] = 0.80
                info["opponent_self_destruct"] = True
            else:
                reward = self.WIN_REWARD * 0.5
                info["official_score"] = 0.10

        info["truncated"] = truncated
        return self._get_observation(), reward, terminated, truncated, info

    def valid_actions(self) -> np.ndarray:
        mask = np.zeros(4, dtype=bool)
        reverse = self.OPPOSITE[self.current_direction]
        for a in range(4):
            if a == reverse:
                continue
            dr, dc = self.DIRECTIONS[a]
            r, c = self.my_head[0] + dr, self.my_head[1] + dc
            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                continue
            if self.walls[r, c] or self.my_trail[r, c] or self.opponent_trail[r, c]:
                continue
            mask[a] = True
        if not mask.any():
            mask[self.current_direction] = True
        return mask
    
    def _move_opponent(self):
        """Move opponent using the selected opponent strategy."""
        # Get opponent action from the opponent class
        action = self.opponent.get_action(
            obs=self._get_grid_obs(),
            my_head=self.my_head,
            opp_head=self.opponent_head,
            grid=self._get_grid(),
            current_dir=self.opponent_direction,
            my_dir=self.current_direction,
        )
        if action != self.OPPOSITE[self.opponent_direction]:
            self.opponent_direction = action
    
    def _count_space(self, start: Tuple[int, int]) -> int:
        grid = self._get_grid()
        h, w = grid.shape
        visited = set()
        stack = [start]
        count = 0
        while stack and count < 2000:
            pos = stack.pop()
            if pos in visited:
                continue
            visited.add(pos)
            count += 1
            r, c = pos
            for dr, dc in self.DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return count

    def _get_grid(self) -> np.ndarray:
        """Get internal grid representation for opponent AI.
        
        Returns:
            grid: (32, 32) int array where:
                0 = empty
                1 = wall
                2 = my_trail
                3 = opponent_trail
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.walls] = 1
        grid[self.my_trail] = 2
        grid[self.opponent_trail] = 3
        return grid
    
    def _check_collision(self, position: Tuple[int, int]) -> Tuple[bool, str]:
        """Check if a position results in collision. Returns (is_collision, collision_type)."""
        r, c = position
        
        # Out of bounds (shouldn't happen with walls, but check anyway)
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True, "wall"
        
        # Hit wall
        if self.walls[r, c]:
            return True, "wall"
        
        # Hit my trail
        if self.my_trail[r, c]:
            return True, "my_trail"
        
        # Hit opponent trail
        if self.opponent_trail[r, c]:
            return True, "opponent_trail"
        
        return False, ""

    def _get_grid_obs(self) -> np.ndarray:
        full = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        full[0] = self.walls.astype(np.float32)
        full[1] = self.my_trail.astype(np.float32)
        full[2] = self.opponent_trail.astype(np.float32)
        full[3, self.my_head[0], self.my_head[1]] = 1.0
        full[4, self.opponent_head[0], self.opponent_head[1]] = 1.0
        return crop_sandbox_obs_np(full)

    def _get_observation(self) -> np.ndarray:
        return self._get_grid_obs()
    
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None
    
    def _render_human(self):
        """Render the environment in human-readable mode."""
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for human rendering. Install with: pip install pygame")
        
        cs = self.cell_size
        w = self.grid_size * cs
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((w, w))
            cap = "Tron Light Cycles"
            if self.render_my_label and self.render_opp_label:
                cap = f"Tron: {self.render_my_label} vs {self.render_opp_label}"
            pygame.display.set_caption(cap)
            self.clock = pygame.time.Clock()
            self._label_font = pygame.font.SysFont("Arial", 11, bold=True)
            self._status_font = pygame.font.SysFont("Arial", 24, bold=True)
            self._big_font = pygame.font.SysFont("Arial", 48, bold=True)
            self._hint_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        black = (0, 0, 0)
        wall = (70, 70, 70)
        grid_line = (28, 28, 28)
        blue = (0, 120, 255)
        red = (255, 50, 50)
        my_head = (100, 200, 255)
        opp_head = (255, 100, 100)
        white = (255, 255, 255)

        self.window.fill(black)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x, y = c * cs, r * cs
                rect = pygame.Rect(x, y, cs, cs)
                if self.walls[r, c]:
                    pygame.draw.rect(self.window, wall, rect)
                elif self.my_trail[r, c]:
                    pygame.draw.rect(self.window, blue, rect)
                elif self.opponent_trail[r, c]:
                    pygame.draw.rect(self.window, red, rect)

        for c in range(self.grid_size + 1):
            x = c * cs
            pygame.draw.line(self.window, grid_line, (x, 0), (x, w))
        for r in range(self.grid_size + 1):
            y = r * cs
            pygame.draw.line(self.window, grid_line, (0, y), (w, y))

        pygame.draw.rect(
            self.window, my_head,
            (self.my_head[1] * cs, self.my_head[0] * cs, cs, cs)
        )
        pygame.draw.rect(
            self.window, opp_head,
            (self.opponent_head[1] * cs, self.opponent_head[0] * cs, cs, cs)
        )

        def head_label(pos, text):
            surf = self._label_font.render(text, True, white)
            cx = pos[1] * cs + cs // 2
            cy = pos[0] * cs + cs // 2
            self.window.blit(surf, surf.get_rect(center=(cx, cy)))

        if self.render_my_label:
            head_label(self.my_head, self.render_my_label)
        if self.render_opp_label:
            head_label(self.opponent_head, self.render_opp_label)

        status = f"Step {self.step_count}"
        if self.render_my_label and self.render_opp_label:
            status = f"{self.render_my_label} vs {self.render_opp_label}  |  {status}"
        self.window.blit(self._status_font.render(status, True, white), (10, 10))

        if self.render_message:
            msg_surf = self._big_font.render(self.render_message, True, white)
            self.window.blit(msg_surf, msg_surf.get_rect(center=(w // 2, w // 2)))
            hint = "Press SPACE for next duel  |  ESC to quit"
            hint_surf = self._hint_font.render(hint, True, (200, 200, 200))
            self.window.blit(hint_surf, hint_surf.get_rect(center=(w // 2, w // 2 + 50)))

        pygame.display.flip()
        self.clock.tick(30 if self.render_message else self.metadata["render_fps"])
        
        return self.window
    
    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array."""
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Walls (gray)
        img[self.walls] = [100, 100, 100]
        
        # My trail (blue)
        img[self.my_trail] = [0, 0, 255]
        
        # Opponent trail (red)
        img[self.opponent_trail] = [255, 0, 0]
        
        # My head (light blue)
        img[self.my_head] = [100, 100, 255]
        
        # Opponent head (pink)
        img[self.opponent_head] = [255, 100, 100]
        
        # Scale up for better visibility
        img = np.repeat(np.repeat(img, self.cell_size, axis=0), self.cell_size, axis=1)
        
        return img
    
    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None
            self.clock = None


# Register the environment
gym.register(
    id="Tron-v0",
    entry_point="tron_solution.env.tron_env:TronEnv",
    max_episode_steps=500,
)
