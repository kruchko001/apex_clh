"""
Custom Gymnasium Environment for Tron Light Cycles Game.

Observation Space: (5, 32, 32) float tensor
  - Channel 0: Walls (border)
  - Channel 1: My trail
  - Channel 2: Opponent trail
  - Channel 3: My head position
  - Channel 4: Opponent head position

Action Space: Discrete(4)
  - 0: UP
  - 1: RIGHT
  - 2: DOWN
  - 3: LEFT

Reward Shaping:
  - Clean kill (opponent hits my trail): +2.0
  - Opponent self-destructs (hits wall/own trail): +1.5
  - Mutual destruction (head-on or simultaneous): +0.5
  - Timeout draw (500 steps): 0.0
  - Die alone (hit wall/own trail): -2.0
  - Step survival: +0.01
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any
from .opponents import get_opponent, BaseOpponent


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
    
    # Direction vectors
    DIRECTIONS = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1),
    }
    
    def __init__(self, grid_size: int = 32, max_steps: int = 500, render_mode: Optional[str] = None,
                 opponent_type: str = "heuristic"):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Initialize opponent
        self.opponent = get_opponent(opponent_type)
        
        # Observation space: (5, 32, 32) float tensor
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, grid_size, grid_size),
            dtype=np.float32
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
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize trails as boolean grids
        self.my_trail = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.opponent_trail = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Initialize walls (border)
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.walls[0, :] = True  # Top border
        self.walls[-1, :] = True  # Bottom border
        self.walls[:, 0] = True  # Left border
        self.walls[:, -1] = True  # Right border
        
        # Random starting positions (not on borders)
        margin = 2
        my_start = self.np_random.integers(margin, self.grid_size - margin, size=2)
        opponent_start = self.np_random.integers(margin, self.grid_size - margin, size=2)
        
        # Ensure they don't start at the same position
        while np.array_equal(my_start, opponent_start):
            opponent_start = self.np_random.integers(margin, self.grid_size - margin, size=2)
        
        self.my_head = tuple(my_start)
        self.opponent_head = tuple(opponent_start)
        
        # Mark initial positions in trails
        self.my_trail[self.my_head] = True
        self.opponent_trail[self.opponent_head] = True
        
        # Random initial directions
        self.current_direction = self.np_random.choice([self.UP, self.RIGHT, self.DOWN, self.LEFT])
        self.opponent_direction = self.np_random.choice([self.UP, self.RIGHT, self.DOWN, self.LEFT])
        
        self.step_count = 0
        
        return self._get_observation(), {"step_count": 0}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Prevent 180-degree turns
        opposite_directions = {
            self.UP: self.DOWN,
            self.DOWN: self.UP,
            self.LEFT: self.RIGHT,
            self.RIGHT: self.LEFT,
        }
        
        if action != opposite_directions.get(self.current_direction):
            self.current_direction = action
        
        # Move opponent with simple AI (avoid immediate collisions)
        self._move_opponent()
        
        # Calculate new positions
        dr, dc = self.DIRECTIONS[self.current_direction]
        new_my_head = (self.my_head[0] + dr, self.my_head[1] + dc)
        
        dr_opp, dc_opp = self.DIRECTIONS[self.opponent_direction]
        new_opponent_head = (self.opponent_head[0] + dr_opp, self.opponent_head[1] + dc_opp)
        
        # Check collisions
        my_collision, my_collision_type = self._check_collision(new_my_head)
        opp_collision, opp_collision_type = self._check_collision(new_opponent_head)
        
        # Determine outcome and rewards
        reward = 0.0
        terminated = False
        info = {
            "my_collision_type": my_collision_type,
            "opponent_collision_type": opp_collision_type,
            "clean_kill": False,
            "opponent_self_destruct": False,
            "mutual_destruction": False,
            "timeout": False,
        }
        
        # Both survive this step
        if not my_collision and not opp_collision:
            # Update positions
            self.my_head = new_my_head
            self.opponent_head = new_opponent_head
            
            # Add to trails
            self.my_trail[self.my_head] = True
            self.opponent_trail[self.opponent_head] = True
            
            # Survival reward
            reward = 0.01
            self.step_count += 1
            
            # Check timeout
            if self.step_count >= self.max_steps:
                terminated = True
                info["timeout"] = True
                # Draw: no additional reward
                
        # Both collide (mutual destruction)
        elif my_collision and opp_collision:
            terminated = True
            reward = 0.5
            info["mutual_destruction"] = True
            
        # Only I collide (I die)
        elif my_collision:
            terminated = True
            reward = -2.0
            
            # Check if opponent hit my trail (clean kill for them, but we track it)
            if opp_collision_type == "trail" and new_opponent_head == new_my_head:
                # Head-on collision already handled above
                pass
                
        # Only opponent collides (I win)
        elif opp_collision:
            terminated = True
            
            # Determine type of opponent collision
            if opp_collision_type == "trail" and new_opponent_head in [tuple(p) for p in np.argwhere(self.my_trail)]:
                # Opponent hit my trail - clean kill
                reward = 2.0
                info["clean_kill"] = True
            else:
                # Opponent self-destructed (hit wall or own trail)
                reward = 1.5
                info["opponent_self_destruct"] = True
        
        obs = self._get_observation()
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _move_opponent(self):
        """Move opponent using the selected opponent strategy."""
        # Get opponent action from the opponent class
        action = self.opponent.get_action(
            obs=self._get_observation(),
            my_head=self.my_head,
            opp_head=self.opponent_head,
            grid=self._get_grid()
        )
        self.opponent_direction = action
    
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
    
    def _get_observation(self) -> np.ndarray:
        """Get observation as (5, 32, 32) float tensor."""
        obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Walls
        obs[0] = self.walls.astype(np.float32)
        
        # Channel 1: My trail
        obs[1] = self.my_trail.astype(np.float32)
        
        # Channel 2: Opponent trail
        obs[2] = self.opponent_trail.astype(np.float32)
        
        # Channel 3: My head
        obs[3, self.my_head[0], self.my_head[1]] = 1.0
        
        # Channel 4: Opponent head
        obs[4, self.opponent_head[0], self.opponent_head[1]] = 1.0
        
        return obs
    
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
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption("Tron Light Cycles")
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((0, 0, 0))
        
        # Draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.walls[r, c]:
                    pygame.draw.rect(
                        self.window,
                        (100, 100, 100),
                        (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    )
        
        # Draw my trail (blue)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.my_trail[r, c]:
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 255),
                        (c * self.cell_size, r * self.cell_size, self.cell_size - 1, self.cell_size - 1)
                    )
        
        # Draw opponent trail (red)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.opponent_trail[r, c]:
                    pygame.draw.rect(
                        self.window,
                        (255, 0, 0),
                        (c * self.cell_size, r * self.cell_size, self.cell_size - 1, self.cell_size - 1)
                    )
        
        # Draw my head (light blue)
        pygame.draw.rect(
            self.window,
            (100, 100, 255),
            (self.my_head[1] * self.cell_size, self.my_head[0] * self.cell_size, self.cell_size, self.cell_size)
        )
        
        # Draw opponent head (pink)
        pygame.draw.rect(
            self.window,
            (255, 100, 100),
            (self.opponent_head[1] * self.cell_size, self.opponent_head[0] * self.cell_size, self.cell_size, self.cell_size)
        )
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
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
