"""
Custom Gymnasium Environment for Tron (Lightcycles) game.
Designed for Bittensor Subnet 1 RL competition.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum


class Direction(IntEnum):
    """Direction enumeration for actions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class TronEnv(gym.Env):
    """
    Custom Gymnasium Environment for Tron (Lightcycles) game.
    
    Observation Space: (5, 32, 32) tensor
        - Channel 0: Walls (1.0 where wall, 0.0 elsewhere)
        - Channel 1: My trail (1.0 where my trail exists)
        - Channel 2: Opponent trail (1.0 where opponent trail exists)
        - Channel 3: My head position (1.0 at current cell)
        - Channel 4: Opponent head position (1.0 at opponent's current cell)
    
    Action Space: Discrete(4)
        - 0: UP
        - 1: RIGHT
        - 2: DOWN
        - 3: LEFT
    
    Reward Shaping (per step):
        - Clean kill (I live, opponent hits my trail): +2.0
        - Opponent self-destructs (hits wall/own trail): +1.5
        - Mutual destruction (head-on or simultaneous trail kill): +0.5
        - Timeout draw (both alive at 500 steps): 0.0
        - Die alone (hit wall or own trail): -2.0
        - Step survival: +0.01
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        grid_size: int = 32,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 4 directions [UP, RIGHT, DOWN, LEFT]
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (5, 32, 32) float tensor
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, grid_size, grid_size),
            dtype=np.float32,
        )
        
        # Direction vectors: [UP, RIGHT, DOWN, LEFT]
        # Note: In array coordinates, row 0 is top, so UP decreases row
        self.direction_vectors = {
            Direction.UP: (-1, 0),
            Direction.RIGHT: (0, 1),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
        }
        
        # State variables (initialized in reset)
        self.my_trail: Optional[np.ndarray] = None
        self.opponent_trail: Optional[np.ndarray] = None
        self.walls: Optional[np.ndarray] = None
        self.my_head: Optional[Tuple[int, int]] = None
        self.opponent_head: Optional[Tuple[int, int]] = None
        self.my_direction: Optional[Direction] = None
        self.opponent_direction: Optional[Direction] = None
        self.current_step: int = 0
        self.game_over: bool = False
        
    def _get_opposite_direction(self, direction: Direction) -> Direction:
        """Get the opposite direction (180-degree turn)."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[direction]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize walls (border)
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.walls[0, :] = 1.0  # Top wall
        self.walls[-1, :] = 1.0  # Bottom wall
        self.walls[:, 0] = 1.0  # Left wall
        self.walls[:, -1] = 1.0  # Right wall
        
        # Initialize empty trails
        self.my_trail = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.opponent_trail = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Place players on opposite sides
        # My head starts on the left side, facing right
        self.my_head = (self.grid_size // 2, 2)
        self.my_direction = Direction.RIGHT
        
        # Opponent head starts on the right side, facing left
        self.opponent_head = (self.grid_size // 2, self.grid_size - 3)
        self.opponent_direction = Direction.LEFT
        
        # Mark initial positions on trails
        self.my_trail[self.my_head] = 1.0
        self.opponent_trail[self.opponent_head] = 1.0
        
        # Reset step counter and game state
        self.current_step = 0
        self.game_over = False
        
        # Get initial observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            "my_head": self.my_head,
            "opponent_head": self.opponent_head,
            "step": self.current_step,
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation tensor of shape (5, 32, 32).
        
        Channels:
            0: Walls
            1: My trail
            2: Opponent trail
            3: My head position
            4: Opponent head position
        """
        obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Walls
        obs[0] = self.walls
        
        # Channel 1: My trail
        obs[1] = self.my_trail
        
        # Channel 2: Opponent trail
        obs[2] = self.opponent_trail
        
        # Channel 3: My head position
        if self.my_head:
            obs[3, self.my_head[0], self.my_head[1]] = 1.0
        
        # Channel 4: Opponent head position
        if self.opponent_head:
            obs[4, self.opponent_head[0], self.opponent_head[1]] = 1.0
        
        return obs
    
    def _move_player(
        self,
        head: Tuple[int, int],
        direction: Direction,
    ) -> Tuple[int, int]:
        """Move player head in the given direction."""
        dr, dc = self.direction_vectors[direction]
        new_row = head[0] + dr
        new_col = head[1] + dc
        return (new_row, new_col)
    
    def _check_collision(
        self,
        position: Tuple[int, int],
        my_trail: np.ndarray,
        opponent_trail: np.ndarray,
        walls: np.ndarray,
    ) -> bool:
        """Check if position results in collision."""
        row, col = position
        
        # Check bounds (wall collision)
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        
        # Check wall collision
        if walls[row, col] == 1.0:
            return True
        
        # Check my trail collision
        if my_trail[row, col] == 1.0:
            return True
        
        # Check opponent trail collision
        if opponent_trail[row, col] == 1.0:
            return True
        
        return False
    
    def _check_collision_detailed(
        self,
        position: Tuple[int, int],
        my_trail: np.ndarray,
        opponent_trail: np.ndarray,
        walls: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Check if position results in collision and return collision type.
        
        Returns:
            (is_collision, collision_type) where collision_type is one of:
            - "wall": hit wall or bounds
            - "self": hit own trail
            - "opponent": hit opponent trail
            - "": no collision
        """
        row, col = position
        
        # Check bounds (wall collision)
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True, "wall"
        
        # Check wall collision
        if walls[row, col] == 1.0:
            return True, "wall"
        
        # Check my trail collision
        if my_trail[row, col] == 1.0:
            return True, "self"
        
        # Check opponent trail collision
        if opponent_trail[row, col] == 1.0:
            return True, "opponent"
        
        return False, ""
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            
        Returns:
            observation: (5, 32, 32) tensor
            reward: Float reward based on outcome
            terminated: Whether the episode ended
            truncated: Whether the episode was truncated (timeout)
            info: Additional information
        """
        if self.game_over:
            # Episode already ended
            return self._get_observation(), 0.0, True, False, {"game_over": True}
        
        # Convert action to direction
        my_action = Direction(action)
        
        # Prevent 180-degree turn (action masking handled by platform, but we enforce here too)
        opposite = self._get_opposite_direction(self.my_direction)
        if my_action == opposite:
            # Keep current direction if trying to reverse
            my_action = self.my_direction
        
        # Simple AI for opponent: continues in same direction
        # (Can be enhanced with more sophisticated behavior)
        opponent_action = self.opponent_direction
        
        # Calculate new positions
        new_my_head = self._move_player(self.my_head, my_action)
        new_opponent_head = self._move_player(self.opponent_head, opponent_action)
        
        # Check for head-on collision (both move to same cell)
        head_on_collision = (new_my_head == new_opponent_head)
        
        # Check collisions for each player with detailed type
        my_collision, my_collision_type = self._check_collision_detailed(
            new_my_head,
            self.my_trail,
            self.opponent_trail,
            self.walls,
        )
        
        opponent_collision, opponent_collision_type = self._check_collision_detailed(
            new_opponent_head,
            self.opponent_trail,
            self.my_trail,
            self.walls,
        )
        
        # Special case: head-on collision (both move to same empty cell)
        if head_on_collision and not my_collision and not opponent_collision:
            my_collision = True
            opponent_collision = True
            my_collision_type = "headon"
            opponent_collision_type = "headon"
        
        # Determine outcome and calculate reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # Determine cause of death for reward calculation
        my_died_alone = False
        clean_kill = False
        opponent_self_destruct = False
        mutual_destruction = False
        
        if my_collision and opponent_collision:
            # Mutual destruction (head-on or both hit trails/walls)
            mutual_destruction = True
            reward = 0.5
            terminated = True
            self.game_over = True
        elif my_collision:
            # I die alone
            my_died_alone = True
            reward = -2.0
            terminated = True
            self.game_over = True
        elif opponent_collision:
            # Opponent dies - determine if clean kill or self-destruct
            if opponent_collision_type == "opponent":
                # Opponent hit my trail - clean kill
                clean_kill = True
                reward = 2.0
            else:
                # Opponent hit wall or own trail - self-destruct
                opponent_self_destruct = True
                reward = 1.5
            terminated = True
            self.game_over = True
        else:
            # Both survive - update game state
            reward = 0.01  # Step survival reward
            
            # Update directions
            self.my_direction = my_action
            self.opponent_direction = opponent_action
            
            # Update heads
            self.my_head = new_my_head
            self.opponent_head = new_opponent_head
            
            # Add new positions to trails
            self.my_trail[self.my_head] = 1.0
            self.opponent_trail[self.opponent_head] = 1.0
            
            # Increment step counter
            self.current_step += 1
            
            # Check for timeout
            if self.current_step >= self.max_steps:
                # Timeout draw
                reward = 0.0
                truncated = True
                self.game_over = True
        
        # Get observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            "my_head": self.my_head,
            "opponent_head": self.opponent_head,
            "step": self.current_step,
            "my_collision": my_collision if terminated else False,
            "opponent_collision": opponent_collision if terminated else False,
            "head_on_collision": head_on_collision if terminated else False,
            "my_collision_type": my_collision_type if terminated else "",
            "opponent_collision_type": opponent_collision_type if terminated else "",
            "clean_kill": clean_kill,
            "opponent_self_destruct": opponent_self_destruct,
            "mutual_destruction": mutual_destruction,
            "my_died_alone": my_died_alone,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render the environment in human-readable format."""
        # Create a simple ASCII representation
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        # Mark walls
        grid[self.walls == 1.0] = '#'
        
        # Mark trails
        grid[(self.my_trail == 1.0) & (self.walls == 0.0)] = 'm'
        grid[(self.opponent_trail == 1.0) & (self.walls == 0.0)] = 'o'
        
        # Mark heads
        if self.my_head:
            grid[self.my_head] = 'M'
        if self.opponent_head:
            grid[self.opponent_head] = 'O'
        
        print(f"Step: {self.current_step}/{self.max_steps}")
        print("-" * (self.grid_size + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.grid_size + 2))
    
    def _render_rgb_array(self):
        """Render the environment as an RGB array."""
        # Create RGB image (3 channels)
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Background: white
        # Walls: black
        img[self.walls == 1.0] = [0.0, 0.0, 0.0]
        
        # My trail: blue
        mask = (self.my_trail == 1.0) & (self.walls == 0.0)
        img[mask] = [0.0, 0.0, 1.0]
        
        # Opponent trail: red
        mask = (self.opponent_trail == 1.0) & (self.walls == 0.0)
        img[mask] = [1.0, 0.0, 0.0]
        
        # My head: bright blue
        if self.my_head:
            img[self.my_head] = [0.0, 0.5, 1.0]
        
        # Opponent head: bright red
        if self.opponent_head:
            img[self.opponent_head] = [1.0, 0.5, 0.0]
        
        return img
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment with gymnasium
gym.register(
    id="Tron-v0",
    entry_point="tron_env:TronEnv",
    max_episode_steps=500,
)


if __name__ == "__main__":
    # Test the environment
    env = TronEnv(render_mode="human")
    
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps
    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {t}: Action={action}, Reward={reward:.2f}")
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished at step {t}")
            break
    
    env.close()
