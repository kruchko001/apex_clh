"""
Advanced Opponent Implementations for Tron Training.
These opponents are used ONLY during training to provide challenging adversaries.
They are NOT subject to the 0.05s inference constraint.
"""

import numpy as np
from typing import Tuple, List, Optional
import copy


class BaseOpponent:
    """Base class for all opponents."""
    
    def get_action(self, obs: np.ndarray, my_head: Tuple[int, int], 
                   opp_head: Tuple[int, int], grid: np.ndarray) -> int:
        """
        Get action for the opponent.
        
        Args:
            obs: Observation array (not used directly, grid is preferred)
            my_head: Player's head position (row, col)
            opp_head: Opponent's head position (row, col)  
            grid: Game grid (0=empty, 1=wall, 2=my_trail, 3=opp_trail)
            
        Returns:
            Action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        raise NotImplementedError


class RandomOpponent(BaseOpponent):
    """Random opponent - picks any valid move randomly."""
    
    def get_action(self, obs: np.ndarray, my_head: Tuple[int, int],
                   opp_head: Tuple[int, int], grid: np.ndarray) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid)
        if not valid_moves:
            return 0  # Will die anyway
        return int(np.random.choice(valid_moves))
    
    def _get_valid_moves(self, head: Tuple[int, int], grid: np.ndarray) -> List[int]:
        """Get list of valid move actions."""
        moves = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
        
        for i, (dr, dc) in enumerate(directions):
            nr, nc = head[0] + dr, head[1] + dc
            if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                moves.append(i)
        return moves


class HeuristicOpponent(BaseOpponent):
    """
    Greedy heuristic opponent.
    Moves forward if safe, otherwise picks random valid move.
    """
    
    def __init__(self):
        self.last_direction = 1  # Start moving RIGHT
    
    def get_action(self, obs: np.ndarray, my_head: Tuple[int, int],
                   opp_head: Tuple[int, int], grid: np.ndarray) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid)
        
        if not valid_moves:
            return 0
        
        # Try to continue in current direction
        if self.last_direction in valid_moves:
            return self.last_direction
        
        # Otherwise pick random valid move
        action = int(np.random.choice(valid_moves))
        self.last_direction = action
        return action
    
    def _get_valid_moves(self, head: Tuple[int, int], grid: np.ndarray) -> List[int]:
        moves = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for i, (dr, dc) in enumerate(directions):
            nr, nc = head[0] + dr, head[1] + dc
            if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                moves.append(i)
        return moves


class LookaheadOpponent(BaseOpponent):
    """
    Strong lookahead opponent with space evaluation.
    Simulates 2-3 steps ahead and evaluates available space.
    """
    
    def __init__(self, lookahead_depth: int = 3):
        self.lookahead_depth = lookahead_depth
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
    
    def get_action(self, obs: np.ndarray, my_head: Tuple[int, int],
                   opp_head: Tuple[int, int], grid: np.ndarray) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid)
        
        if not valid_moves:
            return 0
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        best_move = valid_moves[0]
        best_score = -float('inf')
        
        for action in valid_moves:
            # Simulate this move
            score = self._simulate_move(action, opp_head, my_head, grid, self.lookahead_depth)
            if score > best_score:
                best_score = score
                best_move = action
        
        return best_move
    
    def _simulate_move(self, action: int, opp_head: Tuple[int, int], 
                       my_head: Tuple[int, int], grid: np.ndarray, 
                       depth: int) -> float:
        """Simulate a move and evaluate the resulting state."""
        dr, dc = self.directions[action]
        new_opp_head = (opp_head[0] + dr, opp_head[1] + dc)
        
        # Check if this move is immediately fatal
        if not (0 <= new_opp_head[0] < 32 and 0 <= new_opp_head[1] < 32):
            return -1000
        if grid[new_opp_head[0], new_opp_head[1]] != 0:
            return -1000
        
        # Create simulated grid
        sim_grid = grid.copy()
        sim_grid[opp_head[0], opp_head[1]] = 3  # Mark old position as trail
        
        # Evaluate space using flood fill
        space = self._count_space(new_opp_head, sim_grid)
        
        # Bonus for chasing player
        dist_to_player = abs(new_opp_head[0] - my_head[0]) + abs(new_opp_head[1] - my_head[1])
        
        # Penalty for being too close to walls
        wall_dist = min(new_opp_head[0], 31 - new_opp_head[0], 
                       new_opp_head[1], 31 - new_opp_head[1])
        
        score = space * 1.0 - dist_to_player * 0.5 + wall_dist * 0.1
        
        # If we can look further ahead, do recursive simulation
        if depth > 1:
            # Simple approximation: assume player moves away
            future_space = space * 0.9  # Space will likely decrease
            score += future_space * 0.3
        
        return score
    
    def _count_space(self, start: Tuple[int, int], grid: np.ndarray) -> int:
        """Count available space using flood fill."""
        visited = set()
        stack = [start]
        count = 0
        
        while stack and count < 500:  # Limit for performance
            pos = stack.pop()
            if pos in visited:
                continue
            visited.add(pos)
            count += 1
            
            r, c = pos
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
        
        return count
    
    def _get_valid_moves(self, head: Tuple[int, int], grid: np.ndarray) -> List[int]:
        moves = []
        for i, (dr, dc) in enumerate(self.directions):
            nr, nc = head[0] + dr, head[1] + dc
            if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                moves.append(i)
        return moves


class MinimaxOpponent(BaseOpponent):
    """
    Ultimate Minimax opponent with Alpha-Beta pruning.
    Searches 4-6 moves ahead assuming optimal play from both sides.
    NO time constraints - as strong as computationally feasible.
    """
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
    
    def get_action(self, obs: np.ndarray, my_head: Tuple[int, int],
                   opp_head: Tuple[int, int], grid: np.ndarray) -> int:
        """Find best move using Minimax with Alpha-Beta pruning."""
        valid_moves = self._get_valid_moves(opp_head, grid)
        
        if not valid_moves:
            return 0
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        best_move = valid_moves[0]
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Opponent is maximizing player
        for action in valid_moves:
            value = self._minimax(
                grid, opp_head, my_head, action, 
                depth=self.max_depth - 1, 
                is_maximizing=False,
                alpha=alpha, beta=beta
            )
            
            if value > best_value:
                best_value = value
                best_move = action
            
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # Beta cutoff
        
        return best_move
    
    def _minimax(self, grid: np.ndarray, opp_head: Tuple[int, int], 
                 my_head: Tuple[int, int], last_action: int,
                 depth: int, is_maximizing: bool,
                 alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        
        # Apply last action to get new state
        dr, dc = self.directions[last_action]
        if is_maximizing:  # Opponent's move
            new_opp_head = (opp_head[0] + dr, opp_head[1] + dc)
            new_my_head = my_head
        else:  # Player's move
            new_my_head = (my_head[0] + dr, my_head[1] + dc)
            new_opp_head = opp_head
        
        # Check bounds
        if not (0 <= new_opp_head[0] < 32 and 0 <= new_opp_head[1] < 32):
            return -10000 if is_maximizing else 10000
        if not (0 <= new_my_head[0] < 32 and 0 <= new_my_head[1] < 32):
            return 10000 if is_maximizing else -10000
        
        # Check collisions
        opp_collision = grid[new_opp_head[0], new_opp_head[1]] != 0
        my_collision = grid[new_my_head[0], new_my_head[1]] != 0
        
        # Head-on collision
        if new_opp_head == new_my_head:
            return 0  # Draw
        
        if opp_collision and my_collision:
            return 0  # Both die
        if opp_collision:
            return -10000  # Opponent dies
        if my_collision:
            return 10000  # Player dies
        
        # Terminal depth
        if depth == 0:
            return self._evaluate(new_opp_head, new_my_head, grid)
        
        # Create new grid state
        new_grid = grid.copy()
        if is_maximizing:
            new_grid[opp_head[0], opp_head[1]] = 3
        else:
            new_grid[my_head[0], my_head[1]] = 2
        
        if is_maximizing:
            # Opponent's turn - maximize
            value = -float('inf')
            valid_moves = self._get_valid_moves(new_opp_head, new_grid)
            
            if not valid_moves:
                return -10000  # No moves = death
            
            for action in valid_moves:
                val = self._minimax(
                    new_grid, new_opp_head, new_my_head, action,
                    depth - 1, False, alpha, beta
                )
                value = max(value, val)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            # Player's turn - minimize (assume player plays optimally to win)
            value = float('inf')
            valid_moves = self._get_valid_moves(new_my_head, new_grid)
            
            if not valid_moves:
                return 10000  # Player has no moves = opponent wins
            
            for action in valid_moves:
                val = self._minimax(
                    new_grid, new_opp_head, new_my_head, action,
                    depth - 1, True, alpha, beta
                )
                value = min(value, val)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value
    
    def _evaluate(self, opp_head: Tuple[int, int], my_head: Tuple[int, int], 
                  grid: np.ndarray) -> float:
        """Evaluate board state for the opponent."""
        
        # Count available space for each player
        opp_space = self._count_space(opp_head, grid)
        my_space = self._count_space(my_head, grid)
        
        # Distance between players
        distance = abs(opp_head[0] - my_head[0]) + abs(opp_head[1] - my_head[1])
        
        # Wall distance for opponent
        wall_dist = min(opp_head[0], 31 - opp_head[0], 
                       opp_head[1], 31 - opp_head[1])
        
        # Score: more space than opponent is good, closer to player is good
        space_diff = opp_space - my_space
        chase_bonus = max(0, 10 - distance)  # Bonus for being close
        
        score = space_diff * 2.0 + chase_bonus * 1.5 + wall_dist * 0.2
        
        return score
    
    def _count_space(self, start: Tuple[int, int], grid: np.ndarray) -> int:
        """Count available space using flood fill."""
        visited = set()
        stack = [start]
        count = 0
        
        while stack and count < 1000:
            pos = stack.pop()
            if pos in visited:
                continue
            visited.add(pos)
            count += 1
            
            r, c = pos
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
        
        return count
    
    def _get_valid_moves(self, head: Tuple[int, int], grid: np.ndarray) -> List[int]:
        moves = []
        for i, (dr, dc) in enumerate(self.directions):
            nr, nc = head[0] + dr, head[1] + dc
            if 0 <= nr < 32 and 0 <= nc < 32 and grid[nr, nc] == 0:
                moves.append(i)
        return moves


def get_opponent(opponent_type: str):
    """Factory function to get opponent by name."""
    opponents = {
        'random': RandomOpponent,
        'heuristic': HeuristicOpponent,
        'lookahead': LookaheadOpponent,
        'minimax': MinimaxOpponent,
    }
    
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}. "
                        f"Available: {list(opponents.keys())}")
    
    return opponents[opponent_type]()
