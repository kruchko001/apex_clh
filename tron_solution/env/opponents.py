"""
Advanced Opponent Implementations for Tron Training.
These opponents are used ONLY during training to provide challenging adversaries.
They are NOT subject to the 0.05s inference constraint.
"""

import numpy as np
from typing import Tuple, List, Optional
from tronbot.python import MyTronBot, TIMEOUT_SEC, FIRST_MOVE_TIMEOUT_SEC

DEFAULT_OPPONENT_TYPE = "minimax"
DEFAULT_MINIMAX_DEPTH = 14
PLAY_MINIMAX_DEPTHS = {"easy": 6, "medium": 10, "hard": 14}


class BaseOpponent:
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def get_action(self, obs=None, my_head=None, opp_head=None, grid=None,
                   current_dir=None, my_dir=None) -> int:
        if isinstance(obs, dict):
            state = obs
            grid = np.zeros((state['height'], state['width']), dtype=np.int32)
            raw = state['grid']
            grid[raw == -1] = 1
            grid[raw == 1] = 2
            grid[raw == 2] = 3
            my_head = state['p1_pos']
            opp_head = state['p2_pos']
            current_dir = state.get('p2_dir', current_dir)
            my_dir = state.get('p1_dir', my_dir)
            obs = None
        return self._choose_action(obs, my_head, opp_head, grid, current_dir, my_dir)

    def _choose_action(self, obs: Optional[np.ndarray], my_head: Tuple[int, int],
                       opp_head: Tuple[int, int], grid: np.ndarray,
                       current_dir: Optional[int] = None,
                       my_dir: Optional[int] = None) -> int:
        raise NotImplementedError

    def _get_valid_moves(self, head: Tuple[int, int], grid: np.ndarray,
                         current_dir: Optional[int] = None) -> List[int]:
        moves = []
        for i, (dr, dc) in enumerate(self.DIRECTIONS):
            if current_dir is not None and i == self.OPPOSITE[current_dir]:
                continue
            nr, nc = head[0] + dr, head[1] + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                moves.append(i)
        return moves

    def _mobility(self, head: Tuple[int, int], grid: np.ndarray,
                  current_dir: Optional[int] = None) -> int:
        return len(self._get_valid_moves(head, grid, current_dir))


class RandomOpponent(BaseOpponent):
    """Random opponent - picks any valid move randomly."""
    
    def _choose_action(self, obs: Optional[np.ndarray], my_head: Tuple[int, int],
                       opp_head: Tuple[int, int], grid: np.ndarray,
                       current_dir: Optional[int] = None,
                       my_dir: Optional[int] = None) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid, current_dir)
        if not valid_moves:
            return current_dir if current_dir is not None else 0
        return int(np.random.choice(valid_moves))


class HeuristicOpponent(BaseOpponent):
    def _choose_action(self, obs: Optional[np.ndarray], my_head: Tuple[int, int],
                       opp_head: Tuple[int, int], grid: np.ndarray,
                       current_dir: Optional[int] = None,
                       my_dir: Optional[int] = None) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid, current_dir)
        if not valid_moves:
            return current_dir if current_dir is not None else 0
        if current_dir is not None and current_dir in valid_moves:
            return current_dir
        return int(np.random.choice(valid_moves))


class LookaheadOpponent(BaseOpponent):
    """
    Strong lookahead opponent with space evaluation.
    Simulates 2-3 steps ahead and evaluates available space.
    """
    
    def __init__(self, lookahead_depth: int = 3):
        self.lookahead_depth = lookahead_depth
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
    
    def _choose_action(self, obs: Optional[np.ndarray], my_head: Tuple[int, int],
                       opp_head: Tuple[int, int], grid: np.ndarray,
                       current_dir: Optional[int] = None,
                       my_dir: Optional[int] = None) -> int:
        valid_moves = self._get_valid_moves(opp_head, grid, current_dir)
        
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


class MinimaxOpponent(BaseOpponent):
    def __init__(self, max_depth: int = 10, depth: int = None):
        self.max_depth = depth if depth is not None else max_depth

    def _same_component(self, opp_head, my_head, grid):
        from collections import deque
        h, w = grid.shape
        seen = set()
        q = deque([opp_head, my_head])
        seen.add(opp_head)
        seen.add(my_head)
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return (opp_head in seen and my_head in seen)

    def _fillable_in_component(self, head, grid):
        from collections import deque
        h, w = grid.shape
        if grid[head[0], head[1]] != 0:
            return 0
        seen = {head}
        q = deque([head])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return len(seen)

    def _opponent_trail_penalty(self, nh, grid, opp_trail_val=2):
        penalty = 0.0
        h, w = grid.shape
        for dr, dc in self.DIRECTIONS:
            r, c = nh[0] + dr, nh[1] + dc
            if 0 <= r < h and 0 <= c < w and grid[r, c] == opp_trail_val:
                penalty += 18.0
        return penalty

    def _squeeze_penalty(self, nh, grid, current_dir, action, voronoi):
        if voronoi > -30:
            return 0.0
        h, w = grid.shape
        r, c = nh
        on_border = r <= 2 or r >= h - 3 or c <= 2 or c >= w - 3
        if not on_border:
            return 0.0
        penalty = 0.0
        if action == current_dir:
            penalty += 12.0
        if r <= 2 and action == self.DOWN:
            penalty += 20.0
        if r >= h - 3 and action == self.UP:
            penalty += 20.0
        if c <= 2 and action == self.RIGHT:
            penalty += 20.0
        if c >= w - 3 and action == self.LEFT:
            penalty += 20.0
        return penalty

    def _inward_bonus(self, head, nh, shape):
        h, w = shape
        cr, cc = (h - 1) / 2, (w - 1) / 2
        old_edge = min(head[0], h - 1 - head[0], head[1], w - 1 - head[1])
        new_edge = min(nh[0], h - 1 - nh[0], nh[1], w - 1 - nh[1])
        old_center = abs(head[0] - cr) + abs(head[1] - cc)
        new_center = abs(nh[0] - cr) + abs(nh[1] - cc)
        bonus = 0.0
        if new_edge > old_edge:
            bonus += 3.0
        if new_center < old_center:
            bonus += 2.0
        return bonus

    def _wall_slide_penalty(self, head, nh, current_dir, action, shape):
        h, w = shape
        r, c = nh
        on_edge = r <= 2 or r >= h - 3 or c <= 2 or c >= w - 3
        if not on_edge:
            return 0.0
        penalty = 0.0
        if action == current_dir:
            penalty += 4.0
        if r <= 2 or r >= h - 3:
            if action in (self.LEFT, self.RIGHT):
                penalty += 5.0
        if c <= 2 or c >= w - 3:
            if action in (self.UP, self.DOWN):
                penalty += 5.0
        return penalty

    def _partition_bonus(self, opp_head, my_head, action):
        dr, dc = self.DIRECTIONS[action]
        bonus = 0.0
        if opp_head[1] <= 4 and my_head[0] > opp_head[0] + 6:
            bonus += max(0, dc) * 8.0
        if opp_head[0] <= 4 and my_head[1] > opp_head[1] + 6:
            bonus += max(0, dr) * 8.0
        if opp_head[0] >= 27 and my_head[1] < opp_head[1] - 6:
            bonus += max(0, -dr) * 8.0
        if opp_head[1] >= 27 and my_head[0] < opp_head[0] - 6:
            bonus += max(0, -dc) * 8.0
        return bonus

    def _head_on_penalty(self, opp_head, my_head, grid, opp_dir, my_dir, action):
        dr, dc = self.DIRECTIONS[action]
        nh = (opp_head[0] + dr, opp_head[1] + dc)
        if not (0 <= nh[0] < grid.shape[0] and 0 <= nh[1] < grid.shape[1]):
            return 0.0
        if grid[nh[0], nh[1]] != 0:
            return 0.0
        penalty = 0.0
        for ma in self._get_valid_moves(my_head, grid, my_dir):
            mdr, mdc = self.DIRECTIONS[ma]
            mh = (my_head[0] + mdr, my_head[1] + mdc)
            if mh == nh:
                penalty += 8000.0
            if abs(mh[0] - nh[0]) + abs(mh[1] - nh[1]) == 1:
                penalty += 400.0
        return penalty

    def _apply_simultaneous(self, grid, opp_head, my_head, opp_dir, my_dir, opp_act, my_act):
        if opp_act == self.OPPOSITE[opp_dir]:
            opp_act = opp_dir
        if my_act == self.OPPOSITE[my_dir]:
            my_act = my_dir
        dr_o, dc_o = self.DIRECTIONS[opp_act]
        dr_m, dc_m = self.DIRECTIONS[my_act]
        no = (opp_head[0] + dr_o, opp_head[1] + dc_o)
        nm = (my_head[0] + dr_m, my_head[1] + dc_m)
        h, w = grid.shape

        def blocked(pos):
            r, c = pos
            if not (0 <= r < h and 0 <= c < w):
                return True
            return grid[r, c] != 0

        if no == nm:
            return 0, None
        ob, mb = blocked(no), blocked(nm)
        if ob and mb:
            return 0, None
        if ob:
            return -10000, None
        if mb:
            return 10000, None
        new_grid = grid.copy()
        new_grid[opp_head[0], opp_head[1]] = 3
        new_grid[my_head[0], my_head[1]] = 2
        return None, (new_grid, no, nm, opp_act, my_act)

    def _model_response(self, grid, opp_head, my_head, opp_dir, my_dir, opp_act):
        moves = self._ordered_moves(my_head, grid, my_dir, trail_val=2, my_head=opp_head, my_dir=opp_dir)[:3]
        if not moves:
            return my_dir
        best_act = moves[0]
        best_val = float("inf")
        for ma in moves:
            terminal, state = self._apply_simultaneous(
                grid, opp_head, my_head, opp_dir, my_dir, opp_act, ma,
            )
            if terminal is not None:
                val = terminal
            else:
                ng, no, nm, nod, nmd = state
                val = self._evaluate(no, nm, ng, nod, nmd)
            if val < best_val:
                best_val = val
                best_act = ma
        return best_act

    def _cutoff_bonus(self, opp_head, my_head, action):
        dr, dc = self.DIRECTIONS[action]
        bonus = 0.0
        row_gap = my_head[0] - opp_head[0]
        col_gap = my_head[1] - opp_head[1]
        if opp_head[1] <= 3 and col_gap > 10:
            bonus += max(0, dc) * 3.0
        if opp_head[1] >= 28 and col_gap < -10:
            bonus += max(0, -dc) * 3.0
        if opp_head[0] <= 3 and row_gap > 10:
            bonus += max(0, dr) * 3.0
        if opp_head[0] >= 28 and row_gap < -10:
            bonus += max(0, -dr) * 3.0
        if abs(row_gap) <= 2 and abs(col_gap) > 8:
            bonus += max(0, abs(dc)) * 2.0
        if abs(col_gap) <= 2 and abs(row_gap) > 8:
            bonus += max(0, abs(dr)) * 2.0
        return bonus

    def _move_score(self, action, opp_head, my_head, grid, current_dir, trail_val=3, my_dir=None):
        dr, dc = self.DIRECTIONS[action]
        nh = (opp_head[0] + dr, opp_head[1] + dc)
        if not (0 <= nh[0] < grid.shape[0] and 0 <= nh[1] < grid.shape[1]):
            return -9999
        if grid[nh[0], nh[1]] != 0:
            return -9999
        sim = grid.copy()
        sim[opp_head[0], opp_head[1]] = trail_val
        vor = self._voronoi(opp_head, my_head, grid)
        space = self._count_space(nh, sim)
        mob = len(self._get_valid_moves(nh, sim, action))
        inward = self._inward_bonus(opp_head, nh, grid.shape)
        wall_pen = self._wall_slide_penalty(opp_head, nh, current_dir, action, grid.shape)
        cut = self._cutoff_bonus(opp_head, my_head, action)
        part = self._partition_bonus(opp_head, my_head, action)
        head_pen = self._head_on_penalty(opp_head, my_head, grid, current_dir, my_dir or self.RIGHT, action)
        trail_pen = self._opponent_trail_penalty(nh, grid, opp_trail_val=2)
        squeeze = self._squeeze_penalty(nh, grid, current_dir, action, vor)
        return space * 1.5 + vor * 12.0 + mob * 6.0 + inward * 4.0 + cut * 5.0 + part * 6.0 - wall_pen * 8.0 - head_pen - trail_pen * 10.0 - squeeze * 8.0

    def _ordered_moves(self, head, grid, current_dir, trail_val=3, my_head=None, my_dir=None):
        moves = self._get_valid_moves(head, grid, current_dir)
        if my_head is None:
            scored = []
            for action in moves:
                dr, dc = self.DIRECTIONS[action]
                nh = (head[0] + dr, head[1] + dc)
                sim = grid.copy()
                sim[head[0], head[1]] = trail_val
                scored.append((self._count_space(nh, sim), action))
            scored.sort(reverse=True)
            return [a for _, a in scored]
        scored = [
            (self._move_score(a, head, my_head, grid, current_dir, trail_val, my_dir), a)
            for a in moves
        ]
        scored.sort(reverse=True)
        return [a for _, a in scored]

    def _choose_action(self, obs: Optional[np.ndarray], my_head: Tuple[int, int],
                       opp_head: Tuple[int, int], grid: np.ndarray,
                       current_dir: Optional[int] = None,
                       my_dir: Optional[int] = None) -> int:
        if current_dir is None:
            current_dir = self.RIGHT
        if my_dir is None:
            my_dir = self.RIGHT
        valid_moves = self._ordered_moves(opp_head, grid, current_dir, trail_val=3, my_head=my_head, my_dir=my_dir)
        if not valid_moves:
            return current_dir if current_dir is not None else self.UP
        if len(valid_moves) == 1:
            return valid_moves[0]

        best_move = valid_moves[0]
        best_value = -float('inf')
        best_heuristic = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for action in valid_moves:
            value = self._minimax(
                grid, opp_head, my_head, current_dir, my_dir, action,
                self.max_depth - 1, True, alpha, beta,
            )
            ma = self._model_response(grid, opp_head, my_head, current_dir, my_dir, action)
            terminal, state = self._apply_simultaneous(
                grid, opp_head, my_head, current_dir, my_dir, action, ma,
            )
            if terminal == 0:
                value = min(value, -800.0)
            elif terminal is not None:
                value = max(value, terminal) if terminal > 0 else min(value, terminal)
            elif state is not None:
                ng, no, nm, nod, nmd = state
                sim_val = self._evaluate(no, nm, ng, nod, nmd)
                value = value * 0.55 + sim_val * 0.45
            h = self._move_score(action, opp_head, my_head, grid, current_dir, my_dir=my_dir)
            if value > best_value + 0.5:
                best_value = value
                best_move = action
                best_heuristic = h
            elif abs(value - best_value) <= 0.5 and h > best_heuristic:
                best_move = action
                best_heuristic = h
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return best_move if best_move in valid_moves else valid_moves[0]

    def get_action_vs_known_opp(self, my_head, opp_head, grid, current_dir, my_dir, opp_action):
        if current_dir is None:
            current_dir = self.RIGHT
        if my_dir is None:
            my_dir = self.RIGHT
        valid_moves = self._ordered_moves(opp_head, grid, current_dir, trail_val=3, my_head=my_head, my_dir=my_dir)
        if not valid_moves:
            return current_dir
        if len(valid_moves) == 1:
            return valid_moves[0]
        best_move = valid_moves[0]
        best_value = -float("inf")
        for action in valid_moves:
            terminal, state = self._apply_simultaneous(
                grid, opp_head, my_head, current_dir, my_dir, action, opp_action,
            )
            if terminal == 10000:
                return action
            if terminal == -10000:
                continue
            if terminal == 0:
                value = -600.0
            elif state is not None:
                ng, no, nm, nod, nmd = state
                value = self._evaluate(no, nm, ng, nod, nmd)
            else:
                value = -float("inf")
            value += self._move_score(action, opp_head, my_head, grid, current_dir, my_dir=my_dir) * 0.05
            if value > best_value:
                best_value = value
                best_move = action
        return best_move if best_move in valid_moves else valid_moves[0]

    def rank_actions(self, my_head, opp_head, grid, current_dir, my_dir):
        valid = self._ordered_moves(opp_head, grid, current_dir, trail_val=3, my_head=my_head, my_dir=my_dir)
        ranked = []
        for action in valid:
            val = self._minimax(
                grid, opp_head, my_head, current_dir, my_dir, action,
                self.max_depth - 1, True, -float("inf"), float("inf"),
            )
            ma = self._model_response(grid, opp_head, my_head, current_dir, my_dir, action)
            terminal, state = self._apply_simultaneous(
                grid, opp_head, my_head, current_dir, my_dir, action, ma,
            )
            if terminal == 0:
                val = min(val, -800.0)
            elif terminal is not None:
                val = max(val, terminal) if terminal > 0 else min(val, terminal)
            elif state is not None:
                ng, no, nm, nod, nmd = state
                val = val * 0.55 + self._evaluate(no, nm, ng, nod, nmd) * 0.45
            val += self._move_score(action, opp_head, my_head, grid, current_dir, my_dir=my_dir) * 0.01
            ranked.append((val, action))
        ranked.sort(reverse=True)
        return ranked

    def _minimax(self, grid: np.ndarray, opp_head: Tuple[int, int],
                 my_head: Tuple[int, int], opp_dir: int, my_dir: int,
                 last_action: int, depth: int, is_opponent_turn: bool,
                 alpha: float, beta: float) -> float:
        dr, dc = self.DIRECTIONS[last_action]
        if is_opponent_turn:
            new_opp_head = (opp_head[0] + dr, opp_head[1] + dc)
            new_my_head = my_head
            new_opp_dir = last_action
            new_my_dir = my_dir
        else:
            new_my_head = (my_head[0] + dr, my_head[1] + dc)
            new_opp_head = opp_head
            new_my_dir = last_action
            new_opp_dir = opp_dir

        if not (0 <= new_opp_head[0] < grid.shape[0] and 0 <= new_opp_head[1] < grid.shape[1]):
            return -10000 if is_opponent_turn else 10000
        if not (0 <= new_my_head[0] < grid.shape[0] and 0 <= new_my_head[1] < grid.shape[1]):
            return 10000 if is_opponent_turn else -10000

        opp_collision = grid[new_opp_head[0], new_opp_head[1]] != 0
        my_collision = grid[new_my_head[0], new_my_head[1]] != 0

        if new_opp_head == new_my_head:
            return 0
        if opp_collision and my_collision:
            return 0
        if opp_collision:
            return -10000
        if my_collision:
            return 10000

        if depth == 0:
            return self._evaluate(new_opp_head, new_my_head, grid, new_opp_dir, new_my_dir)

        new_grid = grid.copy()
        if is_opponent_turn:
            new_grid[opp_head[0], opp_head[1]] = 3
        else:
            new_grid[my_head[0], my_head[1]] = 2

        if is_opponent_turn:
            value = float('inf')
            valid_moves = self._ordered_moves(new_my_head, new_grid, new_my_dir, trail_val=2)
            if not valid_moves:
                return 10000
            for action in valid_moves:
                val = self._minimax(
                    new_grid, new_opp_head, new_my_head, new_opp_dir, new_my_dir, action,
                    depth - 1, False, alpha, beta,
                )
                value = min(value, val)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

        value = -float('inf')
        valid_moves = self._ordered_moves(new_opp_head, new_grid, new_opp_dir, trail_val=3, my_head=new_my_head)
        if not valid_moves:
            return -10000
        for action in valid_moves:
            val = self._minimax(
                new_grid, new_opp_head, new_my_head, new_opp_dir, new_my_dir, action,
                depth - 1, True, alpha, beta,
            )
            value = max(value, val)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    
    def _evaluate(self, opp_head: Tuple[int, int], my_head: Tuple[int, int],
                  grid: np.ndarray, opp_dir: int, my_dir: int) -> float:
        partitioned = not self._same_component(opp_head, my_head, grid)
        if partitioned:
            opp_space = self._fillable_in_component(opp_head, grid)
            my_space = self._fillable_in_component(my_head, grid)
        else:
            opp_space = self._count_space(opp_head, grid)
            my_space = self._count_space(my_head, grid)
        opp_mob = self._mobility(opp_head, grid, opp_dir)
        my_mob = self._mobility(my_head, grid, my_dir)
        voronoi = self._voronoi(opp_head, my_head, grid)

        if my_space > 0 and opp_space == 0:
            return -5000
        if opp_space > 0 and my_space == 0:
            return 5000

        distance = abs(opp_head[0] - my_head[0]) + abs(opp_head[1] - my_head[1])
        wall_dist = min(
            opp_head[0], grid.shape[0] - 1 - opp_head[0],
            opp_head[1], grid.shape[1] - 1 - opp_head[1],
        )
        center_r = abs(opp_head[0] - (grid.shape[0] - 1) / 2)
        center_c = abs(opp_head[1] - (grid.shape[1] - 1) / 2)

        score = (opp_space - my_space) * 8.0
        score += (opp_mob - my_mob) * 12.0
        score += voronoi * 10.0
        score += wall_dist * 2.0
        if wall_dist <= 2:
            score -= 25.0
        score -= (center_r + center_c) * 1.5
        if partitioned:
            score += (opp_space - my_space) * 6.0
            score += opp_mob * 8.0
        if voronoi < -40:
            score += voronoi * 8.0
            score -= max(0, distance - 6) * 3.0
        if voronoi < -100:
            score = (opp_space - my_space) * 14.0 + voronoi * 18.0 + opp_mob * 10.0
        elif voronoi > 100:
            score = (opp_space - my_space) * 10.0 + voronoi * 22.0 + opp_mob * 14.0
            score -= max(0, 14 - distance) * 15.0
        elif voronoi > 50:
            score -= max(0, 10 - distance) * 10.0
        elif my_space < opp_space and distance <= 8:
            score += (8 - distance) * 2.0
        elif my_space > opp_space:
            score -= max(0, 12 - distance) * 2.5
        if my_head[1] - opp_head[1] > 12 and opp_head[1] <= 3:
            score -= 20.0
        if my_head[0] - opp_head[0] > 12 and opp_head[0] <= 3:
            score -= 20.0
        return score

    def _voronoi(self, opp_head, my_head, grid):
        from collections import deque
        h, w = grid.shape
        inf = 10**9
        dist_o = np.full((h, w), inf, dtype=np.int32)
        dist_m = np.full((h, w), inf, dtype=np.int32)
        q = deque([(opp_head, 0), (my_head, 1)])
        dist_o[opp_head] = 0
        dist_m[my_head] = 0
        while q:
            (r, c), side = q.popleft()
            d = dist_o[r, c] if side == 0 else dist_m[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] != 0:
                    continue
                if side == 0:
                    if d + 1 < dist_o[nr, nc]:
                        dist_o[nr, nc] = d + 1
                        q.append(((nr, nc), 0))
                elif d + 1 < dist_m[nr, nc]:
                    dist_m[nr, nc] = d + 1
                    q.append(((nr, nc), 1))
        opp_cells = my_cells = 0
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    continue
                do, dm = dist_o[r, c], dist_m[r, c]
                if do == inf and dm == inf:
                    continue
                if do <= dm:
                    opp_cells += 1
                else:
                    my_cells += 1
        return opp_cells - my_cells
    
    def _count_space(self, start: Tuple[int, int], grid: np.ndarray) -> int:
        """Count available space using flood fill."""
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
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
        
        return count


class TronBotOpponent(BaseOpponent):
    def __init__(
        self,
        move_timeout=TIMEOUT_SEC,
        first_move_timeout=FIRST_MOVE_TIMEOUT_SEC,
    ):
        self.bot = MyTronBot(move_timeout, first_move_timeout)

    def reset(self):
        self.bot.reset()

    def _choose_action(self, obs, my_head, opp_head, grid, current_dir=None, my_dir=None):
        action = self.bot.choose_move(grid, my_head, opp_head)
        valid = self._get_valid_moves(opp_head, grid, current_dir)
        if valid:
            if action not in valid:
                action = valid[0]
        elif current_dir is not None:
            action = current_dir
        return action


def get_opponent(opponent_type: str = None, minimax_depth: int = None, tronbot_timeout: float = None):
    opponent_type = opponent_type or DEFAULT_OPPONENT_TYPE
    opponents = {
        'random': RandomOpponent,
        'heuristic': HeuristicOpponent,
        'lookahead': LookaheadOpponent,
        'minimax': MinimaxOpponent,
        'tronbot': TronBotOpponent,
    }

    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}. "
                        f"Available: {list(opponents.keys())}")

    if opponent_type == 'minimax':
        depth = minimax_depth if minimax_depth is not None else DEFAULT_MINIMAX_DEPTH
        return MinimaxOpponent(depth=depth)
    if opponent_type == 'tronbot':
        timeout = tronbot_timeout if tronbot_timeout is not None else TIMEOUT_SEC
        return TronBotOpponent(move_timeout=timeout, first_move_timeout=timeout * 3)
    return opponents[opponent_type]()
