from collections import deque

import numpy as np
from competition.tron.tron import EMPTY, WALL, PLAYER_TRAIL_START
from tron_solution.env.tronbot_player import TronBotPlayer, default_tronbot_path
from tronbot.python import MyTronBot


def find_fake_opp(grid: np.ndarray, y: int, x: int) -> tuple:
    h, w = grid.shape
    seen = set()
    q = deque([(y, x)])
    seen.add((y, x))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if nr <= 0 or nc <= 0 or nr >= h - 1 or nc >= w - 1:
                continue
            if (nr, nc) in seen:
                continue
            if grid[nr, nc] == EMPTY:
                seen.add((nr, nc))
                q.append((nr, nc))
    for r, c in ((1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)):
        if grid[r, c] == EMPTY and (r, c) not in seen:
            return r, c
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] == EMPTY and (r, c) not in seen:
                return r, c
    return 1, 1


class _StationaryBotView:
    def __init__(self, grid: np.ndarray, y: int, x: int, direction: int, fake_opp: tuple):
        h, w = grid.shape
        self.grid_size = h
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        self.walls = (grid == WALL) | border
        self.my_trail = grid >= PLAYER_TRAIL_START
        self.opponent_trail = np.zeros((h, w), dtype=bool)
        self.my_head = (y, x)
        self.opponent_head = fake_opp
        self.current_direction = direction
        self.opponent_direction = 0
        self.OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}
        self.DIRECTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


class TronBotStationaryTeacher:
    def __init__(
        self,
        backend: str = "cpp",
        bot_path: str = None,
        move_timeout: float = 5.0,
        max_depth: int = 100,
        mode: str = "spacefill",
    ):
        self.backend = backend
        self.fake_opp = (1, 1)
        if backend == "cpp":
            self.bot = TronBotPlayer(
                bot_path or default_tronbot_path(),
                as_player=0,
                move_timeout=move_timeout,
                use_timer=True,
            )
            self.bot.start()
        else:
            self.bot = MyTronBot()

    def reset_episode(self, grid: np.ndarray, y: int, x: int):
        self.fake_opp = find_fake_opp(grid, y, x)
        if self.backend == "py":
            self.bot.reset()

    def _cpp_action(self, grid, y, x, direction, valid_mask):
        view = _StationaryBotView(grid, y, x, direction, self.fake_opp)
        action = self.bot.get_action(view)
        valid = [a for a in range(4) if valid_mask[a]]
        if action in valid:
            return action
        return valid[0] if valid else action

    def _py_action(self, grid, y, x, valid_mask):
        M = np.zeros(grid.shape, dtype=np.int8)
        M[grid != 0] = 1
        h, w = M.shape
        M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 1
        action = self.bot.choose_move(M, (y, x), self.fake_opp)
        valid = [a for a in range(4) if valid_mask[a]]
        if action in valid:
            return action
        return valid[0] if valid else action

    def action(self, grid: np.ndarray, y: int, x: int, valid_mask: np.ndarray, direction: int = 0) -> int:
        if self.backend == "cpp":
            return self._cpp_action(grid, y, x, direction, valid_mask)
        return self._py_action(grid, y, x, valid_mask)
