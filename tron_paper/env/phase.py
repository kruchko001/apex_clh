import importlib.util
import os

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("tron_paper_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent

from collections import deque
from typing import Set, Tuple

import numpy as np

from competition.tron.tron import EMPTY, PLAYER_TRAIL_START, WALL

Pos = Tuple[int, int]


def _empty_neighbors(grid: np.ndarray, y: int, x: int) -> list:
    h, w = grid.shape
    out = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == EMPTY:
            out.append((ny, nx))
    return out


def reachable_empty_cells(grid: np.ndarray, start: Pos) -> Set[Pos]:
    seen: Set[Pos] = set()
    q = deque(_empty_neighbors(grid, start[0], start[1]))
    while q:
        cell = q.popleft()
        if cell in seen:
            continue
        seen.add(cell)
        for n in _empty_neighbors(grid, cell[0], cell[1]):
            if n not in seen:
                q.append(n)
    return seen


def agents_separated(grid: np.ndarray, pos0: Pos, pos1: Pos) -> bool:
    r0 = reachable_empty_cells(grid, pos0)
    r1 = reachable_empty_cells(grid, pos1)
    return r0.isdisjoint(r1)


def agents_separated_from_obs(obs: np.ndarray) -> bool:
    h, w = obs.shape[1:]
    grid = np.zeros((h, w), dtype=np.int32)
    grid[obs[0] > 0.5] = WALL
    grid[obs[1] > 0.5] = PLAYER_TRAIL_START
    grid[obs[2] > 0.5] = PLAYER_TRAIL_START + 1
    my = np.argwhere(obs[3] > 0.5)
    opp = np.argwhere(obs[4] > 0.5)
    if len(my) == 0 or len(opp) == 0:
        return False
    return agents_separated(grid, tuple(my[0]), tuple(opp[0]))


def approximate_survival_steps(policy_fn, obs: np.ndarray, max_steps: int = 500) -> int:
    state = obs.copy()
    state[2] = 0.0
    state[4] = 0.0
    steps = 0
    for _ in range(max_steps):
        action = int(policy_fn(state))
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        head = np.argwhere(state[3] > 0.5)
        if len(head) == 0:
            break
        y, x = int(head[0][0]), int(head[0][1])
        ny, nx = y + dy, x + dx
        if ny <= 0 or nx <= 0 or ny >= state.shape[1] - 1 or nx >= state.shape[2] - 1:
            break
        if state[0, ny, nx] > 0.5 or state[1, ny, nx] > 0.5:
            break
        state[1, y, x] = 1.0
        state[3] = 0.0
        state[3, ny, nx] = 1.0
        steps += 1
    return steps
