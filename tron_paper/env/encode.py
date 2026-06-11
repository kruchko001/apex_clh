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

import numpy as np

from competition.tron.tron import TronGame, PLAYER_TRAIL_START, WALL

GRID = 32
PLAY_SIZE = 30
MARGIN = 1
CHANNELS = 5
STATIONARY_CHANNELS = 2


def encode_official(game: TronGame, player_id: int) -> np.ndarray:
    grid = game.grid
    h, w = grid.shape
    state = np.zeros((CHANNELS, h, w), dtype=np.float32)
    state[0] = (grid == WALL).astype(np.float32)
    state[1] = (grid == PLAYER_TRAIL_START + player_id).astype(np.float32)
    opp = np.zeros_like(grid, dtype=np.float32)
    for oid in range(8):
        if oid != player_id:
            opp += (grid == PLAYER_TRAIL_START + oid).astype(np.float32)
    state[2] = np.clip(opp, 0.0, 1.0)
    me = game.get_player(player_id)
    if me is not None and me.alive:
        state[3, me.y, me.x] = 1.0
    for p in game.players:
        if p.id != player_id and p.alive:
            state[4, p.y, p.x] = 1.0
    return state


def encode_non_stationary(game: TronGame, player_id: int) -> np.ndarray:
    return encode_official(game, player_id)


def encode_stationary(game: TronGame, player_id: int) -> np.ndarray:
    obs = encode_official(game, player_id)
    grid = game.grid
    h, w = grid.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    is_wall = grid == WALL
    obs[0] = border.astype(np.float32)
    obs[2] = np.clip((is_wall & ~border).astype(np.float32), 0.0, 1.0)
    obs[4] = 0.0
    return obs


def encode_from_grid(grid: np.ndarray, my_id: int, my_pos, opp_pos) -> np.ndarray:
    h, w = grid.shape
    state = np.zeros((CHANNELS, h, w), dtype=np.float32)
    state[0] = (grid == WALL).astype(np.float32)
    state[1] = (grid == PLAYER_TRAIL_START + my_id).astype(np.float32)
    opp = np.zeros_like(grid, dtype=np.float32)
    for oid in range(8):
        if oid != my_id:
            opp += (grid == PLAYER_TRAIL_START + oid).astype(np.float32)
    state[2] = np.clip(opp, 0.0, 1.0)
    my_y, my_x = my_pos
    state[3, my_y, my_x] = 1.0
    if opp_pos is not None:
        oy, ox = opp_pos
        state[4, oy, ox] = 1.0
    return state
