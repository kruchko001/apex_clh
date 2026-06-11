import numpy as np
from competition.tron.tron import WALL, PLAYER_TRAIL_START
from tron_solution.env.tronbot_engine import Components


def _grid_to_tronbot_map(grid: np.ndarray) -> np.ndarray:
    M = np.zeros(grid.shape, dtype=np.int8)
    M[grid == WALL] = 1
    M[grid >= PLAYER_TRAIL_START] = 1
    M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 1
    return M


def tronbot_stationary_score(grid: np.ndarray, y: int, x: int) -> float:
    """TronBot endgame heuristic: fillable area + component value - degree - articulation."""
    M = _grid_to_tronbot_map(grid)
    pos = (y, x)
    M[pos] = 1
    cp = Components(M)
    score = (
        cp.fillablearea(pos)
        + 0.1 * cp.connectedvalue(pos)
        - 2 * cp._degree(pos)
        - 4 * cp._potential_articulation(pos)
    )
    return float(score)
