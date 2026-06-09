import random
from typing import Tuple

import numpy as np

import tron_paper  # noqa: F401
from competition.tron.tron import EMPTY, WALL


def generate_stationary_map(width: int = 32, height: int = 32, rng: random.Random = None) -> np.ndarray:
    rng = rng or random.Random()
    grid = np.zeros((height, width), dtype=np.int32)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    sides = ["top", "bottom", "left", "right"]
    start_side = rng.choice(sides)
    inner_w = width - 2
    inner_h = height - 2

    if start_side == "top":
        y, x = 1, rng.randint(1, width - 2)
        prev = (y - 1, x)
    elif start_side == "bottom":
        y, x = height - 2, rng.randint(1, width - 2)
        prev = (y + 1, x)
    elif start_side == "left":
        y, x = rng.randint(1, height - 2), 1
        prev = (y, x - 1)
    else:
        y, x = rng.randint(1, height - 2), width - 2
        prev = (y, x + 1)

    cy, cx = y, x
    last_dir = None
    straight = 0
    chosen_sides = {start_side}

    def on_target_side(py, px):
        if start_side == "top" and py >= height - 2:
            return True
        if start_side == "bottom" and py <= 1:
            return True
        if start_side == "left" and px >= width - 2:
            return True
        if start_side == "right" and px <= 1:
            return True
        return False

    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    opp = {"up": "down", "down": "up", "left": "right", "right": "left"}

    while not on_target_side(cy, cx):
        options = list(dirs.keys())
        if last_dir is not None:
            if last_dir in options:
                options.remove(opp[last_dir])
            if straight >= max(inner_w, inner_h) - 3 and last_dir in options:
                options.remove(last_dir)
        if not options:
            options = list(dirs.keys())
        move = rng.choice(options)
        dy, dx = dirs[move]
        ny, nx = cy + dy, cx + dx
        if ny <= 0 or nx <= 0 or ny >= height - 1 or nx >= width - 1:
            continue
        grid[cy, cx] = WALL
        straight = straight + 1 if move == last_dir else 1
        last_dir = move
        cy, cx = ny, nx

    if start_side in ("top", "bottom"):
        fill_side = "left" if cx < width // 2 else "right"
    else:
        fill_side = "top" if cy < height // 2 else "bottom"

    if fill_side == "left":
        grid[:, 1] = WALL
    elif fill_side == "right":
        grid[:, width - 2] = WALL
    elif fill_side == "top":
        grid[1, :] = WALL
    else:
        grid[height - 2, :] = WALL

    return grid
