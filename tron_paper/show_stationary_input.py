"""Save stationary agent obs (5 ch x 32 x 32) as grayscale PNGs for 20 steps."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tron_paper  # noqa: F401

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from tron_paper.env.encode import PLAY_SIZE
from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper.model.phase_torch import (
    crop_play_obs,
    extract_stationary_input,
    mask_unreachable_as_walls,
)

CH_NAMES = ["walls_border", "my_trail", "blocked", "my_head", "opp_head"]
MASK_CH_NAMES = ["walls_masked", "my_trail", "blocked", "my_head", "opp_head"]
MODEL_CH_NAMES = ["unreachable_added", "my_head"]
OUT = os.path.join(os.path.dirname(__file__), "_stationary_obs_preview")
STEPS = 20
SCALE = 16
VAL = 128
GRID = 255
ML = 52
MT = 28
FONT_SIZE = 14


def _font():
    for name in ("consola.ttf", "Consolas.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            pass
    return ImageFont.load_default()


def add_labels(grid: np.ndarray, n: int = 32) -> Image.Image:
    gh, gw = grid.shape
    canvas = Image.new("L", (ML + gw, MT + gh), 0)
    canvas.paste(Image.fromarray(grid, mode="L"), (ML, MT))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for i in range(n):
        row_text = str(i)
        rb = draw.textbbox((0, 0), row_text, font=font)
        rw, rh = rb[2] - rb[0], rb[3] - rb[1]
        draw.text(
            ((ML - rw) // 2 - rb[0], MT + i * SCALE + (SCALE - rh) // 2 - rb[1]),
            row_text,
            fill=GRID,
            font=font,
        )
        col_text = str(i)
        cb = draw.textbbox((0, 0), col_text, font=font)
        cw, ch = cb[2] - cb[0], cb[3] - cb[1]
        draw.text(
            (ML + i * SCALE + (SCALE - cw) // 2 - cb[0], (MT - ch) // 2 - cb[1]),
            col_text,
            fill=GRID,
            font=font,
        )
    return canvas


def to_grid_image(channel: np.ndarray) -> np.ndarray:
    ch = (np.clip(channel, 0, 1) * VAL).astype(np.uint8)
    h, w = ch.shape
    img = np.repeat(np.repeat(ch, SCALE, axis=0), SCALE, axis=1)
    for y in range(0, h * SCALE, SCALE):
        img[y, :] = GRID
    for x in range(0, w * SCALE, SCALE):
        img[:, x] = GRID
    img[-1, :] = GRID
    img[:, -1] = GRID
    return img


def save_obs(obs: np.ndarray, step: int):
    t = torch.from_numpy(obs).unsqueeze(0).float()
    cropped = crop_play_obs(t)[0].numpy()
    masked = mask_unreachable_as_walls(crop_play_obs(t))[0].numpy()
    model_in = extract_stationary_input(t)[0].numpy()

    d = os.path.join(OUT, f"step_{step:02d}")
    os.makedirs(d, exist_ok=True)
    for c, name in enumerate(CH_NAMES):
        img = add_labels(to_grid_image(obs[c]))
        img.save(os.path.join(d, f"ch{c}_{name}.png"))

    mk = os.path.join(d, "masked")
    os.makedirs(mk, exist_ok=True)
    for c, name in enumerate(MASK_CH_NAMES):
        img = add_labels(to_grid_image(masked[c]), n=PLAY_SIZE)
        img.save(os.path.join(mk, f"ch{c}_{name}.png"))
    extra = np.clip(masked[0] - cropped[0], 0, 1)
    img = add_labels(to_grid_image(extra), n=PLAY_SIZE)
    img.save(os.path.join(mk, "ch0_unreachable_added.png"))

    md = os.path.join(d, "model_input")
    os.makedirs(md, exist_ok=True)
    for c, name in enumerate(MODEL_CH_NAMES):
        img = add_labels(to_grid_image(model_in[c]), n=PLAY_SIZE)
        img.save(os.path.join(md, f"ch{c}_{name}.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    env = StationaryTronEnv()
    obs, _ = env.reset(seed=42)
    save_obs(obs, 0)

    for t in range(1, STEPS):
        valid = [i for i, ok in enumerate(env.action_masks()) if ok]
        action = random.choice(valid) if valid else 0
        obs, _, term, trunc, _ = env.step(action)
        save_obs(obs, t)
        if term or trunc:
            break

    print(f"Saved {STEPS} timesteps -> {OUT}")
    print(f"  raw 5ch (32x32) in step_XX/")
    print(f"  masked 5ch + unreachable_added ({PLAY_SIZE}x{PLAY_SIZE}) in step_XX/masked/")
    print(f"  model input 2ch ({PLAY_SIZE}x{PLAY_SIZE}) in step_XX/model_input/")
    print(f"  {SCALE}x upscale, value x{VAL}, white grid + row/col labels")


if __name__ == "__main__":
    main()
