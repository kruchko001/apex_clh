import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pygame
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "official", "shared", "competition", "src"))

from competition.tron.tron import TronGame, GameConfig, PLAYER_TRAIL_START, WALL
from tron_solution.env.opponents import MinimaxOpponent

DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
PT_MOVE_LIMIT = 0.1
CELL = 16
COLORS = {
    "bg": (12, 12, 18),
    "wall": (90, 90, 100),
    "p0_trail": (0, 180, 255),
    "p1_trail": (255, 100, 180),
    "pt_head": (100, 255, 140),
    "mm_head": (200, 120, 255),
    "text": (230, 230, 230),
}


def _load_encode():
    spec = importlib.util.spec_from_file_location("encode", os.path.join(ROOT, "tron_paper", "env", "encode.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.encode_official


from duel_utils import corner_first_actions, apply_opening_cross, head_label_font, blit_head_label

encode_official = _load_encode()


def launcher_select_action(model, state_tensor, valid_actions):
    with torch.no_grad():
        output = model(state_tensor)
    q = output if output.dim() == 1 else output.squeeze(0)
    q_np = q.cpu().numpy()[:4]
    masked = np.full(4, float("-inf"))
    for a in valid_actions:
        if 0 <= a < 4:
            masked[a] = q_np[a]
    best = int(np.argmax(masked))
    if masked[best] == float("-inf"):
        return valid_actions[0] if valid_actions else 0
    return best


def _minimax_grid(g, pid):
    grid = np.zeros_like(g.grid, dtype=np.int32)
    grid[g.grid == WALL] = 1
    grid[g.grid == PLAYER_TRAIL_START + pid] = 2
    grid[g.grid == PLAYER_TRAIL_START + (1 - pid)] = 3
    return grid


def _outcome(pt_as, g):
    p0, p1 = g.get_player(0), g.get_player(1)
    pt_alive = p0.alive if pt_as == 0 else p1.alive
    mm_alive = p1.alive if pt_as == 0 else p0.alive
    if pt_alive and not mm_alive:
        return "PT wins"
    if mm_alive and not pt_alive:
        return "Minimax wins"
    return "Draw"


def _pick_actions(g, model, minimax, pt_as):
    forced = corner_first_actions(g)
    if forced is not None:
        return forced, None, False
    actions = {}
    pt_ms = None
    pt_slow = False
    mm_as = 1 - pt_as
    for pid in (0, 1):
        p = g.get_player(pid)
        if not p.alive:
            continue
        valid = g.get_valid_actions(pid)
        if not valid:
            actions[pid] = int(p.direction)
            continue
        if pid == pt_as:
            obs = encode_official(g, pid)
            t = torch.from_numpy(obs).unsqueeze(0).float()
            t0 = time.perf_counter()
            a = launcher_select_action(model, t, valid)
            pt_ms = (time.perf_counter() - t0) * 1000
            if pt_ms > PT_MOVE_LIMIT * 1000:
                a = int(p.direction)
                pt_slow = True
            actions[pid] = a if a in valid else valid[0]
        else:
            opp = g.get_player(mm_as if pid == pt_as else pt_as)
            grid = _minimax_grid(g, pid)
            a = minimax.get_action(
                my_head=(p.y, p.x),
                opp_head=(opp.y, opp.x),
                grid=grid,
                current_dir=int(p.direction),
                my_dir=int(p.direction),
            )
            actions[pid] = a if a in valid else valid[0]
    return actions, pt_ms, pt_slow


def _draw_frame(screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines, minimax_depth):
    grid = g.grid
    h, w = grid.shape
    label_font = head_label_font()
    screen.fill(COLORS["bg"])
    for y in range(h):
        for x in range(w):
            v = int(grid[y, x])
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            if v == WALL:
                pygame.draw.rect(screen, COLORS["wall"], rect)
            elif v == PLAYER_TRAIL_START:
                pygame.draw.rect(screen, COLORS["p0_trail"], rect.inflate(-2, -2))
            elif v == PLAYER_TRAIL_START + 1:
                pygame.draw.rect(screen, COLORS["p1_trail"], rect.inflate(-2, -2))
    for pid in (0, 1):
        p = g.get_player(pid)
        if not p.alive:
            continue
        color = COLORS["pt_head"] if pid == pt_as else COLORS["mm_head"]
        pygame.draw.rect(
            screen, color,
            pygame.Rect(p.x * CELL + 2, p.y * CELL + 2, CELL - 4, CELL - 4),
        )
        tag = "PT" if pid == pt_as else "MM"
        blit_head_label(screen, label_font, p.x, p.y, CELL, tag)
    lines = [
        f"step {step}  seed={status_lines.get('seed')}  duel={status_lines.get('duel')}",
        f"PT=P{pt_as} (green, {PT_MOVE_LIMIT}s)  Minimax=P{1-pt_as} (purple, depth={minimax_depth})",
    ]
    if last_actions:
        parts = []
        for pid, a in sorted(last_actions.items()):
            tag = "PT" if pid == pt_as else "MM"
            parts.append(f"P{pid}({tag})={DIRS[a]}")
        lines.append("  ".join(parts))
    if pt_ms is not None:
        flag = " TIMEOUT->keep dir" if pt_slow else ""
        lines.append(f"PT inference: {pt_ms:.1f}ms{flag}")
    for extra in status_lines.get("extra", []):
        lines.append(extra)
    lines.append("SPACE=next duel  ESC=quit")
    y0 = h * CELL + 8
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, COLORS["text"]), (8, y0 + i * 20))


def _wait_space(clock, screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines, minimax_depth):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
        _draw_frame(screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines, minimax_depth)
        pygame.display.flip()
        clock.tick(30)


def watch_duel(model_path=MODEL, episodes=3, delay_ms=80, seed=None, pt_as=0, swap_spawn=True, minimax_depth=14):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    minimax = MinimaxOpponent(depth=minimax_depth)

    pygame.init()
    g_cfg = GameConfig(width=32, height=32, max_steps=500, num_players=2)
    screen = pygame.display.set_mode((32 * CELL, 32 * CELL + 140))
    pygame.display.set_caption("PT TronBot vs Minimax")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    rng = np.random.RandomState(seed)
    wins = {"PT": 0, "Minimax": 0, "Draw": 0}

    for ep in range(episodes):
        ep_seed = int(rng.randint(0, 10_000))
        side = (pt_as + ep) % 2 if swap_spawn else pt_as
        g = TronGame(g_cfg)
        g.reset(seed=ep_seed)
        opening = apply_opening_cross(g)
        step = 1 if opening else 0
        last_actions = opening or {}
        pt_ms = None
        pt_slow = False
        pt_timeouts = 0
        status = {"seed": ep_seed, "duel": ep + 1, "extra": []}

        while not g.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return

            last_actions, pt_ms, pt_slow = _pick_actions(g, model, minimax, side)
            if pt_slow:
                pt_timeouts += 1
            g.step(last_actions)
            step += 1
            _draw_frame(screen, font, g, side, step, last_actions, pt_ms, pt_slow, status, minimax_depth)
            pygame.display.flip()
            clock.tick(max(1, 1000 // max(delay_ms, 1)))

        outcome = _outcome(side, g)
        if outcome == "PT wins":
            wins["PT"] += 1
        elif outcome == "Minimax wins":
            wins["Minimax"] += 1
        else:
            wins["Draw"] += 1
        status["extra"] = [
            f"Result: {outcome} in {step} steps  (PT was P{side})",
            f"PT timeouts: {pt_timeouts}  Score PT {wins['PT']} / MM {wins['Minimax']} / D {wins['Draw']}",
        ]
        if ep + 1 < episodes:
            if not _wait_space(clock, screen, font, g, side, step, last_actions, pt_ms, pt_slow, status, minimax_depth):
                pygame.quit()
                return

    status["extra"] = [f"Final: PT {wins['PT']}  Minimax {wins['Minimax']}  Draw {wins['Draw']}"]
    _draw_frame(screen, font, g, side, step, last_actions, pt_ms, pt_slow, status, minimax_depth)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q, pygame.K_SPACE):
                break
        else:
            clock.tick(30)
            continue
        break
    pygame.quit()


def main():
    p = argparse.ArgumentParser(description="PT TronBot vs Minimax duel")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pt-as", type=int, default=0, choices=[0, 1])
    p.add_argument("--minimax-depth", type=int, default=14)
    p.add_argument("--no-swap-spawn", action="store_true")
    args = p.parse_args()
    watch_duel(
        args.model, args.episodes, args.delay_ms, args.seed,
        args.pt_as, swap_spawn=not args.no_swap_spawn, minimax_depth=args.minimax_depth,
    )


if __name__ == "__main__":
    main()
