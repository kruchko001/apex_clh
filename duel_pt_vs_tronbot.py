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
from tronbot.python.mytronbot import TIMEOUT_SEC, FIRST_MOVE_TIMEOUT_SEC
from tron_solution.env.tronbot_player import PersistentTronBotPlayer, default_tronbot_path

DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
PT_MOVE_LIMIT = 0.1
TB_SUBPROC_TIMEOUT = FIRST_MOVE_TIMEOUT_SEC + 1.0
CELL = 16
COLORS = {
    "bg": (12, 12, 18),
    "wall": (90, 90, 100),
    "p0_trail": (0, 180, 255),
    "p1_trail": (255, 100, 180),
    "pt_head": (100, 255, 140),
    "tb_head": (255, 160, 60),
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


def _bot_view(grid, pid, p, opp):
    class V:
        pass
    view = V()
    h, w = grid.shape
    view.grid_size = h
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    view.walls = (grid == WALL) | border
    view.my_trail = grid >= PLAYER_TRAIL_START + pid
    view.opponent_trail = grid >= PLAYER_TRAIL_START + (1 - pid)
    view.my_head = (p.y, p.x)
    view.opponent_head = (opp.y, opp.x)
    view.current_direction = int(p.direction)
    view.opponent_direction = int(opp.direction)
    view.OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}
    view.DIRECTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    return view


def _outcome(pt_as, g):
    p0, p1 = g.get_player(0), g.get_player(1)
    pt_alive = p0.alive if pt_as == 0 else p1.alive
    tb_alive = p1.alive if pt_as == 0 else p0.alive
    if pt_alive and not tb_alive:
        return "PT wins"
    if tb_alive and not pt_alive:
        return "TronBot wins"
    return "Draw"


def _make_tronbot(pt_as):
    return PersistentTronBotPlayer(
        default_tronbot_path(),
        as_player=1 - pt_as,
        move_timeout=TB_SUBPROC_TIMEOUT,
        use_timer=True,
    )


def _pick_actions(g, model, bot, pt_as):
    forced = corner_first_actions(g)
    if forced is not None:
        return forced, None, False
    actions = {}
    pt_ms = None
    pt_slow = False
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
            p0, p1 = g.get_player(0), g.get_player(1)
            view = _bot_view(g.grid, 0, p0, p1)
            a = bot.get_action(view)
            actions[pid] = a if a in valid else valid[0]
    return actions, pt_ms, pt_slow


def run_duel(model, seed=0, pt_as=0, verbose=False):
    g = TronGame(GameConfig(width=32, height=32, max_steps=500, num_players=2))
    g.reset(seed=seed)
    bot = _make_tronbot(pt_as)
    bot.start()
    steps = 0
    pt_times = []
    pt_timeouts = 0
    while not g.game_over:
        actions, pt_ms, pt_slow = _pick_actions(g, model, bot, pt_as)
        if pt_ms is not None:
            pt_times.append(pt_ms / 1000)
            if pt_slow:
                pt_timeouts += 1
        if verbose and steps < 15:
            for pid, a in actions.items():
                p = g.get_player(pid)
                role = "PT" if pid == pt_as else "TB"
                print(f"  step {steps:3d}  P{pid}({role})  {DIRS[a]}  head=({p.y},{p.x})")
        g.step(actions)
        steps += 1
    bot.close()
    return _outcome(pt_as, g), steps, pt_times, pt_timeouts


def _draw_frame(screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines):
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
        color = COLORS["pt_head"] if pid == pt_as else COLORS["tb_head"]
        pygame.draw.rect(
            screen, color,
            pygame.Rect(p.x * CELL + 2, p.y * CELL + 2, CELL - 4, CELL - 4),
        )
        tag = "PT" if pid == pt_as else "TB"
        blit_head_label(screen, label_font, p.x, p.y, CELL, tag)
    lines = [
        f"step {step}  seed={status_lines.get('seed')}  duel={status_lines.get('duel')}",
        f"PT=P{pt_as} (green, {PT_MOVE_LIMIT}s limit)  TronBot=P{1-pt_as} (orange, full C++)",
    ]
    if last_actions:
        parts = []
        for pid, a in sorted(last_actions.items()):
            tag = "PT" if pid == pt_as else "TB"
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


def _wait_space(clock, screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
        _draw_frame(screen, font, g, pt_as, step, last_actions, pt_ms, pt_slow, status_lines)
        pygame.display.flip()
        clock.tick(30)


def watch_duel(model_path=MODEL, episodes=3, delay_ms=80, seed=None, pt_as=0, swap_spawn=True):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    pygame.init()
    g_cfg = GameConfig(width=32, height=32, max_steps=500, num_players=2)
    screen = pygame.display.set_mode((32 * CELL, 32 * CELL + 140))
    pygame.display.set_caption("PT (0.1s) vs full C++ TronBot")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    rng = np.random.RandomState(seed)
    wins = {"PT": 0, "TronBot": 0, "Draw": 0}

    for ep in range(episodes):
        ep_seed = int(rng.randint(0, 10_000))
        side = (pt_as + ep) % 2 if swap_spawn else pt_as
        g = TronGame(g_cfg)
        g.reset(seed=ep_seed)
        opening = apply_opening_cross(g)
        bot = _make_tronbot(side)
        bot.start()
        step = 1 if opening else 0
        last_actions = opening or {}
        pt_ms = None
        pt_slow = False
        pt_timeouts = 0
        status = {"seed": ep_seed, "duel": ep + 1, "extra": []}

        while not g.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    bot.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    bot.close()
                    pygame.quit()
                    return

            last_actions, pt_ms, pt_slow = _pick_actions(g, model, bot, side)
            if pt_slow:
                pt_timeouts += 1
            g.step(last_actions)
            step += 1
            _draw_frame(screen, font, g, side, step, last_actions, pt_ms, pt_slow, status)
            pygame.display.flip()
            clock.tick(max(1, 1000 // max(delay_ms, 1)))

        bot.close()
        outcome = _outcome(side, g)
        if outcome == "PT wins":
            wins["PT"] += 1
        elif outcome == "TronBot wins":
            wins["TronBot"] += 1
        else:
            wins["Draw"] += 1
        status["extra"] = [
            f"Result: {outcome} in {step} steps  (PT was P{side})",
            f"PT timeouts: {pt_timeouts}  Score PT {wins['PT']} / TB {wins['TronBot']} / D {wins['Draw']}",
        ]
        if ep + 1 < episodes:
            if not _wait_space(clock, screen, font, g, side, step, last_actions, pt_ms, pt_slow, status):
                pygame.quit()
                return

    status["extra"] = [f"Final: PT {wins['PT']}  TronBot {wins['TronBot']}  Draw {wins['Draw']}"]
    _draw_frame(screen, font, g, side, step, last_actions, pt_ms, pt_slow, status)
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
    p = argparse.ArgumentParser(description="PT submission vs C++ TronBot duel")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pt-as", type=int, default=0, choices=[0, 1])
    p.add_argument("--no-swap-spawn", action="store_true", help="PT always same spawn; default alternates P0/P1")
    args = p.parse_args()

    if args.gui:
        watch_duel(args.model, args.episodes, args.delay_ms, args.seed, args.pt_as, swap_spawn=not args.no_swap_spawn)
        return

    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()
    print(f"=== PT ({PT_MOVE_LIMIT}s limit) vs full C++ TronBot ({TIMEOUT_SEC}s/move) seed=0 ===")
    o, steps, pt_times, pt_to = run_duel(model, seed=0, pt_as=args.pt_as, verbose=True)
    if pt_times:
        print(f"Result: {o} in {steps} steps  PT avg {np.mean(pt_times)*1000:.1f}ms max {max(pt_times)*1000:.1f}ms  timeouts {pt_to}")
    else:
        print(f"Result: {o} in {steps} steps")


if __name__ == "__main__":
    main()
