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

DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
NEW_MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
OLD_MODEL = os.path.join(ROOT, "_test_out", "tron_model_greedy.pt")
MOVE_LIMIT = 0.1
CELL = 16
COLORS = {
    "bg": (12, 12, 18),
    "wall": (90, 90, 100),
    "p0_trail": (0, 180, 255),
    "p1_trail": (255, 100, 180),
    "new_head": (100, 255, 140),
    "old_head": (255, 160, 60),
    "text": (230, 230, 230),
}


def _load_encode():
    spec = importlib.util.spec_from_file_location("encode", os.path.join(ROOT, "tron_paper", "env", "encode.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.encode_official


from duel_utils import corner_first_actions, apply_opening_cross, head_label_font, blit_head_label, head_tag

encode_official = _load_encode()


def export_greedy(path=OLD_MODEL):
    from tronbot.python.jit_mytronbot_greedy import TronBotSubmitGreedy
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.script(TronBotSubmitGreedy()).save(path)
    return path


def load_pt_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


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


def _outcome(new_as, g):
    p0, p1 = g.get_player(0), g.get_player(1)
    new_alive = p0.alive if new_as == 0 else p1.alive
    old_alive = p1.alive if new_as == 0 else p0.alive
    if new_alive and not old_alive:
        return "Enhanced wins"
    if old_alive and not new_alive:
        return "Previous wins"
    return "Draw"


def _pick_actions(g, new_model, old_model, new_as):
    forced = corner_first_actions(g)
    if forced is not None:
        return forced, {}, {}
    actions = {}
    inf_ms = {}
    inf_slow = {}
    for pid in (0, 1):
        p = g.get_player(pid)
        if not p.alive:
            continue
        valid = g.get_valid_actions(pid)
        if not valid:
            actions[pid] = int(p.direction)
            continue
        model = new_model if pid == new_as else old_model
        obs = encode_official(g, pid)
        t = torch.from_numpy(obs).unsqueeze(0).float()
        t0 = time.perf_counter()
        a = launcher_select_action(model, t, valid)
        ms = (time.perf_counter() - t0) * 1000
        slow = ms > MOVE_LIMIT * 1000
        if slow:
            a = int(p.direction)
        actions[pid] = a if a in valid else valid[0]
        inf_ms[pid] = ms
        inf_slow[pid] = slow
    return actions, inf_ms, inf_slow


def _draw_frame(screen, font, g, new_as, step, last_actions, inf_ms, inf_slow, status_lines):
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
        color = COLORS["new_head"] if pid == new_as else COLORS["old_head"]
        pygame.draw.rect(
            screen, color,
            pygame.Rect(p.x * CELL + 2, p.y * CELL + 2, CELL - 4, CELL - 4),
        )
        tag = head_tag("Enhanced") if pid == new_as else head_tag("Previous")
        blit_head_label(screen, label_font, p.x, p.y, CELL, tag)
    lines = [
        f"step {step}  seed={status_lines.get('seed')}  duel={status_lines.get('duel')}",
        f"Enhanced=P{new_as} (green)  Previous=P{1-new_as} (orange)  {MOVE_LIMIT}s limit",
    ]
    if last_actions:
        parts = []
        for pid, a in sorted(last_actions.items()):
            tag = "New" if pid == new_as else "Old"
            parts.append(f"P{pid}({tag})={DIRS[a]}")
        lines.append("  ".join(parts))
    if inf_ms:
        parts = []
        for pid in sorted(inf_ms):
            tag = "New" if pid == new_as else "Old"
            flag = " TIMEOUT" if inf_slow.get(pid) else ""
            parts.append(f"{tag} {inf_ms[pid]:.1f}ms{flag}")
        lines.append("  ".join(parts))
    for extra in status_lines.get("extra", []):
        lines.append(extra)
    lines.append("SPACE=next duel  ESC=quit")
    y0 = h * CELL + 8
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, COLORS["text"]), (8, y0 + i * 20))


def _wait_space(clock, screen, font, g, new_as, step, last_actions, inf_ms, inf_slow, status_lines):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
        _draw_frame(screen, font, g, new_as, step, last_actions, inf_ms, inf_slow, status_lines)
        pygame.display.flip()
        clock.tick(30)


def watch_duel(new_path=NEW_MODEL, old_path=OLD_MODEL, episodes=3, delay_ms=80, seed=None, new_as=0, swap_spawn=True):
    if not os.path.isfile(old_path):
        export_greedy(old_path)
    new_model = load_pt_model(new_path)
    old_model = load_pt_model(old_path)

    pygame.init()
    g_cfg = GameConfig(width=32, height=32, max_steps=500, num_players=2)
    screen = pygame.display.set_mode((32 * CELL, 32 * CELL + 156))
    pygame.display.set_caption("Enhanced PT vs Previous PT (greedy)")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    rng = np.random.RandomState(seed)
    wins = {"Enhanced": 0, "Previous": 0, "Draw": 0}

    for ep in range(episodes):
        ep_seed = int(rng.randint(0, 10_000))
        side = (new_as + ep) % 2 if swap_spawn else new_as
        g = TronGame(g_cfg)
        g.reset(seed=ep_seed)
        opening = apply_opening_cross(g)
        step = 1 if opening else 0
        last_actions = opening or {}
        inf_ms = {}
        inf_slow = {}
        status = {
            "seed": ep_seed,
            "duel": ep + 1,
            "extra": [
                f"new: {os.path.basename(new_path)}",
                f"old: {os.path.basename(old_path)}",
            ],
        }

        while not g.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return

            last_actions, inf_ms, inf_slow = _pick_actions(g, new_model, old_model, side)
            g.step(last_actions)
            step += 1
            _draw_frame(screen, font, g, side, step, last_actions, inf_ms, inf_slow, status)
            pygame.display.flip()
            clock.tick(max(1, 1000 // max(delay_ms, 1)))

        outcome = _outcome(side, g)
        if outcome == "Enhanced wins":
            wins["Enhanced"] += 1
        elif outcome == "Previous wins":
            wins["Previous"] += 1
        else:
            wins["Draw"] += 1
        status["extra"] = [
            f"Result: {outcome} in {step} steps  (Enhanced was P{side})",
            f"Score Enhanced {wins['Enhanced']} / Previous {wins['Previous']} / D {wins['Draw']}",
        ]
        if ep + 1 < episodes:
            if not _wait_space(clock, screen, font, g, side, step, last_actions, inf_ms, inf_slow, status):
                pygame.quit()
                return

    status["extra"] = [f"Final: Enhanced {wins['Enhanced']}  Previous {wins['Previous']}  Draw {wins['Draw']}"]
    _draw_frame(screen, font, g, side, step, last_actions, inf_ms, inf_slow, status)
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
    p = argparse.ArgumentParser(description="Enhanced PT TronBot vs previous 1-ply greedy PT")
    p.add_argument("--new", default=NEW_MODEL, help="Enhanced floodfill PT")
    p.add_argument("--old", default=OLD_MODEL, help="Previous greedy PT (exported if missing)")
    p.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--new-as", type=int, default=0, choices=[0, 1])
    p.add_argument("--no-swap-spawn", action="store_true")
    args = p.parse_args()
    watch_duel(
        args.new, args.old, args.episodes, args.delay_ms, args.seed,
        args.new_as, swap_spawn=not args.no_swap_spawn,
    )


if __name__ == "__main__":
    main()
