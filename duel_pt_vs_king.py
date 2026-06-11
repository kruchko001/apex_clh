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
PT_MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
GREEDY_MODEL = os.path.join(ROOT, "_test_out", "tron_model_greedy.pt")
SPACEFILL_MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
KING_MODEL = os.path.join(ROOT, "kings", "code_submission_v1.pt")
MOVE_LIMIT = 0.1
CELL = 16
COLORS = {
    "bg": (12, 12, 18),
    "wall": (90, 90, 100),
    "p0_trail": (0, 180, 255),
    "p1_trail": (255, 100, 180),
    "pt_head": (100, 255, 140),
    "king_head": (255, 200, 50),
    "text": (230, 230, 230),
}


def _load_encode():
    spec = importlib.util.spec_from_file_location("encode", os.path.join(ROOT, "tron_paper", "env", "encode.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.encode_official


from duel_utils import corner_first_actions, challenger_side, challenger_corner, apply_opening_cross, head_label_font, blit_head_label, head_tag

encode_official = _load_encode()


def load_pt_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    ex = torch.zeros(1, 5, 32, 32)
    ex[0, 3, 1, 1] = 1.0
    ex[0, 4, 30, 30] = 1.0
    with torch.no_grad():
        out = model(ex)
    if out.dim() == 1:
        assert out.shape[0] == 4, f"expected (4,) logits, got {tuple(out.shape)}"
    else:
        assert out.shape[-1] == 4, f"expected last dim 4, got {tuple(out.shape)}"
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


def ensure_spacefill(path=SPACEFILL_MODEL):
    if os.path.isfile(path):
        return path
    from tronbot.python.jit_mytronbot import TronBotSubmit
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.script(TronBotSubmit()).save(path)
    return path


def ensure_greedy(path=GREEDY_MODEL):
    if os.path.isfile(path):
        return path
    from tronbot.python.jit_mytronbot_greedy import TronBotSubmitGreedy
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.script(TronBotSubmitGreedy()).save(path)
    return path


def _outcome(side_as, g, side_label="PT"):
    p0, p1 = g.get_player(0), g.get_player(1)
    side_alive = p0.alive if side_as == 0 else p1.alive
    king_alive = p1.alive if side_as == 0 else p0.alive
    if side_alive and not king_alive:
        return f"{side_label} wins"
    if king_alive and not side_alive:
        return "King wins"
    return "Draw"


def _pick_actions(g, pt_model, king_model, pt_as):
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
        model = pt_model if pid == pt_as else king_model
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


def _draw_frame(screen, font, g, pt_as, step, last_actions, inf_ms, inf_slow, status_lines, side_label="PT"):
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
        color = COLORS["pt_head"] if pid == pt_as else COLORS["king_head"]
        pygame.draw.rect(
            screen, color,
            pygame.Rect(p.x * CELL + 2, p.y * CELL + 2, CELL - 4, CELL - 4),
        )
        tag = head_tag(side_label) if pid == pt_as else "K"
        blit_head_label(screen, label_font, p.x, p.y, CELL, tag)
    lines = [
        f"step {step}  seed={status_lines.get('seed')}  duel={status_lines.get('duel')}",
        f"{side_label}=P{pt_as} (green)  King=P{1-pt_as} (gold)  both {MOVE_LIMIT}s limit",
    ]
    if last_actions:
        parts = []
        for pid, a in sorted(last_actions.items()):
            tag = side_label if pid == pt_as else "King"
            parts.append(f"P{pid}({tag})={DIRS[a]}")
        lines.append("  ".join(parts))
    if inf_ms:
        parts = []
        for pid in sorted(inf_ms):
            tag = side_label if pid == pt_as else "King"
            flag = " TIMEOUT" if inf_slow.get(pid) else ""
            parts.append(f"{tag} {inf_ms[pid]:.1f}ms{flag}")
        lines.append("  ".join(parts))
    for extra in status_lines.get("extra", []):
        lines.append(extra)
    lines.append("SPACE=next duel  ESC=quit")
    y0 = h * CELL + 8
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, COLORS["text"]), (8, y0 + i * 20))


def _wait_space(clock, screen, font, g, pt_as, step, last_actions, inf_ms, inf_slow, status_lines, side_label="PT"):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
        _draw_frame(screen, font, g, pt_as, step, last_actions, inf_ms, inf_slow, status_lines, side_label)
        pygame.display.flip()
        clock.tick(30)


def watch_duel(king_path, pt_path=PT_MODEL, episodes=3, delay_ms=80, seed=None, side_label="PT"):
    king_model = load_pt_model(king_path)
    pt_model = load_pt_model(pt_path)

    pygame.init()
    g_cfg = GameConfig(width=32, height=32, max_steps=500, num_players=2)
    screen = pygame.display.set_mode((32 * CELL, 32 * CELL + 156))
    pygame.display.set_caption(f"{side_label} TronBot vs King QNet")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    rng = np.random.RandomState(seed)
    wins = {side_label: 0, "King": 0, "Draw": 0}
    side_timeouts = 0
    king_timeouts = 0

    for ep in range(episodes):
        ep_seed = int(rng.randint(0, 10_000))
        side = challenger_side(ep)
        corner = challenger_corner(ep)
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
                f"King: {os.path.basename(king_path)}",
                f"run {ep + 1}/{episodes} — challenger {side_label} P{side} @ {corner}",
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

            last_actions, inf_ms, inf_slow = _pick_actions(g, pt_model, king_model, side)
            if inf_slow.get(side):
                side_timeouts += 1
            if inf_slow.get(1 - side):
                king_timeouts += 1
            g.step(last_actions)
            step += 1
            _draw_frame(screen, font, g, side, step, last_actions, inf_ms, inf_slow, status, side_label)
            pygame.display.flip()
            clock.tick(max(1, 1000 // max(delay_ms, 1)))

        outcome = _outcome(side, g, side_label)
        if outcome == f"{side_label} wins":
            wins[side_label] += 1
        elif outcome == "King wins":
            wins["King"] += 1
        else:
            wins["Draw"] += 1
        status["extra"] = [
            f"King: {os.path.basename(king_path)}",
            f"Result: {outcome} in {step} steps  ({side_label} was P{side})",
            f"Timeouts {side_label}/King: {side_timeouts}/{king_timeouts}  Score {side_label} {wins[side_label]} / King {wins['King']} / D {wins['Draw']}",
        ]
        if ep + 1 < episodes:
            if not _wait_space(clock, screen, font, g, side, step, last_actions, inf_ms, inf_slow, status, side_label):
                pygame.quit()
                return

    status["extra"] = [
        f"Final: {side_label} {wins[side_label]}  King {wins['King']}  Draw {wins['Draw']}",
        f"Timeouts {side_label}/King: {side_timeouts}/{king_timeouts}",
    ]
    _draw_frame(screen, font, g, side, step, last_actions, inf_ms, inf_slow, status, side_label)
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
    p = argparse.ArgumentParser(description="PT TronBot vs downloaded King QNet (official duel rules)")
    p.add_argument("--king", default=KING_MODEL, help="Path to king TorchScript .pt (QNet)")
    p.add_argument("--pt", default=PT_MODEL, help="Your PT TronBot submission")
    p.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    watch_duel(args.king, args.pt, args.episodes, args.delay_ms, args.seed)


if __name__ == "__main__":
    main()
