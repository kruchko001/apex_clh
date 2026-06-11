import os
import importlib.util
import sys
import threading
import time

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise ImportError("Could not locate tron_solution package root")
    _d = _parent

ROOT = os.path.dirname(_d)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "official", "shared", "competition", "src"))

import pygame
import torch
import numpy as np
from enum import IntEnum

from competition.tron.tron import TronGame, GameConfig, PLAYER_TRAIL_START, WALL
from duel_utils import challenger_side, challenger_corner, apply_opening_cross, DEFAULT_DUEL_RUNS

CELL = 20
GRID = 32
WINDOW = GRID * CELL
FPS = 60
KING_MODEL = os.path.join(ROOT, "kings", "code_submission_v1.pt")
KING_MOVE_LIMIT = 0.1
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YOU = (100, 200, 255)
KING = (255, 200, 50)
P0_TRAIL = (0, 120, 255)
P1_TRAIL = (255, 100, 180)
WALL_COLOR = (70, 70, 70)
GRID_LINE = (28, 28, 28)


def _load_encode():
    spec = importlib.util.spec_from_file_location("encode", os.path.join(ROOT, "tron_paper", "env", "encode.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.encode_official


encode_official = _load_encode()


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def load_king(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def king_action(model, g, king_as):
    p = g.get_player(king_as)
    valid = g.get_valid_actions(king_as)
    if not valid:
        return int(p.direction)
    obs = encode_official(g, king_as)
    t = torch.from_numpy(obs).unsqueeze(0).float()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    q = out if out.dim() == 1 else out.squeeze(0)
    q_np = q.cpu().numpy()[:4]
    masked = np.full(4, float("-inf"))
    for a in valid:
        masked[a] = q_np[a]
    a = int(np.argmax(masked))
    if (time.perf_counter() - t0) > KING_MOVE_LIMIT:
        a = int(p.direction)
    elif masked[a] == float("-inf"):
        a = valid[0]
    return a if a in valid else valid[0]


class HumanVsKing:
    def __init__(self, king_path=KING_MODEL, runs=DEFAULT_DUEL_RUNS):
        pygame.init()
        pygame.key.set_repeat(0)
        self.screen = pygame.display.set_mode((WINDOW, WINDOW + 40))
        pygame.display.set_caption("Tron Duel: You vs King")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)
        self.label_font = pygame.font.SysFont("Arial", 11, bold=True)
        self.small = pygame.font.SysFont("Arial", 14)

        self.king = load_king(king_path)
        self.king_path = king_path
        self.runs = runs
        self.cfg = GameConfig(width=32, height=32, max_steps=500, num_players=2)
        self.run_idx = 0
        self.wins = {"You": 0, "King": 0, "Draw": 0}
        self.game_over = False
        self.message = ""
        self.human_as = 0
        self.g = None
        self.start_run()

    def start_run(self):
        self.game_over = False
        self.message = ""
        self.human_as = challenger_side(self.run_idx)
        corner = challenger_corner(self.run_idx)
        self.g = TronGame(self.cfg)
        self.g.reset()
        apply_opening_cross(self.g)
        if self.g.game_over:
            self._finish_run()
        self.status = f"Run {self.run_idx + 1}/{self.runs} — YOU P{self.human_as} @ {corner}"

    def key_to_action(self, key):
        p = self.g.get_player(self.human_as)
        d = int(p.direction)
        if key in (pygame.K_w, pygame.K_UP) and d != Action.DOWN:
            return Action.UP
        if key in (pygame.K_s, pygame.K_DOWN) and d != Action.UP:
            return Action.DOWN
        if key in (pygame.K_a, pygame.K_LEFT) and d != Action.RIGHT:
            return Action.LEFT
        if key in (pygame.K_d, pygame.K_RIGHT) and d != Action.LEFT:
            return Action.RIGHT
        return None

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    def _outcome(self):
        king_as = 1 - self.human_as
        h = self.g.get_player(self.human_as)
        k = self.g.get_player(king_as)
        if h.alive and not k.alive:
            return "You"
        if k.alive and not h.alive:
            return "King"
        return "Draw"

    def _finish_run(self):
        self.game_over = True
        who = self._outcome()
        self.wins[who] += 1
        self.message = f"{who} wins run {self.run_idx + 1}"

    def get_king_move(self):
        king_as = 1 - self.human_as
        result = {}

        def work():
            result["a"] = king_action(self.king, self.g, king_as)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            self._pump_events()
            self.draw(thinking=True)
            t.join(0.05)
        return int(result["a"])

    def resolve_step(self, human_a, king_a):
        actions = {self.human_as: int(human_a), 1 - self.human_as: int(king_a)}
        self.g.step(actions)
        if self.g.game_over:
            self._finish_run()

    def draw(self, thinking=False):
        self.screen.fill(BLACK)
        grid = self.g.grid
        for y in range(GRID):
            for x in range(GRID):
                rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
                v = int(grid[y, x])
                if v == WALL:
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                elif v == PLAYER_TRAIL_START:
                    pygame.draw.rect(self.screen, P0_TRAIL, rect)
                elif v == PLAYER_TRAIL_START + 1:
                    pygame.draw.rect(self.screen, P1_TRAIL, rect)
        for c in range(GRID + 1):
            x = c * CELL
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW))
        for r in range(GRID + 1):
            y = r * CELL
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW, y))

        for pid in (0, 1):
            p = self.g.get_player(pid)
            if not p.alive:
                continue
            color = YOU if pid == self.human_as else KING
            pygame.draw.rect(
                self.screen, color,
                pygame.Rect(p.x * CELL + 2, p.y * CELL + 2, CELL - 4, CELL - 4),
            )
            tag = "YOU" if pid == self.human_as else "K"
            s = self.label_font.render(tag, True, BLACK)
            self.screen.blit(s, s.get_rect(center=(p.x * CELL + CELL // 2, p.y * CELL + CELL // 2)))

        y0 = WINDOW + 4
        self.screen.blit(self.small.render(self.status, True, WHITE), (8, y0))
        score = f"Score YOU {self.wins['You']} / King {self.wins['King']} / D {self.wins['Draw']}"
        self.screen.blit(self.small.render(score, True, WHITE), (8, y0 + 18))

        if self.game_over:
            surf = self.big_font.render(self.message, True, WHITE)
            self.screen.blit(surf, surf.get_rect(center=(WINDOW // 2, WINDOW // 2)))
            if self.run_idx + 1 < self.runs:
                sub = self.font.render("SPACE: next run  ESC: quit", True, (200, 200, 200))
            else:
                sub = self.font.render("SPACE: quit  ESC: quit", True, (200, 200, 200))
            self.screen.blit(sub, sub.get_rect(center=(WINDOW // 2, WINDOW // 2 + 44)))
        else:
            status = "King thinking..." if thinking else "Your turn (WASD)"
            self.screen.blit(self.font.render(status, True, WHITE), (10, 10))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and self.game_over:
                        if self.run_idx + 1 < self.runs:
                            self.run_idx += 1
                            self.start_run()
                        else:
                            running = False
                    elif not self.game_over:
                        action = self.key_to_action(event.key)
                        if action is not None:
                            ka = self.get_king_move()
                            self.resolve_step(action, ka)
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def main(king_path=KING_MODEL, runs=DEFAULT_DUEL_RUNS):
    print(f"You vs King — {runs} runs, opening cross each round")
    print("Challenger spawns cycle: (1,1) / (30,30) / (1,1)")
    print("WASD: one key = both players move")
    HumanVsKing(king_path, runs).run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--king", default=KING_MODEL)
    p.add_argument("--runs", type=int, default=DEFAULT_DUEL_RUNS)
    args = p.parse_args()
    main(args.king, args.runs)
