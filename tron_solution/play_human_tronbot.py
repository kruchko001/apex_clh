import os
import importlib.util
import sys
import threading

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pygame
from enum import IntEnum

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.tronbot_player import TronBotPlayer, default_tronbot_path
from duel_utils import challenger_side, challenger_corner, opening_cross_tronenv, DEFAULT_DUEL_RUNS

CELL_SIZE = 20
GRID = 32
WINDOW = GRID * CELL_SIZE
FPS = 60
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play_human_tronbot.log")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 120, 255)
RED = (255, 50, 50)
GRID_LINE = (28, 28, 28)
WALL_COLOR = (70, 70, 70)
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class HumanVsTronBot:
    def __init__(self, tronbot_path=None, move_timeout=10.0, runs=DEFAULT_DUEL_RUNS):
        pygame.init()
        pygame.key.set_repeat(0)
        self.screen = pygame.display.set_mode((WINDOW, WINDOW + 36))
        pygame.display.set_caption("Tron Duel: You vs TronBot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)
        self.label_font = pygame.font.SysFont("Arial", 11, bold=True)
        self.small = pygame.font.SysFont("Arial", 14)

        self.env = TronEnv(opponent_type="random", render_my_label="YOU", render_opp_label="TronBot")
        self.tronbot = TronBotPlayer(tronbot_path or default_tronbot_path(), as_player=1, move_timeout=move_timeout)
        self.tronbot.start()

        self.runs = runs
        self.run_idx = 0
        self.wins = {"You": 0, "TronBot": 0, "Draw": 0}
        self.step_num = 0
        self.game_over = False
        self.message = ""
        self.status = ""
        with open(LOG_PATH, "w") as f:
            f.write("=== play_human_tronbot ===\n")
            f.write(f"tronbot={self.tronbot.bot_path}\n")
        self.start_run()

    def _log(self, msg, console=False):
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    def start_run(self):
        self.game_over = False
        self.message = ""
        self.step_num = 0
        human_as = challenger_side(self.run_idx)
        corner = challenger_corner(self.run_idx)
        self.env.reset(options={"play_as": human_as})
        self.tronbot.as_player = 1 - human_as
        self.status = f"Run {self.run_idx + 1}/{self.runs} — YOU @ {corner}"
        self._log(
            f"run={self.run_idx + 1} | YOU={self.env.my_head} dir={DIR_NAMES[self.env.current_direction]} | "
            f"TronBot={self.env.opponent_head} dir={DIR_NAMES[self.env.opponent_direction]}"
        )
        if opening_cross_tronenv(self.env):
            self.step_num = 1
            self._log("opening cross -> (1,1)=DOWN (30,30)=UP")
            if self._check_end_after_step():
                return

    def _check_end_after_step(self):
        if self.env.step_count >= self.env.max_steps:
            self.game_over = True
            self.message = "Draw (timeout)"
            self.wins["Draw"] += 1
            return True
        return False

    def key_to_action(self, key):
        d = self.env.current_direction
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

    def get_tronbot_move(self):
        result = {}
        def work():
            result["action"] = self.tronbot.get_action(self.env)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        while thread.is_alive():
            self._pump_events()
            self.draw(thinking=True)
            thread.join(0.05)
        return int(result["action"])

    def resolve_step(self, human_action, tb_action):
        _, _, terminated, truncated, info = self.env.step_dual(int(human_action), tb_action)
        self.step_num += 1
        self._log(
            f"step={self.step_num} YOU={DIR_NAMES[human_action]} TronBot={DIR_NAMES[tb_action]} | "
            f"YOU={self.env.my_head} TB={self.env.opponent_head}"
        )
        if terminated or truncated:
            self.game_over = True
            if info.get("clean_kill") or info.get("opponent_self_destruct"):
                self.message = "You Win!"
                self.wins["You"] += 1
                self._log("END -> YOU win", console=True)
            elif info.get("mutual_destruction") or info.get("timeout"):
                self.message = "Draw"
                self.wins["Draw"] += 1
                self._log("END -> draw", console=True)
            else:
                self.message = "TronBot Wins"
                self.wins["TronBot"] += 1
                self._log(f"END -> TronBot wins ({info.get('my_collision_type')})", console=True)

    def draw(self, thinking=False):
        self.screen.fill(BLACK)
        g = self.env.grid
        for r in range(GRID):
            for c in range(GRID):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if g[r, c] == -1:
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                elif g[r, c] == 1:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif g[r, c] == 2:
                    pygame.draw.rect(self.screen, RED, rect)

        for c in range(GRID + 1):
            x = c * CELL_SIZE
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW))
        for r in range(GRID + 1):
            y = r * CELL_SIZE
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW, y))

        my, mx = self.env.my_head
        oy, ox = self.env.opponent_head
        pygame.draw.rect(self.screen, (100, 200, 255), pygame.Rect(mx * CELL_SIZE, my * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, (255, 100, 100), pygame.Rect(ox * CELL_SIZE, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        you_s = self.label_font.render("YOU", True, WHITE)
        tb_s = self.label_font.render("TB", True, WHITE)
        self.screen.blit(you_s, you_s.get_rect(center=(mx * CELL_SIZE + CELL_SIZE // 2, my * CELL_SIZE + CELL_SIZE // 2)))
        self.screen.blit(tb_s, tb_s.get_rect(center=(ox * CELL_SIZE + CELL_SIZE // 2, oy * CELL_SIZE + CELL_SIZE // 2)))

        y0 = WINDOW + 4
        self.screen.blit(self.small.render(self.status, True, WHITE), (8, y0))
        score = f"Score YOU {self.wins['You']} / TB {self.wins['TronBot']} / D {self.wins['Draw']}"
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
            status = "TronBot thinking..." if thinking else "Your turn (WASD)"
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
                            tb = self.get_tronbot_move()
                            self.resolve_step(action, tb)
            self.draw()
            self.clock.tick(FPS)
        self.tronbot.close()
        pygame.quit()


def main(tronbot_path=None, move_timeout=10.0, runs=DEFAULT_DUEL_RUNS):
    print(f"You vs TronBot — {runs} runs, opening cross each round")
    print("Challenger spawns cycle: (1,1) / (30,30) / (1,1)  |  opening: DOWN @ (1,1), UP @ (30,30)")
    print("WASD: one key = both players move simultaneously")
    print(f"Log: {LOG_PATH}")
    HumanVsTronBot(tronbot_path, move_timeout, runs).run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Human vs TronBot (pygame)")
    p.add_argument("--tronbot-path", type=str, default=None)
    p.add_argument("--move-timeout", type=float, default=10.0)
    p.add_argument("--runs", type=int, default=DEFAULT_DUEL_RUNS)
    args = p.parse_args()
    main(args.tronbot_path, args.move_timeout, args.runs)
