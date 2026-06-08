import os
import importlib.util

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

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pygame
import sys
import threading
import numpy as np
from enum import IntEnum

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.opponents import get_opponent

PLAY_DEPTHS = {"easy": 4, "medium": 6, "hard": 8}

# --- Constants ---
CELL_SIZE = 20
GRID_WIDTH = 32
GRID_HEIGHT = 32
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 120, 255)      # Player
RED = (255, 50, 50)       # Opponent (AI)
GRID_LINE = (28, 28, 28)
WALL_COLOR = (70, 70, 70)
TEXT_COLOR = (255, 255, 255)

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play_human.log")
OPPOSITE = {Action.UP: Action.DOWN, Action.DOWN: Action.UP, Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT}
DELTAS = {Action.UP: (-1, 0), Action.RIGHT: (0, 1), Action.DOWN: (1, 0), Action.LEFT: (0, -1)}


def apply_simultaneous(grid, p1, p2, d1, d2, human_action, ai_action):
    if human_action == OPPOSITE[d1]:
        human_action = d1
    if ai_action == OPPOSITE[d2]:
        ai_action = d2
    dr1, dc1 = DELTAS[human_action]
    dr2, dc2 = DELTAS[ai_action]
    n1 = (p1[0] + dr1, p1[1] + dc1)
    n2 = (p2[0] + dr2, p2[1] + dc2)

    def blocked(pos):
        r, c = pos
        if not (0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH):
            return "Wall"
        if grid[r, c] != 0:
            return "Trail"
        return None

    if n1 == n2:
        return None, "Head-on"
    b1, b2 = blocked(n1), blocked(n2)
    if b1 and b2:
        return None, "Head-on"
    if b1:
        return None, ("AI", b1)
    if b2:
        return None, ("Human", b2)
    grid = grid.copy()
    grid[n1] = 1
    grid[n2] = 2
    return (grid, n1, n2, human_action, ai_action), None


def to_ai_grid(raw):
    g = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
    g[raw == -1] = 1
    g[raw == 1] = 2
    g[raw == 2] = 3
    return g


def metrics(opponent, grid, p1, p2, d1, d2):
    g = to_ai_grid(grid)
    ai_space = opponent._count_space(p2, g)
    human_space = opponent._count_space(p1, g)
    vor = opponent._voronoi(p2, p1, g)
    ai_mob = opponent._mobility(p2, g, d2)
    human_mob = opponent._mobility(p1, g, d1)
    return ai_space, human_space, vor, ai_mob, human_mob


class HumanDuel:
    def __init__(self, difficulty='hard', minimax_depth=None):
        pygame.init()
        pygame.key.set_repeat(0)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tron Duel: Human vs Minimax AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.label_font = pygame.font.SysFont('Arial', 11, bold=True)

        # Initialize Environment
        # We use a raw env for control, handling the step logic manually
        self.env = TronEnv()
        
        # Initialize Opponent
        if minimax_depth is None:
            if difficulty not in PLAY_DEPTHS:
                difficulty = 'medium'
            minimax_depth = PLAY_DEPTHS[difficulty]
        self.minimax_depth = minimax_depth
        self.opponent = get_opponent("minimax", minimax_depth)

        self.step_num = 0
        self.round_num = 0
        self.round_history = []
        self.end_reason = ""
        self.analyzed = False
        with open(LOG_PATH, "w") as f:
            f.write("=== play_human session ===\n")
            f.write(f"minimax_depth={self.minimax_depth}\n")
        self.reset_game()

    def _log(self, msg, console=False):
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    def _to_ai_grid(self):
        return to_ai_grid(self.grid)

    def _metrics(self):
        return metrics(self.opponent, self.grid, self.p1_pos, self.p2_pos, self.p1_dir, self.p2_dir)

    def _capture(self):
        ai_s, hum_s, vor, ai_m, hum_m = self._metrics()
        return {
            "step": self.step_num,
            "grid": self.grid.copy(),
            "p1_pos": self.p1_pos,
            "p2_pos": self.p2_pos,
            "p1_dir": self.p1_dir,
            "p2_dir": self.p2_dir,
            "ai_space": ai_s,
            "human_space": hum_s,
            "voronoi": vor,
            "ai_mob": ai_m,
            "human_mob": hum_m,
        }

    def _ai_valid_moves(self):
        return self.opponent._get_valid_moves(self.p2_pos, self._to_ai_grid(), self.p2_dir)

    def reset_game(self):
        self.round_num += 1
        self.step_num = 0
        self.round_history = []
        self.end_reason = ""
        self.analyzed = False
        self.observation, self.info = self.env.reset()
        self.game_over = False
        self.winner = None
        self.message = "Get Ready!"
        self.message_timer = 60 # frames
        
        # Extract state for AI
        self.grid = self.env.grid.copy()
        self.p1_pos = self.env.p1_pos
        self.p2_pos = self.env.p2_pos
        self.p1_dir = self.env.p1_dir
        self.p2_dir = self.env.p2_dir
        self._log(
            f"round={self.round_num} reset | YOU={self.p1_pos} dir={DIR_NAMES[self.p1_dir]} | "
            f"AI={self.p2_pos} dir={DIR_NAMES[self.p2_dir]}"
        )

    def _compute_ai_action(self, g):
        return self.opponent.get_action(
            obs=None,
            my_head=self.p1_pos,
            opp_head=self.p2_pos,
            grid=g,
            current_dir=self.p2_dir,
            my_dir=self.p1_dir,
        )

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    def get_opponent_move(self):
        pre = self._capture()
        g = self._to_ai_grid()
        valid = self.opponent._get_valid_moves(self.p2_pos, g, self.p2_dir)
        result = {}
        def work():
            result["action"] = self._compute_ai_action(g)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        while thread.is_alive():
            self._pump_events()
            self.draw(thinking=True)
            thread.join(0.05)
        action = result["action"]
        if action not in valid:
            self._log(
                f"step={self.step_num} AI INVALID | chose={DIR_NAMES[action]} "
                f"valid={[DIR_NAMES[a] for a in valid]}"
            )
            action = valid[0] if valid else action
        self._log(
            f"step={self.step_num} AI | chose={DIR_NAMES[action]} | "
            f"space AI={pre['ai_space']} YOU={pre['human_space']} vor={pre['voronoi']} | "
            f"YOU={pre['p1_pos']} AI={pre['p2_pos']}"
        )
        self._pending_step = {
            "pre": pre,
            "ai_action": action,
            "valid": valid,
        }
        return action

    def key_to_action(self, key):
        if key in (pygame.K_w, pygame.K_UP) and self.p1_dir != Action.DOWN:
            return Action.UP
        if key in (pygame.K_s, pygame.K_DOWN) and self.p1_dir != Action.UP:
            return Action.DOWN
        if key in (pygame.K_a, pygame.K_LEFT) and self.p1_dir != Action.RIGHT:
            return Action.LEFT
        if key in (pygame.K_d, pygame.K_RIGHT) and self.p1_dir != Action.LEFT:
            return Action.RIGHT
        return None

    def resolve_step(self, human_action, ai_action):
        pre = self._pending_step["pre"]
        result, outcome = apply_simultaneous(
            self.grid, self.p1_pos, self.p2_pos, self.p1_dir, self.p2_dir,
            human_action, ai_action,
        )
        if result is None:
            self.game_over = True
            if outcome == "Head-on":
                self.end_reason = "Head-on"
                self.winner = "Draw"
                self.message = "Head-on collision! Draw."
                self._log(f"step={self.step_num} END | head-on -> draw", console=True)
            else:
                winner, reason = outcome
                self.end_reason = reason
                if winner == "AI":
                    self.winner = "AI"
                    self.message = f"You Crashed! AI Wins. ({reason})"
                    self._log(f"step={self.step_num} END | YOU crashed ({reason}) -> AI wins", console=True)
                else:
                    self.winner = "Human"
                    self.message = f"AI Crashed! You Win! ({reason})"
                    self._log(f"step={self.step_num} END | AI crashed ({reason}) -> YOU win", console=True)
            self._finish_round()
            return

        self.step_num += 1
        self.grid, self.p1_pos, self.p2_pos, self.p1_dir, self.p2_dir = result
        self._log(
            f"step={self.step_num} YOU={DIR_NAMES[human_action]} AI={DIR_NAMES[ai_action]} | "
            f"space AI={pre['ai_space']} YOU={pre['human_space']} vor={pre['voronoi']}"
        )
        self.round_history.append({
            "step": self.step_num,
            "human_action": human_action,
            "ai_action": ai_action,
            "pre": pre,
        })

        self.env.grid = self.grid
        self.env.p1_pos = self.p1_pos
        self.env.p2_pos = self.p2_pos
        self.env.p1_dir = self.p1_dir
        self.env.p2_dir = self.p2_dir

    def _finish_round(self):
        if not getattr(self, "analyzed", False):
            self._analyze_round()
            self.analyzed = True

    def _analyze_round(self):
        if not self.round_history:
            return
        lines = [f"=== ANALYSIS round={self.round_num} | winner={self.winner} ==="]
        if not self.round_history:
            lines.append("No moves recorded.")
            for line in lines:
                self._log(line, console=True)
            return

        losing_territory = None
        for e in self.round_history:
            pre = e["pre"]
            if pre["human_space"] > pre["ai_space"] and losing_territory is None:
                losing_territory = pre["step"]

        if self.winner == "Human":
            lines.append(f"You won: AI crashed ({self.end_reason}).")
            if losing_territory is not None:
                lines.append(
                    f"Territory lost from step {losing_territory}: "
                    f"your reachable space exceeded AI's."
                )
            lines.append(
                "You outplayed AI by cutting space "
                "(cut across AI path before it closes a loop)."
            )
            lines.append(
                "How to beat you next time: AI should partition the board early "
                "(turn toward center, not the wall), then close a barrier before you cross."
            )
        else:
            lines.append(f"AI won: you crashed ({self.end_reason}).")
            lines.append("Your loss pattern: avoid boxing yourself; watch AI trail when turning.")

        for line in lines:
            self._log(line, console=True)

    def draw_grid_lines(self):
        for c in range(GRID_WIDTH + 1):
            x = c * CELL_SIZE
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW_HEIGHT))
        for r in range(GRID_HEIGHT + 1):
            y = r * CELL_SIZE
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW_WIDTH, y))

    def draw_head_label(self, pos, label, color):
        surf = self.label_font.render(label, True, color)
        cx = pos[1] * CELL_SIZE + CELL_SIZE // 2
        cy = pos[0] * CELL_SIZE + CELL_SIZE // 2
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    def draw(self, thinking=False):
        self.screen.fill(BLACK)

        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if r == 0 or r == GRID_HEIGHT - 1 or c == 0 or c == GRID_WIDTH - 1:
                    rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                    continue
                cell_val = self.grid[r, c]
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if cell_val == 1:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif cell_val == 2:
                    pygame.draw.rect(self.screen, RED, rect)

        self.draw_grid_lines()

        # Draw Heads (Brighter)
        p1_rect = pygame.Rect(self.p1_pos[1] * CELL_SIZE, self.p1_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (100, 200, 255), p1_rect)
        self.draw_head_label(self.p1_pos, "YOU", WHITE)
        
        p2_rect = pygame.Rect(self.p2_pos[1] * CELL_SIZE, self.p2_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 100, 100), p2_rect)
        self.draw_head_label(self.p2_pos, "AI", WHITE)

        # Draw UI Text
        if self.game_over:
            text_surf = self.big_font.render(self.message, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(text_surf, text_rect)
            
            sub_surf = self.font.render("Press SPACE to Restart or ESC to Quit", True, (200, 200, 200))
            sub_rect = sub_surf.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
            self.screen.blit(sub_surf, sub_rect)
        else:
            status = "AI thinking..." if thinking else "Your turn"
            score_text = self.font.render(status, True, WHITE)
            self.screen.blit(score_text, (10, 10))

            inst_text = self.font.render("WASD: one simultaneous step per key", True, (150, 150, 150))
            self.screen.blit(inst_text, (10, WINDOW_HEIGHT - 40))

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
                        self.reset_game()
                    elif not self.game_over:
                        action = self.key_to_action(event.key)
                        if action is not None:
                            ai_action = self.get_opponent_move()
                            self.resolve_step(action, ai_action)

            self.draw()
            self.clock.tick(FPS)

        if self.game_over and not getattr(self, "analyzed", False):
            self._finish_round()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Human vs MinimaxOpponent")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--minimax-depth", type=int, default=None)
    args = p.parse_args()

    depth = args.minimax_depth or PLAY_DEPTHS[args.difficulty]
    print("Starting Human vs Minimax Duel...")
    print(f"Minimax depth: {depth} (depth 10+ can take many seconds per move)")
    print("Both players choose from the same board; one key press resolves both moves.")
    print("Press SPACE to restart after game over.")
    print(f"Full log: {LOG_PATH}")
    print("After each round, analysis prints here automatically.")

    game = HumanDuel(difficulty=args.difficulty, minimax_depth=args.minimax_depth)
    game.run()
