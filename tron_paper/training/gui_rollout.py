import glob
import os
import random
import threading
import time

import pygame
import torch

from competition.tron.tron import PLAYER_TRAIL_START, WALL
from tron_paper.env.stationary_env import StationaryTronEnv
from tron_paper.model.mrl_net import MRLActorCritic
from tron_paper_BH.tronbot_teacher import TronBotStationaryTeacher

CELL = 16
DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
COLORS = {
    "bg": (12, 12, 18),
    "wall": (90, 90, 100),
    "trail": (0, 180, 255),
    "head": (255, 220, 60),
    "head_greedy": (255, 220, 60),
    "head_spacefill": (255, 120, 60),
    "head_bc": (100, 255, 140),
    "text": (230, 230, 230),
}


def snapshot_stationary_policy(model) -> dict:
    ac = MRLActorCritic(stationary=True)
    ac.trunk.load_state_dict({k: v.cpu().clone() for k, v in model.policy.features_extractor.ac.trunk.state_dict().items()})
    pi = model.policy.mlp_extractor.policy_net
    ac.pi1.load_state_dict({k: v.cpu().clone() for k, v in pi[0].state_dict().items()})
    ac.pi2.weight.data = model.policy.action_net.weight.data.cpu().clone()
    ac.pi2.bias.data = model.policy.action_net.bias.data.cpu().clone()
    vf = model.policy.mlp_extractor.value_net
    ac.v1.load_state_dict({k: v.cpu().clone() for k, v in vf[0].state_dict().items()})
    ac.v2.load_state_dict({k: v.cpu().clone() for k, v in vf[2].state_dict().items()})
    ac.v3.weight.data = model.policy.value_net.weight.data.cpu().clone()
    ac.v3.bias.data = model.policy.value_net.bias.data.cpu().clone()
    ac.eval()
    return ac.state_dict()


def resolve_stationary_model(path=None, save_dir="./tron_paper_checkpoints"):
    from tron_paper.training.train_stationary import _sb3_to_ac, load_stationary_ac

    if path:
        if path.endswith(".zip"):
            import tron_paper  # noqa: F401
            from stable_baselines3 import PPO
            return _sb3_to_ac(PPO.load(path), stationary=True)
        return load_stationary_ac(path)
    pt = os.path.join(save_dir, "stationary_agent.pt")
    if os.path.isfile(pt):
        return load_stationary_ac(pt)
    zips = sorted(
        (
            p
            for p in glob.glob(os.path.join(save_dir, "stationary_*.zip"))
            if os.path.getsize(p) > 1000
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    for zp in zips:
        try:
            return resolve_stationary_model(zp, save_dir)
        except (ValueError, OSError, RuntimeError):
            continue
    raise FileNotFoundError(f"No stationary model in {save_dir}")


def _draw_stationary_frame(screen, font, env, lines, head_color=None):
    h, w = env.height, env.width
    screen.fill(COLORS["bg"])
    for y in range(h):
        for x in range(w):
            v = int(env.grid[y, x])
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            if v == WALL:
                pygame.draw.rect(screen, COLORS["wall"], rect)
            elif v >= PLAYER_TRAIL_START:
                pygame.draw.rect(screen, COLORS["trail"], rect.inflate(-2, -2))
    hy, hx = env.player.y, env.player.x
    pygame.draw.rect(
        screen,
        head_color or COLORS["head"],
        pygame.Rect(hx * CELL + 2, hy * CELL + 2, CELL - 4, CELL - 4),
    )
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, COLORS["text"]), (8, h * CELL + 8 + i * 22))


def _poll_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    return True


def _play_tronbot_episode(screen, font, clock, env, teacher, map_seed, phase_label, delay_ms, head_color, extra_lines=None):
    env.reset(seed=map_seed)
    teacher.reset_episode(env.grid, env.player.y, env.player.x)
    done = False
    ep_reward = 0.0
    last_r = 0.0
    last_action = None
    last_tb = None
    msg = "playing"
    last_tick = 0
    done_at = None
    extra_lines = extra_lines or []

    while True:
        if not _poll_quit():
            return False

        if not done and pygame.time.get_ticks() - last_tick >= delay_ms:
            mask = env.action_masks()
            valid = [i for i, ok in enumerate(mask) if ok]
            if not valid:
                done = True
                msg = "no valid actions"
                done_at = pygame.time.get_ticks()
            else:
                action = teacher.action(
                    env.grid, env.player.y, env.player.x, mask, int(env.player.direction)
                )
                _, last_r, term, trunc, info = env.step(action)
                ep_reward += last_r
                last_action = action
                last_tb = info.get("tb_score", info.get("tb_terminal_score"))
                done = term or trunc
                msg = "crash" if term else ("timeout" if trunc else "ok")
                if done:
                    done_at = pygame.time.get_ticks()
            last_tick = pygame.time.get_ticks()

        if done and done_at and pygame.time.get_ticks() - done_at > 1200:
            return True

        act = DIRS[last_action] if last_action is not None else "-"
        tb = f"{last_tb:.1f}" if last_tb is not None else "-"
        lines = [
            f"{phase_label}  steps={env.step_count}  action={act}  tb={tb}",
            f"step_r={last_r:+.2f}  ep_r={ep_reward:.2f}  {msg}  (q/esc quit)",
        ] + extra_lines
        _draw_stationary_frame(screen, font, env, lines, head_color=head_color)
        pygame.display.flip()
        clock.tick(60)


def _resolve_bc_model(model_path=None, save_dir="./tron_paper_BH_checkpoints"):
    from tron_paper.training.train_stationary import load_stationary_ac

    if model_path:
        return load_stationary_ac(model_path)
    pt = os.path.join(save_dir, "stationary_agent.pt")
    if os.path.isfile(pt):
        return load_stationary_ac(pt)
    raise FileNotFoundError(f"No BC model at {pt} — run: python main.py paper-bc train")


def _play_bc_episode(screen, font, clock, env, ac, map_seed, phase_label, delay_ms, head_color, extra_lines=None):
    obs, _ = env.reset(seed=map_seed)
    done = False
    ep_reward = 0.0
    last_r = 0.0
    last_action = None
    last_tb = None
    msg = "playing"
    last_tick = 0
    done_at = None
    extra_lines = extra_lines or []

    while True:
        if not _poll_quit():
            return False

        if not done and pygame.time.get_ticks() - last_tick >= delay_ms:
            mask = env.action_masks()
            valid = [i for i, ok in enumerate(mask) if ok]
            if not valid:
                done = True
                msg = "no valid actions"
                done_at = pygame.time.get_ticks()
            else:
                action = ac.act_greedy(obs)
                if not mask[action]:
                    action = valid[0]
                obs, last_r, term, trunc, info = env.step(action)
                ep_reward += last_r
                last_action = action
                last_tb = info.get("tb_score", info.get("tb_terminal_score"))
                done = term or trunc
                msg = "crash" if term else ("timeout" if trunc else "ok")
                if done:
                    done_at = pygame.time.get_ticks()
            last_tick = pygame.time.get_ticks()

        if done and done_at and pygame.time.get_ticks() - done_at > 1200:
            return True

        act = DIRS[last_action] if last_action is not None else "-"
        tb = f"{last_tb:.1f}" if last_tb is not None else "-"
        lines = [
            f"{phase_label}  steps={env.step_count}  action={act}  tb={tb}",
            f"step_r={last_r:+.2f}  ep_r={ep_reward:.2f}  {msg}  (q/esc quit)",
        ] + extra_lines
        _draw_stationary_frame(screen, font, env, lines, head_color=head_color)
        pygame.display.flip()
        clock.tick(60)


def _pause_banner(screen, font, env, lines, ms=1500):
    deadline = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < deadline:
        if not _poll_quit():
            return False
        _draw_stationary_frame(screen, font, env, lines)
        pygame.display.flip()
        pygame.time.wait(16)
    return True


def watch_bc_vs_greedy_stationary(delay_ms=80, seed=None, model_path=None, save_dir="./tron_paper_BH_checkpoints"):
    ac = _resolve_bc_model(model_path, save_dir)
    greedy = TronBotStationaryTeacher(backend="py", mode="greedy", max_depth=1)

    pygame.init()
    env = StationaryTronEnv()
    w, h = env.width, env.height
    screen = pygame.display.set_mode((w * CELL, h * CELL + 132))
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    pygame.display.set_caption("Stationary duel — BC model vs greedy TronBot (same map)")

    round_num = 0
    while True:
        round_num += 1
        map_seed = seed if seed is not None else random.randint(0, 999999)
        meta = [f"map seed={map_seed}  round={round_num}"]

        if not _play_bc_episode(
            screen, font, clock, env, ac, map_seed,
            "BC clone", delay_ms, COLORS["head_bc"], meta,
        ):
            break

        if not _pause_banner(
            screen, font, env,
            meta + ["BC finished — same map, greedy TronBot (py) next..."],
            ms=1800,
        ):
            break

        if not _play_tronbot_episode(
            screen, font, clock, env, greedy, map_seed,
            "greedy TronBot", delay_ms, COLORS["head_greedy"], meta,
        ):
            break

        if not _pause_banner(
            screen, font, env,
            meta + ["greedy finished — next round..."],
            ms=1200,
        ):
            break

    pygame.quit()


def watch_tronbot_stationary_compare(delay_ms=80, seed=None, spacefill_depth=1, backend="py"):
    greedy = TronBotStationaryTeacher(backend=backend, mode="greedy", max_depth=1)
    spacefill = TronBotStationaryTeacher(backend=backend, mode="spacefill", max_depth=spacefill_depth)

    pygame.init()
    env = StationaryTronEnv()
    w, h = env.width, env.height
    screen = pygame.display.set_mode((w * CELL, h * CELL + 132))
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    pygame.display.set_caption("TronBot stationary — greedy then spacefill (same map)")

    round_num = 0
    while True:
        round_num += 1
        map_seed = seed if seed is not None else random.randint(0, 999999)
        meta = [f"map seed={map_seed}  round={round_num}"]

        if not _play_tronbot_episode(
            screen, font, clock, env, greedy, map_seed,
            "greedy", delay_ms, COLORS["head_greedy"], meta,
        ):
            break

        if not _pause_banner(
            screen, font, env,
            meta + ["greedy finished — same map, spacefill (depth=1) next..."],
            ms=1800,
        ):
            break

        if not _play_tronbot_episode(
            screen, font, clock, env, spacefill, map_seed,
            "spacefill d=1", delay_ms, COLORS["head_spacefill"], meta,
        ):
            break

        if not _pause_banner(
            screen, font, env,
            meta + ["spacefill finished — next round..."],
            ms=1200,
        ):
            break

    pygame.quit()


def watch_tronbot_stationary(delay_ms=80, seed=None, max_depth=1, mode="spacefill", backend="py", compare=False):
    if compare:
        watch_tronbot_stationary_compare(delay_ms, seed, max_depth, backend)
        return

    teacher = TronBotStationaryTeacher(backend=backend, max_depth=max_depth, mode=mode)

    pygame.init()
    env = StationaryTronEnv()
    w, h = env.width, env.height
    screen = pygame.display.set_mode((w * CELL, h * CELL + 110))
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    pygame.display.set_caption(f"TronBot stationary ({backend}, {mode}, depth={max_depth})")

    ep = 0
    while True:
        ep += 1
        map_seed = seed if seed is not None else random.randint(0, 999999)
        head = COLORS["head_greedy"] if mode == "greedy" else COLORS["head_spacefill"]
        if not _play_tronbot_episode(
            screen, font, clock, env, teacher, map_seed,
            f"TronBot ep {ep} ({mode})", delay_ms, head,
        ):
            break

    pygame.quit()


def watch_stationary(model_path=None, save_dir="./tron_paper_checkpoints", delay_ms=80, seed=None):
    ac = resolve_stationary_model(model_path, save_dir)
    ac.eval()

    pygame.init()
    env = StationaryTronEnv()
    w, h = env.width, env.height
    screen = pygame.display.set_mode((w * CELL, h * CELL + 88))
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    pygame.display.set_caption("tron_paper stationary")

    ep = 0
    running = True
    while running:
        ep += 1
        obs, _ = env.reset(seed=seed if seed is not None else random.randint(0, 999999))
        done = False
        ep_reward = 0.0
        last_r = 0.0
        last_action = None
        msg = "playing"
        last_tick = 0
        done_at = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break

            if not running:
                break

            if not done and pygame.time.get_ticks() - last_tick >= delay_ms:
                mask = env.action_masks()
                valid = [i for i, ok in enumerate(mask) if ok]
                if not valid:
                    done = True
                    msg = "no valid actions"
                    done_at = pygame.time.get_ticks()
                else:
                    action = ac.act_greedy(obs)
                    if not mask[action]:
                        action = random.choice(valid)
                    obs, last_r, term, trunc, _ = env.step(action)
                    ep_reward += last_r
                    last_action = action
                    done = term or trunc
                    msg = "crash" if term else ("timeout" if trunc else "ok")
                    if done:
                        done_at = pygame.time.get_ticks()
                last_tick = pygame.time.get_ticks()

            if done and done_at and pygame.time.get_ticks() - done_at > 800:
                break

            act = DIRS[last_action] if last_action is not None else "-"
            _draw_stationary_frame(
                screen,
                font,
                env,
                [
                    f"episode {ep}  steps={env.step_count}  action={act}",
                    f"step_r={last_r:+.2f}  ep_r={ep_reward:.2f}  {msg}  (q/esc quit)",
                ],
            )
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()


class StationaryGUIWorker:
    def __init__(self, delay_ms: int = 50):
        self.delay_ms = delay_ms
        self._lock = threading.Lock()
        self._latest = 0
        self._state = None
        self._last_played = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def push(self, model, update_id: int):
        state = snapshot_stationary_policy(model)
        with self._lock:
            self._latest = update_id
            self._state = state

    def pending(self) -> int:
        with self._lock:
            return self._latest - self._last_played

    def _loop(self):
        pygame.init()
        screen = None
        font = None
        clock = pygame.time.Clock()
        while self._running:
            with self._lock:
                update = self._latest
                state = self._state
            if update > self._last_played and state is not None:
                self._last_played = update
                screen, font = self._play_episode(state, update, screen, font, clock)
            else:
                time.sleep(0.02)
        pygame.quit()

    def _play_episode(self, state_dict, update_num, screen, font, clock):
        ac = MRLActorCritic(stationary=True)
        ac.load_state_dict(state_dict)
        ac.eval()

        env = StationaryTronEnv()
        obs, _ = env.reset(seed=random.randint(0, 999999))
        w, h = env.width, env.height
        if screen is None:
            screen = pygame.display.set_mode((w * CELL, h * CELL + 88))
            font = pygame.font.SysFont("consolas", 15)

        pygame.display.set_caption(f"tron_paper stationary — update #{update_num} (latest)")
        done = False
        ep_reward = 0.0
        last_r = 0.0
        last_action = None
        msg = "rollout"
        last_tick = 0
        done_at = None

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return screen, font

            if not done and pygame.time.get_ticks() - last_tick >= self.delay_ms:
                mask = env.action_masks()
                valid = [i for i, ok in enumerate(mask) if ok]
                if not valid:
                    done = True
                    msg = "no valid actions"
                    done_at = pygame.time.get_ticks()
                else:
                    action = ac.act_greedy(obs)
                    if not mask[action]:
                        action = random.choice(valid)
                    obs, last_r, term, trunc, _ = env.step(action)
                    ep_reward += last_r
                    last_action = action
                    done = term or trunc
                    msg = "crash" if term else ("timeout" if trunc else "ok")
                    if done:
                        done_at = pygame.time.get_ticks()
                last_tick = pygame.time.get_ticks()

            if done and done_at and pygame.time.get_ticks() - done_at > 600:
                return screen, font

            screen.fill(COLORS["bg"])
            for y in range(h):
                for x in range(w):
                    v = int(env.grid[y, x])
                    rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
                    if v == WALL:
                        pygame.draw.rect(screen, COLORS["wall"], rect)
                    elif v >= PLAYER_TRAIL_START:
                        pygame.draw.rect(screen, COLORS["trail"], rect.inflate(-2, -2))
            hy, hx = env.player.y, env.player.x
            pygame.draw.rect(screen, COLORS["head"], pygame.Rect(hx * CELL + 2, hy * CELL + 2, CELL - 4, CELL - 4))
            pending = self.pending()
            act = DIRS[last_action] if last_action is not None else "-"
            lines = [
                f"update #{update_num}  pending={pending}  steps={env.step_count}  action={act}",
                f"step_r={last_r:+.2f}  ep_r={ep_reward:.2f}  {msg}",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, COLORS["text"]), (8, h * CELL + 8 + i * 22))
            pygame.display.flip()
            clock.tick(60)

        return screen, font
