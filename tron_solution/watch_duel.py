import os
import glob
import importlib.util
import argparse
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

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.opponents import PLAY_MINIMAX_DEPTHS
from tron_solution.env.frame_stack_wrapper import GridFrameStackWrapper
from tron_solution.model.obs import N_STACK, to_sandbox_obs_np, valid_mask_from_actions
from tron_solution.model.frame_stack import FrameStack

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_duel.log")
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def _unwrap_tron_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _get_tron_from_venv(venv):
    return _unwrap_tron_env(venv.envs[0])


def _outcome_text(info):
    if info.get("official_score", 0) >= 1.0 or info.get("clean_kill"):
        return "MODEL wins", "win"
    if info.get("official_score", 0) >= 0.80 or info.get("opponent_self_destruct"):
        return "MODEL wins", "win"
    if info.get("timeout") or info.get("truncated"):
        return "Draw (timeout)", "draw"
    if info.get("mutual_destruction"):
        return "Draw", "draw"
    if info.get("my_collision_type") and not info.get("opponent_collision_type"):
        return "AI wins", "loss"
    return "Draw", "draw"


def _poll_quit():
    import pygame
    if not pygame.display.get_init():
        return False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _wait_for_space(render_fn, tron_env, message):
    import pygame
    tron_env.render_message = message
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    tron_env.render_message = None
                    return True
        render_fn()
        tron_env.clock.tick(30)


class DuelLogger:
    def __init__(self, model_path, minimax_depth):
        self.model_path = model_path
        self.minimax_depth = minimax_depth
        self.rounds = []
        self.current = None

    def start(self):
        with open(LOG_PATH, "w") as f:
            f.write("=== watch_duel session ===\n")
            f.write(f"model={self.model_path}\n")
            f.write(f"minimax_depth={self.minimax_depth}\n")

    def _write(self, msg, console=False):
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    def start_duel(self, duel_id):
        self.current = {"id": duel_id, "steps": []}
        self._write(f"\n--- duel={duel_id} start ---")

    def log_step(self, tron, model_action, reward, info, done):
        opp = tron.opponent
        step_n = info.get("log_step", tron.step_count) if done else tron.step_count
        grid = tron._get_grid()
        ranked = []
        if hasattr(opp, "rank_actions"):
            ranked = opp.rank_actions(
                tron.my_head, tron.opponent_head, grid,
                tron.opponent_direction, tron.current_direction,
            )
        best_val, best_act = ranked[0] if ranked else (0, tron.opponent_direction)
        chosen_val = next((v for v, a in ranked if a == tron.opponent_direction), None)
        cv = f"{chosen_val:.1f}" if chosen_val is not None else "?"
        ms = info.get("my_space", 0)
        os_ = info.get("opp_space", 0)
        vor = info.get("voronoi", 0)
        line = (
            f"step={step_n} | MODEL={DIR_NAMES[int(model_action)]} "
            f"@{tron.my_head} dir={DIR_NAMES[tron.current_direction]} | "
            f"AI={DIR_NAMES[tron.opponent_direction]} score={cv} "
            f"best={DIR_NAMES[best_act]}({best_val:.1f}) @{tron.opponent_head} | "
            f"space MODEL={ms:.0f} AI={os_:.0f} vor={vor:.0f} | r={reward:.3f}"
        )
        if done:
            line += (
                f" | END model={info.get('my_collision_type','')} "
                f"ai={info.get('opponent_collision_type','')}"
            )
        self._write(line)
        self.current["steps"].append({
            "step": step_n,
            "model_action": int(model_action),
            "ai_action": tron.opponent_direction,
            "ranked": ranked,
            "my_space": ms,
            "opp_space": os_,
            "voronoi": vor,
            "my_head": tron.my_head,
            "opp_head": tron.opponent_head,
            "done": done,
            "info": dict(info),
        })

    def end_duel(self, outcome, kind, steps, total_r, info):
        self._write(
            f"duel={self.current['id']} result={outcome} steps={steps} "
            f"reward={total_r:.2f} model_hit={info.get('my_collision_type','')} "
            f"ai_hit={info.get('opponent_collision_type','')}",
            console=True,
        )
        self.current["outcome"] = kind
        self.current["steps_count"] = steps
        self.rounds.append(self.current)
        self.current = None

    def analyze(self):
        if not self.rounds:
            return
        wins = sum(1 for r in self.rounds if r["outcome"] == "win")
        losses = sum(1 for r in self.rounds if r["outcome"] == "loss")
        draws = sum(1 for r in self.rounds if r["outcome"] == "draw")
        lines = [
            "",
            "=== SESSION ANALYSIS ===",
            f"Record: MODEL {wins}W {losses}L {draws}D / {len(self.rounds)}",
        ]

        end_types = {}
        head_on_steps = []
        territory_flips = []
        ai_mistakes = []

        for rnd in self.rounds:
            info = rnd["steps"][-1]["info"] if rnd["steps"] else {}
            key = f"{info.get('my_collision_type','')} / {info.get('opponent_collision_type','')}"
            end_types[key] = end_types.get(key, 0) + 1
            if info.get("my_collision_type") == "head_on":
                head_on_steps.append(rnd["steps_count"])

            losing_from = None
            for s in rnd["steps"]:
                if s["opp_space"] < s["my_space"] and losing_from is None:
                    losing_from = s["step"]
                ranked = s.get("ranked") or []
                if not ranked:
                    continue
                bv, ba = ranked[0]
                cv = next((v for v, a in ranked if a == s["ai_action"]), bv)
                if s["ai_action"] != ba and bv - cv > 50:
                    ai_mistakes.append((rnd["id"], s["step"], s["ai_action"], cv, ba, bv, s))

            if losing_from is not None and rnd["outcome"] == "win":
                territory_flips.append((rnd["id"], losing_from))

        lines.append(f"End types: {end_types}")
        if head_on_steps:
            lines.append(
                f"Head-on draws: {len(head_on_steps)} duels, avg step {sum(head_on_steps)/len(head_on_steps):.0f}"
            )

        if territory_flips:
            lines.append("AI lost territory before MODEL won:")
            for did, st in territory_flips:
                lines.append(f"  duel {did}: AI behind in space from step {st}")

        if ai_mistakes:
            lines.append(f"AI suboptimal picks ({len(ai_mistakes)} total, top 5):")
            for did, st, ch, cv, ba, bv, s in sorted(ai_mistakes, key=lambda x: x[5] - x[3], reverse=True)[:5]:
                lines.append(
                    f"  duel {did} step {st}: played {DIR_NAMES[ch]} ({cv:.0f}) "
                    f"not {DIR_NAMES[ba]} ({bv:.0f}) | space MODEL={s['my_space']:.0f} AI={s['opp_space']:.0f}"
                )
        else:
            lines.append("No large minimax ranking errors; losses are heuristic/simultaneous-move gaps.")

        if wins >= losses:
            lines.append("Pattern: MODEL wins or draws — AI needs simultaneous-move awareness and head-on avoidance.")
            lines.append("Fix: simulate both players moving together; penalize head-on; stronger cutoff when MODEL leads space.")
        elif losses > wins:
            lines.append("Pattern: AI winning more — current depth/heuristics sufficient.")

        for line in lines:
            self._write(line, console=True)


def find_vec_normalize(model_path, save_dir="./ppo_tron_checkpoints"):
    name = os.path.basename(model_path)
    if "final_" in name:
        ts = name.replace("tron_ppo_final_", "").replace(".zip", "")
        p = os.path.join(save_dir, f"vec_normalize_final_{ts}.pkl")
        if os.path.isfile(p):
            return p
    files = glob.glob(os.path.join(save_dir, "vec_normalize_final_*.pkl"))
    return sorted(files)[-1] if files else None


def make_sb3_env(minimax_depth, render_mode):
    return Monitor(GridFrameStackWrapper(TronEnv(
        grid_size=32, max_steps=500, render_mode=render_mode,
        opponent_type="minimax", minimax_depth=minimax_depth,
        render_my_label="MODEL", render_opp_label="AI",
    ), n_stack=N_STACK))


def load_sb3_duel_env(model_path, vec_path, minimax_depth, render_mode="human"):
    venv = DummyVecEnv([lambda: make_sb3_env(minimax_depth, render_mode)])
    if vec_path and os.path.isfile(vec_path):
        venv = VecNormalize.load(vec_path, venv)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path)
    return model, venv


def _run_sb3_episodes(model, venv, tron, episodes, delay_ms, render, logger, wait_between):
    wins = losses = draws = 0
    for ep in range(episodes):
        logger.start_duel(ep + 1)
        obs = venv.reset()
        done = False
        steps = 0
        total_r = 0.0
        info = {}

        while not done:
            if render and _poll_quit():
                return wins, losses, draws, False
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            act = int(action[0])
            total_r += float(reward[0])
            steps += 1
            info = info[0] if isinstance(info, list) else info
            logger.log_step(tron, act, float(reward[0]), info, bool(done[0]))
            if render:
                venv.render()
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
            done = bool(done[0])

        outcome, kind = _outcome_text(info)
        if kind == "win":
            wins += 1
        elif kind == "loss":
            losses += 1
        else:
            draws += 1
        logger.end_duel(outcome, kind, steps, total_r, info)

        if render and wait_between:
            hint = outcome if ep + 1 < episodes else f"{outcome}  (done)"
            if not _wait_for_space(venv.render, tron, hint):
                return wins, losses, draws, False

    return wins, losses, draws, True


def run_sb3_duel(model_path, vec_path, minimax_depth, episodes, delay_ms, headless=False):
    import warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    render = not headless
    model, venv = load_sb3_duel_env(model_path, vec_path, minimax_depth, "human" if render else None)
    tron = _get_tron_from_venv(venv)
    logger = DuelLogger(model_path, minimax_depth)
    logger.start()

    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_path or '(none)'}")
    print(f"Opponent: minimax depth {minimax_depth}")
    print(f"Log: {LOG_PATH}")
    if render:
        print("Press SPACE after each duel for the next. ESC or close window to quit.\n")
    else:
        print("Running headless...\n")

    wins, losses, draws, ok = _run_sb3_episodes(
        model, venv, tron, episodes, delay_ms, render, logger, wait_between=render,
    )
    if ok:
        print(f"\nSummary: {wins}W {losses}L {draws}D / {episodes}")
        logger.analyze()
    venv.close()


def run_pt_duel(model_path, minimax_depth, episodes, delay_ms, headless=False):
    import warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    render = not headless
    model = torch.jit.load(model_path)
    model.eval()
    wrapped = False
    raw_core = False
    try:
        with torch.no_grad():
            model(torch.randn(1, 5, 32, 32))
        wrapped = True
    except Exception:
        pass
    if not wrapped:
        try:
            with torch.no_grad():
                model(torch.randn(1, 16, 30, 30))
            raw_core = True
        except Exception:
            pass
    env = TronEnv(
        grid_size=32, max_steps=500, render_mode="human" if render else None,
        opponent_type="minimax", minimax_depth=minimax_depth,
        render_my_label="MODEL", render_opp_label="AI",
    )
    logger = DuelLogger(model_path, minimax_depth)
    logger.start()
    wins = losses = draws = 0
    stack = FrameStack() if raw_core else None
    mode = "sandbox wrapper" if wrapped else ("16ch+stack" if raw_core else "unknown")
    print(f"Model: {model_path} ({mode})")
    print(f"Opponent: minimax depth {minimax_depth}")
    print(f"Log: {LOG_PATH}\n")
    for ep in range(episodes):
        logger.start_duel(ep + 1)
        obs, _ = env.reset(seed=ep)
        if raw_core:
            stacked = stack.reset(obs)
        done = False
        steps = 0
        total_r = 0.0
        info = {}

        while not done:
            if render and _poll_quit():
                env.close()
                return
            if raw_core:
                x = torch.from_numpy(stacked).float().unsqueeze(0)
            elif wrapped:
                x = torch.from_numpy(to_sandbox_obs_np(obs, env.walls)).float().unsqueeze(0)
            else:
                raise RuntimeError("Unsupported model format")
            with torch.no_grad():
                out = model(x)
                logits = out if out.dim() == 1 else out.squeeze(0)
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            if raw_core:
                stacked = stack.step(obs)
            total_r += reward
            steps += 1
            logger.log_step(env, action, reward, info, terminated or truncated)
            if render:
                env.render()
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
            done = terminated or truncated

        outcome, kind = _outcome_text(info)
        if kind == "win":
            wins += 1
        elif kind == "loss":
            losses += 1
        else:
            draws += 1
        logger.end_duel(outcome, kind, steps, total_r, info)

        if render:
            hint = outcome if ep + 1 < episodes else f"{outcome}  (done)"
            if not _wait_for_space(env.render, env, hint):
                env.close()
                return

    print(f"\nSummary: {wins}W {losses}L {draws}D / {episodes}")
    logger.analyze()
    env.close()


def main():
    p = argparse.ArgumentParser(description="Watch trained model vs MinimaxOpponent")
    p.add_argument("--model_path", type=str, default="./ppo_tron_checkpoints/best_model.zip")
    p.add_argument("--vec_normalize", type=str, default=None)
    p.add_argument("--minimax-depth", type=int, default=PLAY_MINIMAX_DEPTHS["hard"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--delay-ms", type=int, default=80, help="Ms between steps (0=fast)")
    p.add_argument("--headless", action="store_true", help="No window; log and analyze only")
    args = p.parse_args()

    if args.model_path.endswith(".pt"):
        run_pt_duel(args.model_path, args.minimax_depth, args.episodes, args.delay_ms, args.headless)
    else:
        vec = args.vec_normalize or find_vec_normalize(args.model_path)
        run_sb3_duel(args.model_path, vec, args.minimax_depth, args.episodes, args.delay_ms, args.headless)


if __name__ == "__main__":
    main()
