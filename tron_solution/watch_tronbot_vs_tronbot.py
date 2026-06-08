import os
import importlib.util
import argparse
import time
import json

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

from tron_solution.env.tron_env import TronEnv
from tron_solution.env.tronbot_player import TronBotPlayer, default_tronbot_path

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_tronbot_vs_tronbot.log")
JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tronbot_vs_tronbot.jsonl")
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


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


def _outcome(info):
    if info.get("clean_kill"):
        return "P0 (blue) wins", "p0"
    if info.get("opponent_self_destruct"):
        return "P0 (blue) wins", "p0"
    if info.get("my_collision_type") and not info.get("opponent_collision_type"):
        return "P1 (red) wins", "p1"
    if info.get("mutual_destruction"):
        return "Draw", "draw"
    if info.get("timeout"):
        return "Draw (timeout)", "draw"
    return "Draw", "draw"


def run_tronbot_vs_tronbot(
    tronbot_path=None,
    episodes=5,
    delay_ms=80,
    headless=False,
    move_timeout=5.0,
    jsonl_path=None,
):
    render = not headless
    jsonl_path = jsonl_path or JSONL_PATH
    bot_path = tronbot_path or default_tronbot_path()
    bot0 = TronBotPlayer(bot_path, as_player=0, move_timeout=move_timeout)
    bot1 = TronBotPlayer(bot_path, as_player=1, move_timeout=move_timeout)
    bot0.start()
    bot1.start()

    env = TronEnv(
        grid_size=32, max_steps=500,
        render_mode="human" if render else None,
        opponent_type="random",
        render_my_label="BOT-0",
        render_opp_label="BOT-1",
    )

    print(f"TronBot: {bot_path}")
    print(f"Episodes: {episodes}")
    print(f"Log: {LOG_PATH}")
    print(f"JSONL: {jsonl_path}")
    if render:
        print("Press SPACE after each duel. ESC to quit.\n")
    else:
        print("Running headless...\n")

    p0_w = p1_w = dr = 0
    try:
        with open(LOG_PATH, "w") as lf:
            lf.write(f"tronbot={bot_path}\nepisodes={episodes}\njsonl={jsonl_path}\n")
        with open(jsonl_path, "w") as jf:
            for ep in range(episodes):
                obs, _ = env.reset(seed=ep)
                done = False
                steps = 0
                while not done:
                    if render and _poll_quit():
                        return
                    a0 = bot0.get_action(env)
                    a1 = bot1.get_action(env)
                    jf.write(json.dumps({
                        "type": "move",
                        "duel": ep + 1,
                        "seed": ep,
                        "step": steps,
                        "p0_head": list(env.my_head),
                        "p1_head": list(env.opponent_head),
                        "p0_dir": env.current_direction,
                        "p1_dir": env.opponent_direction,
                        "p0_action": a0,
                        "p1_action": a1,
                        "p0_action_name": DIR_NAMES[a0],
                        "p1_action_name": DIR_NAMES[a1],
                        "same_action": a0 == a1,
                    }) + "\n")
                    _, _, terminated, truncated, info = env.step_dual(a0, a1)
                    steps += 1
                    if render:
                        env.render()
                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000.0)
                    done = terminated or truncated

                outcome, kind = _outcome(info)
                p0_death = info.get("my_collision_type") or ""
                p1_death = info.get("opponent_collision_type") or ""
                end_rec = {
                    "type": "end",
                    "duel": ep + 1,
                    "seed": ep,
                    "steps": steps,
                    "winner": kind,
                    "outcome": outcome,
                    "p0_death": p0_death,
                    "p1_death": p1_death,
                    "timeout": info.get("timeout", False),
                }
                jf.write(json.dumps(end_rec) + "\n")
                line = f"duel={ep+1} {outcome} steps={steps} p0_death={p0_death} p1_death={p1_death}"
                print(line)
                with open(LOG_PATH, "a") as lf:
                    lf.write(line + "\n")
                if kind == "p0":
                    p0_w += 1
                elif kind == "p1":
                    p1_w += 1
                else:
                    dr += 1

                if render:
                    hint = outcome if ep + 1 < episodes else f"{outcome}  (done)"
                    if not _wait_for_space(env.render, env, hint):
                        return
    finally:
        bot0.close()
        bot1.close()
        env.close()

    print(f"\nSummary: P0 {p0_w}W  P1 {p1_w}W  {dr}D / {episodes}")
    return jsonl_path


def main():
    p = argparse.ArgumentParser(description="Watch TronBot vs TronBot")
    p.add_argument("--tronbot-path", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--move-timeout", type=float, default=5.0)
    p.add_argument("--jsonl", type=str, default=None)
    args = p.parse_args()
    run_tronbot_vs_tronbot(
        args.tronbot_path, args.episodes, args.delay_ms, args.headless, args.move_timeout,
        args.jsonl,
    )


if __name__ == "__main__":
    main()
