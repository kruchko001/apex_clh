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
from tron_solution.env.opponents import MinimaxOpponent, PLAY_MINIMAX_DEPTHS
from tron_solution.env.tronbot_player import TronBotPlayer, default_tronbot_path
from duel_utils import opening_cross_tronenv

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_tronbot_duel.log")
JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tronbot_duel.jsonl")
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


def _outcome(tronbot_side, info):
    tb_wins = (
        (tronbot_side == "my" and (info.get("clean_kill") or info.get("opponent_self_destruct")))
        or (tronbot_side == "opp" and info.get("my_collision_type") and not info.get("opponent_collision_type"))
    )
    tb_loses = (
        (tronbot_side == "my" and info.get("my_collision_type") and not info.get("opponent_collision_type"))
        or (tronbot_side == "opp" and (info.get("clean_kill") or info.get("opponent_self_destruct")))
    )
    if tb_wins:
        return "TRONBOT wins", "tronbot"
    if tb_loses:
        return "MINIMAX wins", "minimax"
    return "Draw", "draw"


def _minimax_metrics(minimax, env):
    grid = env._get_grid()
    mm_head, tb_head = env.opponent_head, env.my_head
    mm_space = minimax._count_space(mm_head, grid)
    tb_space = minimax._count_space(tb_head, grid)
    vor = minimax._voronoi(mm_head, tb_head, grid)
    partitioned = not minimax._same_component(mm_head, tb_head, grid)
    return mm_space, tb_space, vor, partitioned


def _log_move(jf, duel, seed, step, env, minimax, mm_act, tb_act, tronbot_side):
    mm_space, tb_space, vor, partitioned = _minimax_metrics(minimax, env)
    ranked = minimax.rank_actions(
        env.my_head if tronbot_side == "my" else env.opponent_head,
        env.opponent_head if tronbot_side == "my" else env.my_head,
        env._get_grid(),
        env.opponent_direction if tronbot_side == "my" else env.current_direction,
        env.current_direction if tronbot_side == "my" else env.opponent_direction,
    )
    mm_chose_top = bool(ranked and ranked[0][1] == mm_act)
    tb_head = env.my_head if tronbot_side == "my" else env.opponent_head
    mm_head = env.opponent_head if tronbot_side == "my" else env.my_head
    tb_dir = env.current_direction if tronbot_side == "my" else env.opponent_direction
    mm_dir = env.opponent_direction if tronbot_side == "my" else env.current_direction
    rec = {
        "type": "move",
        "duel": duel,
        "seed": seed,
        "step": step,
        "tronbot_head": list(tb_head),
        "minimax_head": list(mm_head),
        "tronbot_dir": tb_dir,
        "minimax_dir": mm_dir,
        "tronbot_action": tb_act,
        "minimax_action": mm_act,
        "mm_space": mm_space,
        "tb_space": tb_space,
        "mm_voronoi": vor,
        "partitioned": partitioned,
        "dist": abs(tb_head[0] - mm_head[0]) + abs(tb_head[1] - mm_head[1]),
        "mm_ranked": [[v, a] for v, a in ranked[:4]],
        "mm_chose_top": mm_chose_top,
    }
    jf.write(json.dumps(rec) + "\n")


def run_tronbot_duel(
    tronbot_path=None,
    minimax_depth=14,
    episodes=3,
    delay_ms=80,
    headless=False,
    tronbot_side="my",
    move_timeout=5.0,
    jsonl_path=None,
):
    import warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    render = not headless
    jsonl_path = jsonl_path or JSONL_PATH
    tronbot = TronBotPlayer(
        tronbot_path or default_tronbot_path(),
        as_player=0 if tronbot_side == "my" else 1,
        move_timeout=move_timeout,
    )
    minimax = MinimaxOpponent(depth=minimax_depth)

    if tronbot_side == "my":
        my_label, opp_label = "TRONBOT", "MINIMAX"
    else:
        my_label, opp_label = "MINIMAX", "TRONBOT"

    env = TronEnv(
        grid_size=32, max_steps=500,
        render_mode="human" if render else None,
        opponent_type="random",
        render_my_label=my_label,
        render_opp_label=opp_label,
    )

    with open(LOG_PATH, "w") as f:
        f.write("=== watch_tronbot_duel ===\n")
        f.write(f"tronbot={tronbot.bot_path}\n")
        f.write(f"minimax_depth={minimax_depth}\n")
        f.write(f"tronbot_side={tronbot_side}\n")
        f.write(f"jsonl={jsonl_path}\n")

    print(f"TronBot: {tronbot.bot_path}")
    print(f"Minimax depth: {minimax_depth}")
    print(f"TronBot plays: {'MY (left)' if tronbot_side == 'my' else 'OPP (right)'}")
    print(f"Log: {LOG_PATH}")
    print(f"JSONL: {jsonl_path}")
    if render:
        print("Press SPACE after each duel. ESC to quit.\n")
    else:
        print("Running headless...\n")

    tb_w = tb_l = dr = 0
    try:
        with open(jsonl_path, "w") as jf:
            for ep in range(episodes):
                tronbot.start()
                obs, _ = env.reset(seed=ep)
                steps = 1 if opening_cross_tronenv(env) else 0
                done = False

                while not done:
                    if render and _poll_quit():
                        return
                    tb_act = tronbot.get_action(env)
                    mm_act = minimax.get_action(
                        obs=env._get_grid_obs(),
                        my_head=env.my_head if tronbot_side == "my" else env.opponent_head,
                        opp_head=env.opponent_head if tronbot_side == "my" else env.my_head,
                        grid=env._get_grid(),
                        current_dir=env.opponent_direction if tronbot_side == "my" else env.current_direction,
                        my_dir=env.current_direction if tronbot_side == "my" else env.opponent_direction,
                    )
                    if tronbot_side == "my":
                        _, _, terminated, truncated, info = env.step_dual(tb_act, mm_act)
                    else:
                        _, _, terminated, truncated, info = env.step_dual(mm_act, tb_act)
                    _log_move(jf, ep + 1, ep, steps, env, minimax, mm_act, tb_act, tronbot_side)
                    steps += 1
                    if render:
                        env.render()
                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000.0)
                    done = terminated or truncated

                outcome, kind = _outcome(tronbot_side, info)
                tb_death = info.get("my_collision_type") if tronbot_side == "my" else info.get("opponent_collision_type", "")
                mm_death = info.get("opponent_collision_type") if tronbot_side == "my" else info.get("my_collision_type", "")
                end_rec = {
                    "type": "end",
                    "duel": ep + 1,
                    "seed": ep,
                    "steps": steps,
                    "winner": kind,
                    "outcome": outcome,
                    "tronbot_death": tb_death,
                    "minimax_death": mm_death,
                    "timeout": info.get("timeout", False),
                }
                jf.write(json.dumps(end_rec) + "\n")
                line = (
                    f"duel={ep+1} result={outcome} steps={steps} "
                    f"tronbot_hit={tb_death} minimax_hit={mm_death}"
                )
                print(line)
                with open(LOG_PATH, "a") as f:
                    f.write(line + "\n")
                if kind == "tronbot":
                    tb_w += 1
                elif kind == "minimax":
                    tb_l += 1
                else:
                    dr += 1

                if render:
                    hint = outcome if ep + 1 < episodes else f"{outcome}  (done)"
                    if not _wait_for_space(env.render, env, hint):
                        return
    finally:
        tronbot.close()
        env.close()

    print(f"\nSummary: TRONBOT {tb_w}W {tb_l}L {dr}D / {episodes}")
    return jsonl_path


def main():
    p = argparse.ArgumentParser(description="Watch TronBot (SOTA) vs MinimaxOpponent")
    p.add_argument("--tronbot-path", type=str, default=None)
    p.add_argument("--minimax-depth", type=int, default=PLAY_MINIMAX_DEPTHS["hard"])
    p.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--tronbot-side", choices=["my", "opp"], default="my")
    p.add_argument("--move-timeout", type=float, default=5.0)
    p.add_argument("--jsonl", type=str, default=None)
    args = p.parse_args()
    run_tronbot_duel(
        args.tronbot_path, args.minimax_depth, args.episodes,
        args.delay_ms, args.headless, args.tronbot_side, args.move_timeout,
        args.jsonl,
    )


if __name__ == "__main__":
    main()
