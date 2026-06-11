import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tron_paper  # noqa: F401

from tron_paper.training.gui_rollout import watch_tronbot_stationary


def main():
    p = argparse.ArgumentParser(description="Watch TronBot in stationary env (pygame)")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--mode", choices=["spacefill", "greedy"], default="spacefill")
    p.add_argument("--backend", choices=["py", "cpp"], default="py", help="py=Python TronBotEngine, cpp=C++ MyTronBot.exe")
    p.add_argument("--compare", action="store_true", help="Same map: greedy first, then spacefill depth=1")
    args = p.parse_args()
    watch_tronbot_stationary(args.delay_ms, args.seed, args.max_depth, args.mode, args.backend, args.compare)


if __name__ == "__main__":
    main()
