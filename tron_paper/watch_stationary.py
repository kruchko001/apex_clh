import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tron_paper  # noqa: F401

from tron_paper.training.gui_rollout import watch_stationary


def main():
    p = argparse.ArgumentParser(description="Watch tron_paper stationary agent (pygame)")
    p.add_argument("--model", type=str, default=None, help=".pt or .zip; default: latest in save-dir")
    p.add_argument("--save-dir", type=str, default="./tron_paper_checkpoints")
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    watch_stationary(args.model, args.save_dir, args.delay_ms, args.seed)


if __name__ == "__main__":
    main()
