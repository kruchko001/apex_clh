"""
Main CLI entry point for Tron RL Solution.

Usage:
    python main.py train --timesteps 100000
    python main.py export --model_path ./ppo_tron_checkpoints/tron_ppo_final_*.zip
    python main.py test --model_path tron_model.pt
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Tron RL Solution CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO agent")
    train_parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps (curriculum default: 500000)")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    train_parser.add_argument("--save-dir", type=str, default="./ppo_tron_checkpoints", help="Save directory")
    train_parser.add_argument("--opponent", type=str, default="minimax", help="Training opponent type")
    train_parser.add_argument("--minimax-depth", type=int, default=14, help="Minimax depth when opponent=minimax")
    train_parser.add_argument("--n-envs", type=int, default=None, help="Parallel envs (default 4 for depth>=12)")
    train_parser.add_argument("--resume", type=str, default=None, help="SB3 .zip to fine-tune from")
    train_parser.add_argument("--vec-normalize", type=str, default=None, help="VecNormalize .pkl for resume")
    train_parser.add_argument("--curriculum", action="store_true", help="Minimax depth curriculum 4->6->10->12->14")
    train_parser.add_argument("--curriculum-stages", type=str, default=None,
                              help='Custom "depth:steps,..." e.g. "6:100000,14:200000"')
    train_parser.add_argument("--curriculum-run-id", type=str, default=None,
                              help="Resume interrupted curriculum (e.g. 20260608_104639)")
    train_parser.add_argument("--stage-gate-win-rate", type=float, default=0.25,
                              help="Min win rate vs current depth to advance (default 0.25)")
    train_parser.add_argument("--stage-gate-episodes", type=int, default=20)
    train_parser.add_argument("--stage-gate-extra-steps", type=int, default=100000,
                              help="Extra timesteps on same stage if 25%% gate fails (default 100000, 0=stop)")
    train_parser.add_argument("--no-stage-gate", action="store_true")
    train_parser.add_argument("--warmup-timesteps", type=int, default=200000)
    train_parser.add_argument("--no-warmup", action="store_true")
    train_parser.add_argument("--no-early-stop", action="store_true")
    train_parser.add_argument("--early-stop-patience", type=int, default=6)
    train_parser.add_argument("--eval-freq", type=int, default=20000)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to TorchScript")
    export_parser.add_argument("--model_path", type=str, help="Path to trained SB3 model")
    export_parser.add_argument("--output", type=str, default="tron_model.pt", help="Output path")
    export_parser.add_argument("--train_and_export", action="store_true", help="Train then export")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test exported model")
    test_parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    test_parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    
    watch_parser = subparsers.add_parser("watch", help="Watch model vs MinimaxOpponent")
    watch_parser.add_argument("--model_path", type=str, default="./ppo_tron_checkpoints/best_model.zip")
    watch_parser.add_argument("--vec_normalize", type=str, default=None)
    watch_parser.add_argument("--minimax-depth", type=int, default=14)
    watch_parser.add_argument("--episodes", type=int, default=5)
    watch_parser.add_argument("--delay-ms", type=int, default=80)

    tronbot_parser = subparsers.add_parser("watch-tronbot", help="Watch TronBot vs MinimaxOpponent")
    tronbot_parser.add_argument("--tronbot-path", type=str, default=None)
    tronbot_parser.add_argument("--minimax-depth", type=int, default=14)
    tronbot_parser.add_argument("--episodes", type=int, default=5)
    tronbot_parser.add_argument("--delay-ms", type=int, default=80)
    tronbot_parser.add_argument("--headless", action="store_true")
    tronbot_parser.add_argument("--tronbot-side", choices=["my", "opp"], default="my")
    tronbot_parser.add_argument("--move-timeout", type=float, default=5.0)

    tbvstb_parser = subparsers.add_parser("watch-tronbot-vs-tronbot", help="Watch TronBot vs TronBot")
    tbvstb_parser.add_argument("--tronbot-path", type=str, default=None)
    tbvstb_parser.add_argument("--episodes", type=int, default=5)
    tbvstb_parser.add_argument("--delay-ms", type=int, default=80)
    tbvstb_parser.add_argument("--headless", action="store_true")
    tbvstb_parser.add_argument("--move-timeout", type=float, default=5.0)

    serve_parser = subparsers.add_parser("serve", help="Launch player HTTP API (official sandbox format)")
    serve_parser.add_argument("--model", type=str, required=True, help="TorchScript .pt")
    serve_parser.add_argument("--port", type=int, default=8001)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--server-stack", action="store_true",
                              help="Stack frames server-side for raw 20-channel core models")

    paper_parser = subparsers.add_parser("train-paper", help="Train MRL paper agent (stationary + non-stationary PPO)")
    paper_parser.add_argument("--stationary-steps", type=int, default=300000)
    paper_parser.add_argument("--non-stationary-steps", type=int, default=500000)
    paper_parser.add_argument("--save-dir", type=str, default="./tron_paper_checkpoints")
    paper_parser.add_argument("--n-envs", type=int, default=4)
    paper_parser.add_argument("--lr", type=float, default=3e-4)
    paper_parser.add_argument("--stage", choices=["all", "stationary", "non-stationary"], default="all")
    paper_parser.add_argument("--stationary-weights", type=str, default=None)

    paper_export = subparsers.add_parser("export-paper", help="Export MRL cursor model to TorchScript")
    paper_export.add_argument("--non-stationary", type=str, default="./tron_paper_checkpoints/non_stationary_agent.pt")
    paper_export.add_argument("--stationary", type=str, default="./tron_paper_checkpoints/stationary_agent.pt")
    paper_export.add_argument("--output", type=str, default="tron_model.pt")

    paper_test = subparsers.add_parser("test-paper", help="Test MRL .pt with official launcher encode/mask flow")
    paper_test.add_argument("--model_path", type=str, default="tron_model.pt")
    paper_test.add_argument("--episodes", type=int, default=5)
    paper_test.add_argument("--seed", type=int, default=0)
    paper_test.add_argument("--grid", type=int, default=32)
    
    args = parser.parse_args()
    
    if args.command == "train":
        from tron_solution.training.train_ppo import train, train_curriculum, DEFAULT_CURRICULUM_TOTAL
        timesteps = args.timesteps
        if timesteps is None:
            timesteps = DEFAULT_CURRICULUM_TOTAL if (args.curriculum or args.curriculum_stages) else 100000
        if args.curriculum or args.curriculum_stages:
            train_curriculum(
                curriculum_stages=args.curriculum_stages,
                total_timesteps=timesteps,
                learning_rate=args.lr,
                verbose=args.verbose,
                save_dir=args.save_dir,
                eval_freq=args.eval_freq,
                n_envs=args.n_envs,
                resume_path=args.resume,
                vec_normalize_path=args.vec_normalize,
                curriculum_run_id=args.curriculum_run_id,
                early_stop=not args.no_early_stop,
                early_stop_patience=args.early_stop_patience,
                stage_gate_win_rate=args.stage_gate_win_rate,
                stage_gate_episodes=args.stage_gate_episodes,
                stage_gate_extra_steps=args.stage_gate_extra_steps,
                no_stage_gate=args.no_stage_gate,
                warmup_opponent=None if args.no_warmup else "heuristic",
                warmup_timesteps=0 if args.no_warmup else args.warmup_timesteps,
            )
        else:
            train(
                total_timesteps=timesteps,
                learning_rate=args.lr,
                verbose=args.verbose,
                save_dir=args.save_dir,
                eval_freq=args.eval_freq,
                opponent_type=args.opponent,
                minimax_depth=args.minimax_depth,
                n_envs=args.n_envs,
                resume_path=args.resume,
                vec_normalize_path=args.vec_normalize,
                early_stop=not args.no_early_stop,
                early_stop_patience=args.early_stop_patience,
            )
    
    elif args.command == "export":
        from tron_solution.export.export_model import export_model
        export_model(
            model_path=args.model_path,
            output_path=args.output,
            train_first=args.train_and_export,
        )
    
    elif args.command == "test":
        from tron_solution.test.test_model import test_model
        test_model(args.model_path, args.episodes)
    
    elif args.command == "watch-tronbot":
        from tron_solution.watch_tronbot_duel import run_tronbot_duel
        run_tronbot_duel(
            args.tronbot_path, args.minimax_depth, args.episodes,
            args.delay_ms, args.headless, args.tronbot_side, args.move_timeout,
        )
    
    elif args.command == "watch-tronbot-vs-tronbot":
        from tron_solution.watch_tronbot_vs_tronbot import run_tronbot_vs_tronbot
        run_tronbot_vs_tronbot(
            args.tronbot_path, args.episodes, args.delay_ms, args.headless, args.move_timeout,
        )
    
    elif args.command == "serve":
        from tron_solution.launch_player import make_app
        import uvicorn
        uvicorn.run(make_app(args.model, args.server_stack), host=args.host, port=args.port)
    
    elif args.command == "train-paper":
        from tron_paper.training.train_stationary import train_stationary
        from tron_paper.training.train_non_stationary import train_non_stationary, train_all
        if args.stage == "all":
            train_all(args.stationary_steps, args.non_stationary_steps, args.save_dir, args.n_envs, args.lr)
        elif args.stage == "stationary":
            train_stationary(args.stationary_steps, args.save_dir, args.n_envs, args.lr)
        else:
            train_non_stationary(args.non_stationary_steps, args.save_dir, args.stationary_weights, args.lr)

    elif args.command == "export-paper":
        from tron_paper.export.export_model import export_mrl
        export_mrl(args.non_stationary, args.stationary, args.output)

    elif args.command == "test-paper":
        from tron_paper.test.test_model import test_paper_model
        test_paper_model(args.model_path, args.episodes, args.seed, args.grid)

    elif args.command == "watch":
        from tron_solution.watch_duel import run_sb3_duel, run_pt_duel, find_vec_normalize
        if args.model_path.endswith(".pt"):
            run_pt_duel(args.model_path, args.minimax_depth, args.episodes, args.delay_ms)
        else:
            vec = args.vec_normalize or find_vec_normalize(args.model_path)
            run_sb3_duel(args.model_path, vec, args.minimax_depth, args.episodes, args.delay_ms)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py train --timesteps 100000 --verbose")
        print("  python main.py export --train_and_export")
        print("  python main.py test --model_path tron_model.pt")
        print("  python main.py watch --model_path ./ppo_tron_checkpoints/best_model.zip")


if __name__ == "__main__":
    main()
