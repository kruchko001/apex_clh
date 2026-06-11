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

    train_dqn_parser = subparsers.add_parser("train-dqn", help="Train DQN agent (same env/curriculum as tron_solution PPO)")
    train_dqn_parser.add_argument("--timesteps", type=int, default=None)
    train_dqn_parser.add_argument("--lr", type=float, default=3e-4)
    train_dqn_parser.add_argument("--verbose", type=int, default=1)
    train_dqn_parser.add_argument("--save-dir", type=str, default="./dqn_tron_checkpoints")
    train_dqn_parser.add_argument("--opponent", type=str, default="minimax")
    train_dqn_parser.add_argument("--minimax-depth", type=int, default=14)
    train_dqn_parser.add_argument("--n-envs", type=int, default=None)
    train_dqn_parser.add_argument("--resume", type=str, default=None)
    train_dqn_parser.add_argument("--vec-normalize", type=str, default=None)
    train_dqn_parser.add_argument("--buffer-size", type=int, default=100000)
    train_dqn_parser.add_argument("--learning-starts", type=int, default=10000)
    train_dqn_parser.add_argument("--batch-size", type=int, default=64)
    train_dqn_parser.add_argument("--exploration-fraction", type=float, default=0.3)
    train_dqn_parser.add_argument("--curriculum", action="store_true")
    train_dqn_parser.add_argument("--curriculum-stages", type=str, default=None)
    train_dqn_parser.add_argument("--curriculum-run-id", type=str, default=None)
    train_dqn_parser.add_argument("--stage-gate-win-rate", type=float, default=0.25)
    train_dqn_parser.add_argument("--stage-gate-episodes", type=int, default=20)
    train_dqn_parser.add_argument("--stage-gate-extra-steps", type=int, default=100000)
    train_dqn_parser.add_argument("--no-stage-gate", action="store_true")
    train_dqn_parser.add_argument("--warmup-timesteps", type=int, default=200000)
    train_dqn_parser.add_argument("--no-warmup", action="store_true")
    train_dqn_parser.add_argument("--no-early-stop", action="store_true")
    train_dqn_parser.add_argument("--early-stop-patience", type=int, default=6)
    train_dqn_parser.add_argument("--eval-freq", type=int, default=20000)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to TorchScript")
    export_parser.add_argument("--model_path", type=str, help="Path to trained SB3 model")
    export_parser.add_argument("--output", type=str, default="tron_model.pt", help="Output path")
    export_parser.add_argument("--train_and_export", action="store_true", help="Train then export")

    export_dqn_parser = subparsers.add_parser("export-dqn", help="Export DQN model to TorchScript")
    export_dqn_parser.add_argument("--model_path", type=str, help="Path to trained SB3 DQN model")
    export_dqn_parser.add_argument("--output", type=str, default="tron_model.pt")
    export_dqn_parser.add_argument("--train_and_export", action="store_true")
    export_dqn_parser.add_argument("--timesteps", type=int, default=100000)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test exported model")
    test_parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    test_parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    
    watch_parser = subparsers.add_parser("watch", help="Watch model vs MinimaxOpponent")
    watch_parser.add_argument("--model_path", type=str, default="./ppo_tron_checkpoints/best_model.zip")
    watch_parser.add_argument("--vec_normalize", type=str, default=None)
    watch_parser.add_argument("--minimax-depth", type=int, default=14)
    watch_parser.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_parser.add_argument("--delay-ms", type=int, default=80)

    play_tronbot_parser = subparsers.add_parser("play-tronbot", help="Play vs TronBot (pygame, YOU vs C++ bot)")
    play_tronbot_parser.add_argument("--tronbot-path", type=str, default=None)
    play_tronbot_parser.add_argument("--move-timeout", type=float, default=10.0)
    play_tronbot_parser.add_argument("--runs", type=int, default=3)

    play_king_parser = subparsers.add_parser("play-king", help="Play vs current king QNet (pygame, YOU vs King)")
    play_king_parser.add_argument("--king", type=str, default="./kings/code_submission_v1.pt")
    play_king_parser.add_argument("--runs", type=int, default=3)

    tronbot_parser = subparsers.add_parser("watch-tronbot", help="Watch TronBot vs MinimaxOpponent")
    tronbot_parser.add_argument("--tronbot-path", type=str, default=None)
    tronbot_parser.add_argument("--minimax-depth", type=int, default=14)
    tronbot_parser.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    tronbot_parser.add_argument("--delay-ms", type=int, default=80)
    tronbot_parser.add_argument("--headless", action="store_true")
    tronbot_parser.add_argument("--tronbot-side", choices=["my", "opp"], default="my")
    tronbot_parser.add_argument("--move-timeout", type=float, default=5.0)

    tbvstb_parser = subparsers.add_parser("watch-tronbot-vs-tronbot", help="Watch TronBot vs TronBot")
    tbvstb_parser.add_argument("--tronbot-path", type=str, default=None)
    tbvstb_parser.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
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
    paper_parser.add_argument("--ent-coef", type=float, default=0.02, help="PPO entropy bonus (stationary)")
    paper_parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Save stationary_agent.pt every N steps")
    paper_parser.add_argument("--stage", choices=["all", "stationary", "non-stationary"], default="all")
    paper_parser.add_argument("--stationary-weights", type=str, default=None)
    paper_parser.add_argument("--show-gui", action="store_true", help="Show GUI rollout after each PPO update (stationary)")
    paper_parser.add_argument("--gui-delay-ms", type=int, default=50)
    paper_parser.add_argument("--gui-every-updates", type=int, default=1)

    paper_export = subparsers.add_parser("export-paper", help="Export MRL cursor model to TorchScript")
    paper_export.add_argument("--non-stationary", type=str, default="./tron_paper_checkpoints/non_stationary_agent.pt")
    paper_export.add_argument("--stationary", type=str, default="./tron_paper_checkpoints/stationary_agent.pt")
    paper_export.add_argument("--output", type=str, default="tron_model.pt")

    paper_test = subparsers.add_parser("test-paper", help="Test MRL .pt with official launcher encode/mask flow")
    paper_test.add_argument("--model_path", type=str, default="tron_model.pt")
    paper_test.add_argument("--episodes", type=int, default=5)
    paper_test.add_argument("--seed", type=int, default=0)
    paper_test.add_argument("--grid", type=int, default=32)

    paper_bc = subparsers.add_parser("paper-bc", help="Stationary behavior cloning (TronBot labels)")
    paper_bc_sub = paper_bc.add_subparsers(dest="bc_action", required=True)
    bc_collect = paper_bc_sub.add_parser("collect", help="Collect TronBot stationary dataset")
    bc_collect.add_argument("--episodes", type=int, default=500)
    bc_collect.add_argument("--seed", type=int, default=0)
    bc_collect.add_argument("--output", type=str, default="./tron_paper_BH/data/stationary_bc.npz")
    bc_collect.add_argument("--backend", choices=["cpp", "py"], default="cpp", help="cpp=full MyTronBot.exe (strongest)")
    bc_collect.add_argument("--tronbot-path", type=str, default=None)
    bc_collect.add_argument("--move-timeout", type=float, default=5.0, help="Per-move cap (cpp uses internal ~1s/3s timer too)")
    bc_collect.add_argument("--max-depth", type=int, default=100, help="Python backend only; cpp uses full search")
    bc_collect.add_argument("--mode", choices=["spacefill", "greedy"], default="spacefill", help="Python backend only")
    bc_train = paper_bc_sub.add_parser("train", help="Train stationary BC model")
    bc_train.add_argument("--data", type=str, default="./tron_paper_BH/data/stationary_bc.npz")
    bc_train.add_argument("--save-dir", type=str, default="./tron_paper_BH_checkpoints")
    bc_train.add_argument("--epochs", type=int, default=20)
    bc_train.add_argument("--batch-size", type=int, default=256)
    bc_train.add_argument("--lr", type=float, default=3e-4)
    bc_train.add_argument("--patience", type=int, default=5, help="Early stop after N epochs without val loss improvement (0=off)")
    bc_train.add_argument("--min-epochs", type=int, default=3, help="Minimum epochs before early stopping")

    prep_sub = subparsers.add_parser("prepare-submission", help="Bundle Python TronBot + export Apex tron_model.pt")
    prep_sub.add_argument("--output-dir", type=str, default="./for_submission")
    prep_sub.add_argument("--mode", choices=["bc", "tronbot"], default="tronbot", help="tronbot=scripted Python MyTronBot export for Apex")
    prep_sub.add_argument("--stationary", type=str, default="./tron_paper_BH_checkpoints/stationary_agent.pt")
    prep_sub.add_argument("--non-stationary", type=str, default="./tron_paper_checkpoints/non_stationary_agent.pt")

    watch_paper = subparsers.add_parser("watch-paper", help="Watch tron_paper stationary agent (pygame GUI)")
    watch_paper.add_argument("--model", type=str, default=None, help=".pt or .zip; default: latest in save-dir")
    watch_paper.add_argument("--save-dir", type=str, default="./tron_paper_checkpoints")
    watch_paper.add_argument("--delay-ms", type=int, default=80)
    watch_paper.add_argument("--seed", type=int, default=None)

    watch_tb_st = subparsers.add_parser("watch-tronbot-stationary", help="Watch TronBot in stationary env (pygame GUI)")
    watch_tb_st.add_argument("--delay-ms", type=int, default=80)
    watch_tb_st.add_argument("--seed", type=int, default=None)
    watch_tb_st.add_argument("--max-depth", type=int, default=1)
    watch_tb_st.add_argument("--mode", choices=["spacefill", "greedy"], default="spacefill")
    watch_tb_st.add_argument("--backend", choices=["py", "cpp"], default="py")
    watch_tb_st.add_argument("--compare", action="store_true", help="Same map: greedy first, then spacefill depth=1")

    watch_bc_greedy = subparsers.add_parser(
        "watch-bc-vs-greedy-stationary",
        help="Same map: BC clone then greedy Python TronBot",
    )
    watch_bc_greedy.add_argument("--model", type=str, default=None, help="BC .pt; default: tron_paper_BH_checkpoints/stationary_agent.pt")
    watch_bc_greedy.add_argument("--save-dir", type=str, default="./tron_paper_BH_checkpoints")
    watch_bc_greedy.add_argument("--delay-ms", type=int, default=80)
    watch_bc_greedy.add_argument("--seed", type=int, default=None)

    watch_pt_tb = subparsers.add_parser("watch-pt-vs-tronbot", help="GUI: submission .pt (0.1s) vs full C++ TronBot")
    watch_pt_tb.add_argument("--model", type=str, default="./for_submission/tron_model.pt")
    watch_pt_tb.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_pt_tb.add_argument("--delay-ms", type=int, default=80)
    watch_pt_tb.add_argument("--seed", type=int, default=None)
    watch_pt_tb.add_argument("--pt-as", type=int, default=0, choices=[0, 1])

    watch_pt_py = subparsers.add_parser("watch-pt-vs-py-tronbot", help="GUI: submission .pt (0.1s) vs Python TronBot spacefill")
    watch_pt_py.add_argument("--model", type=str, default="./for_submission/tron_model.pt")
    watch_pt_py.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_pt_py.add_argument("--delay-ms", type=int, default=80)
    watch_pt_py.add_argument("--seed", type=int, default=None)
    watch_pt_py.add_argument("--pt-as", type=int, default=0, choices=[0, 1])
    watch_pt_py.add_argument("--no-swap-spawn", action="store_true")

    watch_py_king = subparsers.add_parser("watch-py-tronbot-vs-king", help="GUI: full Python TronBot vs king QNet")
    watch_py_king.add_argument("--king", type=str, default="./kings/code_submission_v1.pt")
    watch_py_king.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_py_king.add_argument("--delay-ms", type=int, default=80)
    watch_py_king.add_argument("--seed", type=int, default=None)

    watch_tb_king = subparsers.add_parser("watch-tronbot-vs-king", help="GUI: full C++ TronBot vs king QNet")
    watch_tb_king.add_argument("--king", type=str, default="./kings/code_submission_v1.pt")
    watch_tb_king.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_tb_king.add_argument("--delay-ms", type=int, default=80)
    watch_tb_king.add_argument("--seed", type=int, default=None)
    watch_tb_king.add_argument("--move-timeout", type=float, default=10.0)
    watch_tb_king.add_argument("--no-timer", action="store_true", help="Disable C++ internal timer (not play-tronbot setup)")

    watch_pt_mm = subparsers.add_parser("watch-pt-vs-minimax", help="GUI: PT TronBot (0.1s) vs Minimax")
    watch_pt_mm.add_argument("--model", type=str, default="./for_submission/tron_model.pt")
    watch_pt_mm.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_pt_mm.add_argument("--delay-ms", type=int, default=80)
    watch_pt_mm.add_argument("--seed", type=int, default=None)
    watch_pt_mm.add_argument("--pt-as", type=int, default=0, choices=[0, 1])
    watch_pt_mm.add_argument("--minimax-depth", type=int, default=14)
    watch_pt_mm.add_argument("--no-swap-spawn", action="store_true")

    watch_king = subparsers.add_parser("watch-pt-vs-king", help="GUI: your PT TronBot vs downloaded king QNet .pt")
    watch_king.add_argument("--king", type=str, default="./kings/code_submission_v1.pt", help="Path to king TorchScript .pt")
    watch_king.add_argument("--pt", type=str, default="./for_submission/tron_model.pt")
    watch_king.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_king.add_argument("--delay-ms", type=int, default=80)
    watch_king.add_argument("--seed", type=int, default=None)

    watch_greedy_king = subparsers.add_parser("watch-greedy-vs-king", help="GUI: 1-ply greedy TronBot .pt vs king QNet")
    watch_greedy_king.add_argument("--king", type=str, default="./kings/code_submission_v1.pt")
    watch_greedy_king.add_argument("--greedy", type=str, default="./_test_out/tron_model_greedy.pt")
    watch_greedy_king.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_greedy_king.add_argument("--delay-ms", type=int, default=80)
    watch_greedy_king.add_argument("--seed", type=int, default=None)

    watch_spacefill_king = subparsers.add_parser("watch-spacefill-vs-king", help="GUI: spacefill TronBot .pt vs king QNet")
    watch_spacefill_king.add_argument("--king", type=str, default="./kings/code_submission_v1.pt")
    watch_spacefill_king.add_argument("--spacefill", type=str, default="./for_submission/tron_model.pt")
    watch_spacefill_king.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_spacefill_king.add_argument("--delay-ms", type=int, default=80)
    watch_spacefill_king.add_argument("--seed", type=int, default=None)

    watch_pt_pt = subparsers.add_parser("watch-pt-vs-pt", help="GUI: enhanced PT vs previous 1-ply greedy PT")
    watch_pt_pt.add_argument("--new", type=str, default="./for_submission/tron_model.pt")
    watch_pt_pt.add_argument("--old", type=str, default="./_test_out/tron_model_greedy.pt")
    watch_pt_pt.add_argument("--runs", "--episodes", type=int, default=3, dest="episodes")
    watch_pt_pt.add_argument("--delay-ms", type=int, default=80)
    watch_pt_pt.add_argument("--seed", type=int, default=None)
    watch_pt_pt.add_argument("--new-as", type=int, default=0, choices=[0, 1])
    watch_pt_pt.add_argument("--no-swap-spawn", action="store_true")

    dqn_parser = subparsers.add_parser("train-paper-dqn", help="Train MRL paper agent with DQN (stationary + non-stationary)")
    dqn_parser.add_argument("--stationary-steps", type=int, default=300000)
    dqn_parser.add_argument("--non-stationary-steps", type=int, default=500000)
    dqn_parser.add_argument("--save-dir", type=str, default="./tron_paper_dqn_checkpoints")
    dqn_parser.add_argument("--n-envs", type=int, default=1)
    dqn_parser.add_argument("--lr", type=float, default=3e-4)
    dqn_parser.add_argument("--buffer-size", type=int, default=100000)
    dqn_parser.add_argument("--learning-starts", type=int, default=10000)
    dqn_parser.add_argument("--batch-size", type=int, default=64)
    dqn_parser.add_argument("--exploration-fraction", type=float, default=0.3)
    dqn_parser.add_argument("--stage", choices=["all", "stationary", "non-stationary"], default="all")
    dqn_parser.add_argument("--stationary-weights", type=str, default=None)

    dqn_export = subparsers.add_parser("export-paper-dqn", help="Export MRL DQN cursor model to TorchScript")
    dqn_export.add_argument("--non-stationary", type=str, default="./tron_paper_dqn_checkpoints/non_stationary_agent.pt")
    dqn_export.add_argument("--stationary", type=str, default="./tron_paper_dqn_checkpoints/stationary_agent.pt")
    dqn_export.add_argument("--output", type=str, default="tron_model.pt")

    dqn_test = subparsers.add_parser("test-paper-dqn", help="Test MRL DQN .pt with official launcher flow")
    dqn_test.add_argument("--model_path", type=str, default="tron_model.pt")
    dqn_test.add_argument("--episodes", type=int, default=5)
    dqn_test.add_argument("--seed", type=int, default=0)
    dqn_test.add_argument("--grid", type=int, default=32)
    
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
    
    elif args.command == "train-dqn":
        from tron_solution.training.train_ppo import DEFAULT_CURRICULUM_TOTAL
        from tron_solution_dqn.training.train_dqn import train as train_dqn, train_curriculum as train_dqn_curriculum
        timesteps = args.timesteps
        if timesteps is None:
            timesteps = DEFAULT_CURRICULUM_TOTAL if (args.curriculum or args.curriculum_stages) else 100000
        dqn_kw = dict(
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            exploration_fraction=args.exploration_fraction,
            verbose=args.verbose,
            save_dir=args.save_dir,
        )
        if args.curriculum or args.curriculum_stages:
            train_dqn_curriculum(
                curriculum_stages=args.curriculum_stages,
                total_timesteps=timesteps,
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
                **dqn_kw,
            )
        else:
            train_dqn(
                total_timesteps=timesteps,
                eval_freq=args.eval_freq,
                opponent_type=args.opponent,
                minimax_depth=args.minimax_depth,
                n_envs=args.n_envs,
                resume_path=args.resume,
                vec_normalize_path=args.vec_normalize,
                early_stop=not args.no_early_stop,
                early_stop_patience=args.early_stop_patience,
                **dqn_kw,
            )

    elif args.command == "export-dqn":
        from tron_solution_dqn.export.export_model import export_model as export_dqn_model
        export_dqn_model(
            model_path=args.model_path,
            output_path=args.output,
            train_first=args.train_and_export,
            timesteps=args.timesteps,
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
    
    elif args.command == "play-tronbot":
        from tron_solution.play_human_tronbot import main as play_tronbot
        play_tronbot(args.tronbot_path, args.move_timeout, args.runs)

    elif args.command == "play-king":
        from tron_solution.play_human_king import main as play_king
        play_king(args.king, args.runs)

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
            train_all(
                args.stationary_steps, args.non_stationary_steps, args.save_dir, args.n_envs, args.lr,
                ent_coef=args.ent_coef, checkpoint_freq=args.checkpoint_freq,
                show_gui=args.show_gui, gui_delay_ms=args.gui_delay_ms, gui_every_updates=args.gui_every_updates,
            )
        elif args.stage == "stationary":
            train_stationary(
                args.stationary_steps, args.save_dir, args.n_envs, args.lr,
                ent_coef=args.ent_coef, checkpoint_freq=args.checkpoint_freq,
                show_gui=args.show_gui, gui_delay_ms=args.gui_delay_ms, gui_every_updates=args.gui_every_updates,
            )
        else:
            train_non_stationary(args.non_stationary_steps, args.save_dir, args.stationary_weights, args.lr)

    elif args.command == "export-paper":
        from tron_paper.export.export_model import export_mrl
        export_mrl(args.non_stationary, args.stationary, args.output)

    elif args.command == "test-paper":
        from tron_paper.test.test_model import test_paper_model
        test_paper_model(args.model_path, args.episodes, args.seed, args.grid)

    elif args.command == "paper-bc":
        if args.bc_action == "collect":
            from tron_paper_BH.collect import collect_dataset
            collect_dataset(
                args.episodes, args.seed, args.output,
                args.backend, args.tronbot_path, args.move_timeout, args.max_depth, args.mode,
            )
        else:
            from tron_paper_BH.train_bc import train_bc
            train_bc(
                args.data, args.save_dir, args.epochs, args.batch_size, args.lr,
                patience=args.patience, min_epochs=args.min_epochs,
            )

    elif args.command == "prepare-submission":
        from for_submission.export import export_for_submission
        export_for_submission(args.output_dir, mode=args.mode, stationary=args.stationary, non_stationary=args.non_stationary)

    elif args.command == "watch-paper":
        from tron_paper.training.gui_rollout import watch_stationary
        watch_stationary(args.model, args.save_dir, args.delay_ms, args.seed)

    elif args.command == "watch-tronbot-stationary":
        from tron_paper.training.gui_rollout import watch_tronbot_stationary
        watch_tronbot_stationary(args.delay_ms, args.seed, args.max_depth, args.mode, args.backend, args.compare)

    elif args.command == "watch-bc-vs-greedy-stationary":
        from tron_paper.training.gui_rollout import watch_bc_vs_greedy_stationary
        watch_bc_vs_greedy_stationary(args.delay_ms, args.seed, args.model, args.save_dir)

    elif args.command == "watch-pt-vs-tronbot":
        from duel_pt_vs_tronbot import watch_duel
        watch_duel(args.model, args.episodes, args.delay_ms, args.seed, args.pt_as)

    elif args.command == "watch-pt-vs-py-tronbot":
        from duel_pt_vs_py_tronbot import watch_duel
        watch_duel(args.model, args.episodes, args.delay_ms, args.seed, args.pt_as, swap_spawn=not args.no_swap_spawn)

    elif args.command == "watch-py-tronbot-vs-king":
        from duel_py_tronbot_vs_king import watch_duel
        watch_duel(args.king, args.episodes, args.delay_ms, args.seed)

    elif args.command == "watch-tronbot-vs-king":
        from duel_tronbot_vs_king import watch_duel
        watch_duel(
            args.king, args.episodes, args.delay_ms, args.seed,
            move_timeout=args.move_timeout, use_timer=not args.no_timer,
        )

    elif args.command == "watch-pt-vs-minimax":
        from duel_pt_vs_minimax import watch_duel
        watch_duel(
            args.model, args.episodes, args.delay_ms, args.seed,
            args.pt_as, swap_spawn=not args.no_swap_spawn, minimax_depth=args.minimax_depth,
        )

    elif args.command == "watch-pt-vs-king":
        from duel_pt_vs_king import watch_duel
        watch_duel(args.king, args.pt, args.episodes, args.delay_ms, args.seed)

    elif args.command == "watch-greedy-vs-king":
        from duel_pt_vs_king import ensure_greedy, watch_duel
        watch_duel(
            args.king, ensure_greedy(args.greedy), args.episodes, args.delay_ms, args.seed,
            side_label="Greedy",
        )

    elif args.command == "watch-spacefill-vs-king":
        from duel_pt_vs_king import ensure_spacefill, watch_duel
        watch_duel(
            args.king, ensure_spacefill(args.spacefill), args.episodes, args.delay_ms, args.seed,
            side_label="Spacefill",
        )

    elif args.command == "watch-pt-vs-pt":
        from duel_pt_vs_pt import watch_duel
        watch_duel(
            args.new, args.old, args.episodes, args.delay_ms, args.seed,
            args.new_as, swap_spawn=not args.no_swap_spawn,
        )

    elif args.command == "train-paper-dqn":
        from tron_paper_dqn.training.train_stationary import train_stationary as train_dqn_stationary
        from tron_paper_dqn.training.train_non_stationary import train_non_stationary as train_dqn_ns, train_all as train_dqn_all
        if args.stage == "all":
            train_dqn_all(
                args.stationary_steps, args.non_stationary_steps, args.save_dir, args.n_envs, args.lr,
            )
        elif args.stage == "stationary":
            train_dqn_stationary(
                args.stationary_steps, args.save_dir, args.n_envs, args.lr,
                args.buffer_size, args.learning_starts, args.batch_size, args.exploration_fraction,
            )
        else:
            train_dqn_ns(
                args.non_stationary_steps, args.save_dir, args.stationary_weights, args.lr,
                args.buffer_size, args.learning_starts, args.batch_size, args.exploration_fraction,
            )

    elif args.command == "export-paper-dqn":
        from tron_paper_dqn.export.export_model import export_mrl_dqn
        export_mrl_dqn(args.non_stationary, args.stationary, args.output)

    elif args.command == "test-paper-dqn":
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
