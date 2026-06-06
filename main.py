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
    train_parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    train_parser.add_argument("--save-dir", type=str, default="./ppo_tron_checkpoints", help="Save directory")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to TorchScript")
    export_parser.add_argument("--model_path", type=str, help="Path to trained SB3 model")
    export_parser.add_argument("--output", type=str, default="tron_model.pt", help="Output path")
    export_parser.add_argument("--train_and_export", action="store_true", help="Train then export")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test exported model")
    test_parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    test_parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from tron_solution.training.train_ppo import train
        train(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            verbose=args.verbose,
            save_dir=args.save_dir,
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
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py train --timesteps 100000 --verbose")
        print("  python main.py export --train_and_export")
        print("  python main.py test --model_path tron_model.pt")


if __name__ == "__main__":
    main()
