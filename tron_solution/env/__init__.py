"""Tron Environment Package."""

from .tron_env import TronEnv
from .opponents import get_opponent, DEFAULT_OPPONENT_TYPE, DEFAULT_MINIMAX_DEPTH, PLAY_MINIMAX_DEPTHS

__all__ = ["TronEnv", "get_opponent", "DEFAULT_OPPONENT_TYPE", "DEFAULT_MINIMAX_DEPTH", "PLAY_MINIMAX_DEPTHS"]
