import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tron_paper.model.mish import Mish
from tron_paper_dqn.model.q_net import MRLQNet


class MRLQFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, stationary: bool = False):
        super().__init__(observation_space, features_dim=128)
        self.qnet = MRLQNet(stationary=stationary)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.qnet.trunk_features(obs)


def dqn_policy_kwargs(stationary: bool = False):
    return {
        "features_extractor_class": MRLQFeaturesExtractor,
        "features_extractor_kwargs": {"stationary": stationary},
        "net_arch": [64],
        "activation_fn": Mish,
    }
