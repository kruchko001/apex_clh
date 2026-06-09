import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tron_paper.model.mish import Mish
from tron_paper.model.mrl_net import MRLActorCritic


class MRLFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, stationary: bool = False):
        super().__init__(observation_space, features_dim=128)
        self.ac = MRLActorCritic(stationary=stationary)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.ac.trunk_features(obs)


class MRLPolicy(ActorCriticPolicy):
    def __init__(self, *args, stationary: bool = False, **kwargs):
        kwargs["features_extractor_class"] = MRLFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"stationary": stationary}
        kwargs["net_arch"] = dict(pi=[64], vf=[64, 16])
        kwargs["activation_fn"] = Mish
        super().__init__(*args, **kwargs)
