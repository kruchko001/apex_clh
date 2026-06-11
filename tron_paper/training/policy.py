import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tron_paper.model.mish import Mish
from tron_paper.model.mrl_net import MRLActorCritic, _MASK_FILL
from tron_paper.model.phase_torch import compute_action_mask


class MRLFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, stationary: bool = False):
        super().__init__(observation_space, features_dim=128)
        self.ac = MRLActorCritic(stationary=stationary)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.ac.trunk_features(obs)


class MRLPolicy(ActorCriticPolicy):
    def __init__(self, *args, stationary: bool = False, **kwargs):
        self._stationary = stationary
        kwargs["features_extractor_class"] = MRLFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"stationary": stationary}
        kwargs["net_arch"] = dict(pi=[64], vf=[64, 16])
        kwargs["activation_fn"] = Mish
        super().__init__(*args, **kwargs)

    def _mask_logits(self, logits: th.Tensor, obs: th.Tensor) -> th.Tensor:
        mask = compute_action_mask(obs, self._stationary)
        return logits.masked_fill(~mask, _MASK_FILL)

    def _distribution(self, obs: th.Tensor, latent_pi: th.Tensor):
        logits = self._mask_logits(self.action_net(latent_pi), obs)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy
