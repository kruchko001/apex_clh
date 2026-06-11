import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork

from tron_solution.model.obs import apply_action_mask
from tron_solution.training.train_ppo import TronFeaturesExtractor


class TronMaskedQNetwork(QNetwork):
    def forward(self, obs):
        q_values = super().forward(obs)
        return apply_action_mask(q_values, obs["valid"])


class TronMaskedDQNPolicy(MultiInputPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return TronMaskedQNetwork(**net_args).to(self.device)


class MaskedDQN(DQN):
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                valid = observation["valid"]
                n_batch = valid.shape[0]
                action = np.array([self._sample_valid(valid[i]) for i in range(n_batch)])
            else:
                action = np.array(self._sample_valid(observation["valid"]))
            return action, state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_valid(self, valid):
        idx = np.flatnonzero(valid.astype(bool))
        if len(idx) == 0:
            return self.action_space.sample()
        return int(np.random.choice(idx))


def dqn_policy_kwargs():
    return {
        "features_extractor_class": TronFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "normalize_images": False,
        "net_arch": [],
    }
