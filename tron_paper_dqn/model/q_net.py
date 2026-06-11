import torch
import torch.nn as nn

from tron_paper.env.encode import CHANNELS, PLAY_SIZE, STATIONARY_CHANNELS
from tron_paper.model.mish import Mish
from tron_paper.model.mrl_net import MRLTrunk
from tron_paper.model.phase_torch import extract_non_stationary_input, extract_stationary_input


class MRLQNet(nn.Module):
    def __init__(self, stationary: bool = False, spatial: int = None):
        super().__init__()
        self.stationary = stationary
        in_ch = STATIONARY_CHANNELS if stationary else CHANNELS
        if spatial is None:
            spatial = PLAY_SIZE
        self.trunk = MRLTrunk(in_ch=in_ch, spatial=spatial)
        self.act = Mish()
        self.q1 = nn.Linear(128, 64)
        self.q2 = nn.Linear(64, 4)

    def _prep(self, x):
        if self.stationary:
            return extract_stationary_input(x)
        return extract_non_stationary_input(x)

    def forward(self, x):
        h = self.trunk(self._prep(x))
        return self.q2(self.act(self.q1(h)))

    def trunk_features(self, x):
        return self.trunk(self._prep(x))

    def act_greedy(self, obs):
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            return int(self.forward(t).argmax(dim=-1).item())
