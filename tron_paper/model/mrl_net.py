import torch
import torch.nn as nn

from tron_paper.model.mish import Mish
from tron_paper.env.encode import CHANNELS, GRID


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = Mish()

    def forward(self, x):
        return self.act(self.conv(x) + x)


class MRLTrunk(nn.Module):
    def __init__(self, in_ch: int = CHANNELS, spatial: int = GRID):
        super().__init__()
        self.act = Mish()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.r1 = ResBlock(32)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.r2 = ResBlock(64)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.c5 = nn.Conv2d(64, 64, 7, stride=2, padding=3)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.drop = nn.Dropout(0.2)

        with torch.no_grad():
            n_flat = self._conv_out(torch.zeros(1, in_ch, spatial, spatial)).view(1, -1).shape[1]

        self.fc1 = nn.Linear(n_flat, 256)
        self.fc2 = nn.Linear(256, 128)

    def _conv_out(self, x):
        x = self.act(self.c1(x))
        x = self.r1(x)
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.r2(x)
        x = self.act(self.c4(x))
        x = self.act(self.c5(x))
        return self.pool(x)

    def forward(self, x):
        x = self._conv_out(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return x


class MRLActorCritic(nn.Module):
    def __init__(self, stationary: bool = False, spatial: int = GRID):
        super().__init__()
        self.stationary = stationary
        self.trunk = MRLTrunk(spatial=spatial)
        self.act = Mish()
        self.pi1 = nn.Linear(128, 64)
        self.pi2 = nn.Linear(64, 4)
        self.v1 = nn.Linear(128, 64)
        self.v2 = nn.Linear(64, 16)
        self.v3 = nn.Linear(16, 1)

    def _mask(self, x):
        if not self.stationary:
            return x
        y = x.clone()
        y[:, 2] = 0.0
        y[:, 4] = 0.0
        return y

    def trunk_features(self, x):
        return self.trunk(self._mask(x))

    def forward(self, x):
        h = self.trunk_features(x)
        logits = self.pi2(self.act(self.pi1(h)))
        v = self.v3(self.act(self.v2(self.act(self.v1(h)))))
        return logits, v.squeeze(-1)

    def act_greedy(self, obs):
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            logits, _ = self.forward(t)
            return int(logits.argmax(dim=-1).item())
