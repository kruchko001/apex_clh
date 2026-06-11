import torch
import torch.nn as nn

from tron_paper.model.phase_torch import phase_separated
from tron_paper_dqn.model.q_net import MRLQNet


class MRLQCursor(nn.Module):
    def __init__(self, non_stationary: MRLQNet, stationary: MRLQNet):
        super().__init__()
        self.non_stationary = non_stationary
        self.stationary = stationary

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        sep = phase_separated(obs)
        ns_q = self.non_stationary(obs)
        st_obs = obs.clone()
        st_obs[:, 4] = 0.0
        s_q = self.stationary(st_obs)
        q = s_q * sep + ns_q * (1.0 - sep)
        return q.squeeze(0)


def export_cursor(non_stationary: MRLQNet, stationary: MRLQNet, path: str):
    cursor = MRLQCursor(non_stationary, stationary)
    cursor.eval()
    torch.jit.script(cursor).save(path)
