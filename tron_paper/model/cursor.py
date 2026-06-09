import torch
import torch.nn as nn

from tron_paper.model.mrl_net import MRLActorCritic
from tron_paper.model.phase_torch import phase_separated


class MRLCursor(nn.Module):
    def __init__(self, non_stationary: MRLActorCritic, stationary: MRLActorCritic):
        super().__init__()
        self.non_stationary = non_stationary
        self.stationary = stationary

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        sep = phase_separated(obs)
        ns_logits, _ = self.non_stationary(obs)
        st_obs = obs.clone()
        st_obs[:, 2] = 0.0
        st_obs[:, 4] = 0.0
        s_logits, _ = self.stationary(st_obs)
        logits = s_logits * sep + ns_logits * (1.0 - sep)
        return logits.squeeze(0)


def export_cursor(non_stationary: MRLActorCritic, stationary: MRLActorCritic, path: str, grid: int = 32):
    cursor = MRLCursor(non_stationary, stationary)
    cursor.eval()
    scripted = torch.jit.script(cursor)
    scripted.save(path)
