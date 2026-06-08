import torch
import torch.nn as nn

from tron_solution.model.obs import GRID_CHANNELS, N_STACK, PLAY_SIZE


class StackedTronPolicy(nn.Module):
    def __init__(self, core: nn.Module, n_stack: int = N_STACK):
        super().__init__()
        self.core = core
        self.n_stack = n_stack
        self.register_buffer(
            "frames",
            torch.zeros(n_stack, GRID_CHANNELS, PLAY_SIZE, PLAY_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame = x[0, 1:5, 1:31, 1:31]
        trails = frame[0] + frame[1]
        if trails.sum() <= 2.0:
            self.frames.zero_()
        rolled = torch.roll(self.frames, shifts=-1, dims=0)
        rolled[-1] = frame
        self.frames.copy_(rolled)
        stacked = self.frames.reshape(1, -1, 30, 30)
        logits, _ = self.core(stacked)
        return logits.squeeze(0)


def export_stacked_policy(core: nn.Module, path: str) -> None:
    wrapper = StackedTronPolicy(core)
    wrapper.eval()
    torch.jit.script(wrapper).save(path)
