import os

import torch as th

from tron_paper.model.cursor import MRLCursor, export_cursor
from tron_paper.model.mrl_net import MRLActorCritic


def export_mrl(
    non_stationary_path: str,
    stationary_path: str,
    output: str = "tron_model.pt",
):
    ns = MRLActorCritic(stationary=False)
    st = MRLActorCritic(stationary=True)
    ns.load_state_dict(th.load(non_stationary_path, map_location="cpu", weights_only=True))
    st.load_state_dict(th.load(stationary_path, map_location="cpu", weights_only=True))
    ns.eval()
    st.eval()
    export_cursor(ns, st, output)
    print(f"Exported MRL cursor model to {output}")


if __name__ == "__main__":
    export_mrl(
        "./tron_paper_checkpoints/non_stationary_agent.pt",
        "./tron_paper_checkpoints/stationary_agent.pt",
    )
