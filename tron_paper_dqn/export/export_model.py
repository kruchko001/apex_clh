import torch as th

from tron_paper_dqn.model.cursor import export_cursor
from tron_paper_dqn.model.q_net import MRLQNet


def export_mrl_dqn(
    non_stationary_path: str,
    stationary_path: str,
    output: str = "tron_model.pt",
):
    ns = MRLQNet(stationary=False)
    st = MRLQNet(stationary=True)
    ns.load_state_dict(th.load(non_stationary_path, map_location="cpu", weights_only=True))
    st.load_state_dict(th.load(stationary_path, map_location="cpu", weights_only=True))
    ns.eval()
    st.eval()
    export_cursor(ns, st, output)
    print(f"Exported MRL DQN cursor model to {output}")
