import os
import importlib.util

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise ImportError("Could not locate tron_solution_dqn package root")
    _d = _parent

import glob
import torch
from stable_baselines3 import DQN

from tron_solution.model.tron_cnn import TronCNN
from tron_solution.model.obs import INPUT_CHANNELS, PLAY_SIZE
from tron_solution.model.stacked_policy import export_stacked_policy
from tron_solution_dqn.training.policy import MaskedDQN


def extract_q_weights(sb3_model: DQN) -> dict:
    state_dict = {}
    feat = sb3_model.q_net.features_extractor
    if hasattr(feat, "conv1"):
        state_dict["conv1.weight"] = feat.conv1.weight.data.cpu()
        state_dict["conv1.bias"] = feat.conv1.bias.data.cpu()
    if hasattr(feat, "conv2"):
        state_dict["conv2.weight"] = feat.conv2.weight.data.cpu()
        state_dict["conv2.bias"] = feat.conv2.bias.data.cpu()
    if hasattr(feat, "shared_fc"):
        state_dict["shared_fc.weight"] = feat.shared_fc.weight.data.cpu()
        state_dict["shared_fc.bias"] = feat.shared_fc.bias.data.cpu()
    q_head = sb3_model.q_net.q_net
    if isinstance(q_head, torch.nn.Sequential):
        for module in q_head:
            if isinstance(module, torch.nn.Linear):
                state_dict["actor.weight"] = module.weight.data.cpu()
                state_dict["actor.bias"] = module.bias.data.cpu()
                break
    elif isinstance(q_head, torch.nn.Linear):
        state_dict["actor.weight"] = q_head.weight.data.cpu()
        state_dict["actor.bias"] = q_head.bias.data.cpu()
    return state_dict


def export_model(
    model_path: str = None,
    output_path: str = "tron_model.pt",
    train_first: bool = False,
    timesteps: int = 100000,
):
    if train_first:
        from tron_solution_dqn.training.train_dqn import train
        train(total_timesteps=timesteps, verbose=1)
        checkpoint_dir = "./dqn_tron_checkpoints"
        model_files = glob.glob(os.path.join(checkpoint_dir, "tron_dqn_final_*.zip"))
        if not model_files:
            raise RuntimeError("Training completed but no model file found!")
        model_path = sorted(model_files)[-1]

    if model_path is None:
        checkpoint_dir = "./dqn_tron_checkpoints"
        model_files = glob.glob(os.path.join(checkpoint_dir, "tron_dqn_final_*.zip"))
        if model_files:
            model_path = sorted(model_files)[-1]
        else:
            raise ValueError("No model path provided and no trained models found.")

    print(f"Loading model from {model_path}...")
    sb3_model = MaskedDQN.load(model_path)
    custom_model = TronCNN(input_channels=INPUT_CHANNELS, spatial=PLAY_SIZE)
    weights = extract_q_weights(sb3_model)
    for key, value in weights.items():
        if key in custom_model.state_dict() and custom_model.state_dict()[key].shape == value.shape:
            custom_model.state_dict()[key].copy_(value)
            print(f"Loaded {key}")
    custom_model.eval()
    export_stacked_policy(custom_model, output_path)
    print(f"Model exported to {output_path}")
    return output_path
