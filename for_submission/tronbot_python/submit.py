"""Export: Python MyTronBot source + Apex TorchScript submit file."""

import os
import shutil
import sys

import torch

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tronbot.python.player import obs_to_logits
from tronbot.python.jit_model import TronBotSubmit

DEFAULT_OUT = os.path.join(ROOT, "for_submission")
DEFAULT_STATIONARY = os.path.join(ROOT, "tron_paper_BH_checkpoints", "stationary_agent.pt")
DEFAULT_NON_STATIONARY = os.path.join(ROOT, "tron_paper_checkpoints", "non_stationary_agent.pt")


def _copy_sources(out_dir: str) -> str:
    dst = os.path.join(out_dir, "tronbot_python")
    src = PKG
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    if os.path.isdir(dst):
        for name in os.listdir(src):
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(s, d)
    else:
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return dst


def export_submission(
    output_dir: str = DEFAULT_OUT,
    output_name: str = "tron_model.pt",
    mode: str = "tronbot",
    stationary: str = DEFAULT_STATIONARY,
    non_stationary: str = DEFAULT_NON_STATIONARY,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    dst = _copy_sources(output_dir)
    out = os.path.join(output_dir, output_name)

    if mode == "bc":
        from tron_paper.export.export_model import export_mrl
        export_mrl(non_stationary, stationary, out)
    elif mode == "tronbot":
        model = TronBotSubmit()
        model.eval()
        torch.jit.script(model).save(out)
    else:
        raise ValueError(f"unknown mode: {mode}")

    loaded = torch.jit.load(out)
    loaded.eval()
    ex = torch.zeros(1, 5, 32, 32)
    ex[0, 3, 1, 1] = 1.0
    ex[0, 4, 30, 30] = 1.0
    with torch.no_grad():
        ref = obs_to_logits(ex)
        jit = loaded(ex)

    print("Step 1 — Python TronBot (same as C++ MyTronBot.cc)")
    print(f"  source: tronbot/python/mytronbot.py")
    print(f"  bundle: {dst}/")
    print("Step 2 — Apex submit file (scripted MyTronBot)")
    print(f"  file:   {out}")
    print(f"  mode:   {mode}")
    print(f"  shape:  (1,5,32,32) -> (4,)  MyTronBot={ref.tolist()}  pt={jit.tolist()}")
    return out
