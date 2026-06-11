import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tronbot.python.submit import export_submission

DIR = os.path.dirname(os.path.abspath(__file__))


def export_for_submission(
    output_dir: str = DIR,
    output_name: str = "tron_model.pt",
    mode: str = "tronbot",
    stationary: str = None,
    non_stationary: str = None,
):
    kw = {"output_dir": output_dir, "output_name": output_name, "mode": mode}
    if stationary:
        kw["stationary"] = stationary
    if non_stationary:
        kw["non_stationary"] = non_stationary
    return export_submission(**kw)
