import importlib.util
import os

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    _p = os.path.join(_d, "_path.py")
    if os.path.isfile(_p):
        _s = importlib.util.spec_from_file_location("tron_paper_path", _p)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m.setup_path(__file__)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent

from .encode import GRID, CHANNELS, encode_official, encode_stationary, encode_non_stationary
from .phase import agents_separated, approximate_survival_steps
from .stationary_env import StationaryTronEnv
from .duel_env import MRLDuelEnv
