import importlib.util
import os

_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("tron_paper_bh_path", os.path.join(_d, "_path.py"))
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
_m.setup_path(__file__)
