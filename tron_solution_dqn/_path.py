import os
import sys


def setup_path(caller_file):
    d = os.path.dirname(os.path.abspath(caller_file))
    while True:
        if os.path.basename(d) == "tron_solution_dqn":
            root = os.path.dirname(d)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
        parent = os.path.dirname(d)
        if parent == d:
            raise ImportError("Could not locate tron_solution_dqn package root")
        d = parent
