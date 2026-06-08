import os
import subprocess
from typing import Optional

TRONBOT_TO_ACTION = {1: 0, 2: 1, 3: 2, 4: 3}


def default_tronbot_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tronbot", "cpp"))
    for name in ("MyTronBot.exe", "MyTronBot"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return os.path.join(root, "MyTronBot.exe")


def encode_tronbot_map(tron_env, as_player: int) -> str:
    if as_player == 0:
        self_head, opp_head = tron_env.my_head, tron_env.opponent_head
    else:
        self_head, opp_head = tron_env.opponent_head, tron_env.my_head
    h = w = tron_env.grid_size
    lines = [f"{w} {h}"]
    for r in range(h):
        row = []
        for c in range(w):
            if tron_env.walls[r, c]:
                row.append("#")
            elif self_head == (r, c):
                row.append("1")
            elif opp_head == (r, c):
                row.append("2")
            elif tron_env.my_trail[r, c] or tron_env.opponent_trail[r, c]:
                row.append("#")
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines) + "\n"


class TronBotPlayer:
    def __init__(self, bot_path: str = None, as_player: int = 0, move_timeout: float = 5.0, use_timer: bool = True):
        self.bot_path = bot_path or default_tronbot_path()
        self.as_player = as_player
        self.move_timeout = move_timeout
        self.use_timer = use_timer

    def start(self):
        if not os.path.isfile(self.bot_path):
            raise FileNotFoundError(
                f"TronBot binary not found at {self.bot_path}. "
                f"Run: powershell -File tronbot/cpp/build.ps1 -Fast"
            )

    def _fallback_action(self, tron_env) -> int:
        if self.as_player == 0:
            head, direction = tron_env.my_head, tron_env.current_direction
            trail, other = tron_env.my_trail, tron_env.opponent_trail
        else:
            head, direction = tron_env.opponent_head, tron_env.opponent_direction
            trail, other = tron_env.opponent_trail, tron_env.my_trail
        reverse = tron_env.OPPOSITE[direction]
        for action in (direction, (direction + 1) % 4, (direction + 3) % 4, reverse):
            if action == reverse:
                continue
            dr, dc = tron_env.DIRECTIONS[action]
            r, c = head[0] + dr, head[1] + dc
            if r < 0 or r >= tron_env.grid_size or c < 0 or c >= tron_env.grid_size:
                continue
            if tron_env.walls[r, c] or trail[r, c] or other[r, c]:
                continue
            return action
        return direction

    def get_action(self, tron_env) -> int:
        if not os.path.isfile(self.bot_path):
            self.start()
        payload = encode_tronbot_map(tron_env, self.as_player)
        no_timer = "0" if self.use_timer else "1"
        proc = subprocess.Popen(
            [self.bot_path, "0", no_timer],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=self.move_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return self._fallback_action(tron_env)
        if proc.returncode != 0 or not stdout.strip():
            hint = stderr.strip()[-200:] if stderr and stderr.strip() else f"exit code {proc.returncode}"
            if proc.returncode == 3221225781:
                hint = "missing DLL (rebuild: powershell -File tronbot/cpp/build.ps1 -Fast)"
            elif not os.path.isfile(self.bot_path):
                hint = "missing binary; run: powershell -File tronbot/cpp/build.ps1 -Fast"
            if not getattr(self, "_warned", False):
                print(f"TronBot move failed ({hint}), using fallback")
                self._warned = True
            return self._fallback_action(tron_env)
        try:
            move = int(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return self._fallback_action(tron_env)
        if move not in TRONBOT_TO_ACTION:
            return self._fallback_action(tron_env)
        return TRONBOT_TO_ACTION[move]

    def close(self):
        pass
