import os
import sys
import time

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "official", "shared", "competition", "src"))

import requests
from competition.tron.tron import GameConfig, run_duel_game

MODEL = os.path.join(ROOT, "for_submission", "tron_model.pt")
PORTS = (8011, 8012)
BASE = "http://127.0.0.1"


def wait_health(port, timeout=30):
    url = f"{BASE}:{port}"
    sess = requests.Session()
    sess.trust_env = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = sess.get(f"{url}/health", timeout=1)
            if r.status_code == 200 and r.json().get("ok"):
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def smoke(port):
    url = f"{BASE}:{port}"
    sess = requests.Session()
    sess.trust_env = False
    gid = "smoke-test"
    grid = [[1 if x == 0 or x == 31 or y == 0 or y == 31 else 0 for x in range(32)] for y in range(32)]
    r = sess.post(f"{url}/game", json={
        "game_id": gid, "player_id": 0,
        "config": {"width": 32, "height": 32, "max_steps": 500, "num_players": 2},
        "grid": grid, "your_position": [1, 1], "your_direction": 2,
        "opponent_positions": [[30, 30]],
    }, timeout=5)
    r.raise_for_status()
    r = sess.post(f"{url}/move", json={
        "game_id": gid, "step": 0, "grid": grid,
        "your_position": [1, 1], "your_direction": 2, "your_alive": True,
        "opponent_positions": [[30, 30]], "opponent_alive": [True],
        "valid_actions": [1, 2, 3],
    }, timeout=60)
    r.raise_for_status()
    action = r.json()["action"]
    if action not in (1, 2, 3):
        raise RuntimeError(f"invalid action {action}")
    return action


def main():
    if not os.path.isfile(MODEL):
        raise FileNotFoundError(MODEL)
    for port in PORTS:
        if not wait_health(port):
            raise RuntimeError(f"launcher not up on {port} — start: python launch_tron_rl.py --model {MODEL} --port {port}")
    print(f"Model: {MODEL}")
    for port in PORTS:
        print(f"  health {port}: OK")
    action = smoke(PORTS[0])
    print(f"  /game + /move smoke: action={action}")
    result = run_duel_game(
        f"{BASE}:{PORTS[0]}",
        f"{BASE}:{PORTS[1]}",
        config=GameConfig(width=32, height=32, max_steps=500, num_players=2),
        seed=42,
        move_timeout=60.0,
        startup_health_check_timeout_in_seconds=30,
    )
    print(f"  duel: {result.game_result}  steps={result.steps}  winner={result.winner}")
    if "failed" in result.game_result.lower() or "Health check" in result.game_result:
        raise RuntimeError(result.game_result)
    print("PASS — submission works with launch_tron_rl.py")


if __name__ == "__main__":
    main()
