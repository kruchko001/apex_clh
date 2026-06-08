import os
import importlib.util
import argparse
from typing import Any, Dict, List, Optional

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
        raise ImportError("Could not locate tron_solution package root")
    _d = _parent

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from tron_solution.model.frame_stack import FrameStack
from tron_solution.model.obs import crop_sandbox_obs_np, to_sandbox_obs_np

WALL = 1
PLAYER_TRAIL_START = 2


def encode_state(
    grid: List[List[int]],
    player_id: int,
    my_position: List[int],
    opponent_positions: List[List[int]],
    opponent_alive: List[bool],
    height: int,
    width: int,
) -> np.ndarray:
    grid_arr = np.array(grid, dtype=np.int32)
    state = np.zeros((5, height, width), dtype=np.float32)
    state[0] = (grid_arr == WALL).astype(np.float32)
    state[1] = (grid_arr == PLAYER_TRAIL_START + player_id).astype(np.float32)
    opp = np.zeros_like(grid_arr, dtype=np.float32)
    for oid in range(8):
        if oid != player_id:
            opp += (grid_arr == PLAYER_TRAIL_START + oid).astype(np.float32)
    state[2] = np.clip(opp, 0.0, 1.0)
    my_y, my_x = my_position
    if 0 <= my_y < height and 0 <= my_x < width:
        state[3, my_y, my_x] = 1.0
    for opp_pos, alive in zip(opponent_positions, opponent_alive):
        if alive:
            oy, ox = opp_pos
            if 0 <= oy < height and 0 <= ox < width:
                state[4, oy, ox] = 1.0
    return state


class GameRequest(BaseModel):
    game_id: str
    player_id: int
    config: Dict[str, Any]
    grid: List[List[int]]
    your_position: List[int]
    your_direction: int
    opponent_positions: List[List[int]]


class MoveRequest(BaseModel):
    game_id: str
    step: int
    grid: List[List[int]]
    your_position: List[int]
    your_direction: int
    your_alive: bool
    opponent_positions: List[List[int]]
    opponent_alive: List[bool]
    valid_actions: List[int]


class MoveResponse(BaseModel):
    action: int


class GameSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    game_id: str
    player_id: int
    width: int
    height: int
    frame_stack: Optional[FrameStack] = None


def _pick_action(output: torch.Tensor, valid_actions: List[int], fallback: int) -> int:
    if output.dim() == 1:
        q = output
    elif output.dim() == 2:
        q = output.squeeze(0)
    else:
        q = output.flatten()[:4]
    masked = np.full(4, float("-inf"))
    q_np = q.detach().cpu().numpy()[:4]
    for a in valid_actions:
        if 0 <= a < 4:
            masked[a] = q_np[a]
    best = int(np.argmax(masked))
    if masked[best] == float("-inf"):
        return valid_actions[0] if valid_actions else fallback
    return best


def make_app(model_path: str, server_stack: bool = False) -> FastAPI:
    app = FastAPI(title="Tron RL Player API")
    device = torch.device("cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    sessions: Dict[str, GameSession] = {}

    @app.get("/health")
    def health():
        return {"ok": True, "model_path": model_path, "server_stack": server_stack}

    @app.post("/game")
    def game(req: GameRequest):
        sessions[req.game_id] = GameSession(
            game_id=req.game_id,
            player_id=req.player_id,
            width=req.config.get("width", 32),
            height=req.config.get("height", 32),
            frame_stack=FrameStack() if server_stack else None,
        )
        return {"ok": True}

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest):
        sess = sessions.get(req.game_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown game_id")
        if not req.your_alive or not req.valid_actions:
            return MoveResponse(action=req.your_direction)

        obs5 = encode_state(
            req.grid, sess.player_id, req.your_position,
            req.opponent_positions, req.opponent_alive, sess.height, sess.width,
        )
        grid = crop_sandbox_obs_np(obs5)
        if server_stack:
            fs = sess.frame_stack
            stacked = fs.reset(grid) if req.step == 0 else fs.step(grid)
            x = torch.from_numpy(stacked).unsqueeze(0).to(device)
        else:
            x = torch.from_numpy(to_sandbox_obs_np(grid, obs5[0])).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        return MoveResponse(action=_pick_action(out, req.valid_actions, req.your_direction))

    return app


def main():
    p = argparse.ArgumentParser(description="Launch Tron RL player (official sandbox API)")
    p.add_argument("--model", required=True, help="TorchScript .pt (stacked submission or raw core)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--server-stack", action="store_true",
                     help="Stack frames server-side for raw 16-channel core models")
    args = p.parse_args()
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    uvicorn.run(make_app(args.model, args.server_stack), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
