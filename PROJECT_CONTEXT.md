# Project Context: Bittensor Subnet 1 RL Tron Miner

## Goal
Build a Reinforcement Learning (PPO) agent to play the game "Tron" (Lightcycles) for the Bittensor Subnet 1 competition. 

## Strict Technical Constraints (DO NOT VIOLATE)
1. **Input Shape:** The model MUST accept a tensor of shape `(1, 5, 32, 32)`.
   - Channel 0: Walls (1.0 where wall, 0.0 elsewhere)
   - Channel 1: My trail (1.0 where my trail exists)
   - Channel 2: Opponent trail (1.0 where opponent trail exists)
   - Channel 3: My head position (1.0 at current cell)
   - Channel 4: Opponent head position (1.0 at opponent's current cell)
2. **Output Shape:** The model MUST output a tensor of shape `(4,)` representing raw logits for `[UP, RIGHT, DOWN, LEFT]`.
3. **Inference Speed:** The sandbox is CPU-only. Inference MUST take < 0.05 seconds. Use a very shallow, lightweight CNN (e.g., 2-3 Conv2d layers with MaxPool, followed by a small Linear layer).
4. **Export Format:** The final submission MUST be a TorchScript `.pt` file exported via `torch.jit.script()` or `torch.jit.trace()`. The PyTorch code must be strictly compatible with this (no dynamic Python lists, no complex control flow in `forward()`).
5. **Action Masking:** The platform's launcher handles action masking (e.g., preventing 180-degree turns). The model only needs to output raw logits; do not build masking logic into the PyTorch `forward()` pass.

## RL & Reward Design
- **Algorithm:** PPO (Proximal Policy Optimization) with an Actor-Critic CNN architecture.
- **Reward Shaping (per step):**
  - Clean kill (I live, opponent hits my trail): +2.0
  - Opponent self-destructs (hits wall/own trail): +1.5
  - Mutual destruction (head-on or simultaneous trail kill): +0.5 (Better than dying alone)
  - Timeout draw (both alive at 500 steps): 0.0
  - Die alone (hit wall or own trail): -2.0
  - Step survival: +0.01 (to encourage staying alive)

## Development Workflow
1. Create a custom `gymnasium.Env` that perfectly mimics these rules.
2. Define the lightweight PyTorch `nn.Module` (Actor-Critic).
3. Train locally using Stable Baselines3 (or a custom CleanRL loop).
4. Extract the trained PyTorch `nn.Module` and export it to `tron_model.pt` using `torch.jit.script()`.
5. Verify the `.pt` file loads and runs inference on a random `(1, 5, 32, 32)` tensor in < 0.05s.

## Current Task
The user wants to start with Step 1: Write the custom `gymnasium.Env` for this Tron game, ensuring the observation space is `(5, 32, 32)` and the action space is `Discrete(4)`.