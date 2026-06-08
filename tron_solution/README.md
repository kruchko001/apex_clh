# Tron RL Solution

Reinforcement Learning agent for Bittensor Subnet 1 (Tron Light Cycles game).

## Features

- **Custom Gymnasium Environment**: `(5, 32, 32)` observation space with proper reward shaping
- **Lightweight CNN Model**: <5ms inference time, TorchScript compatible
- **Advanced Minimax Opponent**: Voronoi territory, wall-cut heuristics (same AI as `play_human.py`)
- **PPO Training**: Stable Baselines3, trains against minimax by default
- **Export Pipeline**: Easy export to TorchScript for deployment

## Installation

```bash
pip install gymnasium torch stable-baselines3 pygame numpy
```

## Usage

### Train

```bash
# Using CLI (trains vs minimax opponent, depth 10)
python main.py train --timesteps 100000 --verbose

# Or directly
python -m tron_solution.training.train_ppo --timesteps 100000 --opponent minimax --minimax-depth 10
```

### Export

```bash
# Train and export in one step
python main.py export --train_and_export

# Export existing model
python main.py export --model_path ./ppo_tron_checkpoints/tron_ppo_final_*.zip
```

### Test

```bash
python main.py test --model_path tron_model.pt --episodes 5
```

## Project Structure

```
tron_solution/
├── env/           # Gymnasium environment
│   ├── __init__.py
│   └── tron_env.py
├── model/         # CNN model
│   ├── __init__.py
│   └── tron_cnn.py
├── training/      # PPO training
│   ├── __init__.py
│   └── train_ppo.py
├── export/        # Model export
│   ├── __init__.py
│   └── export_model.py
└── test/          # Testing utilities
    ├── __init__.py
    └── test_model.py
```

## Environment Details

### Observation Space
- Shape: `(5, 32, 32)` float tensor
- Channels:
  1. Walls (border)
  2. My trail
  3. Opponent trail
  4. My head position
  5. Opponent head position

### Action Space
- `Discrete(4)`: UP, RIGHT, DOWN, LEFT

### Reward Shaping
- Clean kill (opponent hits my trail): **+3.0**
- Opponent self-destructs: **+2.0**
- Mutual destruction: **0.0**
- Timeout draw (500 steps): **-1.0**
- Die alone: **-3.0**
- Per-step (alive): **+0.002** survival + territory / voronoi / mobility shaping

## Model Architecture

```
Input: (batch, 5, 32, 32)
  ↓
Conv2d(5→16, 3x3) + ReLU + MaxPool(2x2)
  ↓
Conv2d(16→32, 3x3) + ReLU + MaxPool(2x2)
  ↓
Flatten → Linear(2048→128) + ReLU
  ↓
  ├─→ Actor: Linear(128→4)
  └─→ Critic: Linear(128→1)
```

## Deployment

The exported TorchScript model (`tron_model.pt`) can be loaded directly:

```python
import torch

model = torch.jit.load("tron_model.pt")
model.eval()

obs = torch.randn(1, 5, 32, 32)  # Your observation
with torch.no_grad():
    logits, value = model(obs)
    action = torch.argmax(logits, dim=-1)
```

## License

MIT
