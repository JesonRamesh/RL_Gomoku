# RL Gomoku

A reinforcement learning framework for developing and evaluating Gomoku (Five in a Row) agents on a 9×9 board.

## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.11, NumPy, Pygame, PyTorch

## Project Structure

```
agents/
├── base_agent.py          # BaseAgent abstract class
├── random_agent.py        # Uniform-random baseline agent
├── threatening_agent.py   # Curriculum opponent: blocks threats at tunable probability
├── strategic_agent.py     # Curriculum opponent: full offence + defence heuristics
├── dqn_simple_jeson.py    # DQN agent (Simple CNN — production model)
└── dqn_jeson.py           # DQN agent (Residual CNN — experimental architecture)

game/
├── logic.py               # Core Gomoku rules (default board size: 9×9)
├── gomoku_env.py          # RL environment wrapper (sparse + shaped rewards)
├── board.py               # Pygame visualisation (window: 650×550)
├── match.py               # Headless evaluation utility
└── threat_detector.py     # Standalone threat-detection helpers

train_sparse_jeson.py      # Stage 1: baseline training vs RandomAgent (sparse rewards)
train_phase1_shaped.py     # Stage 2: shaped-reward fine-tuning vs RandomAgent
train_phase2_continue.py   # Stage 3: curriculum vs ThreateningAgent
train_phase2_selfplay.py   # Stage 3 (alt): self-play training
train_phase3_mixed.py      # Stage 4: mixed-opponent curriculum
train_phase4_threeway.py   # Stage 5: three-way curriculum (Random + Threatening + Strategic)

evaluate_baseline.py       # Evaluate agent win-rate against baselines
evaluate_threatening.py    # Evaluate agent vs ThreateningAgent at various skill levels
test_agent.py              # Quick sanity-check script
progress_log.md            # Detailed training notes and results per stage

main.py                    # Entry point (visual PyGame mode or headless evaluation)
```

## Agent Architecture

### Simple DQN (`agents/dqn_simple_jeson.py`) — Production Model

The model that achieved 95–100% win rate vs RandomAgent and strong performance across curriculum stages.

```
Input: (batch, 3, 9, 9)
  Channel 0: Agent's own pieces
  Channel 1: Opponent's pieces
  Channel 2: Constant plane = player ID (+1 or -1)

Conv2D(3 → 64,  kernel=3, pad=1) + BatchNorm + ReLU
Conv2D(64 → 128, kernel=3, pad=1) + BatchNorm + ReLU
Conv2D(128 → 128, kernel=3, pad=1) + BatchNorm + ReLU
Flatten → FC(128×9×9 → 512) + BatchNorm + ReLU
FC(512 → 81)   ← Q-value for each board cell
```

Training algorithm: **Double DQN**

- Online network selects actions; target network evaluates them
- Replay buffer: 100,000 experiences
- Optimizer: Adam (lr = 1e-4), gamma = 0.99
- Gradient clipping: max norm 1.0
- Target network sync: every 1,000 steps

### Residual DQN (`agents/dqn_jeson.py`) — Experimental

Deeper architecture using residual blocks, developed as an alternative for stronger strategic play.

```
Input: (batch, 3, board_size, board_size)
Conv2D(3 → 128, kernel=3, pad=1) + ReLU
5 × ResidualBlock(128)
  └─ Conv2D(128→128) + ReLU → Conv2D(128→128) + skip connection
Conv2D(128 → 32, kernel=1) + ReLU
Flatten → FC(32×9×9 → 81)   ← Q-values
```

## Curriculum Opponents

| Agent              | File                          | Strategy                                               |
| ------------------ | ----------------------------- | ------------------------------------------------------ |
| `RandomAgent`      | `agents/random_agent.py`      | Uniform-random valid moves                             |
| `ThreateningAgent` | `agents/threatening_agent.py` | Blocks 4-in-a-row with configurable probability        |
| `StrategicAgent`   | `agents/strategic_agent.py`   | Wins, blocks, extends sequences, uses opening patterns |

### ThreateningAgent

Parameterised by `block_probability` (0.0–1.0). Only detects and blocks immediate 4-in-a-row threats. Designed for gradual curriculum learning.

```python
from agents.threatening_agent import ThreateningAgent
opp = ThreateningAgent(player_id=-1, block_probability=0.5, board_size=9)
```

### StrategicAgent

Priority: win immediately → block opponent win → extend 4-in-a-row → block 4-in-a-row → extend 3-in-a-row → opening pattern → random fallback. Parameterised by `skill_level` (0.0 = random, 1.0 = always strategic).

```python
from agents.strategic_agent import StrategicAgent
opp = StrategicAgent(player_id=-1, skill_level=0.8, board_size=9)
```

## Training Pipeline

Training progressed through multiple stages. All models target a **9×9 board** with a **5-in-a-row win condition**.

### Stage 1 — Sparse Rewards vs RandomAgent (`train_sparse_jeson.py`)

- Rewards: `+1` win, `-1` loss, `0` draw/ongoing
- Episodes: ~20,000
- Epsilon: 1.0 → 0.02
- Result: **95–97% win rate** vs RandomAgent

### Stage 2 — Shaped Rewards vs RandomAgent (`train_phase1_shaped.py`)

Fine-tunes the Stage 1 model with intermediate rewards:

- Created 3-in-a-row: `+0.15`
- Created 4-in-a-row: `+0.40`
- Blocked opponent 3-in-a-row: `+0.10`
- Blocked opponent 4-in-a-row: `+0.30`

### Stage 3 — Curriculum vs ThreateningAgent (`train_phase2_continue.py` / `train_phase2_selfplay.py`)

- Gradually increases opponent `block_probability`
- Also includes self-play variant for diversity

### Stage 4 — Mixed Opponents (`train_phase3_mixed.py`)

- Mix of RandomAgent and ThreateningAgent at varying skill levels
- Prevents over-fitting to a single opponent type

### Stage 5 — Three-Way Curriculum (`train_phase4_threeway.py`)

Final stage using RandomAgent, ThreateningAgent, and StrategicAgent simultaneously. Best model saved to `models_phase4_v2/phase4_best_strategic.pt`.

## Environment: GomokuEnv

`game/gomoku_env.py` wraps `GomokuLogic` with a standard RL interface.

```python
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv

env = GomokuEnv(GomokuLogic(board_size=9), use_sparse_rewards=True)

state = env.reset()
next_state, reward, done, info = env.step((row, col))
```

`use_sparse_rewards=True` (default): only terminal rewards (`±1`).
`use_sparse_rewards=False`: adds shaped intermediate rewards via `_evaluate_threat_value` and `_evaluate_blocking_move`.

## Usage

### Play Against the Agent (Visual Mode)

```bash
python main.py
```

This launches a PyGame window where you (Human) play against the trained DQN agent loaded from `models_phase4_v2/phase4_best_strategic.pt`.

### Headless Evaluation

```bash
python main.py --headless
```

Runs 100 games between two agents and prints win/loss/draw statistics.

### Training

```bash
python train_sparse_jeson.py        # Stage 1: baseline
python train_phase1_shaped.py       # Stage 2: shaped rewards
python train_phase2_continue.py     # Stage 3: curriculum
python train_phase3_mixed.py        # Stage 4: mixed
python train_phase4_threeway.py     # Stage 5: three-way curriculum
```

### AlphaZero Training (`trainAlphaZero.py`)

The AlphaZero script now supports a two-phase schedule so you can train with
stronger search first, then fine-tune for lower-latency deployment.

- Phase 1 (pretraining): use higher `--num-simulations` for cleaner MCTS targets
- Phase 2 (fine-tuning): switch to lower `--finetune-num-simulations` to adapt
  the network to your deployment search budget (for example 50 sims/move)

Example (recommended when deploying at 50 sims/move):

```bash
python trainAlphaZero.py \
  --iterations 100 \
  --games-per-iteration 35 \
  --epochs-per-iteration 3 \
  --batch-size 256 \
  --replay-size 120000 \
  --learning-rate 3e-4 \
  --num-simulations 100 \
  --finetune-start-iteration 71 \
  --finetune-num-simulations 50 \
  --finetune-learning-rate 1.5e-4 \
  --save-dir models_alphazero_two_phase \
  --save-every 10
```

Notes:

- If `--finetune-num-simulations` is omitted, training stays single-phase.
- If `--finetune-start-iteration` is omitted, it defaults to 70% of total
  iterations.
- Iteration logs now include phase (`P1`/`FT`) and active MCTS simulations.

### Evaluation

```bash
python evaluate_baseline.py         # Win rate vs RandomAgent
python evaluate_threatening.py      # Win rate vs ThreateningAgent
python test_agent.py                # Quick sanity check
```

## Developing Your Own Agent

Create `agents/my_agent.py` and inherit from `BaseAgent`:

```python
from agents.base_agent import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def __init__(self, player_id):
        super().__init__(player_id)

    def predict(self, board_state):
        """
        Return next move as (row, col) tuple.

        board_state: numpy array (9×9)
            1  = your pieces
            -1 = opponent pieces
            0  = empty cells
        """
        valid_moves = list(zip(*np.where(board_state == 0)))
        return valid_moves[0]  # Replace with your logic
```

## Key Classes

| Class                 | Location                     | Description                                      |
| --------------------- | ---------------------------- | ------------------------------------------------ |
| `BaseAgent`           | `agents/base_agent.py`       | Abstract base — implement `predict(board_state)` |
| `GomokuLogic`         | `game/logic.py`              | Game rules, `make_move()`, win detection         |
| `GomokuEnv`           | `game/gomoku_env.py`         | RL env — `reset()`, `step(action)`               |
| `DQNAgent` (simple)   | `agents/dqn_simple_jeson.py` | Production DQN agent                             |
| `DQNAgent` (residual) | `agents/dqn_jeson.py`        | Experimental deeper DQN agent                    |
| `eval_agents()`       | `game/match.py`              | Headless evaluation, alternates first move       |

## Notable Changes vs Baseline Repository

| Area         | Change                                                                              |
| ------------ | ----------------------------------------------------------------------------------- |
| Board size   | Default changed from 15×15 to **9×9**                                               |
| Window size  | Pygame window reduced from 900×700 to **650×550**                                   |
| `GomokuEnv`  | Added shaped reward methods, `use_sparse_rewards` flag                              |
| `main.py`    | Now loads trained DQN agent for human-vs-AI play                                    |
| `.gitignore` | Added `*.pt`, model directories, archive, and dev artefacts                         |
| New agents   | `dqn_jeson.py`, `dqn_simple_jeson.py`, `threatening_agent.py`, `strategic_agent.py` |
| New scripts  | Full training pipeline (5 stages) + evaluation scripts                              |

## Setup for UCL remote GPU access for training

Navigate to /scratch0/$USER
Clone Repo: https://github.com/JesonRamesh/RL_Gomoku.git
run: cd RL_Gomoku

```bash
cd /scratch0/$USER # replace $USER with user
git clone https://github.com/JesonRamesh/RL_Gomoku.git
cd RL_Gomoku
```

### uv setup

Configure ~/.bashrc

```bash
nano ~/.bashrc
```

```bash
# 1. Define your workspace on the big drive (ALL CAPS)
export WORKSPACE="/scratch0/$USER$"

# 2. Force Singularity to use Scratch for caching
export SINGULARITY_CACHEDIR="$WORKSPACE/.cache/singularity"
export SINGULARITY_TMPDIR="$WORKSPACE/.tmp"

# 3. Create the directories safely
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# 4. uv environment variables
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.local/share/uv/python"

# 5. Safely load the uv tool environment ONLY if it exists
if [ -f "$WORKSPACE/uv_tool/env" ]; then
    . "$WORKSPACE/uv_tool/env"
fi

. "/scratch0/$USER/uv_tool/env"

. "$HOME/.local/bin/env"

```

Download uv and refresh bash (recommend use bash)

```bash
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/scratch0/$USER$/uv_tool" sh
source ~/.bashrc
```

### Installing torch and other dependencies

Check cuda version (if using)

```bash
nvidia-smi
```

Add uv packages

```bash
uv sync
```

If for whatever reason the required dependencies are not added by default:

```bash
uv add pygame numpy matplotlib
```

```bash
uv add torch --index https://download.pytorch.org/whl/$CUDAVERSION # replace with cuda version e.g. cu126
```

#### If that fails copy this into pyproject.toml under dependencies:

```bash
[[tool.uv.index]]
name = "pytorch-cuda"
# Replace with whatever cuda version from nvidia-smi
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
```

Then run:

```bash
uv add torch
```

Activate uv environment

```bash
source .venv/bin/activate
```

### Setting up tmux for background processes

```bash
# Initialise tmux instance called training
tmux new -s training
```

Run training script e.g.

```bash
bash
source .venv/bin/activate
python trainAlphaZero.py
```

Press Ctrl + B, D to exit

```bash
# To reconnect
tmux attach -t training
```
