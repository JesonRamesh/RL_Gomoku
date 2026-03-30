# RL Gomoku — COMP0215 Coursework

A reinforcement learning project for playing Gomoku (Five in a Row) on a 9×9 board. The project progresses through four generations of agents, from a baseline DQN to an AlphaZero implementation, each building on the lessons of the previous.

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, NumPy, PyTorch, Pygame, Matplotlib

---

## Quick Start — Play Against an Agent

```bash
python main.py
```

This opens a PyGame window where you play as Black (Player 1) against the Gen 3 Dueling DQN agent by default. To switch to a different agent, open `main.py` and uncomment the relevant option block.

**Available opponents in `main.py`:**

| Option | Agent | Model file |
|--------|-------|------------|
| Option 1 (default) | Gen 3 — Dueling DQN + PER | `submission_models/gen3_final.pt` |
| Option 2 | Gen 2 — Curriculum DQN | `submission_models/gen2_best.pt` |
| Option 3 | Gen 4 — AlphaZero (MCTS) | `submission_models/gen4_alphazero.pt` |
| Option 4 | Minimax agent | — |
| Option 5 | Strategic heuristic agent | — |
| Option 6 | Random agent | — |

### Headless Evaluation

```bash
python main.py --headless --num-games 100
```

Runs 100 games between two agents (configured inside `main.py`) and prints win/draw/loss statistics.

---

## Project Structure

```
submission_models/
├── gen2_best.pt           # Gen 2 best checkpoint (SimpleDQN, curriculum-trained)
├── gen3_final.pt          # Gen 3 final model (Dueling DQN + PER)
└── gen4_alphazero.pt      # Gen 4 AlphaZero network weights

agents/
├── base_agent.py          # Abstract BaseAgent — implement predict(board_state)
├── random_agent.py        # Uniform-random baseline
├── strategic_agent.py     # Heuristic opponent: win/block/extend with tunable skill
├── minimax_agent.py       # Minimax search opponent with configurable depth
├── dqn_simple.py          # Gen 1 & 2 agent: Simple CNN + Double DQN
├── dqn_agent.py           # Gen 3 agent: Dueling DQN + Prioritized Replay
├── alphazero_net.py        # Gen 4 network: ResNet with policy + value heads
└── alphazero_mcts.py      # Gen 4 agent: PUCT MCTS wrapping AlphaZeroNet

game/
├── logic.py               # Core Gomoku rules and win detection
├── board.py               # PyGame board rendering and input handling
├── match.py               # Headless evaluation — eval_agents()
├── gomoku_env.py          # RL environment wrapper (sparse rewards)
└── gomoku_env_shaped.py   # RL environment with shaped intermediate rewards

Training scripts (Gen 1 → 4):
├── train_sparse.py            # Gen 1: baseline DQN vs RandomAgent
├── train_phase2_selfplay.py   # Gen 2 Phase A: self-play curriculum
├── train_phase3_mixed.py      # Gen 2 Phase B: mixed opponents (self-play + random)
├── train_phase4_threeway.py   # Gen 2 Phase C: three-way curriculum
├── trainDQN.py                # Gen 3: Dueling DQN with sparse rewards
├── train_shaped.py            # Gen 3: shaped rewards + adaptive curriculum
├── train_combined_v2.py       # Gen 3: combined best approach (final Gen 3 run)
└── train_alphazero.py         # Gen 4: AlphaZero self-play training

Evaluation:
├── evaluate_baseline.py       # Win rate vs RandomAgent and StrategicAgent
├── evaluate_checkpoints.py    # Evaluate a series of checkpoints
├── test_agent.py              # Quick sanity check for a single agent
└── plot_training_progress.py  # Generate training curve plot (Gen 1 & 2)

main.py                    # Entry point for visual and headless play
progress_log.md            # Training notes and results per stage
training_progress.png      # Win-rate learning curve (Gen 1 → Gen 2)
```

---

## Agent Generations

### Gen 1 — Baseline DQN

**Architecture (`agents/dqn_simple.py`):** Simple CNN with Double DQN.

```
Input: (batch, 3, 9, 9)
  Channel 0: Agent's own pieces
  Channel 1: Opponent's pieces
  Channel 2: Constant plane = player identity (+1 or -1)

Conv2D(3 → 64,   k=3, pad=1) + BatchNorm + ReLU
Conv2D(64 → 128, k=3, pad=1) + BatchNorm + ReLU
Conv2D(128 → 128, k=3, pad=1) + BatchNorm + ReLU
Flatten → FC(10368 → 512) + BatchNorm + ReLU
FC(512 → 81)   ← Q-value for each board cell
```

<<<<<<< HEAD
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
=======
**Training (`train_sparse.py`):**
- Sparse rewards only: `+1` win, `-1` loss, `0` ongoing
- ~20,000 episodes vs RandomAgent
>>>>>>> submission
- Epsilon: 1.0 → 0.02
- **Result:** ~95–97% win rate vs RandomAgent

---

<<<<<<< HEAD
Fine-tunes the Stage 1 model with intermediate rewards:

- Created 3-in-a-row: `+0.15`
- Created 4-in-a-row: `+0.40`
- Blocked opponent 3-in-a-row: `+0.10`
- Blocked opponent 4-in-a-row: `+0.30`
=======
### Gen 2 — Curriculum DQN
>>>>>>> submission

Same SimpleDQN architecture, trained in three progressive phases.

**Phase A — Self-Play (`train_phase2_selfplay.py`):**
- Loads the Gen 1 model and continues training against a frozen copy of itself
- Develops more strategic play beyond what random opponents expose

**Phase B — Mixed Opponents (`train_phase3_mixed.py`):**
- 70% frozen self-play, 30% RandomAgent
- Buffer warmup: 500 episodes before weight updates begin
- Prevents the strategy collapse seen in pure self-play

**Phase C — Three-Way Curriculum (`train_phase4_threeway.py`):**
- 60% self-play, 25% RandomAgent, 15% StrategicAgent-0.3
- Tracks a separate best-strategic checkpoint
- **Submission model:** `submission_models/gen2_best.pt`

```bash
# Reproduce Gen 2 training (run in order):
python train_phase2_selfplay.py
python train_phase3_mixed.py
python train_phase4_threeway.py
```

---

### Gen 3 — Dueling DQN with Prioritized Experience Replay

**Architecture (`agents/dqn_agent.py`):** Deeper CNN with Dueling DQN heads and PER.

```
Input: (batch, 3, 9, 9)

Conv2D(3 → 64,   k=3, pad=1) + BatchNorm + ReLU
Conv2D(64 → 128,  k=3, pad=1) + BatchNorm + ReLU  ─┐ residual
Conv2D(128 → 128, k=3, pad=1) + BatchNorm + ReLU  ─┘
Conv2D(128 → 128, k=3, pad=1) + BatchNorm + ReLU
Flatten (→ 10368)

Value stream:   FC(10368 → 256) + ReLU → FC(256 → 1)
Advantage stream: FC(10368 → 256) + ReLU → FC(256 → 81)
Q(s,a) = V(s) + [A(s,a) − mean(A(s,·))]
```

**Key improvements over Gen 2:**
- **Dueling architecture:** Separates state value from per-action advantage — more stable learning
- **Prioritized Experience Replay (PER):** Samples by TD error; important transitions replayed more often
- **Shaped rewards:** Intermediate rewards for creating/blocking 3-in-a-row and 4-in-a-row sequences

**Training:**
```bash
python trainDQN.py            # Sparse rewards, strong baseline
python train_shaped.py        # Shaped rewards + adaptive curriculum
python train_combined_v2.py   # Combined approach (produced the submission model)
```

**Submission model:** `submission_models/gen3_final.pt`
**Result:** ~100% win rate vs RandomAgent, strong performance vs StrategicAgent

---

### Gen 4 — AlphaZero

**Architecture (`agents/alphazero_net.py`):** ResNet tower with dual policy and value heads.

```
Input: (batch, 3, 9, 9)

Conv2D(3 → 128, k=3, pad=1) + BatchNorm + ReLU      ← input stem
6 × ResidualBlock(128 filters)
  └─ Conv→BN→ReLU→Conv→BN + skip connection→ReLU

Policy head:
  Conv2D(128 → 2, k=1) + BN + ReLU → Flatten → FC(162 → 81) → log-softmax

Value head:
  Conv2D(128 → 1, k=1) + BN + ReLU → Flatten → FC(81 → 64) + ReLU → FC(64 → 1) → tanh
```

**Agent (`agents/alphazero_mcts.py`):** PUCT MCTS using the network for both move selection and position evaluation.
- Selection: UCB with prior probabilities from policy head
- Evaluation: value head replaces rollouts
- Default: 400 simulations per move

**Training (`train_alphazero.py`):**
- Self-play with Dirichlet noise for exploration
- Network updated from MCTS visit counts (policy target) and game outcomes (value target)

**Submission model:** `submission_models/gen4_alphazero.pt`

```bash
python train_alphazero.py
```

> **Note:** The Gen 4 model must be loaded via `AlphaZeroNet` (not `AlphaZeroAgent`). See Option 3 in `main.py` for the correct loading pattern.

---

## Environment

`game/gomoku_env.py` wraps `GomokuLogic` with a standard RL interface:

```python
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv

env = GomokuEnv(GomokuLogic(board_size=9), use_sparse_rewards=True)
state = env.reset()                          # Returns (3, 9, 9) numpy array
next_state, reward, done, info = env.step((row, col))
```

`game/gomoku_env_shaped.py` extends this with intermediate rewards used during Gen 3 training:

| Event | Reward |
|-------|--------|
| Win | +1.0 |
| Loss | −1.0 |
| Create 4-in-a-row | +0.40 |
| Block opponent 4-in-a-row | +0.30 |
| Create 3-in-a-row | +0.15 |
| Block opponent 3-in-a-row | +0.10 |
| Ignore a critical threat | −0.25 |

---

## Evaluation Scripts

```bash
# Win rate vs RandomAgent and StrategicAgent at multiple skill levels
python evaluate_baseline.py

# Evaluate a list of checkpoint files and print a results table
python evaluate_checkpoints.py

# Quick sanity check — loads a model and plays 10 games
python test_agent.py

# Generate the Gen 1 → Gen 2 training progress plot
python plot_training_progress.py
```

---

## Writing Your Own Agent

<<<<<<< HEAD
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
=======
Inherit from `BaseAgent` and implement `predict`:
>>>>>>> submission

```python
from agents.base_agent import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def __init__(self, player_id):
        super().__init__(player_id)

    def predict(self, board_state: np.ndarray) -> tuple:
        """
        board_state: (9, 9) numpy array
            +1 = your pieces,  -1 = opponent pieces,  0 = empty
        Returns: (row, col) move
        """
        valid_moves = list(zip(*np.where(board_state == 0)))
        return valid_moves[0]  # replace with your logic
```

Then plug it into `main.py` or `eval_agents()` from `game/match.py`.

<<<<<<< HEAD
| Class                 | Location                     | Description                                      |
| --------------------- | ---------------------------- | ------------------------------------------------ |
| `BaseAgent`           | `agents/base_agent.py`       | Abstract base — implement `predict(board_state)` |
| `GomokuLogic`         | `game/logic.py`              | Game rules, `make_move()`, win detection         |
| `GomokuEnv`           | `game/gomoku_env.py`         | RL env — `reset()`, `step(action)`               |
| `DQNAgent` (simple)   | `agents/dqn_simple_jeson.py` | Production DQN agent                             |
| `DQNAgent` (residual) | `agents/dqn_jeson.py`        | Experimental deeper DQN agent                    |
| `eval_agents()`       | `game/match.py`              | Headless evaluation, alternates first move       |
=======
---
>>>>>>> submission

## Submission Models Summary

<<<<<<< HEAD
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
=======
| Model | File | Architecture | Win rate vs Random |
|-------|------|--------------|--------------------|
| Gen 2 | `submission_models/gen2_best.pt` | SimpleDQN (3-conv, Double DQN) | ~98% |
| Gen 3 | `submission_models/gen3_final.pt` | DQNetwork (4-conv, Dueling + PER) | ~100% |
| Gen 4 | `submission_models/gen4_alphazero.pt` | AlphaZeroNet (ResNet + MCTS) | — |
>>>>>>> submission
