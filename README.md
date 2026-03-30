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

This opens a PyGame window where you play as Black (Player 1) against the Gen 3 Dueling DQN agent by default.

### Choosing Your Opponent

Use `--opponent-model` to pick a named preset (automatically selects the correct agent type and model path):

```bash
python main.py --opponent-model gen3   # Play vs Gen 3 DQN
python main.py --opponent-model gen2   # Play vs Gen 2 DQN
python main.py --opponent-model gen4   # Play vs Gen 4 AlphaZero
```

Or use `--opponent` to specify an agent type directly (no model file needed for non-neural agents):

```bash
python main.py --opponent random
python main.py --opponent minimax
python main.py --opponent strategic
```

**Available model presets:**

| Preset | Agent | Model file |
|--------|-------|------------|
| `gen2` | Gen 2 — Curriculum DQN | `Model/gen2_best.pt` |
| `gen3` | Gen 3 — Dueling DQN + PER | `Model/gen3_best.pt` |
| `gen4` | Gen 4 — AlphaZero (MCTS) | `Model/gen4_best.pt` |

List all presets at any time:
```bash
python main.py --list-models
```

---

### Watch Two AI Agents Play (Interactive UI)

Use `--player1-model` to replace the human with an AI for Player 1, combined with `--opponent-model` for Player 2:

```bash
python main.py --player1-model gen2 --opponent-model gen3
python main.py --player1-model gen3 --opponent-model gen4
```

You can also pit a model against a non-neural agent:

```bash
python main.py --player1-model gen3 --opponent random
python main.py --player1-model gen4 --opponent minimax
```

If `--player1-model` is omitted, Player 1 defaults to human as normal.

---

### Slowing Down AI Moves for Recording

Use `--move-delay` to insert a pause (in seconds) between each AI move — useful when screen recording:

```bash
python main.py --player1-model gen3 --opponent-model gen4 --move-delay 1.5
python main.py --player1-model gen2 --opponent random --move-delay 2.0
```

The default delay is `1.0` second. Set to `0` for no delay:

```bash
python main.py --player1-model gen3 --opponent-model gen4 --move-delay 0
```

---

### Headless Evaluation

```bash
python main.py --headless --num-games 100
```

Runs 100 games between two agents and prints win/draw/loss statistics. Configure agents using presets:

```bash
python main.py --headless --agent1-model gen2 --agent2-model gen3
python main.py --headless --agent1-model gen3 --agent2-model gen4 --num-games 50
```

Or against non-neural agents:

```bash
python main.py --headless --agent1-model gen2 --agent2 random
python main.py --headless --agent1-model gen4 --agent2 strategic
```

---

### Full CLI Reference (Interactive Mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--opponent-model` | — | Named preset for Player 2 (`gen2`, `gen3`, `gen4`) — auto-selects agent type |
| `--opponent` | `dqn` | Agent type for Player 2 (`dqn`, `dqn_simple`, `alphazero`, `minimax`, `strategic`, `random`) |
| `--player1-model` | — | Named preset for Player 1 AI — omit to play as human |
| `--move-delay` | `1.0` | Seconds to pause between AI moves |
| `--az-simulations` | `200` | MCTS simulations per move for AlphaZero opponent |
| `--board-size` | `9` | Board size |
| `--model-path` | — | Custom model path for opponent (overridden by `--opponent-model`) |

### Full CLI Reference (Headless Mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | — | Run without UI |
| `--num-games` | `100` | Number of games to play |
| `--agent1-model` | `gen3` | Named preset for agent 1 |
| `--agent2-model` | `gen4` | Named preset for agent 2 |
| `--agent1` | `dqn` | Agent type for agent 1 (if not using preset) |
| `--agent2` | `alphazero` | Agent type for agent 2 (if not using preset) |
| `--agent1-az-simulations` | `20` | MCTS simulations for agent 1 if AlphaZero |
| `--agent2-az-simulations` | `50` | MCTS simulations for agent 2 if AlphaZero |
| `--board-size` | `9` | Board size |

---

## Project Structure

```
Model/
├── gen2_best.pt           # Gen 2 best checkpoint (SimpleDQN, curriculum-trained)
├── gen3_best.pt          # Gen 3 final model (Dueling DQN + PER)
└── gen4_best.pt      # Gen 4 AlphaZero network weights

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

**Training (`train_sparse.py`):**
- Sparse rewards only: `+1` win, `-1` loss, `0` ongoing
- ~20,000 episodes vs RandomAgent
- Epsilon: 1.0 → 0.02
- **Result:** ~95–97% win rate vs RandomAgent

---

### Gen 2 — Curriculum DQN

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

Inherit from `BaseAgent` and implement `predict`:

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

---

## Submission Models Summary

| Model | File | Architecture | Win rate vs Random |
|-------|------|--------------|--------------------|
| Gen 2 | `submission_models/gen2_best.pt` | SimpleDQN (3-conv, Double DQN) | ~98% |
| Gen 3 | `submission_models/gen3_best.pt` | DQNetwork (4-conv, Dueling + PER) | ~100% |
| Gen 4 | `submission_models/gen4_best.pt` | AlphaZeroNet (ResNet + MCTS) | — |