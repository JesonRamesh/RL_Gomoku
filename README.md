# RL Gomoku

A reinforcement learning framework for developing and evaluating Gomoku (Five in a Row) agents.

## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.11, NumPy, Pygame

## Project Structure

```
agents/              # Agent implementations
├── base_agent.py    # BaseAgent class to inherit from
└── random_agent.py  # Example agent

game/
├── logic.py         # Core Gomoku game logic
├── gomoku_env.py    # RL environment wrapper (for training)
└── match.py         # Agent evaluation utilities

main.py              # Entry point (visual/headless modes)
```

## Developing Your RL Agent

### 1. Create Your Agent Class

Create `agents/my_agent.py` and inherit from `BaseAgent`:

```python
from agents.base_agent import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def __init__(self, player_id):
        super().__init__(player_id)
        # Initialize your model/network here
    
    def predict(self, board_state):
        """
        Return next move as (row, col) tuple.
        
        board_state: numpy array
            - 1 = player 1's pieces
            - -1 = player 2's pieces
            - 0 = empty cells
        """
        valid_moves = list(zip(*np.where(board_state == 0)))
        if not valid_moves:
            return None
        
        # Your RL logic here
        return valid_moves[0]  # Replace with your model's prediction
```

### 2. Training Your Agent

Use `GomokuEnv` for training (standard RL interface):

```python
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv

env = GomokuEnv(GomokuLogic(board_size=15))

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = your_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        your_agent.update(state, action, reward, next_state, done)
        state = next_state
```

**Create training/evaluation scripts in `agents/` directory:**
- `agents/train_dqn.py` - Your training script
- `agents/eval_during_training.py` - Evaluation during training

This keeps training code organized and separate from final evaluation.

### 3. Final Agent Comparison

Use `game/match.py` for standardized final evaluation:

```python
from game.match import eval_agents
from agents.my_agent import MyAgent
from agents.random_agent import RandomAgent

my_agent = MyAgent(player_id=1)
random_agent = RandomAgent(player_id=-1)

# Run evaluation (returns dict with wins/losses/draws)
results = eval_agents(my_agent, random_agent, num_games=100)
# {'agent1_wins': 65, 'agent2_wins': 30, 'draws': 5}

print(f"Win rate: {results['agent1_wins'] / 100 * 100:.1f}%")
```

**Features:**
- Headless (fast)
- Fair (alternates first move)
- Returns win/loss/draw statistics

## Usage

### Visual Mode
```bash
python main.py
```

### Headless Evaluation
```bash
python main.py --headless
```

## Key Classes

- **`BaseAgent`**: Inherit and implement `predict(board_state) -> (row, col)`
- **`GomokuEnv`**: RL environment with `reset()` and `step(action)`
- **`eval_agents()`**: Final evaluation function in `game/match.py`

## Workflow Summary

1. **Training**: Create scripts in `agents/` directory using `GomokuEnv`
2. **Training Evaluation**: Use custom eval scripts in `agents/` for progress tracking
3. **Final Comparison**: Use `game/match.py` for standardized agent evaluation
