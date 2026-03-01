"""
Phase 1 Training: Shaped Rewards vs RandomAgent

Continues from the 20k baseline model, adding intermediate (shaped)
rewards to teach the agent to build and block threats during the game.

Why this matters:
    The baseline only received +1 or -1 at the very end of a 30-move game.
    The agent had no idea which individual move was good or bad.
    With shaped rewards, the agent gets feedback DURING the game for every
    3-in-a-row it builds or blocks — a much clearer learning signal.

Files created: models_phase1/
Files modified: None
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple_jeson import DQNAgent
from agents.random_agent import RandomAgent


# -------------------------------------------------------
# Shaped Reward Helpers
# -------------------------------------------------------

def count_consecutive(board, row, col, player):
    """
    Count the longest consecutive sequence of `player` stones
    that passes through (row, col) in any of the 4 directions.

    Assumes the stone at (row, col) already belongs to `player`.
    Returns 1 if the stone is completely isolated.
    """
    board_size = board.shape[0]
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_count = 1

    for dr, dc in directions:
        count = 1  # Start with the stone we just placed

        # Count forward in this direction
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r += dr
            c += dc

        # Count backward in this direction
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc

        max_count = max(max_count, count)

    return max_count


def count_blocked_opponent(board, row, col, opponent):
    """
    Count the longest opponent sequence that runs through (row, col).

    Our stone now sits at (row, col). We check how many opponent stones
    are adjacent in each direction to see what threat we just cut off.
    Returns the highest connected count in any single direction.
    """
    board_size = board.shape[0]
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_blocked = 0

    for dr, dc in directions:
        count = 0

        # Count opponent stones forward (not including our stone at row, col)
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == opponent:
            count += 1
            r += dr
            c += dc

        # Count opponent stones backward
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == opponent:
            count += 1
            r -= dr
            c -= dc

        max_blocked = max(max_blocked, count)

    return max_blocked


def compute_shaped_reward(board, row, col, player):
    """
    Compute an intermediate reward for placing a stone at (row, col).

    Offensive rewards (building our own threats):
        3-in-a-row created: +0.15
        4-in-a-row created: +0.40

    Defensive rewards (blocking opponent threats):
        Blocked opponent 3-in-a-row: +0.10
        Blocked opponent 4-in-a-row: +0.30

    These are sized to be meaningful next to terminal ±1.0 win/loss rewards
    but small enough not to override the main winning objective.

    Note: this is only called when the game is still ongoing (not done).
    """
    reward = 0.0
    opponent = -player

    # Offensive: reward for building a longer sequence
    own_length = count_consecutive(board, row, col, player)
    if own_length == 3:
        reward += 0.15
    elif own_length == 4:
        reward += 0.40

    # Defensive: reward for blocking an opponent sequence
    blocked_length = count_blocked_opponent(board, row, col, opponent)
    if blocked_length == 3:
        reward += 0.10
    elif blocked_length >= 4:
        reward += 0.30

    return reward


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_agent(agent, opponent, board_size, num_games=50):
    """
    Play num_games games and return the agent's win percentage.
    Agent plays with epsilon=0 (pure exploitation, no random moves).
    Opponent alternates going first to get a fair measurement.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0

    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)

        # Alternate who goes first each game
        if game_num % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
        else:
            agent.player_id = -1
            opponent.player_id = 1

        state = env.reset()
        done = False

        while not done:
            if env.logic.current_player == agent.player_id:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    wins += 1
            else:
                action = opponent.predict(state)
                state, _, done, _ = env.step(action)

    agent.epsilon = original_epsilon
    return (wins / num_games) * 100


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_phase1(num_episodes=5000, batch_size=32, train_frequency=4,
                 save_frequency=500, eval_frequency=100, eval_games=50,
                 board_size=9, save_dir="models_phase1"):
    """
    Phase 1: Continue training from the 20k baseline with shaped rewards.

    Key differences from baseline training:
        - Loads the existing model (does not train from scratch)
        - Starts with low epsilon (agent exploits what it already knows)
        - Adds shaped intermediate rewards on top of win/loss signals
        - Still trains against RandomAgent (safe, no curriculum gap)

    What we expect to see:
        - Win rate stays above 95% vs Random (no regression)
        - Agent starts building 3-in-a-row patterns proactively
        - Shapes a better learned value function for Phase 2 (self-play)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the trained 20k baseline model
    agent = DQNAgent(player_id=1, board_size=board_size)
    baseline_path = "models_baseline_9x9/dqn_baseline_final_20k.pt"
    agent.load_model(baseline_path)

    # Low epsilon: we want the agent to mostly exploit what it knows,
    # not randomly explore the board again from scratch.
    agent.epsilon = 0.05
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9995

    opponent = RandomAgent(player_id=-1)

    # Tracking
    episode_rewards = []
    win_rates = []
    losses = []

    print("=" * 60)
    print("PHASE 1: Shaped Reward Training vs RandomAgent")
    print("=" * 60)
    print(f"Baseline model:    {baseline_path}")
    print(f"Device:            {agent.device}")
    print(f"Starting epsilon:  {agent.epsilon}  ->  {agent.epsilon_end}")
    print(f"Episodes:          {num_episodes}")
    print(f"Offensive rewards: 3-in-row +0.15 | 4-in-row +0.40")
    print(f"Defensive rewards: block-3  +0.10 | block-4  +0.30")
    print("=" * 60 + "\n")

    start_time = time.time()
    step_count = 0
    best_win_rate = 0.0

    for episode in range(num_episodes):

        # Alternate who goes first each episode
        if episode % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
            agent_goes_first = True
        else:
            agent.player_id = -1
            opponent.player_id = 1
            agent_goes_first = False

        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        state = env.reset()
        episode_reward = 0.0
        done = False

        # If the opponent goes first, let them make one move before the loop
        if not agent_goes_first:
            opponent_action = opponent.predict(state)
            state, _, done, _ = env.step(opponent_action)

        while not done:
            action = agent.predict(state)
            row, col = action
            next_state, terminal_reward, done, _ = env.step(action)

            # Compute shaped reward only while the game is still ongoing.
            # When done=True, terminal_reward is already ±1.0 — no shaping needed.
            shaped = 0.0
            if not done:
                shaped = compute_shaped_reward(next_state, row, col, agent.player_id)

            total_reward = terminal_reward + shaped
            episode_reward += total_reward
            step_count += 1

            # Store the combined reward in the replay buffer
            agent.store_experience(state, action, total_reward, next_state, done)

            # Train every train_frequency steps
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

            if done:
                break

            # Opponent's turn
            state = next_state
            opponent_action = opponent.predict(state)
            next_state, _, done, _ = env.step(opponent_action)
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode + 1:>5}/{num_episodes} | "
                  f"Reward: {avg_reward:>6.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # Evaluate every eval_frequency episodes
        if (episode + 1) % eval_frequency == 0:
            win_rate = evaluate_agent(agent, opponent, board_size, eval_games)
            win_rates.append(win_rate)

            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            print(f"Win Rate vs Random: {win_rate:.1f}%")
            print(f"Best so far:        {max(best_win_rate, win_rate):.1f}%\n")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save_model(os.path.join(save_dir, "phase1_best.pt"))
                print(f"New best model saved ({win_rate:.1f}%)\n")

        # Save checkpoint every save_frequency episodes
        if (episode + 1) % save_frequency == 0:
            agent.save_model(os.path.join(save_dir, f"phase1_ep{episode + 1}.pt"))

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Phase 1 Training Complete")
    print(f"Best Win Rate vs Random: {best_win_rate:.1f}%")
    print(f"Total Time: {elapsed / 3600:.2f} hours")
    print("=" * 60)

    agent.save_model(os.path.join(save_dir, "phase1_final.pt"))

    # Plot training results
    if win_rates:
        eval_episodes = [(i + 1) * eval_frequency for i in range(len(win_rates))]

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(eval_episodes, win_rates, marker="o", color="green", linewidth=2)
        plt.axhline(y=95, color="red", linestyle="--", label="Baseline (95%)")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.title("Win Rate vs RandomAgent")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(losses, alpha=0.4, color="blue")
        if len(losses) > 100:
            smoothed = np.convolve(losses, np.ones(100) / 100, mode="valid")
            plt.plot(smoothed, color="darkblue", label="Smoothed")
            plt.legend()
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "phase1_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    train_phase1(
        num_episodes=5000,
        batch_size=32,
        train_frequency=4,
        save_frequency=500,
        eval_frequency=100,
        eval_games=50,
        board_size=9,
        save_dir="models_phase1"
    )
