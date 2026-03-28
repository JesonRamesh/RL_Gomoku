"""
Phase 2 Training: Self-Play with Sparse Rewards

The agent trains against a frozen copy of itself.
Every sync_frequency episodes, the frozen copy updates to match the agent.

Why this works:
    - The opponent is always the same strength as the agent (no skill gap).
    - Strategic patterns emerge naturally: simple 3-in-a-rows get blocked,
      so the agent must discover forks and deeper patterns to win.
    - Sparse rewards only — the win/loss signal stays clean and unbiased.
    - Difficulty scales automatically as the agent improves.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent


# Evaluation helpers

def evaluate_vs_random(agent, board_size, num_games=50):
    """
    Play num_games against RandomAgent and return the agent's win percentage.
    Used every eval_frequency episodes to check the agent has not regressed.
    Agent plays with epsilon=0 (pure exploitation, no random moves).
    """
    random_opponent = RandomAgent(player_id=-1)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0
    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)

        # Alternate who goes first for a fair measurement
        if game_num % 2 == 0:
            agent.player_id = 1
            random_opponent.player_id = -1
        else:
            agent.player_id = -1
            random_opponent.player_id = 1

        state = env.reset()
        done = False

        while not done:
            if env.logic.current_player == agent.player_id:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    wins += 1
            else:
                action = random_opponent.predict(state)
                state, _, done, _ = env.step(action)

    agent.epsilon = original_epsilon
    return (wins / num_games) * 100


def evaluate_vs_strategic(agent, board_size, skill_level, num_games=100):
    """
    Play num_games against StrategicAgent at a given skill level.
    Only used for the final evaluation after training is complete.
    Agent plays with epsilon=0 (pure exploitation, no random moves).
    """
    strategic_opponent = StrategicAgent(
        player_id=-1, skill_level=skill_level, board_size=board_size
    )
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0
    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)

        if game_num % 2 == 0:
            agent.player_id = 1
            strategic_opponent.player_id = -1
        else:
            agent.player_id = -1
            strategic_opponent.player_id = 1

        state = env.reset()
        done = False

        while not done:
            if env.logic.current_player == agent.player_id:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    wins += 1
            else:
                action = strategic_opponent.predict(state)
                state, _, done, _ = env.step(action)

    agent.epsilon = original_epsilon
    return (wins / num_games) * 100


# Training

def train_selfplay(num_episodes=6000, batch_size=32, train_frequency=4,
                   sync_frequency=250, save_frequency=500,
                   eval_frequency=100, eval_games=50,
                   board_size=9, save_dir="models_phase2"):
    """
    Self-play training with sparse rewards.

    The training agent plays against a frozen copy of itself.
    The frozen copy is updated (synced) every sync_frequency episodes.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the 20k baseline as the starting point for the training agent
    agent = DQNAgent(player_id=1, board_size=board_size)
    baseline_path = "models_baseline_9x9/dqn_baseline_final_20k.pt"
    agent.load_model(baseline_path)

    # Small epsilon: mostly exploit existing skills, explore a little
    agent.epsilon = 0.10
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9995

    # Create the frozen opponent from the same starting weights.
    # epsilon=0.0 so it always plays its best move — a consistent opponent.
    frozen_opponent = DQNAgent(player_id=-1, board_size=board_size)
    frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
    frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
    frozen_opponent.epsilon = 0.0

    # Tracking
    episode_rewards = []
    losses = []
    win_rates_vs_random = []
    sync_episodes = []

    print("=" * 60)
    print("PHASE 2: Self-Play Training (Sparse Rewards)")
    print("=" * 60)
    print(f"Baseline model:    {baseline_path}")
    print(f"Device:            {agent.device}")
    print(f"Starting epsilon:  {agent.epsilon}  to  {agent.epsilon_end}")
    print(f"Episodes:          {num_episodes}")
    print(f"Opponent sync:     every {sync_frequency} episodes")
    print(f"Rewards:           Sparse only (Win +1, Loss -1, Ongoing 0)")
    print("=" * 60 + "\n")

    start_time = time.time()
    step_count = 0
    best_win_rate_vs_random = 0.0

    for episode in range(num_episodes):

        # Alternate who goes first each episode
        if episode % 2 == 0:
            agent.player_id = 1
            frozen_opponent.player_id = -1
            agent_goes_first = True
        else:
            agent.player_id = -1
            frozen_opponent.player_id = 1
            agent_goes_first = False

        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        state = env.reset()
        episode_reward = 0.0
        done = False

        # If the frozen opponent goes first, let it make one move before the loop
        if not agent_goes_first:
            opp_action = frozen_opponent.predict(state)
            state, _, done, _ = env.step(opp_action)

        while not done:

            # Agent's turn
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            step_count += 1

            # Store experience — sparse reward only, no shaping
            agent.store_experience(state, action, reward, next_state, done)

            # Train every train_frequency steps
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

            if done:
                break

            # Frozen opponent's turn
            state = next_state
            opp_action = frozen_opponent.predict(state)
            next_state, _, done, _ = env.step(opp_action)
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Sync frozen opponent every sync_frequency episodes
        if (episode + 1) % sync_frequency == 0:
            frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
            frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
            sync_episodes.append(episode + 1)
            print(f"Frozen opponent updated at episode {episode + 1}")

        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode + 1:>5}/{num_episodes} | "
                  f"Reward: {avg_reward:>6.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # Evaluate vs RandomAgent every eval_frequency episodes
        if (episode + 1) % eval_frequency == 0:
            win_rate = evaluate_vs_random(agent, board_size, eval_games)
            win_rates_vs_random.append(win_rate)

            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            print(f"Win Rate vs Random:  {win_rate:.1f}%")
            print(f"Best vs Random:      {max(best_win_rate_vs_random, win_rate):.1f}%\n")

            if win_rate > best_win_rate_vs_random:
                best_win_rate_vs_random = win_rate
                agent.save_model(os.path.join(save_dir, "phase2_best.pt"))
                print(f"New best model saved ({win_rate:.1f}% vs Random)\n")

        # Save checkpoint every save_frequency episodes
        if (episode + 1) % save_frequency == 0:
            agent.save_model(os.path.join(save_dir, f"phase2_ep{episode + 1}.pt"))

    elapsed = time.time() - start_time

    # Final evaluation against all opponent types
    print("\n" + "=" * 60)
    print("Phase 2 Training Complete")
    print("=" * 60)
    print(f"Total Time: {elapsed / 3600:.2f} hours")
    print(f"Best Win Rate vs Random: {best_win_rate_vs_random:.1f}%")

    print("\nRunning final evaluations (100 games each) ...")
    final_vs_random = evaluate_vs_random(agent, board_size, num_games=100)
    final_vs_strategic_03 = evaluate_vs_strategic(agent, board_size, skill_level=0.3, num_games=100)
    final_vs_strategic_05 = evaluate_vs_strategic(agent, board_size, skill_level=0.5, num_games=100)

    print(f"\nFinal Results:")
    print(f"vs RandomAgent: {final_vs_random:.1f}%")
    print(f"vs StrategicAgent-0.3: {final_vs_strategic_03:.1f}%")
    print(f"vs StrategicAgent-0.5: {final_vs_strategic_05:.1f}%")
    print("=" * 60)

    agent.save_model(os.path.join(save_dir, "phase2_final.pt"))

    # Plot training results
    if win_rates_vs_random:
        eval_episodes = [(i + 1) * eval_frequency for i in range(len(win_rates_vs_random))]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(eval_episodes, win_rates_vs_random, marker="o", color="green", linewidth=2)
        plt.axhline(y=95, color="red", linestyle="--", label="Baseline (95%)")
        for ep in sync_episodes:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.title("Win Rate vs Random\n(orange = opponent sync points)")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(losses, alpha=0.3, color="blue")
        if len(losses) > 100:
            smoothed = np.convolve(losses, np.ones(100) / 100, mode="valid")
            plt.plot(smoothed, color="darkblue", label="Smoothed")
            plt.legend()
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        labels = ["Random\n(baseline)", "Random\n(phase 2)", "Strategic\n0.3", "Strategic\n0.5"]
        values = [95.0, final_vs_random, final_vs_strategic_03, final_vs_strategic_05]
        colors = ["gray", "green", "steelblue", "purple"]
        plt.bar(labels, values, color=colors, alpha=0.7)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% line")
        plt.ylabel("Win Rate (%)")
        plt.title("Final Results Comparison")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "phase2_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    train_selfplay(
        num_episodes=6000,
        batch_size=32,
        train_frequency=4,
        sync_frequency=250,
        save_frequency=500,
        eval_frequency=100,
        eval_games=50,
        board_size=9,
        save_dir="models_phase2"
    )
