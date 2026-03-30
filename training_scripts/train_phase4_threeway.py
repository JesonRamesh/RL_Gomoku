"""
Phase 4 Training: Three-Way Mixed Opponents (Self-Play + Random + Strategic-0.3)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent


# Evaluation helpers
def evaluate_vs_random(agent, board_size, num_games=50):
    """Play num_games against RandomAgent and return the agent's win percentage."""
    random_opponent = RandomAgent(player_id=-1)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0
    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)

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
    """Play num_games against StrategicAgent at a given skill level."""
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


# Helper: play one episode against a given opponent
def play_episode(agent, opponent, board_size):
    """
    Play one full episode between the agent and the given opponent.
    Returns (episode_reward, list_of_experiences).
    """
    game_logic = GomokuLogic(board_size=board_size)
    env = GomokuEnv(game_logic, use_sparse_rewards=True)
    state = env.reset()
    episode_reward = 0.0
    experiences = []
    done = False

    # Randomly decide who goes first
    agent_goes_first = random.random() < 0.5
    if agent_goes_first:
        agent.player_id = 1
        opponent.player_id = -1
    else:
        agent.player_id = -1
        opponent.player_id = 1
        opp_action = opponent.predict(state)
        state, _, done, _ = env.step(opp_action)

    while not done:
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        experiences.append((state, action, reward, next_state, done))

        if done:
            break

        state = next_state
        opp_action = opponent.predict(state)
        next_state, _, done, _ = env.step(opp_action)
        state = next_state

    return episode_reward, experiences


# Training
def train_threeway(num_episodes=6000, warmup_episodes=500, batch_size=32,
                   train_frequency=4, self_play_ratio=0.60, random_ratio=0.25,
                   sync_frequency=500, save_frequency=500,
                   eval_frequency=100, eval_games=50, eval_strategic_frequency=500,
                   board_size=9, save_dir="models_phase4_v2"):

    assert self_play_ratio + random_ratio <= 1.0, "Ratios must sum to at most 1.0"
    strategic_ratio = 1.0 - self_play_ratio - random_ratio

    os.makedirs(save_dir, exist_ok=True)

    # Load the best Phase 3 model as our starting point
    agent = DQNAgent(player_id=1, board_size=board_size)
    start_model_path = "models_phase3/phase3_best.pt"
    agent.load_model(start_model_path)

    agent.epsilon = 0.05
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9998

    # Frozen self-play opponent — same weights, always greedy
    frozen_opponent = DQNAgent(player_id=-1, board_size=board_size)
    frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
    frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
    frozen_opponent.epsilon = 0.0

    # Random opponent (skill anchor)
    random_opponent = RandomAgent(player_id=-1)

    # Strategic opponent (direct exposure to rule-based patterns)
    strategic_opponent = StrategicAgent(player_id=-1, skill_level=0.3, board_size=board_size)

    # Tracking
    episode_rewards = []
    losses = []
    win_rates_vs_random = []
    win_rates_vs_strategic = []
    strategic_eval_episodes = []
    sync_episodes = []

    print("=" * 60)
    print("PHASE 4: Three-Way Mixed Opponent Training")
    print("=" * 60)
    print(f"Starting model: {start_model_path}")
    print(f"Device: {agent.device}")
    print(f"Starting epsilon: {agent.epsilon}  ->  {agent.epsilon_end}")
    print(f"Warmup episodes: {warmup_episodes} (no weight updates)")
    print(f"Training episodes: {num_episodes}")
    print(f"Opponent split: {int(self_play_ratio*100)}% self-play  |  "
          f"{int(random_ratio*100)}% RandomAgent  |  "
          f"{int(strategic_ratio*100)}% StrategicAgent-0.3")
    print(f"Sync frequency: every {sync_frequency} episodes")
    print(f"Strategic eval: every {eval_strategic_frequency} episodes")
    print(f"Rewards: Sparse only (Win +1, Loss -1, Ongoing 0)")
    print("=" * 60 + "\n")

    start_time = time.time()

    # WARMUP PHASE: fill the buffer before training begins
    print(f" Warmup Phase: {warmup_episodes} episodes (no weight updates) ")

    for episode in range(warmup_episodes):
        opponent = _select_opponent(
            frozen_opponent, random_opponent, strategic_opponent,
            self_play_ratio, random_ratio
        )
        _, experiences = play_episode(agent, opponent, board_size)

        for state, action, reward, next_state, done in experiences:
            agent.store_experience(state, action, reward, next_state, done)

        if (episode + 1) % 100 == 0:
            print(f"  Warmup episode {episode + 1}/{warmup_episodes} | "
                  f"Buffer size: {len(agent.replay_buffer)}")

    print(f"\nWarmup complete. Buffer contains {len(agent.replay_buffer)} experiences.")
    print("Starting training...\n")

    # TRAINING PHASE
    step_count = 0
    best_win_rate_vs_random = 0.0
    best_win_rate_vs_strategic = 0.0

    for episode in range(num_episodes):

        opponent = _select_opponent(
            frozen_opponent, random_opponent, strategic_opponent,
            self_play_ratio, random_ratio
        )
        opponent_label = _opponent_label(
            opponent, frozen_opponent, random_opponent, strategic_opponent
        )

        episode_reward, experiences = play_episode(agent, opponent, board_size)

        for state, action, reward, next_state, done in experiences:
            agent.store_experience(state, action, reward, next_state, done)
            step_count += 1

            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Sync the frozen opponent every sync_frequency episodes
        if (episode + 1) % sync_frequency == 0:
            frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
            frozen_opponent.target_network.load_state_dict(
                agent.target_network.state_dict()
            )
            sync_episodes.append(episode + 1)
            print(f"Frozen opponent updated at episode {episode + 1}")

        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode + 1:>5}/{num_episodes} | "
                  f"Reward: {avg_reward:>6.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Last: {opponent_label}")

        # Evaluate vs RandomAgent every eval_frequency episodes
        if (episode + 1) % eval_frequency == 0:
            win_rate = evaluate_vs_random(agent, board_size, eval_games)
            win_rates_vs_random.append(win_rate)

            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            print(f"Win Rate vs Random:  {win_rate:.1f}%")
            print(f"Best vs Random:      {max(best_win_rate_vs_random, win_rate):.1f}%\n")

            if win_rate > best_win_rate_vs_random:
                best_win_rate_vs_random = win_rate
                agent.save_model(os.path.join(save_dir, "phase4_best.pt"))
                print(f"New best model saved ({win_rate:.1f}% vs Random)\n")

        # Evaluate vs Strategic-0.3 every eval_strategic_frequency episodes
        if (episode + 1) % eval_strategic_frequency == 0:
            strat_rate = evaluate_vs_strategic(
                agent, board_size, skill_level=0.3, num_games=50
            )
            win_rates_vs_strategic.append(strat_rate)
            strategic_eval_episodes.append(episode + 1)
            print(f"\n--- Strategic Eval at Episode {episode + 1} ---")
            print(f"Win Rate vs Strategic-0.3: {strat_rate:.1f}%")
            print(f"Best vs Strategic-0.3:     {max(best_win_rate_vs_strategic, strat_rate):.1f}%\n")

            if strat_rate > best_win_rate_vs_strategic:
                best_win_rate_vs_strategic = strat_rate
                agent.save_model(os.path.join(save_dir, "phase4_best_strategic.pt"))
                print(f"New best strategic model saved ({strat_rate:.1f}% vs Strategic-0.3)\n")

        # Save checkpoint every save_frequency episodes
        if (episode + 1) % save_frequency == 0:
            agent.save_model(os.path.join(save_dir, f"phase4_ep{episode + 1}.pt"))

    elapsed = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("Phase 4 Training Complete")
    print("=" * 60)
    print(f"Total Time: {elapsed / 3600:.2f} hours")
    print(f"Best Win Rate vs Random: {best_win_rate_vs_random:.1f}%")
    print(f"Best Win Rate vs Strategic-0.3: {best_win_rate_vs_strategic:.1f}%")

    print("\nRunning final evaluations (100 games each) ...")
    final_vs_random = evaluate_vs_random(agent, board_size, num_games=100)
    final_vs_strategic_03 = evaluate_vs_strategic(
        agent, board_size, skill_level=0.3, num_games=100
    )
    final_vs_strategic_05 = evaluate_vs_strategic(
        agent, board_size, skill_level=0.5, num_games=100
    )

    print(f"\nFinal Results:")
    print(f"vs RandomAgent:        {final_vs_random:.1f}%")
    print(f"vs StrategicAgent-0.3: {final_vs_strategic_03:.1f}%")
    print(f"vs StrategicAgent-0.5: {final_vs_strategic_05:.1f}%")
    print("=" * 60)

    agent.save_model(os.path.join(save_dir, "phase4_final.pt"))

    # Plot training results
    if win_rates_vs_random:
        eval_episodes = [(i + 1) * eval_frequency for i in range(len(win_rates_vs_random))]

        plt.figure(figsize=(16, 4))

        # Left: win rate vs Random
        plt.subplot(1, 4, 1)
        plt.plot(eval_episodes, win_rates_vs_random, marker="o", color="green", linewidth=2)
        plt.axhline(y=98, color="gray", linestyle="--", label="Phase 3 (98%)")
        plt.axhline(y=88, color="red", linestyle=":", label="Min acceptable (88%)")
        for ep in sync_episodes:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.title("Win Rate vs Random\n(orange = sync points)")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True)

        # Second: win rate vs Strategic-0.3 over training
        plt.subplot(1, 4, 2)
        if win_rates_vs_strategic:
            plt.plot(strategic_eval_episodes, win_rates_vs_strategic,
                     marker="s", color="purple", linewidth=2)
            plt.axhline(y=42, color="gray", linestyle="--", label="Phase 3 (42%)")
            plt.axhline(y=50, color="red", linestyle=":", label="50% target")
            for ep in sync_episodes:
                plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.title("Win Rate vs Strategic-0.3\n(orange = sync points)")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True)

        # Third: training loss
        plt.subplot(1, 4, 3)
        plt.plot(losses, alpha=0.3, color="blue")
        if len(losses) > 100:
            smoothed = np.convolve(losses, np.ones(100) / 100, mode="valid")
            plt.plot(smoothed, color="darkblue", label="Smoothed")
            plt.legend()
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)

        # Right: final comparison bars
        plt.subplot(1, 4, 4)
        labels = ["vs Random", "vs S-0.3", "vs S-0.5"]
        phase3_values = [98.0, 42.0, 28.0]
        phase4_values = [final_vs_random, final_vs_strategic_03, final_vs_strategic_05]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, phase3_values, width, label="Phase 3", color="steelblue", alpha=0.7)
        plt.bar(x + width / 2, phase4_values, width, label="Phase 4 v2", color="green", alpha=0.7)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% target")
        plt.xticks(x, labels)
        plt.ylabel("Win Rate (%)")
        plt.title("Phase 3 vs Phase 4 v2")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "phase4_v2_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to {plot_path}")
        plt.close()

    return agent


# Opponent selection helpers
def _select_opponent(frozen, random_opp, strategic, self_play_ratio, random_ratio):
    """Sample one opponent from the three-way pool using the configured ratios."""
    roll = random.random()
    if roll < self_play_ratio:
        return frozen
    elif roll < self_play_ratio + random_ratio:
        return random_opp
    else:
        return strategic


def _opponent_label(opponent, frozen, random_opp, strategic):
    if opponent is frozen:
        return "self-play"
    elif opponent is random_opp:
        return "random"
    else:
        return "strategic"


if __name__ == "__main__":
    train_threeway(
        num_episodes=6000,
        warmup_episodes=500,
        batch_size=32,
        train_frequency=4,
        self_play_ratio=0.60,
        random_ratio=0.25,    
        sync_frequency=500,
        save_frequency=500,
        eval_frequency=100,
        eval_games=50,
        eval_strategic_frequency=500,
        board_size=9,
        save_dir="models_phase4_v2"
    )
