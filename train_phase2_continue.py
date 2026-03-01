"""
Phase 2 Continuation: Extended Self-Play

Continues self-play training from phase2_final.pt with two adjustments:
    1. Loads the improved Phase 2 model instead of the baseline.
    2. Sync frequency increased from 250 → 500 episodes.
       This gives the agent twice as long to consolidate each improvement
       before the frozen opponent catches up, producing more meaningful games.

Target: 50%+ win rate vs StrategicAgent-0.3 (up from 40% after Phase 2).

Files created: models_phase2_continue/
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
from agents.strategic_agent import StrategicAgent


# -------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------

def evaluate_vs_random(agent, board_size, num_games=50):
    """
    Play num_games against RandomAgent and return the agent's win percentage.
    Used every eval_frequency episodes to confirm no regression.
    Agent plays with epsilon=0 (pure exploitation).
    """
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
    """
    Play num_games against StrategicAgent at a given skill level.
    Only used for the final evaluation after training is complete.
    Agent plays with epsilon=0 (pure exploitation).
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


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_selfplay_continued(num_episodes=8000, batch_size=32, train_frequency=4,
                              sync_frequency=500, save_frequency=500,
                              eval_frequency=100, eval_games=50,
                              board_size=9, save_dir="models_phase2_continue"):
    """
    Continue self-play training from the Phase 2 final model.

    Key change from Phase 2: sync_frequency increased from 250 → 500.
    The frozen opponent now updates half as often, giving the training agent
    more time to develop and exploit a skill advantage before the opponent
    catches up. This creates more informative games and stronger gradients.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the Phase 2 model as the starting point
    agent = DQNAgent(player_id=1, board_size=board_size)
    start_model_path = "models_phase2/phase2_final.pt"
    agent.load_model(start_model_path)

    # Keep epsilon low — the agent already has strong baseline skills
    agent.epsilon = 0.05
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9998

    # Frozen opponent starts from the same weights, always plays greedily
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
    print("PHASE 2 CONTINUATION: Extended Self-Play")
    print("=" * 60)
    print(f"Starting model:    {start_model_path}")
    print(f"Device:            {agent.device}")
    print(f"Starting epsilon:  {agent.epsilon}  ->  {agent.epsilon_end}")
    print(f"Episodes:          {num_episodes}")
    print(f"Opponent sync:     every {sync_frequency} episodes  (was 250)")
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

            # Store experience — sparse reward only
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
            print(f"  [Sync] Frozen opponent updated at episode {episode + 1}")

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
                agent.save_model(os.path.join(save_dir, "phase2_continue_best.pt"))
                print(f"New best model saved ({win_rate:.1f}% vs Random)\n")

        # Save checkpoint every save_frequency episodes
        if (episode + 1) % save_frequency == 0:
            agent.save_model(os.path.join(save_dir, f"phase2_continue_ep{episode + 1}.pt"))

    elapsed = time.time() - start_time

    # Final evaluation against all opponent types
    print("\n" + "=" * 60)
    print("Phase 2 Continuation Training Complete")
    print("=" * 60)
    print(f"Total Time: {elapsed / 3600:.2f} hours")
    print(f"Best Win Rate vs Random: {best_win_rate_vs_random:.1f}%")

    print("\nRunning final evaluations (100 games each) ...")
    final_vs_random = evaluate_vs_random(agent, board_size, num_games=100)
    final_vs_strategic_03 = evaluate_vs_strategic(agent, board_size, skill_level=0.3, num_games=100)
    final_vs_strategic_05 = evaluate_vs_strategic(agent, board_size, skill_level=0.5, num_games=100)

    print(f"\nFinal Results:")
    print(f"  vs RandomAgent:        {final_vs_random:.1f}%   (Phase 2 was 95%)")
    print(f"  vs StrategicAgent-0.3: {final_vs_strategic_03:.1f}%   (Phase 2 was 40%, target 50%+)")
    print(f"  vs StrategicAgent-0.5: {final_vs_strategic_05:.1f}%   (Phase 2 was 10%)")
    print("=" * 60)

    agent.save_model(os.path.join(save_dir, "phase2_continue_final.pt"))

    # Plot training results
    if win_rates_vs_random:
        eval_episodes = [(i + 1) * eval_frequency for i in range(len(win_rates_vs_random))]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(eval_episodes, win_rates_vs_random, marker="o", color="green", linewidth=2)
        plt.axhline(y=95, color="gray", linestyle="--", label="Phase 2 (95%)")
        plt.axhline(y=90, color="red", linestyle=":", label="Min acceptable (90%)")
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
        labels = ["Random\n(phase 2)", "Random\n(continue)", "Strategic\n0.3", "Strategic\n0.5"]
        phase2_values = [95.0, final_vs_random, final_vs_strategic_03, final_vs_strategic_05]
        prev_values = [95.0, 95.0, 40.0, 10.0]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, prev_values, width, label="Phase 2", color="steelblue", alpha=0.7)
        plt.bar(x + width / 2, phase2_values, width, label="Continuation", color="green", alpha=0.7)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% target")
        plt.xticks(x, labels)
        plt.ylabel("Win Rate (%)")
        plt.title("Phase 2 vs Continuation")
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "phase2_continue_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    train_selfplay_continued(
        num_episodes=8000,
        batch_size=32,
        train_frequency=4,
        sync_frequency=500,
        save_frequency=500,
        eval_frequency=100,
        eval_games=50,
        board_size=9,
        save_dir="models_phase2_continue"
    )
