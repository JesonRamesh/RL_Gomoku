"""
Phase 5: Defensive Awakening via Ignore Penalty

Goal: Teach the agent to block 4-in-a-row and winning threats mid-game.

Root cause of missing defense in phase4_best_strategic.pt:
    Sparse rewards (+1/-1 at terminal only) cannot credit-assign across 20+ moves.
    When the agent ignores a threat on move 8 and loses on move 22, the gradient
    update blames move 22, not move 8. The agent never learns "ignoring that threat
    caused my loss."

Fix: GomokuEnvShaped with positive_rewards=False — only the ignore penalty fires.
    - Ignoring a winning threat (4-in-a-row with open end): -0.3 immediately
    - Ignoring a 4-threat (single open end): -0.1 immediately
    - All other non-terminal rewards: 0 (same as sparse)
    - Terminal win/loss: ±1.0 (unchanged, still dominant)

Why this is safe (unlike Stage 2's failure):
    Stage 2 added POSITIVE offensive rewards → inflated offensive Q-values → agent
    chased shapes instead of winning. The ignore penalty is purely NEGATIVE — it only
    lowers Q-values for specific bad actions (ignoring threats). It does NOT inflate
    any Q-values. Result: Q(blocking) stays unchanged; Q(ignoring) goes down.
    Blocking becomes relatively better without corrupting offensive skills.

Changes from Phase 4 v2:
    - Environment: GomokuEnvShaped(positive_rewards=False) instead of GomokuEnv
    - Strategic ratio: 20% (up from 15%) at skill_level=0.5 (up from 0.3)
    - Self-play ratio: 55% (down from 60% to make room for more Strategic)
    - Everything else identical: 25% random, 500-ep warmup, sync every 500

Load:    models_phase4_v2/phase4_best_strategic.pt
Save:    models_phase5/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from game.gomoku_env_shaped import GomokuEnvShaped
from agents.dqn_simple import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent


# -------------------------------------------------------
# Evaluation helpers (always use sparse env for fair comparison)
# -------------------------------------------------------

def evaluate_vs_random(agent, board_size, num_games=50):
    random_opponent = RandomAgent(player_id=-1)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        agent.player_id = 1 if game_num % 2 == 0 else -1
        random_opponent.player_id = -agent.player_id
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
    strategic_opponent = StrategicAgent(player_id=-1, skill_level=skill_level, board_size=board_size)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for game_num in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        agent.player_id = 1 if game_num % 2 == 0 else -1
        strategic_opponent.player_id = -agent.player_id
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
# Play one episode using the SHAPED environment
# -------------------------------------------------------

def play_episode_shaped(agent, opponent, board_size):
    """
    Play one full episode using GomokuEnvShaped(positive_rewards=False).
    The agent receives the ignore penalty when it misses a critical block.
    The opponent always uses a plain GomokuLogic step (we only shape agent rewards).
    """
    game_logic = GomokuLogic(board_size=board_size)
    shaped_env = GomokuEnvShaped(game_logic, positive_rewards=False)

    state = shaped_env.reset()
    episode_reward = 0.0
    experiences = []
    done = False

    agent_goes_first = random.random() < 0.5
    if agent_goes_first:
        agent.player_id = 1
        opponent.player_id = -1
    else:
        agent.player_id = -1
        opponent.player_id = 1
        opp_action = opponent.predict(state)
        # Opponent steps through the same logic (rewards don't matter for opponent)
        state, _, done, _ = shaped_env.step(opp_action)

    while not done:
        action = agent.predict(state)
        next_state, reward, done, _ = shaped_env.step(action)
        episode_reward += reward
        experiences.append((state, action, reward, next_state, done))

        if done:
            break

        state = next_state
        opp_action = opponent.predict(state)
        next_state, _, done, _ = shaped_env.step(opp_action)
        state = next_state

    return episode_reward, experiences


# -------------------------------------------------------
# Opponent selection
# -------------------------------------------------------

def select_opponent(frozen, random_opp, strategic, self_play_ratio, random_ratio):
    roll = random.random()
    if roll < self_play_ratio:
        return frozen
    elif roll < self_play_ratio + random_ratio:
        return random_opp
    else:
        return strategic


def opponent_label(opponent, frozen, random_opp, strategic):
    if opponent is frozen:
        return "self-play"
    elif opponent is random_opp:
        return "random"
    else:
        return "strategic-0.5"


# -------------------------------------------------------
# Main training loop
# -------------------------------------------------------

def train_phase5(
    num_episodes=8000,
    warmup_episodes=500,
    batch_size=32,
    train_frequency=4,
    self_play_ratio=0.55,
    random_ratio=0.25,
    # strategic_ratio = 1 - 0.55 - 0.25 = 0.20 (StrategicAgent-0.5)
    sync_frequency=500,
    save_frequency=500,
    eval_frequency=500,
    eval_games=50,
    board_size=9,
    save_dir="models_phase5",
):
    strategic_ratio = 1.0 - self_play_ratio - random_ratio
    os.makedirs(save_dir, exist_ok=True)

    # Load best Phase 4 model
    agent = DQNAgent(player_id=1, board_size=board_size)
    start_model = "models_phase4_v2/phase4_best_strategic.pt"
    agent.load_model(start_model)
    agent.epsilon = 0.05
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9998

    # Frozen self-play opponent
    frozen_opponent = DQNAgent(player_id=-1, board_size=board_size)
    frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
    frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
    frozen_opponent.epsilon = 0.0

    random_opponent = RandomAgent(player_id=-1)
    strategic_opponent = StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size)

    episode_rewards = []
    losses = []
    win_rates_vs_random = []
    win_rates_vs_strategic_03 = []
    win_rates_vs_strategic_05 = []
    eval_episodes = []
    sync_episodes = []

    print("=" * 60)
    print("PHASE 5: Defensive Awakening (Ignore Penalty Only)")
    print("=" * 60)
    print(f"Load:             {start_model}")
    print(f"Device:           {agent.device}")
    print(f"Environment:      GomokuEnvShaped(positive_rewards=False)")
    print(f"Epsilon:          {agent.epsilon} → {agent.epsilon_end}")
    print(f"Warmup:           {warmup_episodes} episodes (no weight updates)")
    print(f"Training:         {num_episodes} episodes")
    print(f"Opponent split:   {int(self_play_ratio*100)}% self-play | "
          f"{int(random_ratio*100)}% random | "
          f"{int(strategic_ratio*100)}% StrategicAgent-0.5")
    print(f"Sync frequency:   every {sync_frequency} episodes")
    print(f"Save metric:      best vs StrategicAgent-0.5")
    print("=" * 60 + "\n")

    start_time = time.time()
    best_vs_random = 0.0
    best_vs_strategic_05 = 0.0
    best_vs_strategic_03 = 0.0

    # ---------------------------------------------------
    # WARMUP: fill buffer before any weight updates
    # ---------------------------------------------------
    print(f"--- Warmup: {warmup_episodes} episodes (no training) ---")
    for ep in range(warmup_episodes):
        opponent = select_opponent(frozen_opponent, random_opponent, strategic_opponent,
                                   self_play_ratio, random_ratio)
        _, experiences = play_episode_shaped(agent, opponent, board_size)
        for s, a, r, ns, d in experiences:
            agent.store_experience(s, a, r, ns, d)
        if (ep + 1) % 100 == 0:
            print(f"  Warmup {ep+1}/{warmup_episodes} | Buffer: {len(agent.replay_buffer)}")

    print(f"\nWarmup done. Buffer: {len(agent.replay_buffer)} experiences.\n")

    # ---------------------------------------------------
    # TRAINING
    # ---------------------------------------------------
    step_count = 0

    for episode in range(num_episodes):
        opponent = select_opponent(frozen_opponent, random_opponent, strategic_opponent,
                                   self_play_ratio, random_ratio)
        label = opponent_label(opponent, frozen_opponent, random_opponent, strategic_opponent)

        ep_reward, experiences = play_episode_shaped(agent, opponent, board_size)

        for s, a, r, ns, d in experiences:
            agent.store_experience(s, a, r, ns, d)
            step_count += 1
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

        # Sync frozen opponent
        if (episode + 1) % sync_frequency == 0:
            frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
            frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
            sync_episodes.append(episode + 1)
            print(f"  [Sync] Frozen opponent updated at episode {episode + 1}")

        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_l = np.mean(losses[-100:]) if losses else 0.0
            print(f"Ep {episode+1:>5}/{num_episodes} | "
                  f"Reward: {avg_r:>6.3f} | Loss: {avg_l:.4f} | "
                  f"Eps: {agent.epsilon:.3f} | Opp: {label}")

        # Evaluate every eval_frequency episodes
        if (episode + 1) % eval_frequency == 0:
            wr_rand = evaluate_vs_random(agent, board_size, eval_games)
            wr_s03  = evaluate_vs_strategic(agent, board_size, 0.3, eval_games)
            wr_s05  = evaluate_vs_strategic(agent, board_size, 0.5, eval_games)

            win_rates_vs_random.append(wr_rand)
            win_rates_vs_strategic_03.append(wr_s03)
            win_rates_vs_strategic_05.append(wr_s05)
            eval_episodes.append(episode + 1)

            print(f"\n{'='*50}")
            print(f"EVAL at episode {episode+1}")
            print(f"  vs Random:         {wr_rand:.1f}%   (Phase4 baseline: 98.5%)")
            print(f"  vs Strategic-0.3:  {wr_s03:.1f}%   (Phase4 baseline: 64.0%)")
            print(f"  vs Strategic-0.5:  {wr_s05:.1f}%   (Phase4 baseline: 43.0%)")

            # STOP SIGNAL — print clearly if something is wrong
            if wr_rand < 85.0:
                print(f"\n  *** WARNING: vs Random dropped to {wr_rand:.1f}% — "
                      f"below 85% threshold. Consider stopping. ***")
            if wr_s03 < 55.0 and episode + 1 >= 2000:
                print(f"  *** NOTE: vs Strategic-0.3 at {wr_s03:.1f}% — "
                      f"Phase4 baseline was 64%. Monitoring for regression. ***")
            print(f"{'='*50}\n")

            # Save best models
            if wr_rand > best_vs_random:
                best_vs_random = wr_rand
                agent.save_model(os.path.join(save_dir, "phase5_best_random.pt"))
                print(f"  [Save] New best vs Random: {wr_rand:.1f}%")

            if wr_s05 > best_vs_strategic_05:
                best_vs_strategic_05 = wr_s05
                agent.save_model(os.path.join(save_dir, "phase5_best_strategic05.pt"))
                print(f"  [Save] New best vs Strategic-0.5: {wr_s05:.1f}%")

            if wr_s03 > best_vs_strategic_03:
                best_vs_strategic_03 = wr_s03
                agent.save_model(os.path.join(save_dir, "phase5_best_strategic03.pt"))
                print(f"  [Save] New best vs Strategic-0.3: {wr_s03:.1f}%")

        # Checkpoint
        if (episode + 1) % save_frequency == 0:
            agent.save_model(os.path.join(save_dir, f"phase5_ep{episode+1}.pt"))

    elapsed = time.time() - start_time

    # Final save
    agent.save_model(os.path.join(save_dir, "phase5_final.pt"))

    print("\n" + "=" * 60)
    print("Phase 5 Training Complete")
    print("=" * 60)
    print(f"Time:                   {elapsed/60:.1f} min")
    print(f"Best vs Random:         {best_vs_random:.1f}%")
    print(f"Best vs Strategic-0.3:  {best_vs_strategic_03:.1f}%")
    print(f"Best vs Strategic-0.5:  {best_vs_strategic_05:.1f}%")

    # Full 200-game evaluation
    print("\nRunning final 200-game evaluation...")
    final_rand  = evaluate_vs_random(agent, board_size, 200)
    final_s03   = evaluate_vs_strategic(agent, board_size, 0.3, 200)
    final_s05   = evaluate_vs_strategic(agent, board_size, 0.5, 200)
    print(f"  vs Random:         {final_rand:.1f}%")
    print(f"  vs Strategic-0.3:  {final_s03:.1f}%")
    print(f"  vs Strategic-0.5:  {final_s05:.1f}%")
    print("=" * 60)

    # Plot
    if eval_episodes:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(eval_episodes, win_rates_vs_random, marker="o", color="green", linewidth=2)
        plt.axhline(y=98.5, color="gray", linestyle="--", label="Phase4 (98.5%)")
        plt.axhline(y=85.0, color="red",  linestyle=":", label="Min threshold (85%)")
        for ep in sync_episodes:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Random\n(orange = sync points)")
        plt.ylim([0, 105]); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(eval_episodes, win_rates_vs_strategic_03, marker="s",
                 color="steelblue", linewidth=2, label="vs S-0.3")
        plt.plot(eval_episodes, win_rates_vs_strategic_05, marker="^",
                 color="purple", linewidth=2, label="vs S-0.5")
        plt.axhline(y=64.0, color="steelblue", linestyle="--", alpha=0.5, label="Phase4 S-0.3 (64%)")
        plt.axhline(y=43.0, color="purple",    linestyle="--", alpha=0.5, label="Phase4 S-0.5 (43%)")
        for ep in sync_episodes:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Strategic Opponents")
        plt.ylim([0, 105]); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(losses, alpha=0.3, color="blue")
        if len(losses) > 100:
            smoothed = np.convolve(losses, np.ones(100) / 100, mode="valid")
            plt.plot(smoothed, color="darkblue", label="Smoothed (100-step)")
            plt.legend()
        plt.xlabel("Training Step"); plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "phase5_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8000,
                        help="Number of training episodes (default: 8000; use 1000 for a quick test)")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="models_phase5")
    args = parser.parse_args()

    train_phase5(
        num_episodes=args.episodes,
        warmup_episodes=args.warmup,
        save_dir=args.save_dir,
    )
