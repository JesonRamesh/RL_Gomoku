"""
Phase 5b: Defensive Continuation — Corrected Approach

Loads from Phase 5's best checkpoint (phase5_best_random.pt: 99.5% / 78.0% / 51.0%).

What Phase 5 taught us:
    - The ignore penalty approach WORKS — early checkpoints improved +12pp vs S-0.3
    - Peak performance occurs at ~500–1000 episodes; longer runs collapse
    - Collapse mechanism: frozen opponent absorbs defensive training over many syncs,
      making self-play too hard for the 25% random anchor to stabilise
    - Two variables were changed simultaneously (Strategic % AND skill) — broke the rule

Corrections (one variable at a time):
    - Random anchor: 25% → 30% (restored to Phase 3's proven minimum)
    - Strategic ratio: 20% → 15% (reduced to safe level)
    - Strategic skill: kept at 0.5 (ONE variable changed vs Phase 4)
    - Episodes: 8,000 → 3,000 (stop before the collapse window)
    - Eval frequency: 500 → 250 (catch the peak before it passes)

Load:    models_phase5/phase5_best_random.pt
Save:    models_phase5b/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from game.gomoku_env_shaped import GomokuEnvShaped
from agents.dqn_simple_jeson import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent


def evaluate_vs_random(agent, board_size, num_games=50):
    opponent = RandomAgent(player_id=-1)
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for g in range(num_games):
        logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(logic, use_sparse_rewards=True)
        agent.player_id = 1 if g % 2 == 0 else -1
        opponent.player_id = -agent.player_id
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
    agent.epsilon = orig_eps
    return wins / num_games * 100


def evaluate_vs_strategic(agent, board_size, skill, num_games=50):
    opponent = StrategicAgent(player_id=-1, skill_level=skill, board_size=board_size)
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for g in range(num_games):
        logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(logic, use_sparse_rewards=True)
        agent.player_id = 1 if g % 2 == 0 else -1
        opponent.player_id = -agent.player_id
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
    agent.epsilon = orig_eps
    return wins / num_games * 100


def play_episode_shaped(agent, opponent, board_size):
    logic = GomokuLogic(board_size=board_size)
    env = GomokuEnvShaped(logic, positive_rewards=False)
    state = env.reset()
    episode_reward = 0.0
    experiences = []
    done = False

    agent_goes_first = random.random() < 0.5
    agent.player_id = 1 if agent_goes_first else -1
    opponent.player_id = -agent.player_id

    if not agent_goes_first:
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


def select_opponent(frozen, random_opp, strategic, self_play_ratio, random_ratio):
    roll = random.random()
    if roll < self_play_ratio:
        return frozen
    elif roll < self_play_ratio + random_ratio:
        return random_opp
    return strategic


def train_phase5b(
    num_episodes=3000,
    warmup_episodes=500,
    batch_size=32,
    train_frequency=4,
    self_play_ratio=0.55,
    random_ratio=0.30,      # restored to Phase 3's proven anchor
    # strategic_ratio = 0.15 at skill=0.5
    sync_frequency=500,
    eval_frequency=250,     # was 500 — catch peaks before they pass
    eval_games=50,
    board_size=9,
    save_dir="models_phase5b",
):
    strategic_ratio = 1.0 - self_play_ratio - random_ratio
    os.makedirs(save_dir, exist_ok=True)

    agent = DQNAgent(player_id=1, board_size=board_size)
    start_model = "models_phase5/phase5_best_random.pt"
    agent.load_model(start_model)
    agent.epsilon = 0.05
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.9998

    frozen_opponent = DQNAgent(player_id=-1, board_size=board_size)
    frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
    frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
    frozen_opponent.epsilon = 0.0

    random_opponent = RandomAgent(player_id=-1)
    strategic_opponent = StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size)

    episode_rewards, losses = [], []
    wr_random, wr_s03, wr_s05 = [], [], []
    eval_eps, sync_eps = [], []

    print("=" * 60)
    print("PHASE 5b: Defensive Continuation (Corrected)")
    print("=" * 60)
    print(f"Load:            {start_model}")
    print(f"Device:          {agent.device}")
    print(f"Environment:     GomokuEnvShaped(positive_rewards=False)")
    print(f"Epsilon:         {agent.epsilon} → {agent.epsilon_end}")
    print(f"Warmup:          {warmup_episodes} episodes")
    print(f"Training:        {num_episodes} episodes")
    print(f"Opponent split:  {int(self_play_ratio*100)}% self-play | "
          f"{int(random_ratio*100)}% random | "
          f"{int(strategic_ratio*100)}% StrategicAgent-0.5")
    print(f"Eval frequency:  every {eval_frequency} episodes")
    print("=" * 60 + "\n")

    start_time = time.time()
    best_rand, best_s03, best_s05 = 0.0, 0.0, 0.0

    # WARMUP
    print(f"--- Warmup: {warmup_episodes} episodes ---")
    for ep in range(warmup_episodes):
        opp = select_opponent(frozen_opponent, random_opponent, strategic_opponent,
                              self_play_ratio, random_ratio)
        _, experiences = play_episode_shaped(agent, opp, board_size)
        for s, a, r, ns, d in experiences:
            agent.store_experience(s, a, r, ns, d)
        if (ep + 1) % 100 == 0:
            print(f"  Warmup {ep+1}/{warmup_episodes} | Buffer: {len(agent.replay_buffer)}")
    print(f"\nWarmup done. Buffer: {len(agent.replay_buffer)}\n")

    step_count = 0

    for episode in range(num_episodes):
        opp = select_opponent(frozen_opponent, random_opponent, strategic_opponent,
                              self_play_ratio, random_ratio)
        ep_r, experiences = play_episode_shaped(agent, opp, board_size)

        for s, a, r, ns, d in experiences:
            agent.store_experience(s, a, r, ns, d)
            step_count += 1
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

        agent.decay_epsilon()
        episode_rewards.append(ep_r)

        if (episode + 1) % sync_frequency == 0:
            frozen_opponent.q_network.load_state_dict(agent.q_network.state_dict())
            frozen_opponent.target_network.load_state_dict(agent.target_network.state_dict())
            sync_eps.append(episode + 1)
            print(f"  [Sync] Frozen opponent updated at episode {episode + 1}")

        if (episode + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_l = np.mean(losses[-100:]) if losses else 0.0
            print(f"Ep {episode+1:>5}/{num_episodes} | "
                  f"Reward: {avg_r:>6.3f} | Loss: {avg_l:.4f} | Eps: {agent.epsilon:.3f}")

        if (episode + 1) % eval_frequency == 0:
            r = evaluate_vs_random(agent, board_size, eval_games)
            s03 = evaluate_vs_strategic(agent, board_size, 0.3, eval_games)
            s05 = evaluate_vs_strategic(agent, board_size, 0.5, eval_games)
            wr_random.append(r); wr_s03.append(s03); wr_s05.append(s05)
            eval_eps.append(episode + 1)

            print(f"\n{'='*55}")
            print(f"EVAL at episode {episode+1}")
            print(f"  vs Random:        {r:.1f}%   (Phase5 start: 99.5%)")
            print(f"  vs Strategic-0.3: {s03:.1f}%   (Phase5 start: 78.0%)")
            print(f"  vs Strategic-0.5: {s05:.1f}%   (Phase5 start: 51.0%)")

            if r < 90.0:
                print(f"\n  *** STOP WARNING: vs Random at {r:.1f}% < 90% threshold ***")
            if s03 < 70.0 and episode + 1 >= 1000:
                print(f"  *** REGRESSION: vs S-0.3 at {s03:.1f}% < 70% (Phase5 best was 78%) ***")
            print(f"{'='*55}\n")

            if r > best_rand:
                best_rand = r
                agent.save_model(os.path.join(save_dir, "phase5b_best_random.pt"))
                print(f"  [Save] New best vs Random: {r:.1f}%")
            if s05 > best_s05:
                best_s05 = s05
                agent.save_model(os.path.join(save_dir, "phase5b_best_s05.pt"))
                print(f"  [Save] New best vs S-0.5: {s05:.1f}%")
            if s03 > best_s03:
                best_s03 = s03
                agent.save_model(os.path.join(save_dir, "phase5b_best_s03.pt"))
                print(f"  [Save] New best vs S-0.3: {s03:.1f}%")

        if (episode + 1) % 500 == 0:
            agent.save_model(os.path.join(save_dir, f"phase5b_ep{episode+1}.pt"))

    agent.save_model(os.path.join(save_dir, "phase5b_final.pt"))
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Phase 5b Complete")
    print("=" * 60)
    print(f"Time:                  {elapsed/60:.1f} min")
    print(f"Best vs Random:        {best_rand:.1f}%")
    print(f"Best vs S-0.3:         {best_s03:.1f}%")
    print(f"Best vs S-0.5:         {best_s05:.1f}%")
    print("\nRunning final 200-game evaluation on last model...")
    fr = evaluate_vs_random(agent, board_size, 200)
    fs03 = evaluate_vs_strategic(agent, board_size, 0.3, 200)
    fs05 = evaluate_vs_strategic(agent, board_size, 0.5, 200)
    print(f"  vs Random:        {fr:.1f}%")
    print(f"  vs Strategic-0.3: {fs03:.1f}%")
    print(f"  vs Strategic-0.5: {fs05:.1f}%")
    print("=" * 60)

    if eval_eps:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(eval_eps, wr_random, marker="o", color="green", linewidth=2)
        plt.axhline(y=99.5, color="gray", linestyle="--", label="Phase5 start (99.5%)")
        plt.axhline(y=90.0, color="red",  linestyle=":", label="Stop threshold (90%)")
        for ep in sync_eps:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Random")
        plt.ylim([0, 105]); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(eval_eps, wr_s03, marker="s", color="steelblue", linewidth=2, label="vs S-0.3")
        plt.plot(eval_eps, wr_s05, marker="^", color="purple",    linewidth=2, label="vs S-0.5")
        plt.axhline(y=78.0, color="steelblue", linestyle="--", alpha=0.5, label="Phase5 best S-0.3 (78%)")
        plt.axhline(y=51.0, color="purple",    linestyle="--", alpha=0.5, label="Phase5 best S-0.5 (51%)")
        for ep in sync_eps:
            plt.axvline(x=ep, color="orange", linestyle=":", alpha=0.5)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Strategic Opponents")
        plt.ylim([0, 105]); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(losses, alpha=0.3, color="blue")
        if len(losses) > 100:
            sm = np.convolve(losses, np.ones(100)/100, mode="valid")
            plt.plot(sm, color="darkblue", label="Smoothed"); plt.legend()
        plt.xlabel("Training Step"); plt.ylabel("Loss")
        plt.title("Training Loss"); plt.grid(True)

        plt.tight_layout()
        path = os.path.join(save_dir, "phase5b_curves.png")
        plt.savefig(path, dpi=150)
        print(f"\nPlot saved to {path}")
        plt.close()

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--warmup",   type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="models_phase5b")
    args = parser.parse_args()
    train_phase5b(
        num_episodes=args.episodes,
        warmup_episodes=args.warmup,
        save_dir=args.save_dir,
    )
