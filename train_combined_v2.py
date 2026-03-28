"""
Combined Long Run v2: Rohan's Adaptive Curriculum + Jeson's Lessons

Why train_combined_longrun.py kept failing:
    - Fixed phase schedules have no feedback mechanism.
    - Sparse Phase A → no gradient signal (Loss = 0, agent learns nothing).
    - Positive shaped Phase A → reward hacking (agent chases shaped rewards,
      win rate declines 58%→42% while episode reward stays constant at ~1.4).
    - Root cause: without adaptive curriculum, once reward hacking starts,
      nothing stops it. The schedule runs regardless of agent performance.

Why Rohan's training never had this problem:
    - Adaptive curriculum with demotion: when win rate < 25%, agent drops back
      to an easier opponent. Against easier opponents, wins are frequent and
      terminal ±1.0 dominates shaped rewards again. Q-values self-correct.

This script = train_rohan.py with 3 targeted improvements from our failures:

    1. Ignore penalty enabled from episode 1 — same as Rohan's proven approach.
       The "ignore penalty causes negative spiral at high ε" failure only occurred
       with FIXED phases that forced strategic opponents from the start. With
       adaptive curriculum at Level 0 (Random), the random opponent creates threats
       rarely (~2-3 per game), so the penalty fires rarely and keeps losses clearly
       negative. Disabling it (previous attempt) made losses near-zero, destroying
       the Q-network's ability to distinguish wins from losses → 28% vs Random.

    2. Minimum 25% random anchor (Jeson Stage 6 lesson):
       - Rohan's anchor dropped to 15% at high levels
       - Stage 6 proved that 20% causes strategy collapse; 25% is the floor
       - New formula: max(0.25, 0.4 - level * 0.02) — never drops below 25%

    3. Win detection by actual game outcome, not episode_reward (bug fix):
       - Rohan used episode_reward > 0.5 to detect wins
       - Shaped rewards inflate episode_reward even for losses
       - A loss with many shaped rewards: episode_reward ≈ +0.5 → false "win"
       - This causes over-eager promotion and curriculum instability
       - Fix: track terminal reward separately; curriculum only uses real wins

Architecture: DQNAgentRohan (Dueling DQN + PER + AdamW) — unchanged.
Environment: GomokuEnvShaped — unchanged, just passes ignore_penalty_enabled.
Curriculum:  11 levels (Random → Strat-0.9 → MM-0.7) — same as Rohan's.

Load: nothing — fresh start
Save: models_combined_v2/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import deque

from game.logic import GomokuLogic
from game.gomoku_env_shaped import GomokuEnvShaped
from game.gomoku_env import GomokuEnv
from agents.dqn_rohan import DQNAgentRohan
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from agents.minimax_agent import MinimaxAgent


# ── Evaluation (always sparse, always ε=0) ───────────────────────────────────

def evaluate_agent(agent, opponent, board_size, num_games=30):
    """Evaluate agent win rate with sparse rewards."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0
    for g in range(num_games):
        logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(logic, use_sparse_rewards=True)

        if g % 2 == 0:
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
    return wins / num_games * 100


def run_standard_eval(agent, board_size, num_games, label):
    """Run and print the standard evaluation suite."""
    r   = evaluate_agent(agent, RandomAgent(-1), board_size, num_games)
    s03 = evaluate_agent(agent, StrategicAgent(-1, 0.3, board_size), board_size, num_games)
    s05 = evaluate_agent(agent, StrategicAgent(-1, 0.5, board_size), board_size, num_games)
    s07 = evaluate_agent(agent, StrategicAgent(-1, 0.7, board_size), board_size, num_games)
    print(f"\n{'─'*50}")
    print(f"EVAL {label} ({num_games} games each)")
    print(f"  vs Random:        {r:.1f}%")
    print(f"  vs Strategic-0.3: {s03:.1f}%")
    print(f"  vs Strategic-0.5: {s05:.1f}%")
    print(f"  vs Strategic-0.7: {s07:.1f}%")
    print(f"{'─'*50}\n")
    return r, s03, s05, s07


# ── Main training function ────────────────────────────────────────────────────

def train_combined_v2(
    total_episodes=100000,
    batch_size=64,
    train_frequency=4,
    board_size=9,
    save_dir="models_combined_v2",
    promote_threshold=0.60,   # promote when win rate ≥ 60% over last 100+ games
    demote_threshold=0.25,    # demote when win rate < 25%
    min_games_at_level=150,   # must play this many games before promotion/demotion check
):
    os.makedirs(save_dir, exist_ok=True)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = DQNAgentRohan(player_id=1, board_size=board_size)
    agent.epsilon = 1.0
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.99995   # 1.0 → 0.02 over ~100k episodes

    # ── Fixed opponents ───────────────────────────────────────────────────────
    random_opp   = RandomAgent(player_id=-1)
    strat_opp    = StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size)
    minimax_opp  = MinimaxAgent(player_id=-1, board_size=board_size,
                                time_limit=0.05, skill_level=0.5)

    # ── Curriculum levels ─────────────────────────────────────────────────────
    # Levels 0–8: StrategicAgent at increasing skill
    # Levels 9–11: MinimaxAgent at increasing skill
    def make_strat(skill):
        def _get():
            strat_opp.skill_level = skill
            return strat_opp
        return _get

    def make_mm(skill):
        def _get():
            minimax_opp.set_skill_level(skill)
            return minimax_opp
        return _get

    difficulty_levels = [
        ("Random",    lambda: random_opp),
        ("Strat-0.2", make_strat(0.2)),
        ("Strat-0.3", make_strat(0.3)),
        ("Strat-0.4", make_strat(0.4)),
        ("Strat-0.5", make_strat(0.5)),
        ("Strat-0.6", make_strat(0.6)),
        ("Strat-0.7", make_strat(0.7)),
        ("Strat-0.8", make_strat(0.8)),
        ("Strat-0.9", make_strat(0.9)),
        ("MM-0.3",    make_mm(0.3)),
        ("MM-0.5",    make_mm(0.5)),
        ("MM-0.7",    make_mm(0.7)),
    ]

    current_level = 0
    max_level = 0
    recent_wins  = deque(maxlen=100)   # 1 = won, 0 = lost (curriculum games only)
    games_at_level = 0

    # ── Tracking ──────────────────────────────────────────────────────────────
    all_rewards, all_losses = [], []
    level_history = []
    eval_eps, wr_rand, wr_s03, wr_s05, wr_s07 = [], [], [], [], []
    best_rand = best_s03 = best_s05 = 0.0

    step_count = 0

    print("=" * 65)
    print("COMBINED V2: Adaptive Curriculum + Jeson's Lessons")
    print("=" * 65)
    print(f"Architecture:  DQNAgentRohan (Dueling DQN + PER + AdamW)")
    print(f"Device:        {agent.device}")
    print(f"Epsilon:       1.0 → 0.02 over {total_episodes:,} episodes")
    print(f"Rewards:       GomokuEnvShaped (full shaped rewards + ignore penalty from ep 1)")
    print(f"Curriculum:    {len(difficulty_levels)} levels, promote ≥ {promote_threshold:.0%}, demote < {demote_threshold:.0%}")
    print(f"Random anchor: max(0.25, 0.40 - level×0.02)  [minimum 25%]")
    print(f"Win detection: actual game outcome (not episode_reward)")
    print(f"Save:          {save_dir}/")
    print("=" * 65 + "\n")

    start_time = time.time()

    for episode in range(total_episodes):
        # ── Select curriculum opponent ─────────────────────────────────────────
        level_name, opp_getter = difficulty_levels[current_level]
        curriculum_opponent = opp_getter()

        # Random anchor: minimum 25%, starting at 40% at level 0
        # IMPROVEMENT 2: Jeson Stage 6 lesson — 20% anchor caused collapse; 25% is floor
        random_anchor = max(0.25, 0.40 - current_level * 0.02)
        use_random_game = (np.random.random() < random_anchor) and (current_level > 0)

        if use_random_game:
            opponent = random_opp
            is_curriculum_game = False
        else:
            opponent = curriculum_opponent
            is_curriculum_game = True

        # ── Alternate first player ─────────────────────────────────────────────
        agent_first = (episode % 2 == 0)
        agent.player_id    = 1  if agent_first else -1
        opponent.player_id = -1 if agent_first else 1

        # ── Environment ───────────────────────────────────────────────────────
        # Full shaped rewards including ignore penalty from episode 1.
        # The ignore penalty does NOT cause a negative spiral at Level 0 (Random):
        #   a random opponent creates 4-in-a-row threats only ~2-3 times per game,
        #   so the penalty fires rarely and usefully — it keeps losses clearly negative
        #   (-0.7 net reward) so the Q-network can distinguish wins from losses.
        # Disabling the ignore penalty (previous attempt) made losses near-zero reward,
        #   destroying the Q-network's ability to learn that losing is bad → 28% vs Random.
        logic = GomokuLogic(board_size=board_size)
        env   = GomokuEnvShaped(logic, positive_rewards=True,
                                ignore_penalty_enabled=True)

        # ── Play episode ───────────────────────────────────────────────────────
        state = env.reset()
        ep_reward = 0.0
        terminal_win = False   # IMPROVEMENT 3: track actual game outcome
        done = False

        if not agent_first:
            opp_action = opponent.predict(state)
            state, _, done, _ = env.step(opp_action)

        while not done:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            step_count += 1

            agent.store_experience(state, action, reward, next_state, done)

            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    all_losses.append(loss)

            # IMPROVEMENT 3: detect win from terminal reward, not episode_reward.
            # episode_reward can exceed 0.5 even in a loss (shaped rewards inflate it).
            if done and reward > 0:
                terminal_win = True

            if done:
                break

            state = next_state
            opp_action = opponent.predict(state)
            next_state, _, done, _ = env.step(opp_action)
            state = next_state

        agent.decay_epsilon()
        all_rewards.append(ep_reward)
        level_history.append(current_level)

        # ── Curriculum: update win tracking (curriculum games only) ───────────
        if is_curriculum_game:
            recent_wins.append(1 if terminal_win else 0)
            games_at_level += 1

        # ── Curriculum: check promotion / demotion ────────────────────────────
        if len(recent_wins) >= 50 and games_at_level >= min_games_at_level:
            win_rate = np.mean(recent_wins)

            if win_rate >= promote_threshold and current_level < len(difficulty_levels) - 1:
                current_level += 1
                games_at_level = 0
                recent_wins.clear()

                print(f"\n>>> PROMOTED to Level {current_level}: "
                      f"{difficulty_levels[current_level][0]} "
                      f"(win rate: {win_rate:.1%}, ε={agent.epsilon:.3f}) <<<")

                if current_level > max_level:
                    max_level = current_level
                    agent.save_model(
                        os.path.join(save_dir, f"level_{current_level}.pt"))
                    print(f"    [Save] New max level checkpoint\n")

            elif win_rate < demote_threshold and current_level > 0:
                current_level -= 1
                games_at_level = 0
                recent_wins.clear()
                print(f"\n>>> DEMOTED to Level {current_level}: "
                      f"{difficulty_levels[current_level][0]} "
                      f"(win rate: {win_rate:.1%}) <<<\n")

        # ── Logging every 500 episodes ─────────────────────────────────────────
        if (episode + 1) % 500 == 0:
            avg_r  = np.mean(all_rewards[-500:])
            avg_l  = np.mean(all_losses[-1000:]) if all_losses else 0.0
            curr_wr = np.mean(recent_wins) if recent_wins else 0.0
            elapsed = (time.time() - start_time) / 60
            print(f"Ep {episode+1:>6}/{total_episodes} | "
                  f"Lv {current_level}: {level_name:10} | "
                  f"CurrWR: {curr_wr:.1%} | "
                  f"Rew: {avg_r:>5.2f} | Loss: {avg_l:.4f} | "
                  f"Eps: {agent.epsilon:.3f} | {elapsed:.1f}m")

        # ── Evaluation every 2000 episodes ────────────────────────────────────
        if (episode + 1) % 2000 == 0:
            r, s03, s05, s07 = run_standard_eval(
                agent, board_size, 50, f"ep {episode+1}")
            eval_eps.append(episode + 1)
            wr_rand.append(r); wr_s03.append(s03)
            wr_s05.append(s05); wr_s07.append(s07)

            print(f"  Curriculum: Level {current_level} ({level_name}) | "
                  f"Max: {max_level} | ε={agent.epsilon:.3f}")

            if r   > best_rand:
                best_rand = r
                agent.save_model(os.path.join(save_dir, "best_random.pt"))
                print(f"  [Save] New best vs Random: {r:.1f}%")
            if s05 > best_s05:
                best_s05 = s05
                agent.save_model(os.path.join(save_dir, "best_s05.pt"))
                print(f"  [Save] New best vs S-0.5: {s05:.1f}%")
            if s03 > best_s03:
                best_s03 = s03
                agent.save_model(os.path.join(save_dir, "best_s03.pt"))
                print(f"  [Save] New best vs S-0.3: {s03:.1f}%")

        # ── Checkpoint every 10k episodes ─────────────────────────────────────
        if (episode + 1) % 10000 == 0:
            agent.save_model(
                os.path.join(save_dir, f"ep{episode+1}.pt"))

    # ── Final save and evaluation ─────────────────────────────────────────────
    agent.save_model(os.path.join(save_dir, "final.pt"))
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("COMBINED V2 COMPLETE")
    print("=" * 65)
    print(f"Time:         {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)")
    print(f"Max level:    {max_level} ({difficulty_levels[max_level][0]})")
    print(f"Best vs Rand: {best_rand:.1f}%")
    print(f"Best vs S-0.3:{best_s03:.1f}%")
    print(f"Best vs S-0.5:{best_s05:.1f}%")

    print("\nFinal 200-game evaluation...")
    agent.epsilon = 0.0
    fr, fs03, fs05, fs07 = run_standard_eval(agent, board_size, 200, "FINAL")
    mm3 = evaluate_agent(
        agent,
        MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.3),
        board_size, 50)
    mm5 = evaluate_agent(
        agent,
        MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.5),
        board_size, 50)
    print(f"  vs MM-0.3: {mm3:.1f}%")
    print(f"  vs MM-0.5: {mm5:.1f}%")
    print(f"\nPhase4 baseline for comparison:")
    print(f"  vs Random: 98.5% | vs S-0.3: 66.5% | vs S-0.5: 43.5% | vs MM-0.3: ~5%")
    print("=" * 65)

    # ── Training curves plot ───────────────────────────────────────────────────
    if eval_eps:
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.plot(eval_eps, wr_rand, "go-", linewidth=2, markersize=4)
        plt.axhline(y=98.5, color="gray",  linestyle="--", alpha=0.5, label="Phase4 (98.5%)")
        plt.axhline(y=85.0, color="red",   linestyle=":",  alpha=0.7, label="Floor (85%)")
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Random"); plt.ylim([0, 105])
        plt.legend(fontsize=7); plt.grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(eval_eps, wr_s03, "bs-", linewidth=2, markersize=4, label="vs S-0.3")
        plt.plot(eval_eps, wr_s05, "m^-", linewidth=2, markersize=4, label="vs S-0.5")
        plt.plot(eval_eps, wr_s07, "rD-", linewidth=2, markersize=4, label="vs S-0.7")
        plt.axhline(y=66.5, color="blue",   linestyle="--", alpha=0.3)
        plt.axhline(y=43.5, color="purple", linestyle="--", alpha=0.3)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Strategic Opponents"); plt.ylim([0, 105])
        plt.legend(fontsize=7); plt.grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(level_history, alpha=0.4, color="steelblue")
        plt.xlabel("Episode"); plt.ylabel("Difficulty Level")
        plt.title(f"Curriculum (Max Level: {max_level})")
        plt.yticks(range(len(difficulty_levels)),
                   [n for n, _ in difficulty_levels], fontsize=6)
        plt.grid(True)

        plt.subplot(1, 4, 4)
        labels   = ["Rand", "S-0.3", "S-0.5", "S-0.7", "MM-0.3", "MM-0.5"]
        baseline = [98.5,    66.5,    43.5,    20.0,      5.0,      2.0]
        final    = [fr,       fs03,    fs05,    fs07,     mm3,      mm5]
        x = np.arange(len(labels)); w = 0.35
        plt.bar(x - w/2, baseline, w, label="Phase4 baseline", color="steelblue", alpha=0.7)
        plt.bar(x + w/2, final,    w, label="Combined v2",     color="green",     alpha=0.7)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.5)
        plt.xticks(x, labels, fontsize=7)
        plt.ylabel("Win Rate (%)"); plt.title("Baseline vs Combined V2")
        plt.ylim([0, 105]); plt.legend(fontsize=7); plt.grid(True, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=100000)
    parser.add_argument("--save-dir",  type=str,   default="models_combined_v2")
    parser.add_argument("--promote",   type=float, default=0.60)
    parser.add_argument("--demote",    type=float, default=0.25)
    args = parser.parse_args()

    train_combined_v2(
        total_episodes=args.episodes,
        save_dir=args.save_dir,
        promote_threshold=args.promote,
        demote_threshold=args.demote,
    )
