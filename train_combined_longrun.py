"""
Phase 6: Combined Best-of-Both-Worlds Long Run

Architecture (Rohan's proven components):
    - Dueling DQN:  separates V(s) from A(s,a) — faster convergence in positions
                    where most moves are equivalent (most of Gomoku)
    - Prioritized Experience Replay: replays surprising/high-error experiences
                    more often — critical for learning rare but important blocks
    - AdamW + weight decay: prevents weight explosion over 60k episodes
    - Shaped rewards from EPISODE 1: no Q-value mismatch (Stage 2 root cause)

Curriculum (Jeson's discovered insights):
    - Phase A (ep 1–10k):  Foundation — Random-heavy, ε=1.0→0.52, no self-play
    - Phase B (ep 10k–30k): Defensive Mastery — Strategic opponents only,
                             NO self-play. This is the key Phase 5 lesson.
    - Phase C (ep 30k–60k): Strategic Integration — self-play introduced AFTER
                             30k eps of defensive training, sync every 2000 eps.

Why this works where everything else failed:
    Stage 2 failure: shaped rewards on pre-trained sparse model → Q-value mismatch.
        Fix: fresh start, Q-values calibrated to shaped world from ep 1.
    Phase 5 failure: self-play before defensive skills → escalation trap.
        Fix: no self-play until Phase C (30k defensive eps first).
    Phase 5 collapse: short runs + fast sync → opponent too hard too fast.
        Fix: 60k single run, sync every 2000 (4x slower).
    Stage 4/6: empty buffer on reload + random anchor below threshold.
        Fix: never reload (single continuous run), 30-40% random throughout.

Load: nothing — fresh start from random weights
Save: models_combined/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from game.gomoku_env_shaped import GomokuEnvShaped
from agents.dqn_rohan import DQNAgentRohan
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from agents.minimax_agent import MinimaxAgent


# -------------------------------------------------------
# Evaluation (always sparse env for fair comparison)
# -------------------------------------------------------

def evaluate(agent, opponent_factory, board_size, num_games=50):
    """
    Evaluate agent against an opponent. opponent_factory is a callable that
    returns a fresh opponent (needed because StrategicAgent is stateful).
    """
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    for g in range(num_games):
        opp = opponent_factory()
        logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(logic, use_sparse_rewards=True)
        agent.player_id = 1 if g % 2 == 0 else -1
        opp.player_id = -agent.player_id
        state = env.reset()
        done = False
        while not done:
            if env.logic.current_player == agent.player_id:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    wins += 1
            else:
                action = opp.predict(state)
                state, _, done, _ = env.step(action)
    agent.epsilon = orig_eps
    return wins / num_games * 100


def run_eval(agent, board_size, num_games, label):
    """Run standard evaluation suite and print results."""
    r  = evaluate(agent, lambda: RandomAgent(player_id=-1), board_size, num_games)
    s3 = evaluate(agent, lambda: StrategicAgent(player_id=-1, skill_level=0.3, board_size=board_size), board_size, num_games)
    s5 = evaluate(agent, lambda: StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size), board_size, num_games)
    s7 = evaluate(agent, lambda: StrategicAgent(player_id=-1, skill_level=0.7, board_size=board_size), board_size, num_games)
    print(f"\n{'='*55}")
    print(f"EVAL {label} ({num_games} games)")
    print(f"  vs Random:        {r:.1f}%")
    print(f"  vs Strategic-0.3: {s3:.1f}%")
    print(f"  vs Strategic-0.5: {s5:.1f}%")
    print(f"  vs Strategic-0.7: {s7:.1f}%")
    if r < 85.0 and "Phase A" not in label:
        print(f"  *** WARNING: vs Random < 85% — investigate immediately ***")
    elif r < 60.0 and "Phase A" in label:
        print(f"  *** WARNING: vs Random < 60% in Phase A — trend is declining, check rewards ***")
    print(f"{'='*55}\n")
    return r, s3, s5, s7


# -------------------------------------------------------
# Episode runner (shaped env throughout)
# -------------------------------------------------------

def play_episode(agent, opponent, board_size, use_shaped=True, ignore_penalty=True):
    """
    Play one game. Returns (reward, experiences).

    use_shaped=False → sparse rewards (GomokuEnv): not used in the current design
        but kept for compatibility.

    use_shaped=True, ignore_penalty=False → positive shaped rewards only (Phase A):
        blocking rewards, threat creation rewards, and positional bonuses all fire,
        giving a dense learning signal even at high epsilon. The ignore penalty is
        disabled because at ε>0.5 the agent plays randomly more than half the time —
        the penalty cannot distinguish exploration from a deliberate bad decision and
        fires incorrectly on random moves, suppressing all Q-values.

    use_shaped=True, ignore_penalty=True → full shaped rewards (Phase B+):
        ignore penalty enabled once ε<0.52 and the agent is making mostly deliberate
        choices. At that point penalising ignored threats is correct.
    """
    logic = GomokuLogic(board_size=board_size)
    if use_shaped:
        env = GomokuEnvShaped(logic, positive_rewards=True, ignore_penalty_enabled=ignore_penalty)
    else:
        env = GomokuEnv(logic, use_sparse_rewards=True)
    state = env.reset()
    ep_reward = 0.0
    experiences = []
    done = False

    agent_first = random.random() < 0.5
    agent.player_id = 1 if agent_first else -1
    opponent.player_id = -agent.player_id

    if not agent_first:
        opp_action = opponent.predict(state)
        state, _, done, _ = env.step(opp_action)

    while not done:
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        experiences.append((state, action, reward, next_state, done))
        if done:
            break
        state = next_state
        opp_action = opponent.predict(state)
        next_state, _, done, _ = env.step(opp_action)
        state = next_state

    return ep_reward, experiences


# -------------------------------------------------------
# Opponent selectors per phase
# -------------------------------------------------------

def get_phase_a_opponent(random_opp, strat_01):
    """Phase A: 80% Random / 20% Strategic-0.1"""
    return random_opp if random.random() < 0.80 else strat_01


def get_phase_b_opponent(random_opp, strat_03, strat_05):
    """Phase B: 40% Random / 35% Strategic-0.5 / 25% Strategic-0.3"""
    r = random.random()
    if r < 0.40:
        return random_opp
    elif r < 0.75:
        return strat_05
    else:
        return strat_03


def get_phase_c_opponent(random_opp, strat_05, strat_07, frozen):
    """Phase C: 30% Random / 25% Strategic-0.5 / 25% self-play / 20% Strategic-0.7"""
    r = random.random()
    if r < 0.30:
        return random_opp
    elif r < 0.55:
        return strat_05
    elif r < 0.80:
        return frozen
    else:
        return strat_07


# -------------------------------------------------------
# Main training function
# -------------------------------------------------------

def train_combined(
    phase_a_episodes=10000,
    phase_b_episodes=20000,
    phase_c_episodes=30000,
    batch_size=64,
    train_frequency=4,
    sync_frequency=2000,        # Phase C self-play sync (slow — 4x Phase 5)
    eval_frequency=2000,        # Evaluate every 2000 episodes
    board_size=9,
    save_dir="models_combined",
):
    total_episodes = phase_a_episodes + phase_b_episodes + phase_c_episodes
    os.makedirs(save_dir, exist_ok=True)

    # ── Agent: Dueling DQN + PER + AdamW (Rohan's architecture) ──────────────
    agent = DQNAgentRohan(player_id=1, board_size=board_size)
    # Fresh start: epsilon=1.0, no model loaded
    agent.epsilon = 1.0
    agent.epsilon_end = 0.02
    # Continuous decay: 1.0 → 0.02 over total_episodes
    agent.epsilon_decay = (agent.epsilon_end / agent.epsilon) ** (1.0 / total_episodes)

    # ── Fixed opponents ───────────────────────────────────────────────────────
    random_opp = RandomAgent(player_id=-1)
    strat_01   = StrategicAgent(player_id=-1, skill_level=0.1, board_size=board_size)
    strat_03   = StrategicAgent(player_id=-1, skill_level=0.3, board_size=board_size)
    strat_05   = StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size)
    strat_07   = StrategicAgent(player_id=-1, skill_level=0.7, board_size=board_size)
    frozen     = None   # initialised at Phase C start

    # ── Tracking ──────────────────────────────────────────────────────────────
    all_rewards, all_losses = [], []
    eval_eps = []
    wr_rand, wr_s03, wr_s05, wr_s07 = [], [], [], []
    phase_c_ep = 0      # episode counter within Phase C (for sync timing)

    # best checkpoint trackers
    best_rand = best_s03 = best_s05 = 0.0

    print("=" * 65)
    print("COMBINED LONG RUN: Dueling DQN + PER + Shaped Rewards + 3-Phase Curriculum")
    print("=" * 65)
    print(f"Architecture:    DQNAgentRohan (Dueling DQN + Prioritized Replay)")
    print(f"Environment:     Phase A: GomokuEnvShaped(positive_rewards=True, ignore_penalty=False)")
    print(f"Device:          {agent.device}")
    print(f"Epsilon:         1.0 → 0.02 over {total_episodes:,} episodes")
    print(f"Epsilon decay:   {agent.epsilon_decay:.7f} per episode")
    print()
    print(f"Phase A: ep 1 – {phase_a_episodes:,}")
    print(f"         Opponents: 80% Random / 20% StrategicAgent-0.1")
    print(f"         Goal: establish basic offense + first defensive instincts")
    print()
    print(f"Phase B: ep {phase_a_episodes+1:,} – {phase_a_episodes+phase_b_episodes:,}")
    print(f"         Opponents: 40% Random / 35% StrategicAgent-0.5 / 25% StrategicAgent-0.3")
    print(f"         Goal: deep defensive mastery — NO self-play (Stage 5 lesson)")
    print()
    print(f"Phase C: ep {phase_a_episodes+phase_b_episodes+1:,} – {total_episodes:,}")
    print(f"         Opponents: 30% Random / 25% StrategicAgent-0.5 / 25% self-play / 20% StrategicAgent-0.7")
    print(f"         Self-play sync: every {sync_frequency} episodes (4x slower than Phase 5)")
    print(f"         Goal: strategic depth via self-play with defensive foundation")
    print()
    print(f"Eval:    every {eval_frequency} episodes + phase boundaries")
    print(f"Save:    {save_dir}/")
    print("=" * 65 + "\n")

    start_time = time.time()
    step_count = 0
    current_phase = "A"

    for episode in range(total_episodes):
        global_ep = episode + 1  # 1-indexed for display

        # ── Phase routing ─────────────────────────────────────────────────────
        if episode < phase_a_episodes:
            if current_phase != "A":
                current_phase = "A"
            opponent = get_phase_a_opponent(random_opp, strat_01)

        elif episode < phase_a_episodes + phase_b_episodes:
            if current_phase != "B":
                current_phase = "B"
                print(f"\n{'*'*55}")
                print(f"  ENTERING PHASE B at episode {global_ep}")
                print(f"  Opponents: 40% Random / 35% S-0.5 / 25% S-0.3")
                print(f"  Rewards: enabling ignore penalty (was positive-only in Phase A)")
                print(f"  Epsilon now ~{agent.epsilon:.2f} — agent making ~{(1-agent.epsilon)*100:.0f}% deliberate choices")
                print(f"  No self-play until Phase C")
                print(f"{'*'*55}\n")
                run_eval(agent, board_size, 100, f"Phase A→B boundary (ep {global_ep})")
            opponent = get_phase_b_opponent(random_opp, strat_03, strat_05)

        else:
            if current_phase != "C":
                current_phase = "C"
                phase_c_ep = 0
                # Initialise frozen self-play opponent from current model state
                frozen = DQNAgentRohan(player_id=-1, board_size=board_size)
                frozen.q_network.load_state_dict(agent.q_network.state_dict())
                frozen.target_network.load_state_dict(agent.target_network.state_dict())
                frozen.epsilon = 0.0
                print(f"\n{'*'*55}")
                print(f"  ENTERING PHASE C at episode {global_ep}")
                print(f"  Opponents: 30% Random / 25% S-0.5 / 25% self-play / 20% S-0.7")
                print(f"  Self-play frozen copy initialised")
                print(f"  Sync every {sync_frequency} Phase-C episodes")
                print(f"{'*'*55}\n")
                run_eval(agent, board_size, 100, f"Phase B→C boundary (ep {global_ep})")
            opponent = get_phase_c_opponent(random_opp, strat_05, strat_07, frozen)
            phase_c_ep += 1

        # ── Play episode ──────────────────────────────────────────────────────
        # Phase A: positive shaped rewards ONLY — no ignore penalty.
        #   Positive rewards (blocking, threat creation) give dense signal even at ε>0.5.
        #   Ignore penalty disabled: at ε=0.77 most moves are random; the penalty fires
        #   on random exploration moves and suppresses Q-values incorrectly.
        # Phase B+: full shaped rewards including ignore penalty (ε<0.52, mostly deliberate).
        use_ignore_penalty = (current_phase != "A")
        ep_reward, experiences = play_episode(
            agent, opponent, board_size,
            use_shaped=True,
            ignore_penalty=use_ignore_penalty,
        )

        for s, a, r, ns, d in experiences:
            agent.store_experience(s, a, r, ns, d)
            step_count += 1
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    all_losses.append(loss)

        agent.decay_epsilon()
        all_rewards.append(ep_reward)

        # ── Phase C: sync frozen opponent ─────────────────────────────────────
        if current_phase == "C" and phase_c_ep > 0 and phase_c_ep % sync_frequency == 0:
            frozen.q_network.load_state_dict(agent.q_network.state_dict())
            frozen.target_network.load_state_dict(agent.target_network.state_dict())
            print(f"  [Sync] Frozen opponent updated at Phase-C ep {phase_c_ep} (global {global_ep})")

        # ── Logging every 200 episodes ────────────────────────────────────────
        if global_ep % 200 == 0:
            avg_r = np.mean(all_rewards[-200:])
            avg_l = np.mean(all_losses[-500:]) if all_losses else 0.0
            elapsed = (time.time() - start_time) / 60
            print(f"Ep {global_ep:>6}/{total_episodes} | Phase {current_phase} | "
                  f"Reward: {avg_r:>6.3f} | Loss: {avg_l:.4f} | "
                  f"Eps: {agent.epsilon:.3f} | {elapsed:.1f}m")

        # ── Periodic evaluation ───────────────────────────────────────────────
        if global_ep % eval_frequency == 0:
            r, s3, s5, s7 = run_eval(
                agent, board_size, 50, f"ep {global_ep} Phase {current_phase}"
            )
            eval_eps.append(global_ep)
            wr_rand.append(r); wr_s03.append(s3); wr_s05.append(s5); wr_s07.append(s7)

            if r > best_rand:
                best_rand = r
                agent.save_model(os.path.join(save_dir, "combined_best_random.pt"))
                print(f"  [Save] New best vs Random: {r:.1f}%")
            if s5 > best_s05:
                best_s05 = s5
                agent.save_model(os.path.join(save_dir, "combined_best_s05.pt"))
                print(f"  [Save] New best vs S-0.5: {s5:.1f}%")
            if s3 > best_s03:
                best_s03 = s3
                agent.save_model(os.path.join(save_dir, "combined_best_s03.pt"))
                print(f"  [Save] New best vs S-0.3: {s3:.1f}%")

        # ── Checkpoint every 5000 episodes ────────────────────────────────────
        if global_ep % 5000 == 0:
            agent.save_model(os.path.join(save_dir, f"combined_ep{global_ep}.pt"))

    # ── Final save and evaluation ─────────────────────────────────────────────
    agent.save_model(os.path.join(save_dir, "combined_final.pt"))
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("COMBINED LONG RUN COMPLETE")
    print("=" * 65)
    print(f"Total time:          {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)")
    print(f"Best vs Random:      {best_rand:.1f}%")
    print(f"Best vs S-0.3:       {best_s03:.1f}%")
    print(f"Best vs S-0.5:       {best_s05:.1f}%")

    print("\nRunning final 200-game evaluation...")
    fr, fs3, fs5, fs7 = run_eval(agent, board_size, 200, "FINAL (200 games)")

    # Also eval Minimax
    mm3 = evaluate(
        agent,
        lambda: MinimaxAgent(player_id=-1, board_size=board_size,
                             time_limit=0.1, skill_level=0.3),
        board_size, 50
    )
    print(f"  vs MinimaxAgent-0.3: {mm3:.1f}%")

    print(f"\nFull summary:")
    print(f"  vs Random:        {fr:.1f}%  (Phase4 baseline: 98.5%)")
    print(f"  vs Strategic-0.3: {fs3:.1f}%  (Phase4 baseline: 66.5%)")
    print(f"  vs Strategic-0.5: {fs5:.1f}%  (Phase4 baseline: 43.5%)")
    print(f"  vs Strategic-0.7: {fs7:.1f}%  (Phase4 baseline: ~20%)")
    print(f"  vs Minimax-0.3:   {mm3:.1f}%  (Phase4 baseline: ~5%)")
    print("=" * 65)

    # ── Training curves plot ──────────────────────────────────────────────────
    if eval_eps:
        phase_b_start = phase_a_episodes
        phase_c_start = phase_a_episodes + phase_b_episodes

        plt.figure(figsize=(18, 4))

        plt.subplot(1, 4, 1)
        plt.plot(eval_eps, wr_rand, marker="o", color="green", linewidth=2)
        plt.axhline(y=98.5, color="gray", linestyle="--", label="Phase4 (98.5%)")
        plt.axhline(y=85.0, color="red",  linestyle=":", label="Stop (85%)")
        plt.axvline(x=phase_b_start, color="blue",   linestyle="--", alpha=0.5, label="Phase B")
        plt.axvline(x=phase_c_start, color="purple", linestyle="--", alpha=0.5, label="Phase C")
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Random"); plt.ylim([0, 105])
        plt.legend(fontsize=7); plt.grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(eval_eps, wr_s03, marker="s", color="steelblue", linewidth=2, label="vs S-0.3")
        plt.plot(eval_eps, wr_s05, marker="^", color="purple",    linewidth=2, label="vs S-0.5")
        plt.plot(eval_eps, wr_s07, marker="D", color="darkred",   linewidth=2, label="vs S-0.7")
        plt.axhline(y=66.5, color="steelblue", linestyle="--", alpha=0.4, label="P4 S-0.3")
        plt.axhline(y=43.5, color="purple",    linestyle="--", alpha=0.4, label="P4 S-0.5")
        plt.axvline(x=phase_b_start, color="blue",   linestyle="--", alpha=0.5)
        plt.axvline(x=phase_c_start, color="purple", linestyle="--", alpha=0.5)
        plt.xlabel("Episode"); plt.ylabel("Win Rate (%)")
        plt.title("vs Strategic Opponents"); plt.ylim([0, 105])
        plt.legend(fontsize=7); plt.grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(all_losses, alpha=0.2, color="blue")
        if len(all_losses) > 200:
            sm = np.convolve(all_losses, np.ones(200)/200, mode="valid")
            plt.plot(sm, color="darkblue", linewidth=1.5, label="Smoothed (200-step)")
            plt.legend(fontsize=7)
        plt.axvline(x=phase_b_start * 8, color="blue",   linestyle="--", alpha=0.4)
        plt.axvline(x=phase_c_start * 8, color="purple", linestyle="--", alpha=0.4)
        plt.xlabel("Training Step"); plt.ylabel("Loss")
        plt.title("Training Loss"); plt.grid(True)

        plt.subplot(1, 4, 4)
        labels   = ["Random", "S-0.3", "S-0.5", "S-0.7", "MM-0.3"]
        baseline = [98.5,      66.5,    43.5,    20.0,     5.0]
        final    = [fr,         fs3,     fs5,     fs7,    mm3]
        x = np.arange(len(labels)); w = 0.35
        plt.bar(x - w/2, baseline, w, label="Phase 4 baseline", color="steelblue", alpha=0.7)
        plt.bar(x + w/2, final,    w, label="Combined run",     color="green",     alpha=0.7)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50%")
        plt.xticks(x, labels, fontsize=8)
        plt.ylabel("Win Rate (%)"); plt.title("Phase 4 vs Combined Run")
        plt.ylim([0, 105]); plt.legend(fontsize=7); plt.grid(True, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "combined_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to {plot_path}")
        plt.close()

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=int, default=10000,
                        help="Phase A episodes (default: 10000)")
    parser.add_argument("--phase-b", type=int, default=20000,
                        help="Phase B episodes (default: 20000)")
    parser.add_argument("--phase-c", type=int, default=30000,
                        help="Phase C episodes (default: 30000)")
    parser.add_argument("--save-dir", type=str, default="models_combined")
    args = parser.parse_args()

    train_combined(
        phase_a_episodes=args.phase_a,
        phase_b_episodes=args.phase_b,
        phase_c_episodes=args.phase_c,
        save_dir=args.save_dir,
    )
