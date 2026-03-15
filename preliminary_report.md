# COMP0215 Coursework — Preliminary Results Report
**Team:** [Team Name] | **Members:** Jeson Ramesh, Rohan Beya [+ other members]
**Date:** 16 March 2026 | **Task:** Gomoku Challenge Task (9×9, Five-in-a-Row)

---

## 1. Task and Approach

We chose the **Challenge Task**: a Reinforcement Learning agent for Gomoku (Five-in-a-Row) on a 9×9 board. This qualifies for the special incentive exemption from the Tic-Tac-Toe core task.

Our agent is a **Dueling Double DQN with Prioritized Experience Replay (PER)**, trained from zero knowledge via a 11-level adaptive curriculum over 80,000+ episodes.

---

## 2. Agent Architecture

```
Input: 3-channel board tensor (9×9)
  Channel 0: Agent's own pieces
  Channel 1: Opponent's pieces
  Channel 2: Constant plane = player ID (+1 or −1)

Conv2D(3→64, k=3) + BatchNorm + ReLU
Conv2D(64→128, k=3) + BatchNorm + ReLU    ← residual skip from conv1
Conv2D(128→128, k=3) + BatchNorm + ReLU
Conv2D(128→128, k=3) + BatchNorm + ReLU

Flatten → split into two streams (Dueling DQN):
  Value stream:     FC(10368→256) → FC(256→1)       ← how good is this board state?
  Advantage stream: FC(10368→256) → FC(256→81)       ← how much better is each move?

Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
```

**Training algorithm:** Double DQN — online network selects actions, target network evaluates them (updated every 500 steps). Optimizer: AdamW (lr=5×10⁻⁵, weight decay=1×10⁻⁴). Gamma=0.99. Gradient clipping norm=1.0.

**Prioritized Experience Replay:** samples transitions by TD-error magnitude (α=0.6), weighted by importance sampling (β=0.4→1.0). Focuses training on the most informative experiences.

---

## 3. Training Method

Training proceeded in two parallel tracks across 12 documented stages:

**Jeson's track (Stages 1–7, SimpleDQN):** Established baseline with sparse rewards → shaped rewards → curriculum vs ThreateningAgent → StrategicAgent → MinimaxAgent. Identified key failure modes: Bellman inconsistency when adding shaped rewards to a pre-trained sparse model (Stage 2: 98%→46% collapse), and the need for a minimum 25% random anchor to prevent strategy collapse (Stage 6).

**Rohan's track (Stages 8–11, DuelingDQN + PER):** Fresh training with the enhanced architecture using **shaped rewards from scratch** (where Bellman inconsistency is not a problem). Key reward signals: +0.3 for blocking opponent 4-in-a-row (ignore penalty), +0.2 for creating threats, −0.3 for missing an obvious block. Adaptive curriculum with 11 difficulty levels (Random → Strat-0.2 → ... → MM-0.7): promote when win rate ≥60% over last 100 curriculum games, demote when <25%. Dynamic random anchor: max(15%, 30% − level×2%) to prevent catastrophic forgetting.

**Stage 12 — DQN-guided MCTS (AlphaZero-style inference):** Wrapped the trained DQN in MCTS (200 simulations, UCB selection, tanh-squashed Q-value leaf evaluation). Result: performance *decreased* (−45pp vs Strat-0.5). Root cause: DQN Q-values are relative rankings within a position, not calibrated win-probability estimates across positions. AlphaZero works because the value network is trained *jointly with MCTS data* — using a separately-trained DQN as a MCTS leaf evaluator breaks this calibration.

---

## 4. Results

| Agent Version | vs Random | vs Strat-0.5 | vs MM-0.3 | vs MM-0.5 |
|---|---|---|---|---|
| Stage 1 — Jeson baseline (sparse, 20k eps) | 95–97% | ~10% | — | — |
| Stage 7 — Jeson best (shaped + curriculum) | 98.5% | 43% | — | — |
| **Stage 11 — Rohan final (Dueling+PER, 80k eps)** | **~95%** | **~60%** | **~55%** | **~40%** |
| Stage 12 — DQN + MCTS (200 sims) | 90% | 10% | — | 10% |

**Opponent definitions:**
- **StrategicAgent (Strat-X):** A hand-coded heuristic opponent that wins immediately if possible, blocks the agent's win, extends its own sequences, and falls back to random play. The skill level X (0.0–1.0) is the probability it plays strategically on any given turn — Strat-0.5 plays the smart move 50% of the time and random otherwise. Used as a mid-difficulty curriculum opponent.
- **MinimaxAgent (MM-X):** A minimax tree-search opponent with alpha-beta pruning. The skill level X controls the proportion of moves where it plays the full minimax best move vs. a random move — MM-0.5 is a strong tactical opponent; MM-1.0 is near-perfect. Represents a planning-based agent with genuine lookahead, unlike the reactive DQN.

**Best deployed model:** Stage 11 (`final.pt`) — Dueling DQN + PER trained over 80,000 episodes. The model also includes tactical inference-time overrides for forced moves (immediate win, block opponent win, open-four creation/defence) which improve move quality in critical positions without replacing the learned policy.

---

## 5. Key Lessons Learned

| Lesson | Evidence |
|---|---|
| Shaped rewards must be used from scratch — adding them to a pre-trained sparse model collapses performance | Stage 2: 98%→46% in 600 episodes; Phase 6: 88%→23% in 30k episodes |
| Minimum 25% random anchor is required — 20% causes strategy collapse | Stage 5 vs Stage 6 direct comparison |
| Adaptive curriculum (promote/demote by win rate) is essential for reaching Minimax | Fixed-schedule attempts stalled at 0% vs Minimax; Stage 10 reached level 8 |
| Ignore penalty (penalise missing obvious blocks immediately) is the key to defensive play | Stage 11 reliably blocks threats; all prior sparse agents did not |
| DQN Q-values are not compatible with MCTS leaf evaluation without joint training | Stage 12: MCTS hurt performance despite the DQN being strong alone |

---

## 6. Team Contributions

| Member | Contribution |
|---|---|
| Jeson Ramesh | Game environment, shaped reward design, Stages 1–7 training pipeline, curriculum opponent design (ThreateningAgent, StrategicAgent), Phase 5–6 continuation experiments, MCTS failure analysis |
| Rohan Beya | DuelingDQN + PER architecture, GomokuEnvShaped with ignore penalty, adaptive curriculum (train_rohan.py), Stages 8–12, MCTS agent implementation, final model training |

---

## 7. Next Steps (by March 30 Final Submission)

- Finalise technical report with full learning curves and ablation analysis
- Record 3–5 minute demo video (Human vs Agent, DQN vs Minimax)
- Consider sparse fine-tuning of Stage 11 model (3,000 episodes, sparse rewards only) to close the gap vs MM-0.5 without risking catastrophic forgetting
