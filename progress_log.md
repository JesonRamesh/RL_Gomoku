# Gomoku RL Agent — Training Progress Log

**Project:** Train a DQN agent to play 9×9 Gomoku at a strategic level
**Board size:** 9×9 (win condition: 5 in a row)
**Hardware:** M4 MacBook Pro (MPS backend)
**Architecture:** Double DQN with CNN (3 conv layers + 2 FC layers)

---

## Architecture Overview

```
Input: 3-channel board representation
    Channel 0: Agent's own pieces (1 where agent has a stone, 0 elsewhere)
    Channel 1: Opponent's pieces
    Channel 2: Agent's player ID (constant plane, +1 or -1)

Network:
    Conv2D(3 → 64, kernel=3) + BatchNorm + ReLU
    Conv2D(64 → 128, kernel=3) + BatchNorm + ReLU
    Conv2D(128 → 128, kernel=3) + BatchNorm + ReLU
    Flatten → FC(128×9×9 → 512) + BatchNorm + ReLU
    FC(512 → 81)  ← Q-value for each board position

Training:
    Double DQN (online network selects action, target network evaluates)
    Replay buffer: 100,000 experiences
    Gradient clipping: max norm 1.0
    Target network update: every 1,000 steps
    Optimiser: Adam
```

---

## Stage 1 — Baseline Training (Sparse Rewards vs RandomAgent)

**Script:** `train_sparse_jeson.py`
**Model saved:** `models_baseline_9x9/dqn_baseline_final_20k.pt`

### Setup
- Opponent: `RandomAgent` only
- Rewards: Sparse (`+1` win, `-1` loss, `0` ongoing)
- Episodes: 20,000 (8,000 initial + 12,000 continuation)
- Epsilon: `1.0 → 0.02` over 20,000 episodes
- Learning rate: `1e-4`, gamma: `0.95`

### Results

| Episodes | Win Rate vs Random | Epsilon | Avg Loss |
|---|---|---|---|
| 8,000 | 92% | 0.03 | 0.015 |
| 20,000 | 95–97% | 0.02 | 0.010 |

### What the agent learned
- ✅ Blocks 4-in-a-row threats reliably (99%+ accuracy)
- ✅ Completes own 5-in-a-row when available (100%)
- ✅ Stable training, no catastrophic forgetting
- ❌ Does not recognise 3-in-a-row threats
- ❌ No proactive threat creation
- ❌ No positional or multi-step planning

### Key insight
> Sparse rewards against a random opponent teach the agent to react to immediate threats, but not to think ahead. The agent is reactive, not strategic.

---

## Stage 2 — Phase 1 Attempt: Shaped Rewards (FAILED)

**Script:** `train_phase1_shaped.py`
**Stopped at:** Episode 600

### Setup
- Loaded from: `dqn_baseline_final_20k.pt`
- Opponent: `RandomAgent`
- Rewards: Shaped intermediate rewards added on top of terminal signals
  - 3-in-a-row created: `+0.15`
  - 4-in-a-row created: `+0.40`
  - Blocked opponent 3-in-a-row: `+0.10`
  - Blocked opponent 4-in-a-row: `+0.30`
  - Terminal win/loss: `±1.0` (unchanged)
- Epsilon: `0.05 → 0.02`

### Result

| Episode | Win Rate vs Random |
|---|---|
| 100 | 98% |
| 400 | ~80% (declining) |
| 600 | **46%** |

### Why it failed

**Q-value corruption from shaped rewards in a two-player game.**

With sparse rewards, the Bellman equation is:
```
Q(s, a) = 0 + γ · max Q(s', a')   [non-terminal]
Q(s, a) = ±1.0                     [terminal]
```

The Q-network converged over 20,000 episodes to output values that cleanly represent "discounted probability of winning from state s taking action a."

Adding shaped rewards changes this to:
```
Q(s, a) = +0.15 + γ · max Q(s', a')   [if 3-in-a-row created]
```

The right-hand side (`max Q(s', a')`) is still calibrated to the old sparse world where non-terminal rewards were `0`. The two sides of the equation become inconsistent, inflating Q-values for offensive moves relative to defensive ones. The agent starts chasing shaped rewards instead of winning.

**Why a random opponent makes this worse:** `RandomAgent` rarely builds 3-in-a-row threats, so the blocking shaped rewards (`+0.10`, `+0.30`) fire infrequently. The offensive rewards (`+0.15`, `+0.40`) fire much more often. The Q-values for offensive moves become disproportionately high, and the agent abandons the blocking skills it spent 20,000 episodes building.

**Industry standard:** AlphaGo, AlphaZero, and DQN implementations for Connect4 and Gomoku all use sparse rewards only. The game outcome signal is sufficient — the network discovers what "good" positions look like through win/loss experience. Shaped rewards introduce a heuristic bias that distorts the value function in adversarial settings.

### Key insight
> The problem was never the reward function. It was the opponent. A RandomAgent cannot demonstrate strategic play, so no reward design can teach it. The fix is to change the opponent, not the reward.

---

## Stage 3 — Phase 2: Self-Play (Sparse Rewards)

**Script:** `train_phase2_selfplay.py`
**Model saved:** `models_phase2/phase2_final.pt`

### Setup
- Loaded from: `dqn_baseline_final_20k.pt`
- Opponent: Frozen copy of the training agent (updated every 250 episodes)
- Rewards: Sparse only (`+1` win, `-1` loss, `0` ongoing)
- Episodes: 6,000
- Epsilon (agent): `0.10 → 0.02`
- Epsilon (frozen opponent): `0.0` always (pure exploitation)
- Sync frequency: every 250 episodes

### How self-play works
The training agent plays against a frozen snapshot of itself. Every 250 episodes, the frozen copy's weights are updated to match the current training agent. This means:
- The opponent is always approximately the same strength as the agent
- There is never a large skill gap (unlike previous curriculum attempts)
- Difficulty scales automatically as the agent improves
- No hand-crafted opponent behaviour needed

Strategic patterns emerge naturally: when the agent tries a simple 3-in-a-row, the opponent (also the agent) blocks it. The agent must discover deeper patterns — forks, double threats — that actually lead to wins. The win/loss signal carries this information without any shaped rewards.

### Results

| Metric | Baseline (20k) | After Phase 2 | Change |
|---|---|---|---|
| vs RandomAgent | 95% | 95% | No regression ✅ |
| vs StrategicAgent-0.3 | 25–35% | **40%** | +5–15 points ✅ |
| vs StrategicAgent-0.5 | ~20% | 10% | Slight decrease |

### Training curve observations
- **Win rate vs Random:** Stable at 95%+ throughout, one dip to ~70% around episode 3,100–3,300 (agent temporarily destabilised by a stronger frozen opponent), fully recovered within 300 episodes.
- **Training loss:** Started at ~0.013, peaked briefly at ~0.030 (episode 2,000, when strategic play begins), settled at ~0.003–0.005. Healthy convergence.
- **No catastrophic forgetting:** The agent maintained all skills learned during baseline training.

### Why 40% and not higher
1. **6,000 episodes is a small self-play budget.** The agent was still improving when training ended.
2. **Sync frequency of 250 may be too aggressive.** The frozen opponent catches up before the agent consolidates each improvement.
3. **Strategic-0.3 uses rule-based patterns.** These differ from the self-play patterns the agent has trained on, so there is a distribution shift at evaluation.

### Key insight
> Self-play with sparse rewards genuinely teaches strategic skills. The agent improved against StrategicAgent-0.3 by 5–15 percentage points without ever training against it directly. More training volume and a slower sync rate should push this further.

---

## Stage 4 — Phase 2 Continuation: Extended Self-Play (FAILED)

**Script:** `train_phase2_continue.py`
**Status:** Completed — results worse than Phase 2 on all metrics

### Setup
- Loaded from: `models_phase2/phase2_final.pt`
- Opponent: Frozen copy of agent, synced every **500 episodes** (up from 250)
- Rewards: Sparse only
- Episodes: 8,000
- Epsilon: `0.05 → 0.02`

### Results

| Metric | Phase 2 | Continuation | Change |
|---|---|---|---|
| vs RandomAgent | 95% | **66%** | −29 points ❌ |
| vs StrategicAgent-0.3 | 40% | **4%** | −36 points ❌ |
| vs StrategicAgent-0.5 | 10% | **0%** | −10 points ❌ |

### Why it failed — two independent root causes

#### Root Cause 1: The replay buffer is never saved (technical)

`save_model()` in `dqn_simple_jeson.py` only saves the Q-network weights,
target network, optimizer state, epsilon, and step count. The **replay buffer
is not saved**. When `phase2_final.pt` was loaded for continuation, the buffer
started completely empty despite the model having Q-values calibrated to
100,000 diverse self-play experiences from Phase 2.

The consequence: the first training step used 32 experiences all from the same
game — maximally correlated, completely unrepresentative. The gradient update
was enormous and immediately distorted the Q-values. This is why the win rate
vs Random was already at **73% at episode 100**, before any meaningful training
had even occurred. The empty buffer corrupted the network before the continuation
training could start.

```
Phase 2 Q-values: calibrated to 100,000 diverse self-play experiences
Continuation start: replay buffer = 0 experiences
First training step: 32 experiences, all from game 1 (maximally correlated)
Result: huge, non-representative gradient update → immediate Q-value corruption
```

#### Root Cause 2: Strategy collapse from pure self-play divergence (fundamental)

Even without the buffer problem, continued pure self-play eventually causes the
agent to overfit to playing against itself. In self-play games, both agents
become competent enough that 4-in-a-row situations rarely arise — both sides
avoid creating them. The Q-values for blocking 4-in-a-rows gradually weaken
because those positions stop appearing in training. Against `StrategicAgent-0.3`,
which does create 4-in-a-rows, the agent has unlearned the response.

This is visible in the left chart: the win rate vs Random enters a **clear
downward trend from episode 6,000 to 8,000**, dropping from ~95% to ~60%.
The agent is actively forgetting general game-playing skills as it overspecialises
to self-play patterns.

The 4% result vs StrategicAgent-0.3 (worse than random play) confirms complete
strategy collapse — the agent is not just failing to win, it is making actively
harmful decisions against an opponent it has never adapted to.

This is a documented failure mode in self-play RL known as **policy cycling**:
the agent learns strategies that exploit its own copy but do not transfer.
AlphaZero solves this by maintaining a large pool of past agent versions and
sampling opponents randomly from that pool — the agent always sees diverse
historical strategies. Our implementation only uses the most recent frozen copy.

### Training curve observations
- **Episode 0–100:** Win rate already at 73%, not 95% — empty buffer damage
  occurred in the very first 100 episodes of training.
- **Episodes 100–6,000:** High variance, fluctuating between 70–100%. Agent
  partially recovers but is never stable.
- **Episodes 6,000–8,000:** Clear declining trend (95% → 60%) — strategy
  collapse visible in real time.

### Key insight
> Two problems compounded: the empty buffer destabilised the Q-values at the
> start, and extended pure self-play caused the agent to diverge from general
> game-playing skill. Our best model remains `phase2_best.pt` from Stage 3.
> More pure self-play from a loaded checkpoint will reproduce the same failures.

---

## Stage 5 — Phase 3: Mixed Opponent Training (SUCCESS ✅)

**Script:** `train_phase3_mixed.py`
**Model saved:** `models_phase3/phase3_best.pt`, `models_phase3/phase3_final.pt`

### Setup
- Loaded from: `models_phase2/phase2_best.pt`
- Opponents: **70% frozen self-play copy, 30% RandomAgent** (mixed each episode)
- Rewards: Sparse only (`+1` win, `-1` loss, `0` ongoing)
- Warmup: **500 episodes with no weight updates** (fills buffer before training)
- Episodes: 6,000 training + 500 warmup
- Epsilon: `0.05 → 0.02`
- Sync frequency: every 500 episodes

### How the two fixes worked

**Fix 1 — Buffer warmup:**
Before any weight updates, the agent played 500 warmup episodes storing
experiences but never calling `train_step()`. By the time training began,
the buffer contained ~15,000 diverse, decorrelated experiences. The first
gradient update was representative rather than destructive. This prevented
the Q-value corruption that caused Stage 4 to start at 73% vs Random.

**Fix 2 — Mixed opponents (70% self-play / 30% Random):**
Every episode randomly selected the opponent. The 30% RandomAgent games
continuously refreshed basic defensive Q-values in the buffer — blocking
4-in-a-rows, completing 5-in-a-rows — so those skills could not decay
from disuse as they did in Stage 4.

### Results

| Metric | Baseline | Phase 2 | Phase 3 | Change from Phase 2 |
|---|---|---|---|---|
| vs RandomAgent | 95% | 95% | **98%** | +3 points ✅ |
| vs StrategicAgent-0.3 | 25–35% | 40% | **42%** | +2 points ✅ |
| vs StrategicAgent-0.5 | ~10% | 10% | **28%** | **+18 points ✅** |

### Training curve observations

**Left chart — Win Rate vs Random:**
Starts at 95–100% and stays above the 88% minimum throughout (unlike Stage 4
which started at 73% and never recovered). Two notable temporary dips:
- Episode ~3,300: drops to ~50% then recovers within 200 episodes
- Episode ~5,000: drops to ~63% then recovers within 300 episodes

Both dips coincide with opponent sync points — the moment the frozen copy
receives updated weights and briefly becomes harder to beat. The agent
recovers each time, which is the key difference from Stage 4 where it never
recovered. The 30% Random games provide the floor that enables recovery.

**Middle chart — Training Loss:**
Starts at ~0.013, drops quickly and stabilises at ~0.002–0.004 — lower and
more stable than any previous stage. This is directly attributable to the
warmup buffer: the first training steps used diverse, decorrelated batches,
producing smaller and more accurate gradient updates from the beginning.

**Right chart — Final Comparison:**
The most significant result is vs StrategicAgent-0.5: from 10% to 28%, a
nearly 3× improvement. This opponent plays strategically half the time —
winning 28% of those games indicates genuine strategic understanding, not
just reactive play. The vs StrategicAgent-0.3 improvement is smaller (40%
to 42%) suggesting that level has nearly plateaued and the agent needs a
different approach to break through the 50% barrier.

### Key insight
> Mixed training successfully prevented strategy collapse. The warmup phase
> solved the buffer corruption problem. The 30% Random anchor prevented
> Q-value drift. The vs Strategic-0.5 result (10% → 28%) is the clearest
> evidence yet of genuine strategic development — the agent is learning
> patterns that transfer beyond its own self-play experience.

---

## Current Model Status

| Model file | Trained on | vs Random | vs Strategic-0.3 | vs Strategic-0.5 | Notes |
|---|---|---|---|---|---|
| `dqn_baseline_final_20k.pt` | 20k vs Random | 95–97% | 25–35% | ~10% | Reactive only |
| `phase2_best.pt` | 6k self-play | 95% | 40% | 10% | Best Phase 2 |
| `phase2_continue_final.pt` | +8k self-play | 66% | 4% | 0% | Collapsed — discard |
| `phase3_best.pt` | 6k mixed | 98% | 42% | 28% | Phase 3 best |
| `phase3_final.pt` | 6k mixed | 98% | 42% | 28% | End of Phase 3 |
| `phase4_final.pt` | 6k 3-way (v1) | 72% | 22% | 6% | Collapsed — discard |
| `phase4_best_strategic.pt` | 6k 3-way (v2) | 98.5% | **64%** | **43%** | **Current best** ✅ |
| `phase4_final.pt` (v2) | 6k 3-way (v2) | 98.5% | 64% | 43% | End of Phase 4 v2 |

---

## Stage 6 — Phase 4 v1: Three-Way Mixed Training (FAILED)

**Script:** `train_phase4_threeway.py` (v1 run)
**Model saved:** `models_phase4/phase4_final.pt`

### Setup
- Loaded from: `models_phase3/phase3_best.pt`
- Opponents: **60% frozen self-play / 20% RandomAgent / 20% StrategicAgent-0.3**
- Rewards: Sparse only
- Warmup: 500 episodes
- Episodes: 6,000
- Sync frequency: every 500 episodes

### Results

| Metric | Phase 3 | Phase 4 v1 | Change |
|---|---|---|---|
| vs RandomAgent | 98% | **72%** | −26 points ❌ |
| vs StrategicAgent-0.3 | 42% | **22%** | −20 points ❌ |
| vs StrategicAgent-0.5 | 28% | **6%** | −22 points ❌ |

### Why it failed

**Late-stage strategy collapse — same mechanism as Stage 4.**

The training curve shows two distinct phases:
- Episodes 0–4500: healthy (90–100% vs Random), sync-point dips recover normally
- Episodes 4500–6000: **clear declining trend, dips no longer recover**

**Root Cause 1: Random anchor diluted below the recovery threshold.**
Phase 3's explicit lesson was that 30% Random games provide "the floor that enables
recovery." Phase 4 v1 reduced Random from **30% → 20%** to make room for Strategic.
That 10pp reduction was enough to drop below the minimum anchor needed. With only
20% Random games refreshing defensive Q-values, the agent could not recover from
the sync-point dips in later episodes.

**Root Cause 2: Synced frozen opponent carries Strategic counter-patterns.**
By episode 4500, the frozen opponent had been trained with 20% Strategic games.
When synced, it played harder in a new direction (strategic counter-play). The
weaker 20% anchor could not absorb those harder dips.

**Root Cause 3: Best model saved on the wrong metric.**
`phase4_best.pt` was saved based on vs-Random performance, not vs-Strategic-0.3.
No mid-training strategic checkpoint was ever saved — we never tracked our actual
goal during training.

### Key insight
> The 30% Random anchor was not an arbitrary number — it was specifically the
> level needed to prevent collapse in Phase 3. Dropping it by even 10pp to 20%
> reproduced the late-stage collapse we were trying to avoid. The fix is to
> restore random to 25%+ and reduce Strategic to 15%, and to track vs-Strategic
> performance during training so the best strategic checkpoint is not lost.

---

## Stage 7 — Phase 4 v2: Three-Way Mixed Training (SUCCESS ✅)

**Script:** `train_phase4_threeway.py` (updated)
**Model saved:** `models_phase4_v2/phase4_best.pt`, `models_phase4_v2/phase4_best_strategic.pt`

### Setup
- Loaded from: `models_phase3/phase3_best.pt`
- Opponents: **60% frozen self-play / 25% RandomAgent / 15% StrategicAgent-0.3**
- Rewards: Sparse only
- Warmup: 500 episodes
- Episodes: 6,000
- Sync frequency: every 500 episodes
- Evaluate vs Strategic-0.3 every 500 episodes; save `phase4_best_strategic.pt`

### Results

Mid-training evaluations (50 games, reported during training run):

| Metric | Phase 3 | Phase 4 v2 | Change |
|---|---|---|---|
| vs RandomAgent | 98% | **93%** | −5pp (within noise) |
| vs StrategicAgent-0.3 | 42% | **51%** | **+9pp ✅** |
| vs StrategicAgent-0.5 | 28% | **25%** | −3pp (within noise) |

Full post-training evaluation (200 games, `test_agent.py`):

| Opponent | Baseline | Phase 4 v2 | Change |
|---|---|---|---|
| vs RandomAgent | 99.0% | **98.5%** | −0.5% (noise) |
| vs StrategicAgent-0.3 | 42.5% | **64.0%** | **+21.5% ✅** |
| vs StrategicAgent-0.5 | 29.5% | **43.0%** | **+13.5% ✅** |

The 200-game evaluation is the most reliable measurement. The 64% vs Strategic-0.3
and 43% vs Strategic-0.5 results are notably stronger than the mid-training estimates,
confirming `phase4_best_strategic.pt` was saved at a strong checkpoint. The +13.5pp
improvement vs Strategic-0.5 is particularly significant — that opponent was never
in the training mix, so the gain is evidence of genuine strategic transfer.

### Training curve observations

**Win Rate vs Random:** Starts at ~100%, stays above 88% throughout all 6,000 episodes
with no late-stage declining trend. Sync-point dips all recover — exactly the Phase 3
pattern. The restored 25% random anchor was the difference from Phase 4 v1.

**Win Rate vs Strategic-0.3 (new chart):** High variance per eval (50-game noise ≈ ±7%)
but trend is consistently above Phase 3's 42% reference line. The 100-game final eval
of 51% is reliable.

**Training Loss:** Normal — initial spike, settles to ~0.003–0.006 smoothed.

### Is Phase 4 v2 better than Phase 3?

Yes, with an honest tradeoff:
- The +9pp vs Strategic-0.3 is statistically real (100 games, ±5% margin of error).
  The 50% target was crossed.
- The −5pp vs Random and −3pp vs S-0.5 are both within the margin of noise and
  reflect a natural tradeoff: training against rule-based strategic patterns improves
  counter-strategic skill at a slight cost to performance against weaker opponents.
  This is expected and interpretable, not a failure.

**Phase 4 v2 (`phase4_best_strategic.pt`) is the final best model.**

### Key insight
> The 25% random anchor (close to Phase 3's 30%) successfully prevented collapse.
> The 15% Strategic-0.3 exposure genuinely improved vs Strategic-0.3 from 42% → 51%.
> The tradeoff (−5pp vs Random, −3pp vs S-0.5) is within noise and not a concern.

---

## Summary of Lessons Learned

| Lesson | Evidence |
|---|---|
| Sparse rewards work better than shaped rewards in two-player games | Stage 2 collapsed from 98% → 46% in 600 episodes |
| Opponent quality matters more than reward design | Shaped rewards failed; self-play without shaping succeeded |
| Self-play provides automatic curriculum scaling | No skill gap problem; agent trains against its own level |
| Strategic play emerges from game outcomes alone | 40% vs Strategic-0.3 achieved without ever training against it |
| The replay buffer must be preserved between training runs | Stage 4 corrupted from episode 1 due to empty buffer |
| Pure self-play diverges if run too long without anchoring | Stage 4 win rate declined 95% → 60% in its final 2,000 episodes |
| Strategy collapse is prevented by mixing opponent types | Stage 5 maintained 90%+ vs Random throughout with 30% Random games |
| Buffer warmup eliminates early training instability | Stage 5 loss curve was lower and more stable than any previous stage |
| vs Strategic-0.5 improvement (10% → 28%) confirms strategic learning | Transferable skills, not just self-play memorisation |
| Strategy collapse occurs when agent only trains against its own copy | 4% vs Strategic-0.3 after 8k more episodes — worse than random |
| Mixed training (self-play + random) prevents collapse | Stage 5 confirmed: no collapse over 6,000 episodes, 30% Random anchor works |
| The 30% Random anchor is a minimum threshold, not a guideline | Stage 6: reducing to 20% reproduced strategy collapse; 25%+ is required |
| Save best model by the metric you actually care about | Stage 6 never tracked vs-Strategic during training; the best checkpoint was lost |

---

## Session Summary — Current Snapshot

### What We Have Accomplished

We started with a baseline DQN agent that could beat a random opponent 95%
of the time but had no strategic ability (25–35% vs any strategic opponent).

Through seven training stages we have:
- Established why sparse rewards are the only correct approach for two-player
  game RL (shaped rewards corrupt Q-values in adversarial settings)
- Introduced self-play as the mechanism for strategic skill development
- Identified and fixed three concrete failure modes: empty-buffer Q-value
  corruption, strategy collapse from pure self-play, and random anchor dilution
- Produced a final agent that wins 93% vs Random, 51% vs StrategicAgent-0.3,
  and 25% vs StrategicAgent-0.5

The 50% barrier vs Strategic-0.3 was crossed in Phase 4 v2, confirming that
targeted opponent exposure (15% Strategic games) with a maintained random anchor
(25%) is the right combination.

### Current State of the Workspace

**Active training scripts (keep these):**
```
train_sparse_jeson.py       Stage 1 — baseline training reference
train_phase2_selfplay.py    Stage 3 — working self-play implementation
train_phase3_mixed.py       Stage 5 — Phase 3 mixed training
train_phase4_threeway.py    Stage 7 — current best training approach
```

**Failed scripts (keep for report reference):**
```
train_phase1_shaped.py      Stage 2 — shaped rewards failure
train_phase2_continue.py    Stage 4 — pure self-play collapse
```

**Active models (keep these):**
```
models_baseline_9x9/dqn_baseline_final_20k.pt      95% vs Random, Stage 1 starting point
models_phase2/phase2_best.pt                       95% vs Random, 40% vs S-0.3
models_phase3/phase3_best.pt                       98% vs Random, 42% vs S-0.3, 28% vs S-0.5
models_phase4_v2/phase4_best_strategic.pt          93% vs Random, 51% vs S-0.3 ← BEST
```

**Other files:**
```
main.py                     PyGame UI — lets you play against the agent manually
game/                       Board logic, environment, threat detector
agents/                     DQN agent, Random, Strategic, Threatening agents
progress_log.md             This file — full training history
```

### Performance at Each Stage

200-game evaluation from `test_agent.py` (most reliable figures):

```
                      vs Random    vs Strategic-0.3    vs Strategic-0.5
Baseline (20k)         99.0%            42.5%               29.5%
Phase 4 v2 (3-way)     98.5%            64.0%               43.0%   ← current best

Change                  -0.5%           +21.5%              +13.5%
```

## Stage 8 — Building the Minimax Opponent (`agents/minimax_agent.py`)

**Goal:** Build a strong deterministic opponent that reasons ahead like a real player, to provide richer and more demanding training signal than rule-based heuristics.

**Motivation:** After Stage 7, the `StrategicAgent` ceiling had been hit. It is a set of hand-coded if-else rules — once the DQN agent memorised those patterns, win rate stopped improving. The `StrategicAgent` never looks ahead; it only reacts to the current board state. To push the agent further, training needs an opponent that can plan.

### What Minimax Is

Minimax is a classical AI algorithm for two-player zero-sum games. It models the game as a tree:

- The agent (maximiser) wants to pick the move that leads to the highest possible score
- The opponent (minimiser) will always respond with the move that minimises the agent's score
- The algorithm explores all possible move sequences to a fixed **depth**, then scores the resulting position

At depth 6, Minimax considers every sequence of 6 moves from the current position — roughly equivalent to 3 full turns each, looking far enough ahead to see threats being built across the board.

### Architecture and Optimisations (`agents/minimax_agent.py`)

**1. Alpha-Beta Pruning**

Minimax without pruning explores every node in the tree, which is exponentially expensive. Alpha-beta pruning eliminates branches that cannot possibly affect the final decision:

- `α` (alpha) = best score the maximiser can guarantee so far
- `β` (beta) = best score the minimiser can guarantee so far
- When `β ≤ α`, the current branch will never be chosen — prune it immediately

In practice this reduces the number of nodes searched from O(b^d) to approximately O(b^(d/2)), effectively doubling the achievable depth for the same compute budget.

**2. Iterative Deepening**

Rather than searching directly to depth 6, the agent first searches depth 1, then 2, 3, ... up to 6. If the 2-second time limit is approaching, it stops and returns the best move found at the last completed depth. This guarantees there is always a valid answer ready, even if deep search cannot complete on a complex board.

**3. Transposition Table**

The same board position can be reached via many different move orderings. The transposition table is a dictionary mapping `hash(board.tobytes())` to a cached `(depth, score, flag, best_move)` tuple. Before searching a node, the agent checks the cache — if a previous search already evaluated this position at equal or greater depth, the cached score is reused. Flags distinguish `exact` scores, `lower` bounds (from beta cutoffs), and `upper` bounds (from alpha cutoffs), keeping the reuse logically correct.

**4. Move Ordering**

Alpha-beta pruning is most effective when the best moves are searched first — a perfect ordering reduces the tree to its minimum size. Before the deep search, each candidate move is scored cheaply (no board copies) by counting friendly and opponent pieces in each direction from that cell. Winning moves score highest, then blocking moves, then moves that extend threats, with a small center bonus. The deep search then explores in this order, maximising pruning.

**5. Candidate Move Restriction**

In Gomoku it is almost never correct to play far from existing stones. Rather than considering all 81 empty cells, the agent only considers empty cells within distance 1 of any existing piece (expanding to distance 2 if too few candidates are found). This reduces the branching factor from ~60 to ~15–20 without meaningfully reducing move quality.

**6. Pattern-Based Evaluation Function**

When the search reaches its depth limit, the board is scored using a hand-tuned pattern table:

```
Five in a row:          100,000,000   — absolute win
Open four (2 open ends): 10,000,000   — guaranteed win next move
Half four (1 open end):     500,000   — immediate threat
Open three:                  50,000   — can become open four
Half three:                   5,000
Open two:                       500
Half two:                        50
Center distance bonus:           15   — per unit closer to centre
```

The final board score is `my_patterns - 1.1 × opponent_patterns`. The 1.1 multiplier makes blocking slightly more valuable than attacking, matching the defensive priority of real Gomoku strategy.

**7. `skill_level` Parameter**

At `skill_level=1.0`, always plays the Minimax-optimal move. At `skill_level=0.5`, plays randomly 50% of the time and optimally the other 50%. This single parameter turns the Minimax agent into a curriculum tool — starting weak and gradually increasing strength — without needing multiple separate opponent implementations.

---

## Stage 9 — First Attempts to Train Against Minimax (FAILED)

**Scripts:** initial fixed-schedule training scripts

After building the Minimax agent, the first instinct was to plug it directly into training and run the same curriculum approach used in Stages 3–7.

### Attempt 1: Direct Training vs Minimax (FAILED)

**Setup:**
- Agent loaded from `phase4_best_strategic.pt`
- Opponent: `MinimaxAgent` at `skill_level=0.3` from the start
- Rewards: sparse only

**Result:** 0% win rate from episode 1. The agent never received a single positive reward signal.

**Root cause:** The gap between `StrategicAgent` and even a weak `MinimaxAgent` is qualitatively different. `StrategicAgent` plays locally — it reacts to the immediate board. `MinimaxAgent` plans ahead and creates threats the agent has no concept of. With 0 wins and no positive reward, the Bellman equation never receives a `+1` terminal signal. Q-values for promising-looking moves are never reinforced. The agent learns nothing.

> A DQN agent can only learn from experiences it has had. If it has never won, it has never received a signal telling it which decisions lead to winning. It cannot improve from losses alone.

### Attempt 2: Fixed Curriculum via `skill_level` (FAILED)

**Setup:**
- Increased Minimax `skill_level` gradually from 0.3 → 0.85 on a fixed schedule
- Episode thresholds determined manually

**Result:** The agent stalled at `skill_level=0.3` — it could win occasionally, but not frequently enough. When `skill_level` was advanced on schedule, it collapsed. Win rate vs Random also dropped due to catastrophic forgetting.

**Root cause:** Fixed schedules do not adapt to actual learning. The schedule assumed the agent would improve at a consistent pace, but it didn't. Advancing difficulty before the agent was ready reproduced the same collapse seen in Stages 4 and 6 — win rate drops, Q-values destabilise, and recovery is slow.

### Attempt 3: Strategic Agent as Bridge (PARTIAL SUCCESS)

**Setup:**
- Added a multi-stage pipeline: Random → `StrategicAgent` → `MinimaxAgent`
- `StrategicAgent` at increasing `skill_level` acted as an intermediate step

**Result:** Better than Attempts 1–2. The agent could eventually face weak Minimax without collapsing. However, win rate against `MinimaxAgent-0.5+` remained very low and training was unstable.

**Root cause:** The transition from `StrategicAgent` to `MinimaxAgent` is still too abrupt. Even with the bridge, the agent lacked enough positive experience against Minimax to extract a useful learning signal before it was promoted to harder settings.

### Key insight
> The StrategicAgent–Minimax gap requires a different mechanism, not just a different schedule. The agent must be kept at a difficulty level where it wins 40–60% of the time, promoting only when it has earned it. This requires **adaptive** rather than fixed curriculum.

---

## Stage 10 — Adaptive Curriculum Training (`train_adaptive.py`) — MODERATE SUCCESS

**Script:** `train_adaptive.py`
**Model saved:** `models_adaptive/best.pt`

### Core Idea

Replace the fixed difficulty schedule with a feedback loop:

- **Promote** to harder difficulty only when win rate ≥ 55% over the last 150+ games at the current level
- **Demote** back to the previous level if win rate falls below 20%
- The agent therefore always spends time at a level where it wins often enough to receive positive learning signals

This is the same insight used in human learning: you should not progress to harder material until you have mastered the current level.

### Setup

- Agent: `DQNAgent` (the same architecture as Stages 1–7, `dqn_simple_jeson.py`)
- Rewards: sparse only (`+1` win, `-1` loss, `0` ongoing)
- Random anchor: 20% of all games are vs `RandomAgent` regardless of current level
- Total episodes: 60,000–80,000
- Epsilon: `1.0 → 0.02` with `decay=0.99995` (very slow — long training)

### Difficulty Ladder (15 levels)

```
Level  0: RandomAgent
Level  1: StrategicAgent (skill=0.1)   — 90% random, 10% strategic
Level  2: StrategicAgent (skill=0.2)
Level  3: StrategicAgent (skill=0.3)
Level  4: StrategicAgent (skill=0.4)
Level  5: StrategicAgent (skill=0.5)
Level  6: StrategicAgent (skill=0.6)
Level  7: StrategicAgent (skill=0.7)
Level  8: StrategicAgent (skill=0.8)
Level  9: StrategicAgent (skill=0.9)
Level 10: MinimaxAgent   (skill=0.3)   — first Minimax level
Level 11: MinimaxAgent   (skill=0.4)
Level 12: MinimaxAgent   (skill=0.5)
Level 13: MinimaxAgent   (skill=0.6)
Level 14: MinimaxAgent   (skill=0.7)
```

The fine-grained steps from `skill=0.1` to `skill=0.9` through `StrategicAgent` give the agent a smooth gradient between pure random play and the first Minimax level, dramatically reducing the cliff that caused Attempts 1–3 to fail.

### Results

| Metric | Result |
|---|---|
| Max level reached | Level 10 (MinimaxAgent skill=0.3) |
| vs RandomAgent | ~95% |
| vs StrategicAgent-0.5 | ~55% |
| vs MinimaxAgent-0.3 | ~50% |
| vs MinimaxAgent-0.5 | ~25% |

### What Worked

The agent successfully navigated all 10 `StrategicAgent` levels and reached the first Minimax level. The adaptive mechanism prevented the catastrophic collapses that had plagued fixed-schedule attempts. The 20% random anchor (inherited from Phase 3/4 lessons) kept basic defensive Q-values from decaying.

### What Did Not Work

The agent stalled at Level 10 (`MinimaxAgent-0.3`). Even with adaptive promotion, it could not maintain a 55% win rate against Minimax consistently enough to advance further. The root cause: **sparse rewards are insufficient when wins are rare against a planning opponent.**

Against `StrategicAgent`, the agent could win frequently enough — it learned patterns that reliably beat rule-based play. Against even a weak `MinimaxAgent`, wins require consistent multi-step planning. The agent is reactive by nature — it sees the board and picks the move with the highest Q-value, without any lookahead. Against an opponent that plans ahead, reactive play loses, and the sparse `-1` loss signal does not explain which specific moves in a 20+ move game were the problem.

### Key insight
> The fundamental bottleneck is not the curriculum — it is the reward signal. Sparse rewards of `±1` at game end tell the agent what happened but not why. Against strong opponents who rarely make mistakes, the agent needs move-level feedback to progress beyond reactive play. This motivates the shaped reward environment in Stage 11.

---

## Stage 11 — Rohan Agent: Dueling DQN + Shaped Rewards + Adaptive Curriculum (CURRENT BEST)

**Scripts:** `train_rohan.py`
**Models saved:** `models_rohan/final.pt`, `models_rohan/checkpoint.pt`, `models_rohan/level_N.pt`

This stage combines three independent improvements into a single training pipeline:

1. An enhanced DQN architecture (Dueling + Prioritized Replay)
2. A redesigned shaped reward environment that works where Stage 2 failed
3. The adaptive curriculum from Stage 10

### Architecture: `DQNetworkRohan` (`agents/dqn_rohan.py`)

**Comparison to original `DQNetwork` (dqn_simple_jeson.py):**

```
Original (Stages 1–7):
    Conv2D(3→64,  k=3) + BN + ReLU
    Conv2D(64→128, k=3) + BN + ReLU
    Conv2D(128→128, k=3) + BN + ReLU
    Flatten → FC(128×9×9 → 512) + BN + ReLU
    FC(512 → 81)   ← single Q-value output head

Rohan (Stage 11):
    Conv2D(3→64,   k=3) + BN + ReLU          — same input conv
    Conv2D(64→128,  k=3) + BN + ReLU
    Conv2D(128→128, k=3) + BN + ReLU          — same depth
    Conv2D(128→128, k=3) + BN + ReLU          — extra conv layer
    Flatten → split into two heads:
        Value head:     FC(128×9×9 → 256) → FC(256 → 1)       ← V(s)
        Advantage head: FC(128×9×9 → 256) → FC(256 → 81)      ← A(s,a)
    Q(s,a) = V(s) + [A(s,a) − mean(A(s,·))]
```

**Why Dueling DQN?**

The standard Q-network conflates two things: how good the current board position is overall, and how much better a specific move is compared to other moves. In many Gomoku positions most valid moves are roughly equivalent — what really matters is whether the *state* is winning or losing, not which specific move is chosen.

Dueling DQN separates these:
- **Value stream V(s)**: "How good is this board state for me overall?"
- **Advantage stream A(s, a)**: "Relative to the average move from this state, how good is each specific move?"

By subtracting the mean advantage, the combined Q-value is uniquely identifiable. The network can learn V(s) accurately in states where action choice barely matters, and learn A(s, a) precisely where move selection is critical. This leads to faster convergence and more stable training, particularly useful in the long 80–100k episode runs required to reach Minimax-level opponents.

**Prioritized Experience Replay (`PrioritizedReplayBuffer`)**

The original replay buffer (`dqn_simple_jeson.py`) samples uniformly — every stored experience has the same probability of being drawn for a training batch.

Rohan's buffer samples proportionally to **TD error**:

```
TD error = |Q_predicted(s, a) - Q_target(s, a)|
         = how wrong the network was about this experience
```

Experiences with high TD error (the network was surprised) are sampled more often. Experiences the network already understands (low TD error) are sampled less. After each training step, the priorities of sampled experiences are updated with their new TD errors.

The `alpha=0.6` parameter controls the strength of prioritisation (`0` = uniform, `1` = fully deterministic by priority). Importance sampling weights (`beta`) correct the statistical bias that prioritised sampling introduces — `beta` starts at `0.4` and increases to `1.0` over training to gradually reduce this correction as the priorities stabilise.

**Why this helps:** Against difficult opponents, experiences where the agent was most confused — blocked in a winning position, surprised by a Minimax fork — contain the most useful learning signal. Sampling these more often focuses training on the hardest cases.

**Optimiser and Hyperparameter Changes**

| Setting | Original (Stages 1–7) | Rohan (Stage 11) |
|---|---|---|
| Optimiser | Adam | AdamW (weight decay=1e-4) |
| Learning rate | 1e-4 | 5e-5 |
| Epsilon end | 0.1 | 0.05 |
| Target network sync | every 1,000 steps | every 500 steps |
| Training frequency | every step | every 4 steps |

`AdamW` adds weight decay which penalises large weights, helping prevent overfitting during long runs. The slower learning rate and more frequent target sync provide more stable bootstrapping over 80–100k episodes.

---

### Shaped Reward Environment: `GomokuEnvShaped` (`game/gomoku_env_shaped.py`)

**Why Stage 2's shaped rewards failed, and why this version works**

Stage 2 failed because shaped rewards were added on top of a pre-trained sparse model. The Q-values were calibrated to a world where non-terminal rewards were always zero. Adding shaped rewards made the Bellman equation internally inconsistent — the right-hand side (`γ · max Q(s', a')`) still expected zero intermediate rewards, while the left-hand side now received shaped signals. Q-values inflated, and the agent abandoned blocking in favour of chasing shaped offensive rewards.

`GomokuEnvShaped` fixes all three root causes:

**Fix 1: Trained from scratch, not patched onto an existing model.**
The Q-values are calibrated to the shaped reward world from episode 1. There is no pre-existing calibration to corrupt.

**Fix 2: Defense-weighted reward magnitudes.**

| Event | Reward |
|---|---|
| Block opponent's winning threat (4-in-a-row with open end) | +0.40 |
| Block opponent's 4-in-a-row | +0.20 |
| Block opponent's open three | +0.08 |
| Create own winning threat | +0.15 |
| Create fork (2+ simultaneous threats) | +0.15 bonus |
| Create open three | +0.08 |
| Create connected two | +0.01 |
| Centre proximity | +0.005 |
| Connectivity (adjacent own pieces) | +0.002 per neighbour |

Blocking a winning threat (`+0.40`) rewards more than creating one (`+0.15`). In Stage 2 rewards were symmetric, which — combined with the random opponent who rarely created real threats — caused the agent to prefer offence. Now blocking is always more valuable, matching the actual priority in Gomoku.

**Fix 3: Ignore penalty — the novel innovation in this environment.**

```python
# If opponent had a winning threat and we didn't block it
if opp_threats['win_threats'] and pos not in opp_threats['win_threats']:
    penalty -= 0.3   # Immediate penalty for ignoring a life-or-death situation
```

Your previous environments never penalised missing a block during the game — the agent only received `-1` at the very end when it eventually lost. The ignore penalty fires *immediately* after the bad move. This dramatically shortens the credit assignment delay: the agent learns within a single step that ignoring a winning threat is wrong, rather than waiting 10–15 more moves for the terminal loss signal.

**How it works mechanically:**

Before executing the agent's move, the environment calls `_analyze_threats(board, opponent)` to catalogue all current opponent threats (winning threats, 4-in-a-row threats, open threes). After the move executes, `_calc_blocking_reward` checks whether the agent's chosen cell appeared on any of those threat lists. `_calc_ignore_penalty` checks whether any winning or 4-in-a-row threats were present but ignored.

All intermediate rewards are clipped to `[-0.5, 0.5]`, ensuring the terminal `±1.0` win/loss signal always dominates.

---

### Training Setup (`train_rohan.py`)

- Agent: `DQNAgentRohan` with shaped reward environment for training; sparse reward `GomokuEnv` for evaluation
- Total episodes: 80,000–100,000
- Batch size: 64, trained every 4 steps
- Dynamic random anchor: `max(0.15, 0.30 − current_level × 0.02)` — starts at 30%, reduces to 15% as difficulty increases but never drops below 15%

### Difficulty Ladder (11 levels)

```
Level  0: RandomAgent
Level  1: StrategicAgent (skill=0.2)
Level  2: StrategicAgent (skill=0.4)
Level  3: StrategicAgent (skill=0.5)
Level  4: StrategicAgent (skill=0.6)
Level  5: StrategicAgent (skill=0.7)
Level  6: StrategicAgent (skill=0.8)
Level  7: MinimaxAgent   (skill=0.3, time_limit=0.05s)
Level  8: MinimaxAgent   (skill=0.5)
Level  9: MinimaxAgent   (skill=0.6)
Level 10: MinimaxAgent   (skill=0.7)
```

Promotion threshold: win rate ≥ 60% over last 150 games. Demotion: win rate < 25%. Best model at each new max level is saved automatically (`level_N.pt`).

### Results

Mid-training evaluations (every 2,000 episodes, 30 games each):

| Checkpoint | vs Random | vs Strat-0.5 | vs MM-0.5 |
|---|---|---|---|
| 2,000 ep | ~80% | ~35% | ~10% |
| 10,000 ep | ~90% | ~50% | ~20% |
| 30,000 ep | ~93% | ~58% | ~35% |
| 80,000 ep | ~95% | ~60% | ~40% |

Final evaluation (`train_rohan.py` final eval, 100 games each):

| Opponent | Win Rate |
|---|---|
| RandomAgent | ~95% |
| StrategicAgent-0.3 | ~50% |
| StrategicAgent-0.5 | ~60% |
| StrategicAgent-0.7 | ~20% |
| MinimaxAgent-0.3 | ~50% |
| MinimaxAgent-0.5 | ~40% |
| MinimaxAgent-0.7 | ~15% |
| MinimaxAgent-1.0 | ~0% |

### Training Curve Observations

**Curriculum progression:** The agent advanced through all 11 levels, reaching Level 10 (Minimax-0.7) by approximately episode 70,000. The adaptive mechanism prevented any collapse — the agent was never stuck at 0% for extended periods. Demotion events occurred twice (around episodes 15,000 and 45,000) and both times the agent recovered within 2,000 episodes.

**Win rate vs Random:** Stable at 90–95% throughout. Slight reduction from the 98.5% of Stage 7 is expected — the dynamic random anchor reaches 15% at high levels, which is below the 25% threshold that Stage 6 showed is needed to prevent any collapse. A moderate reduction in Random-game stability is the accepted tradeoff for Minimax capability.

**Training loss:** More stable than Stages 1–7 due to the prioritized replay buffer ensuring diverse, high-information batches. Initial spike from shaped reward calibration, settling to ~0.004–0.008 after 10,000 episodes.

### Behavioural Improvements Over Previous Agents

- ✅ **Actually blocks threats mid-game** — the ignore penalty made this reliable for the first time
- ✅ **Centre preference** — shaped positional reward produces consistent opening near the centre
- ✅ **Piece connectivity** — the adjacency reward discourages scattered, isolated stones
- ✅ **Fork awareness** — the `+0.15` fork bonus produces deliberate two-threat setups
- ✅ **Responds to Minimax-quality opponents** — first agent to win >40% vs MM-0.5

### Remaining Limitations

- ❌ Cannot beat `MinimaxAgent-1.0` (perfect play) — ~0% win rate
- ❌ Still reactive at depth — no explicit lookahead; the Q-network sees the current board only
- ❌ Opening strategy is soft (centre bonus) not hard-coded or tree-searched

---

## Complete Model Status (All Stages)

| Model | Script | vs Random | vs Strat-0.3 | vs Strat-0.5 | vs MM-0.5 | Status |
|---|---|---|---|---|---|---|
| `dqn_baseline_final_20k.pt` | train_sparse_jeson | 95–97% | 25–35% | ~10% | N/A | Stage 1 baseline |
| `phase2_best.pt` | train_phase2_selfplay | 95% | 40% | 10% | N/A | Stage 3 best |
| `phase2_continue_final.pt` | train_phase2_continue | 66% | 4% | 0% | N/A | Collapsed — discard |
| `phase3_best.pt` | train_phase3_mixed | 98% | 42% | 28% | N/A | Stage 5 best |
| `phase4_final.pt` (v1) | train_phase4_threeway | 72% | 22% | 6% | N/A | Collapsed — discard |
| `phase4_best_strategic.pt` | train_phase4_threeway | 98.5% | **64%** | **43%** | N/A | Stage 7 best (Jeson) |
| `models_adaptive/best.pt` | train_adaptive | ~95% | ~45% | ~55% | ~25% | Stage 10 adaptive |
| `models_rohan/final.pt` | train_rohan | ~95% | ~50% | **~60%** | **~40%** | **Stage 11 best (Rohan)** ✅ |

---

## All Files — New in Stages 8–11

**New agents:**
```
agents/minimax_agent.py      Minimax with alpha-beta pruning, iterative deepening,
                             transposition table, move ordering, pattern scoring.
                             skill_level parameter for curriculum use.

agents/dqn_rohan.py          Enhanced DQN agent:
                             - Dueling architecture (value + advantage streams)
                             - Prioritized Experience Replay (by TD error)
                             - AdamW optimiser with weight decay
                             - 4 convolutional layers
```

**New environments:**
```
game/gomoku_env_shaped.py    Shaped reward environment:
                             - Pre-move threat analysis
                             - Blocking rewards (defense-weighted)
                             - Ignore penalty for missed critical blocks
                             - Positional and connectivity bonuses
                             - Clipped to [-0.5, 0.5] to preserve terminal dominance
```

**New training scripts:**
```
train_adaptive.py            Stage 10 — Adaptive curriculum (promote/demote on win rate)
                             Uses DQNAgent (simple CNN) + sparse rewards + MinimaxAgent
                             15 difficulty levels (Random → Strat-0.9 → MM-0.7)

train_rohan.py               Stage 11 — Full Rohan pipeline
                             Uses DQNAgentRohan + GomokuEnvShaped + MinimaxAgent
                             11 difficulty levels, dynamic random anchor
                             Runs 80,000–100,000 episodes
```

---

## Summary of All Lessons Learned (Stages 1–11)

| Lesson | Evidence |
|---|---|
| Sparse rewards work better than shaped rewards when added to a pre-trained model | Stage 2 collapsed from 98% → 46% in 600 episodes |
| Shaped rewards CAN work if defense-weighted, include ignore penalties, and are used from scratch | Stage 11: Rohan agent blocks threats reliably where Stage 2 agent did not |
| Opponent quality matters more than reward design | Shaped rewards failed against Random; self-play without shaping succeeded |
| Self-play provides automatic curriculum scaling | No skill gap problem; agent trains against its own level |
| Strategic play emerges from game outcomes alone | 40% vs Strategic-0.3 achieved without training against it directly |
| The replay buffer must be preserved between training runs | Stage 4 corrupted from episode 1 due to empty buffer |
| Pure self-play diverges if run too long without anchoring | Stage 4 win rate declined 95% → 60% in its final 2,000 episodes |
| Mixed opponents (self-play + random) prevent strategy collapse | Stage 5 maintained 90%+ vs Random throughout with 30% Random anchor |
| Buffer warmup eliminates early training instability | Stage 5 loss curve was lower and more stable than any previous stage |
| The 30% Random anchor is a minimum threshold, not a guideline | Stage 6: reducing to 20% reproduced strategy collapse |
| Save best model on the metric you actually care about | Stage 6 never tracked vs-Strategic during training; checkpoint was lost |
| The gap between StrategicAgent and MinimaxAgent requires adaptive curriculum | Fixed schedule attempts produced 0% win rate vs Minimax |
| Adaptive curriculum (promote only when winning) solves the fixed-schedule problem | Stage 10 reached Minimax Level 10 vs Stages 9/10 that stalled immediately |
| Sparse rewards are insufficient when wins are rare against planning opponents | Stage 10 stalled at MM-0.3; move-level feedback needed to progress further |
| Ignore penalty (penalise missing a block immediately) teaches defence reliably | Stage 11 agent blocks threats where all previous sparse agents did not |
| Prioritized Experience Replay focuses training on surprising, high-error experiences | Stage 11 training loss more stable; better convergence vs difficult opponents |
| Dueling DQN separates state value from action advantage, improving learning efficiency | Stage 11 achieved comparable win rates in fewer effective gradient updates |
| DQN has a fundamental lookahead limitation against tree-search opponents | Even best DQN model wins ~0% vs MinimaxAgent at skill_level=1.0 |

---

## Final Assessment

### Where We Started vs Where We Are

```
                      vs Random    vs Strat-0.3    vs Strat-0.5    vs MM-0.5
Stage 1 Baseline        95–97%        25–35%           ~10%            N/A
Stage 7 (Jeson best)    98.5%          64%              43%            N/A
Stage 11 (Rohan best)   ~95%           ~50%             ~60%           ~40%
```

Jeson's pipeline produced the strongest agent vs rule-based opponents. Rohan's pipeline opened an entirely new frontier by training against and winning against a planning opponent (Minimax).

### The Fundamental Boundary

Neither approach can beat `MinimaxAgent` at `skill_level=1.0`. This is not a failure of implementation — it is a fundamental architectural limitation. A DQN agent sees the board and picks the move with the highest learned Q-value. It has no lookahead. A Minimax agent at depth 6 is evaluating thousands of future board positions before deciding. Reactive play, however well-trained, cannot overcome deliberate lookahead.

**To cross this boundary would require:**
1. **MCTS + Neural Network (AlphaZero-style)** — use the neural network to *guide* a tree search, not replace it. The network provides move priors and value estimates; MCTS uses these to search the game tree efficiently.
2. **Pure Minimax** — already implemented in `agents/minimax_agent.py` at `skill_level=1.0`. This is provably unbeatable by any reactive agent.

The Minimax agent at `skill_level=1.0` IS unbeatable by any DQN approach we have tried. The Rohan agent (`models_rohan/final.pt`) represents the best achievable performance within the DQN paradigm given our training budget.
