"""
plot_training_progress.py

Evaluates Gen 1 and Gen 2 checkpoints against three opponents and plots
a unified win-rate learning curve for the technical report.

Usage:
    python plot_training_progress.py

Output:
    training_progress.png  — publication-quality figure (saved in repo root)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from game.logic import GomokuLogic
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent

BOARD_SIZE = 9
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
EVAL_GAMES = 100   # games per opponent per checkpoint (raise for smoother curves)

# ---------------------------------------------------------------------------
# Simple DQN architecture — matches Gen 1 & Gen 2 checkpoint state dicts
# (conv1/bn1, conv2/bn2, conv3/bn3, fc1/bn4, fc2)
# ---------------------------------------------------------------------------

class SimpleDQNetwork(nn.Module):
    def __init__(self, board_size=9):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.fc1   = nn.Linear(128 * board_size * board_size, 512)
        self.bn4   = nn.BatchNorm1d(512)
        self.fc2   = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        return self.fc2(x)


def preprocess(board: np.ndarray, player: int) -> torch.Tensor:
    """3-channel board encoding, returned as (1, 3, H, W) tensor on DEVICE."""
    h, w = board.shape
    state = np.zeros((3, h, w), dtype=np.float32)
    state[0] = (board == player).astype(np.float32)
    state[1] = (board == -player).astype(np.float32)
    state[2] = np.full((h, w), player, dtype=np.float32)
    return torch.tensor(state, device=DEVICE).unsqueeze(0)


def load_network(path: str) -> SimpleDQNetwork:
    net = SimpleDQNetwork(BOARD_SIZE).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ckpt["q_network_state_dict"])
    net.eval()
    return net


def greedy_move(net: SimpleDQNetwork, board: np.ndarray, player: int) -> tuple:
    with torch.no_grad():
        q = net(preprocess(board, player)).squeeze(0).cpu().numpy()
    valid = list(zip(*np.where(board == 0)))
    if not valid:
        return None
    best = max(valid, key=lambda rc: q[rc[0] * BOARD_SIZE + rc[1]])
    return best


def play_game(net: SimpleDQNetwork, opponent, agent_player: int) -> int:
    """
    Play one game. Returns +1 if the agent (net) wins, -1 if it loses, 0 draw.
    agent_player: +1 means agent goes first, -1 means opponent goes first.
    """
    game = GomokuLogic(BOARD_SIZE)
    while not game.game_over:
        if game.current_player == agent_player:
            move = greedy_move(net, game.board, agent_player)
        else:
            move = opponent.predict(game.board)
        if move is None:
            break
        game.make_move(move[0], move[1])

    if game.winner == agent_player:
        return 1
    if game.winner == 0:
        return 0
    return -1


def evaluate(net: SimpleDQNetwork, opponent, n_games: int) -> float:
    """Win rate [0, 1]. Alternates which side the agent plays."""
    wins = 0
    for i in range(n_games):
        # Alternate: even games agent is player 1, odd games player -1
        agent_player = 1 if i % 2 == 0 else -1
        # Rebind opponent player_id to match the other side
        opponent.player_id = -agent_player
        result = play_game(net, opponent, agent_player)
        if result == 1:
            wins += 1
    return wins / n_games


# ---------------------------------------------------------------------------
# Checkpoint definitions
# Each entry: (label, cumulative_episode_count, path, generation)
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    # Gen 1 — Baseline DQN vs Random
    # (label, cumulative_episodes, path, generation, is_best_checkpoint)
    ("8k",    8_000,  "models_baseline_9x9/dqn_baseline_final_8k.pt",      1, False),
    ("20k",  20_000,  "models_baseline_9x9/dqn_baseline_final_20k.pt",     1, True),

    # Gen 2 — Self-Play
    ("+6k",  26_000,  "models_phase2/phase2_best.pt",             2, True),
    ("+8k",  28_000,  "models_phase2/phase2_final.pt",            2, False),

    # Gen 2 — Mixed opponents (Phase 3)
    ("+6k",  34_000,  "models_phase3/phase3_best.pt",             2, True),
    ("+6k",  36_000,  "models_phase3/phase3_final.pt",            2, False),

    # Gen 2 — Three-way curriculum (Phase 4 v2)
    ("+6k",  44_000,  "models_phase4_v2/phase4_best_strategic.pt", 2, True),
    ("end",  46_000,  "models_phase4_v2/phase4_final.pt",          2, False),
]

OPPONENTS = [
    ("vs Random",        RandomAgent(-1)),
    ("vs Strategic-0.3", StrategicAgent(-1, skill_level=0.3, board_size=BOARD_SIZE)),
    ("vs Strategic-0.5", StrategicAgent(-1, skill_level=0.5, board_size=BOARD_SIZE)),
]

COLORS = {
    "vs Random":        "#2ecc71",   # green
    "vs Strategic-0.3": "#3498db",   # blue
    "vs Strategic-0.5": "#e74c3c",   # red
}

GEN_COLORS = {1: "#f0f0f0", 2: "#e8f4fd"}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Evaluating {len(CHECKPOINTS)} checkpoints × {len(OPPONENTS)} opponents × {EVAL_GAMES} games\n")

    # Filter to checkpoints that exist on disk
    checkpoints = [(lbl, ep, path, gen, best)
                   for lbl, ep, path, gen, best in CHECKPOINTS
                   if os.path.exists(path)]

    missing = [path for _, _, path, _, _ in CHECKPOINTS if not os.path.exists(path)]
    if missing:
        print(f"WARNING — skipping {len(missing)} missing checkpoints:")
        for p in missing:
            print(f"  {p}")
        print()

    results  = {opp_name: [] for opp_name, _ in OPPONENTS}
    episodes = []
    labels   = []
    gens     = []
    is_best  = []

    for lbl, ep, path, gen, best in checkpoints:
        print(f"Loading: {path}")
        net = load_network(path)
        episodes.append(ep)
        labels.append(lbl)
        gens.append(gen)
        is_best.append(best)

        for opp_name, opp in OPPONENTS:
            wr = evaluate(net, opp, EVAL_GAMES)
            results[opp_name].append(wr * 100)
            print(f"  {opp_name:<20} {wr*100:5.1f}%")
        print()

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Generation background shading ---
    gen_spans = []
    if episodes:
        current_gen = gens[0]
        span_start  = episodes[0]
        for i in range(1, len(episodes)):
            if gens[i] != current_gen:
                gen_spans.append((span_start, episodes[i - 1], current_gen))
                current_gen = gens[i]
                span_start  = episodes[i]
        gen_spans.append((span_start, episodes[-1], current_gen))

    for x0, x1, g in gen_spans:
        margin = (episodes[-1] - episodes[0]) * 0.008
        ax.axvspan(x0 - margin, x1 + margin, alpha=0.18,
                   color=GEN_COLORS.get(g, "#eeeeee"), zorder=0)

    # --- Generation label banners at the top ---
    gen_banner_labels = {
        1: "Gen 1 — Baseline DQN\nvs Random opponent",
        2: "Gen 2 — Self-Play & Curriculum\n(self-play → mixed → 3-way)",
    }
    for x0, x1, g in gen_spans:
        mid = (x0 + x1) / 2
        ax.text(mid, 108, gen_banner_labels.get(g, f"Gen {g}"),
                ha="center", va="center",
                fontsize=9.5, color="#333333", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=GEN_COLORS.get(g, "#eeeeee"),
                          edgecolor="#bbbbbb", alpha=0.85))

    # --- Generation divider ---
    for i in range(1, len(gens)):
        if gens[i] != gens[i - 1]:
            boundary = (episodes[i - 1] + episodes[i]) / 2
            ax.axvline(boundary, color="#888888", linestyle="--",
                       linewidth=1.4, zorder=1)

    # --- Sub-phase dividers within Gen 2 ---
    phase_boundaries = []
    phase_labels_map = {
        26_000: "Self-play",
        34_000: "Mixed\nopponents",
        44_000: "3-way\ncurriculum",
    }
    for i in range(1, len(episodes)):
        if gens[i] == gens[i - 1] == 2 and labels[i - 1] != labels[i]:
            # Only draw when moving to a new phase label group
            ep_here = episodes[i]
            if ep_here in phase_labels_map:
                ax.axvline(ep_here - 1000, color="#cccccc", linestyle=":",
                           linewidth=1.0, zorder=1)

    # --- Phase annotations inside Gen 2 ---
    for ep, phase_lbl in phase_labels_map.items():
        ax.text(ep + 200, 2, phase_lbl,
                ha="left", va="bottom", fontsize=7.5,
                color="#666666", style="italic")

    # --- Win rate lines ---
    for opp_name, _ in OPPONENTS:
        wr = results[opp_name]
        color = COLORS[opp_name]

        # Draw the connecting line
        ax.plot(episodes, wr, color=color, linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=3)

        # Different markers: filled circle = best checkpoint, open diamond = final
        for i, (ep, w, best) in enumerate(zip(episodes, wr, is_best)):
            if best:
                ax.plot(ep, w, "o", color=color, markersize=9,
                        markeredgecolor="white", markeredgewidth=1.5,
                        zorder=5)
            else:
                ax.plot(ep, w, "D", color="white", markersize=7,
                        markeredgecolor=color, markeredgewidth=2.0,
                        zorder=5)

        # Annotate the final best checkpoint value
        best_indices = [i for i, b in enumerate(is_best) if b]
        if best_indices:
            last_i = best_indices[-1]
            ax.annotate(f"{wr[last_i]:.0f}%",
                        xy=(episodes[last_i], wr[last_i]),
                        xytext=(10, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9,
                        color=color, fontweight="bold")

    # --- 50% reference line ---
    ax.axhline(50, color="#aaaaaa", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(episodes[0] - 800, 51, "50%", fontsize=8, color="#aaaaaa", va="bottom")

    # --- X-axis: one tick per checkpoint, two-row label (episode + sub-label) ---
    ax.set_xticks(episodes)
    tick_labels = []
    for lbl, ep, path, gen, best in checkpoints:
        ep_str = f"{ep // 1000}k ep"
        tick_labels.append(f"{ep_str}\n({lbl})" if not best else f"{ep_str}\n★ {lbl}")
    ax.set_xticklabels(tick_labels, fontsize=7.5, ha="center", linespacing=1.4)

    # --- Legend: lines + marker style explanation ---
    legend_handles = []
    for opp_name, _ in OPPONENTS:
        legend_handles.append(
            plt.Line2D([0], [0], color=COLORS[opp_name], linewidth=2.2,
                       marker="o", markersize=7,
                       markeredgecolor="white", markeredgewidth=1.5,
                       label=opp_name))
    legend_handles.append(
        plt.Line2D([0], [0], color="#888888", linewidth=0,
                   marker="o", markersize=7,
                   markeredgecolor="white", markeredgewidth=1.5,
                   label="● Best checkpoint saved"))
    legend_handles.append(
        plt.Line2D([0], [0], color="#888888", linewidth=0,
                   marker="D", markersize=7, markerfacecolor="white",
                   markeredgecolor="#888888", markeredgewidth=2.0,
                   label="◇ Final checkpoint (end of phase)"))
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=8.5, framealpha=0.92, ncol=1)

    # --- Axes formatting ---
    ax.set_xlim(episodes[0] - 2500, episodes[-1] + 5000)
    ax.set_ylim(0, 117)
    ax.set_xlabel("Cumulative Training Episodes", fontsize=11, labelpad=8)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title(
        "Training Progression: Gen 1 → Gen 2\nWin Rate vs Increasing Opponent Strength",
        fontsize=13, fontweight="bold", pad=14)

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "training_progress.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")

    # -----------------------------------------------------------------------
    # Also print a clean results table for copy-paste into the report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE (copy into report)")
    print("=" * 70)
    header = f"{'Checkpoint':<28} {'Episodes':>10}" + "".join(
        f"  {n:>16}" for n, _ in OPPONENTS)
    print(header)
    print("-" * len(header))
    for i, (lbl, ep, path, gen, best) in enumerate(checkpoints):
        row = f"{'Gen ' + str(gen) + ' ' + os.path.basename(path).replace('.pt',''):<28} {ep:>10,}"
        for opp_name, _ in OPPONENTS:
            row += f"  {results[opp_name][i]:>15.1f}%"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
