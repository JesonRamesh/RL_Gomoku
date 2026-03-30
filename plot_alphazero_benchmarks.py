"""
Create AlphaZero benchmark figures and a short markdown report.

This script evaluates a sequence of AlphaZero checkpoints against fixed opponents,
then produces:
1) Progression plot (win rate vs checkpoint iteration)
2) Final-results bar chart (wider opponent suite)
3) CSV files + markdown summary report

Example:
    python plot_alphazero_benchmarks.py \
      --checkpoints-dir models_alphazero_two_phase \
      --final-model Model/alphazero_final.pt \
      --output-dir results/alphazero_benchmarks \
      --phase-split-iter 70
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from agents.alphazero_agent import AlphaZeroAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from game.match import eval_agents


ITER_RE = re.compile(r"alphazero_iter_(\d+)\.pt$")


@dataclass
class CheckpointEval:
    checkpoint: str
    iteration: int
    vs_random: float
    vs_strategic_05: float
    vs_minimax_05: float


@dataclass
class OpponentEval:
    opponent: str
    win_rate: float
    games: int


def _agent1_win_rate(results: dict[str, int]) -> float:
    total = results["agent1_wins"] + results["agent2_wins"] + results["draws"]
    if total == 0:
        return 0.0
    return 100.0 * results["agent1_wins"] / total


def _evaluate_once(
    agent: AlphaZeroAgent, opponent, board_size: int, num_games: int
) -> float:
    results = eval_agents(agent, opponent, num_games=num_games, board_size=board_size)
    return _agent1_win_rate(results)


def _new_alphazero(
    model_path: str, board_size: int, num_simulations: int
) -> AlphaZeroAgent:
    agent = AlphaZeroAgent(
        player_id=1,
        board_size=board_size,
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=0.0,
        channels=128,
        num_res_blocks=6,
    )
    agent.load_model(model_path)
    return agent


def _discover_checkpoints(checkpoints_dir: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for p in sorted(Path(checkpoints_dir).glob("*.pt")):
        m = ITER_RE.search(p.name)
        if not m:
            continue
        entries.append((int(m.group(1)), str(p)))
    entries.sort(key=lambda x: x[0])
    return entries


def _write_checkpoint_csv(rows: Iterable[CheckpointEval], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "iteration",
                "vs_random",
                "vs_strategic_0.5",
                "vs_minimax_0.5",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.checkpoint,
                    r.iteration,
                    f"{r.vs_random:.2f}",
                    f"{r.vs_strategic_05:.2f}",
                    f"{r.vs_minimax_05:.2f}",
                ]
            )


def _write_final_csv(rows: Iterable[OpponentEval], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "games", "win_rate"])
        for r in rows:
            w.writerow([r.opponent, r.games, f"{r.win_rate:.2f}"])


def _plot_progression(
    rows: list[CheckpointEval], out_path: str, phase_split_iter: int | None
) -> None:
    xs = [r.iteration for r in rows]
    y_rand = [r.vs_random for r in rows]
    y_strat = [r.vs_strategic_05 for r in rows]
    y_mm = [r.vs_minimax_05 for r in rows]

    plt.figure(figsize=(13, 5.5))
    plt.plot(xs, y_rand, "o-", color="#2ca02c", label="vs Random", linewidth=2)
    plt.plot(xs, y_strat, "o-", color="#1f77b4", label="vs Strategic-0.5", linewidth=2)
    plt.plot(xs, y_mm, "o-", color="#d62728", label="vs Minimax-0.5", linewidth=2)

    # Mark best checkpoint for each metric
    best_rand_idx = max(range(len(rows)), key=lambda i: rows[i].vs_random)
    best_strat_idx = max(range(len(rows)), key=lambda i: rows[i].vs_strategic_05)
    best_mm_idx = max(range(len(rows)), key=lambda i: rows[i].vs_minimax_05)

    for idx, ys, color, label in [
        (best_rand_idx, y_rand, "#2ca02c", "Best vs Random"),
        (best_strat_idx, y_strat, "#1f77b4", "Best vs Strategic-0.5"),
        (best_mm_idx, y_mm, "#d62728", "Best vs Minimax-0.5"),
    ]:
        plt.scatter(
            xs[idx],
            ys[idx],
            s=110,
            marker="D",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )

    if phase_split_iter is not None:
        plt.axvline(phase_split_iter, color="gray", linestyle="--", alpha=0.7)
        plt.text(
            phase_split_iter,
            103,
            f" phase split @ iter {phase_split_iter}",
            color="gray",
            fontsize=9,
            va="top",
        )

    plt.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("AlphaZero Checkpoint Progression")
    plt.xlabel("Training Iteration")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_final_bars(rows: list[OpponentEval], out_path: str) -> None:
    labels = [r.opponent for r in rows]
    values = [r.win_rate for r in rows]

    palette = [
        "#2ca02c",  # Random
        "#8ecae6",  # Strat 0.3
        "#219ebc",  # Strat 0.5
        "#126782",  # Strat 0.7
        "#f4a261",  # MM 0.3
        "#e76f51",  # MM 0.5
        "#b56576",  # MM 1.0
    ]

    plt.figure(figsize=(10.5, 5.2))
    bars = plt.bar(labels, values, color=palette[: len(labels)], alpha=0.9)
    plt.axhline(50, color="red", linestyle="--", linewidth=1.2, alpha=0.65)
    plt.ylabel("Win Rate (%)")
    plt.title("AlphaZero Final Model: 100-game style benchmark")
    plt.ylim(0, 105)
    plt.grid(True, axis="y", alpha=0.25)

    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            min(102, v + 2),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_report(
    checkpoint_rows: list[CheckpointEval],
    final_rows: list[OpponentEval],
    final_model: str,
    output_dir: str,
    phase_split_iter: int | None,
) -> None:
    best_random = max(checkpoint_rows, key=lambda r: r.vs_random)
    best_strat = max(checkpoint_rows, key=lambda r: r.vs_strategic_05)
    best_mm = max(checkpoint_rows, key=lambda r: r.vs_minimax_05)

    report_path = os.path.join(output_dir, "alphazero_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AlphaZero Benchmark Summary\n\n")
        f.write(f"Final model evaluated: `{final_model}`\n\n")
        if phase_split_iter is not None:
            f.write(f"Two-phase split marker: iteration **{phase_split_iter}**\n\n")

        f.write("## Best checkpoint (progression suite)\n\n")
        f.write("- Best vs Random: ")
        f.write(f"iter {best_random.iteration} ({best_random.vs_random:.1f}%)\n")
        f.write("- Best vs Strategic-0.5: ")
        f.write(f"iter {best_strat.iteration} ({best_strat.vs_strategic_05:.1f}%)\n")
        f.write("- Best vs Minimax-0.5: ")
        f.write(f"iter {best_mm.iteration} ({best_mm.vs_minimax_05:.1f}%)\n\n")

        f.write("## Final benchmark table\n\n")
        f.write("| Opponent | Games | Win rate |\n")
        f.write("|---|---:|---:|\n")
        for row in final_rows:
            f.write(f"| {row.opponent} | {row.games} | {row.win_rate:.1f}% |\n")

        f.write("\n## Output files\n\n")
        f.write("- `alphazero_progression.png`\n")
        f.write("- `alphazero_final_results.png`\n")
        f.write("- `alphazero_checkpoint_results.csv`\n")
        f.write("- `alphazero_final_results.csv`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaZero checkpoints and make benchmark figures"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="models_alphazero_two_phase"
    )
    parser.add_argument("--final-model", type=str, default="Model/alphazero_final.pt")
    parser.add_argument(
        "--output-dir", type=str, default="results/alphazero_benchmarks"
    )

    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--az-simulations", type=int, default=50)
    parser.add_argument("--phase-split-iter", type=int, default=None)

    parser.add_argument("--progression-games", type=int, default=30)
    parser.add_argument("--final-games", type=int, default=100)

    parser.add_argument("--minimax-time-limit", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = _discover_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching alphazero_iter_*.pt found in: {args.checkpoints_dir}"
        )

    checkpoint_rows: list[CheckpointEval] = []

    print("=" * 70)
    print("CHECKPOINT PROGRESSION EVALUATION")
    print("=" * 70)

    for iteration, ckpt_path in checkpoints:
        print(f"\nEvaluating checkpoint iter {iteration}: {ckpt_path}")
        az = _new_alphazero(
            model_path=ckpt_path,
            board_size=args.board_size,
            num_simulations=args.az_simulations,
        )

        wr_rand = _evaluate_once(
            az, RandomAgent(-1), args.board_size, args.progression_games
        )
        wr_strat = _evaluate_once(
            az,
            StrategicAgent(-1, skill_level=0.5, board_size=args.board_size),
            args.board_size,
            args.progression_games,
        )
        wr_mm = _evaluate_once(
            az,
            MinimaxAgent(
                -1,
                board_size=args.board_size,
                time_limit=args.minimax_time_limit,
                skill_level=0.5,
            ),
            args.board_size,
            args.progression_games,
        )

        checkpoint_rows.append(
            CheckpointEval(
                checkpoint=os.path.basename(ckpt_path),
                iteration=iteration,
                vs_random=wr_rand,
                vs_strategic_05=wr_strat,
                vs_minimax_05=wr_mm,
            )
        )

        print(
            f"iter {iteration}: vs Random={wr_rand:.1f}% | "
            f"vs Strategic-0.5={wr_strat:.1f}% | "
            f"vs Minimax-0.5={wr_mm:.1f}%"
        )

    checkpoint_csv = os.path.join(args.output_dir, "alphazero_checkpoint_results.csv")
    _write_checkpoint_csv(checkpoint_rows, checkpoint_csv)

    progression_plot = os.path.join(args.output_dir, "alphazero_progression.png")
    _plot_progression(checkpoint_rows, progression_plot, args.phase_split_iter)

    final_model = args.final_model
    if not os.path.exists(final_model):
        # fallback to latest checkpoint if final model path does not exist
        final_model = checkpoints[-1][1]
        print(
            f"Final model not found at requested path, falling back to: {final_model}"
        )

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    az_final = _new_alphazero(
        model_path=final_model,
        board_size=args.board_size,
        num_simulations=args.az_simulations,
    )

    final_suite = [
        ("Random", RandomAgent(-1)),
        ("S03", StrategicAgent(-1, skill_level=0.3, board_size=args.board_size)),
        ("S05", StrategicAgent(-1, skill_level=0.5, board_size=args.board_size)),
        ("S07", StrategicAgent(-1, skill_level=0.7, board_size=args.board_size)),
        (
            "MM03",
            MinimaxAgent(
                -1,
                board_size=args.board_size,
                time_limit=args.minimax_time_limit,
                skill_level=0.3,
            ),
        ),
        (
            "MM05",
            MinimaxAgent(
                -1,
                board_size=args.board_size,
                time_limit=args.minimax_time_limit,
                skill_level=0.5,
            ),
        ),
        (
            "MM10",
            MinimaxAgent(
                -1,
                board_size=args.board_size,
                time_limit=args.minimax_time_limit,
                skill_level=1.0,
            ),
        ),
    ]

    final_rows: list[OpponentEval] = []
    for name, opp in final_suite:
        print(f"\nEvaluating final model vs {name}...")
        wr = _evaluate_once(az_final, opp, args.board_size, args.final_games)
        final_rows.append(
            OpponentEval(opponent=name, win_rate=wr, games=args.final_games)
        )
        print(f"vs {name}: {wr:.1f}%")

    final_csv = os.path.join(args.output_dir, "alphazero_final_results.csv")
    _write_final_csv(final_rows, final_csv)

    final_plot = os.path.join(args.output_dir, "alphazero_final_results.png")
    _plot_final_bars(final_rows, final_plot)

    _write_report(
        checkpoint_rows=checkpoint_rows,
        final_rows=final_rows,
        final_model=final_model,
        output_dir=args.output_dir,
        phase_split_iter=args.phase_split_iter,
    )

    print("\n" + "=" * 70)
    print("Done. Outputs written to:")
    print(f"  {args.output_dir}")
    print("Generated files:")
    print("  - alphazero_progression.png")
    print("  - alphazero_final_results.png")
    print("  - alphazero_checkpoint_results.csv")
    print("  - alphazero_final_results.csv")
    print("  - alphazero_report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
