"""
This script trains via:
- self-play games using MCTS visit distributions as policy targets
- supervised updates on (state, policy_target, value_target)
"""

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents.alphazero_agent import AlphaZeroAgent
from game.logic import GomokuLogic


@dataclass
class SelfPlayExample:
    state: np.ndarray  # shape: (3, board_size, board_size)
    policy: np.ndarray  # shape: (board_size * board_size,)
    player: int  # player to move at this state


def encode_board(board: np.ndarray, current_player: int, board_size: int) -> np.ndarray:
    """Encode board from current_player perspective into 3 planes."""
    state = np.zeros((3, board_size, board_size), dtype=np.float32)
    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == -current_player).astype(np.float32)
    state[2] = np.full((board_size, board_size), current_player, dtype=np.float32)
    return state


def run_self_play_game(
    agent: AlphaZeroAgent,
    board_size: int,
    temperature_moves: int,
) -> Tuple[List[SelfPlayExample], int]:
    """
    Run one full self-play game and return training examples + game winner.

    winner values:
    - 1 / -1: winning player
    - 0: draw
    """
    logic = GomokuLogic(board_size=board_size)
    examples: List[SelfPlayExample] = []
    move_idx = 0

    while not logic.game_over:
        to_play = logic.current_player
        agent.player_id = to_play

        # High temperature in opening for exploration, then near-greedy.
        agent.temperature = 1.0 if move_idx < temperature_moves else 0.0

        board_copy = logic.board.copy()
        move, visit_dist = agent.self_play_move(board_copy, add_dirichlet_noise=True)
        if move is None:
            break

        examples.append(
            SelfPlayExample(
                state=encode_board(board_copy, to_play, board_size),
                policy=visit_dist.astype(np.float32),
                player=to_play,
            )
        )

        logic.make_move(move[0], move[1])
        move_idx += 1

    winner = int(logic.winner) if logic.winner is not None else 0
    return examples, winner


def build_training_tensors(
    examples: List[SelfPlayExample],
    winner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert self-play examples from one game into arrays for buffer/training."""
    states = []
    policies = []
    values = []

    for ex in examples:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if ex.player == winner else -1.0
        states.append(ex.state)
        policies.append(ex.policy)
        values.append(z)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(policies, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    )


def train_step(
    agent: AlphaZeroAgent,
    optimizer: torch.optim.Optimizer,
    batch_states: np.ndarray,
    batch_policies: np.ndarray,
    batch_values: np.ndarray,
) -> Tuple[float, float, float]:
    """One gradient step over a mini-batch."""
    states = torch.from_numpy(batch_states).to(agent.device)
    policy_targets = torch.from_numpy(batch_policies).to(agent.device)
    value_targets = torch.from_numpy(batch_values).to(agent.device)

    agent.network.train()
    policy_logits, values = agent.network(states)
    values = values.squeeze(1)

    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def train_alphazero(
    board_size: int,
    iterations: int,
    games_per_iteration: int,
    epochs_per_iteration: int,
    batch_size: int,
    replay_size: int,
    temperature_moves: int,
    learning_rate: float,
    num_simulations: int,
    finetune_start_iteration: int,
    finetune_num_simulations: int | None,
    finetune_learning_rate: float | None,
    save_dir: str,
    save_every: int,
):
    os.makedirs(save_dir, exist_ok=True)

    agent = AlphaZeroAgent(
        player_id=1,
        board_size=board_size,
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        channels=128,
        num_res_blocks=6,
    )

    optimizer = torch.optim.Adam(
        agent.network.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    replay: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=replay_size)

    print("=" * 72)
    print("ALPHAZERO TRAINING")
    print("=" * 72)
    print(f"Device: {agent.device}")
    print(f"Board size: {board_size}")
    print(f"Iterations: {iterations}")
    print(f"Games/iteration: {games_per_iteration}")
    print(f"Epochs/iteration: {epochs_per_iteration}")
    print(f"Batch size: {batch_size}")
    print(f"Replay size: {replay_size}")
    print(f"MCTS simulations/move (phase 1): {num_simulations}")
    if finetune_num_simulations is not None:
        print(
            f"Fine-tune starts at iter: {finetune_start_iteration} | "
            f"MCTS (phase 2): {finetune_num_simulations}"
        )
        if finetune_learning_rate is not None:
            print(f"Learning rate (phase 2): {finetune_learning_rate}")
    else:
        print("Fine-tune phase: disabled")
    print("=" * 72)

    start_all = time.time()
    in_finetune_phase = False

    for it in range(1, iterations + 1):
        it_start = time.time()

        # Optional two-phase schedule: strong-search pretraining then
        # low-search fine-tuning for faster deployment-time inference.
        should_finetune = (
            finetune_num_simulations is not None and it >= finetune_start_iteration
        )
        if should_finetune and not in_finetune_phase:
            in_finetune_phase = True
            if finetune_learning_rate is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = finetune_learning_rate
            print(
                f"\n>>> Entering fine-tune phase at iter {it}: "
                f"num_simulations={finetune_num_simulations}"
                + (
                    f", lr={optimizer.param_groups[0]['lr']:.6g}"
                    if finetune_learning_rate is not None
                    else ""
                )
                + " <<<\n"
            )

        agent.num_simulations = (
            finetune_num_simulations if should_finetune else num_simulations
        )

        # 1) Self-play data generation
        generated_positions = 0
        outcomes = {1: 0, -1: 0, 0: 0}

        agent.network.eval()
        for _ in range(games_per_iteration):
            game_examples, winner = run_self_play_game(
                agent=agent,
                board_size=board_size,
                temperature_moves=temperature_moves,
            )
            outcomes[winner] += 1

            if not game_examples:
                continue

            states, policies, values = build_training_tensors(game_examples, winner)
            generated_positions += len(game_examples)

            for s, p, v in zip(states, policies, values):
                replay.append((s, p, float(v)))

        # 2) Network optimization on replay buffer
        losses = []
        p_losses = []
        v_losses = []

        if len(replay) >= batch_size:
            replay_list = list(replay)
            for _ in range(epochs_per_iteration):
                np.random.shuffle(replay_list)

                for i in range(0, len(replay_list), batch_size):
                    batch = replay_list[i : i + batch_size]
                    if len(batch) < batch_size:
                        continue

                    batch_states = np.asarray([x[0] for x in batch], dtype=np.float32)
                    batch_policies = np.asarray([x[1] for x in batch], dtype=np.float32)
                    batch_values = np.asarray([x[2] for x in batch], dtype=np.float32)

                    loss, p_loss, v_loss = train_step(
                        agent=agent,
                        optimizer=optimizer,
                        batch_states=batch_states,
                        batch_policies=batch_policies,
                        batch_values=batch_values,
                    )
                    losses.append(loss)
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)

        # 3) Logging + checkpoint
        elapsed_it = time.time() - it_start
        avg_loss = np.mean(losses) if losses else float("nan")
        avg_pl = np.mean(p_losses) if p_losses else float("nan")
        avg_vl = np.mean(v_losses) if v_losses else float("nan")

        print(
            f"Iter {it:03d}/{iterations} | "
            f"phase={'FT' if should_finetune else 'P1'} | "
            f"sims={agent.num_simulations} | "
            f"games={games_per_iteration} | "
            f"positions={generated_positions} | "
            f"buffer={len(replay)} | "
            f"W/D/L(P1)={outcomes[1]}/{outcomes[0]}/{outcomes[-1]} | "
            f"loss={avg_loss:.4f} (p={avg_pl:.4f}, v={avg_vl:.4f}) | "
            f"{elapsed_it / 60:.1f}m"
        )

        if it % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"alphazero_iter_{it}.pt")
            agent.save_model(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "alphazero_final.pt")
    agent.save_model(final_path)

    total_elapsed = time.time() - start_all
    print("=" * 72)
    print(f"Training complete. Final model: {final_path}")
    print(f"Total time: {total_elapsed / 3600:.2f}h")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero agent with self-play")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iteration", type=int, default=20)
    parser.add_argument("--epochs-per-iteration", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--temperature-moves", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-simulations", type=int, default=60)
    parser.add_argument(
        "--finetune-start-iteration",
        type=int,
        default=None,
        help="Iteration index (1-based) to start phase-2 fine-tuning. "
        "Default: disabled unless --finetune-num-simulations is set.",
    )
    parser.add_argument(
        "--finetune-num-simulations",
        type=int,
        default=None,
        help="Phase-2 MCTS simulations per move (for deployment-target tuning).",
    )
    parser.add_argument(
        "--finetune-learning-rate",
        type=float,
        default=None,
        help="Optional learning rate for phase-2 fine-tuning.",
    )
    parser.add_argument("--save-dir", type=str, default="models_alphazero")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    if args.finetune_num_simulations is not None:
        if args.finetune_num_simulations <= 0:
            raise ValueError("--finetune-num-simulations must be > 0")
        if args.finetune_start_iteration is None:
            args.finetune_start_iteration = max(1, int(args.iterations * 0.7))
        if (
            args.finetune_start_iteration < 1
            or args.finetune_start_iteration > args.iterations
        ):
            raise ValueError(
                "--finetune-start-iteration must be between 1 and --iterations"
            )
    else:
        # Keep this ignored when no fine-tune phase is configured.
        args.finetune_start_iteration = args.iterations + 1

    return args


if __name__ == "__main__":
    args = parse_args()
    train_alphazero(
        board_size=args.board_size,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        temperature_moves=args.temperature_moves,
        learning_rate=args.learning_rate,
        num_simulations=args.num_simulations,
        finetune_start_iteration=args.finetune_start_iteration,
        finetune_num_simulations=args.finetune_num_simulations,
        finetune_learning_rate=args.finetune_learning_rate,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )
