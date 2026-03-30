"""
AlphaZero self-play training loop for 9x9 Gomoku.

Three-phase iteration:
  1. Self-play:  best network generates games → training examples with 8x augmentation
  2. Train:      challenger network trains on replay buffer samples
  3. Evaluate:   challenger vs best (40 games); promoted if win rate >= 55%

Usage:
  python train_alphazero.py --mode test   # 3 iterations, ~20-30 min  (convergence check)
  python train_alphazero.py --mode m4     # 80 iterations, ~8-10 hrs  (M4 MacBook full run)
  python train_alphazero.py --mode full   # 100 iterations, paper settings (GPU recommended)

Reference: Silver et al. 2018 (Science 362), Silver et al. 2017 (Nature 550).
"""

import os
import json
import random
import argparse
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from agents.alphazero_net  import AlphaZeroNet, preprocess_board
from agents.alphazero_mcts import run_mcts, get_pi_and_move
from game.logic import GomokuLogic


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_example(board: np.ndarray, pi: np.ndarray, z: float,
                    player: int, board_size: int = 9) -> list:
    """
    Apply all 8 D4 symmetries (4 rotations x 2 reflections) to one training example.
    Returns exactly 8 tuples: (board, player, pi_flat, z).

    WHY: Gomoku is symmetric under rotation/reflection. Augmenting gives 8x free
    training data without extra self-play compute.
    Reference: Silver et al. 2017, Methods — 'Data augmentation'.
    """
    pi_board = pi.reshape(board_size, board_size)
    examples = []
    b, p = board.copy(), pi_board.copy()

    for _ in range(4):                   # 0°, 90°, 180°, 270°
        examples.append((b.copy(), player, p.flatten().copy(), z))
        # Horizontal reflection (left-right mirror)
        examples.append((np.fliplr(b).copy(), player, np.fliplr(p).flatten().copy(), z))
        # Rotate 90° counter-clockwise for next iteration
        b = np.rot90(b)
        p = np.rot90(p)

    return examples  # exactly 8 examples


# Self-Play Game
def self_play_game(net: AlphaZeroNet, num_simulations: int, c_puct: float,
                   temp_threshold: int, dirichlet_alpha: float,
                   dirichlet_eps: float, board_size: int = 9) -> list:
    """
    Play one complete self-play game and return augmented training examples.

    For the first `temp_threshold` moves: τ=1 (sample proportionally to visits).
    After that: τ=0 (greedy, pick most-visited). This ensures diverse openings
    while guaranteeing optimal play in critical late-game positions.

    Returns list of (board, player, pi, z) tuples — ~30 moves × 8 symmetries ≈ 240.
    """
    game    = GomokuLogic(board_size)
    history = []  # (board_copy, current_player, pi_vector)

    while not game.game_over:
        move_number = int(np.sum(game.board != 0))
        tau         = 1.0 if move_number < temp_threshold else 0.0

        root, _ = run_mcts(
            board=game.board,
            player=game.current_player,
            net=net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            board_size=board_size
        )

        pi, move = get_pi_and_move(root, board_size, tau)
        history.append((game.board.copy(), game.current_player, pi))
        game.make_move(*move)

    # Assign outcome z to every position retroactively
    winner   = game.winner  # +1, -1, or 0 (draw)
    examples = []
    for board, player, pi in history:
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        examples.extend(augment_example(board, pi, z, player, board_size))

    return examples


# Loss Function
def compute_loss(net: AlphaZeroNet, batch: list,
                 device: torch.device) -> tuple:
    """
    AlphaZero loss: l = (z - v)² - π^T log(p)
    Reference: Silver et al. 2018, Methods — 'Neural network training'.

    Value loss (MSE): trains value head to predict game outcome z ∈ {-1, 0, +1}
    Policy loss (cross-entropy): trains policy head to match MCTS distribution π
      π is a better estimate than raw network output p because it incorporates
      tree search — training p → π improves the network over iterations.

    Returns: (total_loss tensor, value_loss scalar, policy_loss scalar)
    """
    boards, players, pi_targets, z_targets = zip(*batch)

    states = torch.stack([
        torch.FloatTensor(preprocess_board(b, p))
        for b, p in zip(boards, players)
    ]).to(device)                                                    # (batch, 3, 9, 9)

    pi_t = torch.FloatTensor(np.array(pi_targets)).to(device)       # (batch, 81)
    z_t  = torch.FloatTensor(np.array(z_targets)).unsqueeze(1).to(device)  # (batch, 1)

    policy_out, value_out = net(states)                              # (batch, 81), (batch, 1)

    value_loss  = F.mse_loss(value_out, z_t)
    policy_loss = -(pi_t * torch.log(policy_out + 1e-8)).sum(dim=1).mean()
    total_loss  = value_loss + policy_loss

    return total_loss, value_loss.item(), policy_loss.item()


# Evaluation Game
def play_eval_game(net_first: AlphaZeroNet, net_second: AlphaZeroNet,
                   num_sims: int, c_puct: float, board_size: int = 9) -> int:
    """
    Play one evaluation game with no Dirichlet noise and τ=0 (greedy).
    Returns: +1 if first player wins, -1 if second player wins, 0 for draw.
    """
    game = GomokuLogic(board_size)
    nets = {1: net_first, -1: net_second}

    while not game.game_over:
        net = nets[game.current_player]
        root, _ = run_mcts(
            board=game.board,
            player=game.current_player,
            net=net,
            num_simulations=num_sims,
            c_puct=c_puct,
            dirichlet_alpha=0.0,
            dirichlet_eps=0.0,
            board_size=board_size
        )
        best_move = max(root.children, key=lambda m: root.children[m].N)
        game.make_move(*best_move)

    return game.winner


# Benchmark
def run_benchmark(net: AlphaZeroNet, iteration: int, num_sims: int,
                  board_size: int = 9):
    """
    Evaluate the best network against RandomAgent and StrategicAgent to track
    absolute skill progression across iterations.
    """
    from agents.random_agent    import RandomAgent
    from agents.strategic_agent import StrategicAgent
    from agents.alphazero_mcts  import AlphaZeroMCTS

    az_agent = AlphaZeroMCTS(net=net, player_id=1,
                              num_simulations=num_sims, board_size=board_size)

    opponents = {
        'RandomAgent':     RandomAgent(player_id=-1),
        'StrategicAgent-0.5': StrategicAgent(player_id=-1, skill_level=0.5),
    }

    print(f"\n  --- Benchmark at iteration {iteration} ---")
    for name, opp in opponents.items():
        wins = draws = losses = 0
        for _ in range(20):
            game = GomokuLogic(board_size)
            while not game.game_over:
                if game.current_player == 1:
                    az_agent.player_id = 1
                    move = az_agent.predict(game.board)
                else:
                    opp.player_id = -1
                    move = opp.predict(game.board)
                game.make_move(*move)
            if game.winner == 1:
                wins += 1
            elif game.winner == 0:
                draws += 1
            else:
                losses += 1
        print(f"  vs {name}: {wins}W/{draws}D/{losses}L  ({wins/20:.0%} win rate)")


# Main Training Loop
def get_config(mode: str) -> dict:
    """
    Return hyperparameter dict for the selected training mode.

    test:  3 iterations with minimal sims — verifies convergence, no errors (~20-30 min on M4)
    m4:    80 iterations tuned for M4 MPS — practical full training (~8-10 hours)
    full:  100 iterations at paper-scale settings — intended for GPU (4090 ~16 hrs)

    The key tradeoff between modes is MCTS simulations per move:
      - More sims → stronger policy targets → better training signal per game
      - Fewer sims → faster self-play → more iterations in the same wall-clock time
    On M4 the bottleneck is sequential MCTS (one network call per sim), so we
    reduce sims and compensate with more iterations and a larger per-game contribution.
    """
    configs = {
        'test': {
            'NUM_ITERATIONS':      3,
            'NUM_SELF_PLAY_GAMES': 15,    # 15 games per iteration
            'NUM_TRAIN_STEPS':     200,   # 200 gradient steps
            'NUM_EVAL_GAMES':      6,     # 3 challenger-first + 3 best-first
            'PROMOTION_THRESHOLD': 0.55,
            'NUM_SIMS_TRAIN':      50,    # 50 sims/move → fast
            'NUM_SIMS_EVAL':       100,
            'MIN_BUFFER_SIZE':     500,   # low threshold so training starts at iter 1
            'BENCHMARK_EVERY':     1,     # benchmark every iteration (only 3 total)
        },
        'm4': {
            'NUM_ITERATIONS':      80,
            'NUM_SELF_PLAY_GAMES': 50,    # 50 games per iteration
            'NUM_TRAIN_STEPS':     700,   # 700 gradient steps
            'NUM_EVAL_GAMES':      20,    # 10 challenger-first + 10 best-first
            'PROMOTION_THRESHOLD': 0.55,
            'NUM_SIMS_TRAIN':      100,   # 100 sims/move — key M4 reduction
            'NUM_SIMS_EVAL':       200,
            'MIN_BUFFER_SIZE':     5_000,
            'BENCHMARK_EVERY':     10,
        },
        'full': {
            'NUM_ITERATIONS':      100,
            'NUM_SELF_PLAY_GAMES': 100,   # paper-scale
            'NUM_TRAIN_STEPS':     1000,
            'NUM_EVAL_GAMES':      40,
            'PROMOTION_THRESHOLD': 0.55,
            'NUM_SIMS_TRAIN':      200,
            'NUM_SIMS_EVAL':       400,
            'MIN_BUFFER_SIZE':     10_000,
            'BENCHMARK_EVERY':     10,
        },
    }
    if mode not in configs:
        raise ValueError(f"Unknown mode '{mode}'. Choose: test | m4 | full")
    return configs[mode]


def train_alphazero(mode: str = 'm4'):
    cfg    = get_config(mode)
    device = get_device()
    print(f"Training on: {device}  |  mode: {mode}")
    os.makedirs("models_alphazero", exist_ok=True)

    # Hyperparameters
    C_PUCT          = 1.5
    TEMP_THRESHOLD  = 15     # τ=1 for first 15 moves, τ=0 after
    DIRICHLET_ALPHA = 0.3    # Silver 2018 Supplementary: α ≈ 10/|A| for Gomoku
    DIRICHLET_EPS   = 0.25   # Silver 2018 Supplementary: ε = 0.25
    BATCH_SIZE      = 256
    LEARNING_RATE   = 0.002
    BOARD_SIZE      = 9

    # Unpack mode-specific config 
    NUM_ITERATIONS      = cfg['NUM_ITERATIONS']
    NUM_SELF_PLAY_GAMES = cfg['NUM_SELF_PLAY_GAMES']
    NUM_TRAIN_STEPS     = cfg['NUM_TRAIN_STEPS']
    NUM_EVAL_GAMES      = cfg['NUM_EVAL_GAMES']
    PROMOTION_THRESHOLD = cfg['PROMOTION_THRESHOLD']
    NUM_SIMS_TRAIN      = cfg['NUM_SIMS_TRAIN']
    NUM_SIMS_EVAL       = cfg['NUM_SIMS_EVAL']
    MIN_BUFFER_SIZE     = cfg['MIN_BUFFER_SIZE']
    BENCHMARK_EVERY     = cfg['BENCHMARK_EVERY']

    # Initialise 
    net_best = AlphaZeroNet(board_size=BOARD_SIZE, num_filters=128,
                            num_res_blocks=6).to(device)
    replay_buffer = deque(maxlen=500_000)
    training_log  = []

    print(f"Network: {sum(p.numel() for p in net_best.parameters()):,} parameters")

    # Iteration loop
    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{NUM_ITERATIONS}")

        # Phase 1: Self-Play
        net_best.eval()
        new_examples = []
        for g in range(NUM_SELF_PLAY_GAMES):
            examples = self_play_game(
                net=net_best,
                num_simulations=NUM_SIMS_TRAIN,
                c_puct=C_PUCT,
                temp_threshold=TEMP_THRESHOLD,
                dirichlet_alpha=DIRICHLET_ALPHA,
                dirichlet_eps=DIRICHLET_EPS,
                board_size=BOARD_SIZE
            )
            new_examples.extend(examples)
            if (g + 1) % 20 == 0:
                print(f"  Self-play: {g+1}/{NUM_SELF_PLAY_GAMES} games  "
                      f"({len(new_examples)} examples so far)")

        replay_buffer.extend(new_examples)
        print(f"  Buffer: {len(replay_buffer):,} examples  "
              f"(+{len(new_examples)} this iteration)")

        if len(replay_buffer) < MIN_BUFFER_SIZE:
            print(f"  Skipping training: buffer < {MIN_BUFFER_SIZE:,}")
            continue

        # Phase 2: Train
        net_challenger = deepcopy(net_best).to(device)
        net_challenger.train()
        optimizer = torch.optim.AdamW(
            net_challenger.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4   # L2 regularisation (= c in the paper's cθ² term)
        )

        loss_sum = v_loss_sum = p_loss_sum = 0.0
        for _ in range(NUM_TRAIN_STEPS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            loss, v_loss, p_loss = compute_loss(net_challenger, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_challenger.parameters(), 1.0)
            optimizer.step()
            loss_sum   += loss.item()
            v_loss_sum += v_loss
            p_loss_sum += p_loss

        avg_loss   = loss_sum   / NUM_TRAIN_STEPS
        avg_v_loss = v_loss_sum / NUM_TRAIN_STEPS
        avg_p_loss = p_loss_sum / NUM_TRAIN_STEPS
        print(f"  Train loss: {avg_loss:.4f}  "
              f"(value={avg_v_loss:.4f}, policy={avg_p_loss:.4f})")

        # Phase 3: Evaluate 
        net_challenger.eval()
        net_best.eval()
        wins = 0
        for i in range(NUM_EVAL_GAMES):
            # Alternate first/second player to remove first-move bias
            if i % 2 == 0:
                result = play_eval_game(net_challenger, net_best,
                                        NUM_SIMS_EVAL, C_PUCT, BOARD_SIZE)
                if result == 1:  wins += 1   # challenger (player 1) won
            else:
                result = play_eval_game(net_best, net_challenger,
                                        NUM_SIMS_EVAL, C_PUCT, BOARD_SIZE)
                if result == -1: wins += 1   # challenger (player -1) won

        win_rate = wins / NUM_EVAL_GAMES
        promoted = win_rate >= PROMOTION_THRESHOLD
        print(f"  Eval: challenger {win_rate:.0%} ({wins}/{NUM_EVAL_GAMES}) "
              f"→ {'PROMOTED' if promoted else 'Rejected'}")

        if promoted:
            net_best = net_challenger
            ckpt_path = f"models_alphazero/best_iter{iteration:04d}.pt"
            net_best.save_weights(ckpt_path)
            net_best.save_weights("models_alphazero/best.pt")
            print(f"  Saved: {ckpt_path}")

        # Log
        training_log.append({
            'iteration':          iteration,
            'avg_loss':           avg_loss,
            'value_loss':         avg_v_loss,
            'policy_loss':        avg_p_loss,
            'challenger_win_rate': win_rate,
            'promoted':           promoted,
            'buffer_size':        len(replay_buffer)
        })
        with open("models_alphazero/training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        # Absolute benchmark every N iterations
        if iteration % BENCHMARK_EVERY == 0:
            run_benchmark(net_best, iteration, NUM_SIMS_EVAL, BOARD_SIZE)

    print("\nTraining complete.")
    print("Final weights: models_alphazero/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Gomoku training")
    parser.add_argument(
        '--mode', type=str, default='m4',
        choices=['test', 'm4', 'full'],
        help="test: convergence check (~20-30 min) | m4: full M4 run (~8-10 hrs) | full: paper-scale GPU"
    )
    args = parser.parse_args()
    train_alphazero(mode=args.mode)
