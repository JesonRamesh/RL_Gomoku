# main.py
import sys
import os
import threading
import pygame
import argparse

from agents.dqn_agent import DQNAgent
from agents.alphazero_agent import AlphaZeroAgent
from game.logic import GomokuLogic
from game.board import Board
from game.match import eval_agents

from agents.strategic_agent import StrategicAgent
from agents.base_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent


def build_opponent(opponent_name, model_path, board_size, az_simulations):
    """Build opponent agent from CLI options."""
    if opponent_name == "dqn":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DQN model file not found: '{model_path}'. "
                "Pass a valid path with --model-path."
            )
        agent = DQNAgent(player_id=-1, board_size=board_size)
        try:
            agent.load_model(model_path)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Failed to load DQN model from '{model_path}'. "
                "If this is an AlphaZero checkpoint, run with '--opponent alphazero'."
            ) from exc
        agent.epsilon = 0.0
        return agent

    if opponent_name == "alphazero":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"AlphaZero model file not found: '{model_path}'. "
                "Train first or pass a valid path with --model-path."
            )
        agent = AlphaZeroAgent(
            player_id=-1,
            board_size=board_size,
            num_simulations=az_simulations,
        )
        agent.load_model(model_path)
        return agent

    if opponent_name == "minimax":
        return MinimaxAgent(player_id=-1, board_size=board_size, skill_level=1)

    if opponent_name == "strategic":
        return StrategicAgent(player_id=-1, skill_level=1.0, board_size=board_size)

    if opponent_name == "random":
        return RandomAgent(player_id=-1)

    raise ValueError(f"Unsupported opponent: {opponent_name}")


def main(
    headless=False,
    num_games=100,
    opponent="dqn",
    model_path="Model/final.pt",
    az_simulations=20,
):
    if headless:
        agent1 = RandomAgent(player_id=1)  # Replace with whatever agents
        agent2 = RandomAgent(player_id=-1)

        # Run evaluation and get results dictionary
        results = eval_agents(agent1, agent2, num_games=num_games, board_size=9)
        return results

    board_size = 9

    # non-headless mode (PyGame)
    game = GomokuLogic(board_size=board_size)
    board = Board(game)

    player_1 = HumanAgent(player_id=1)

    # Opponent selected by CLI option.
    player_2 = build_opponent(
        opponent_name=opponent,
        model_path=model_path,
        board_size=board_size,
        az_simulations=az_simulations,
    )

    players = {1: player_1, -1: player_2}
    ai_result_lock = threading.Lock()
    ai_result = {"ready": False, "player_id": None, "move": None, "error": None}
    ai_thread = None

    running = True

    def _compute_ai_move(agent, board_snapshot, player_id):
        move = None
        error = None
        try:
            move = agent.predict(board_snapshot)
        except Exception as exc:
            error = str(exc)

        with ai_result_lock:
            ai_result["ready"] = True
            ai_result["player_id"] = player_id
            ai_result["move"] = move
            ai_result["error"] = error

    while running:
        board.draw()

        current_agent = players[game.current_player]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                board.mouse_click(pos)

                if board.game_started and not game.game_over and current_agent.is_human:
                    pass

        # Handle AI turn without blocking the UI thread.
        if board.game_started and not game.game_over and not current_agent.is_human:
            if ai_thread is None or not ai_thread.is_alive():
                with ai_result_lock:
                    ready = ai_result["ready"]

                if not ready:
                    ai_thread = threading.Thread(
                        target=_compute_ai_move,
                        args=(current_agent, game.board.copy(), game.current_player),
                        daemon=True,
                    )
                    ai_thread.start()

            with ai_result_lock:
                ready = ai_result["ready"]
                result_player = ai_result["player_id"]
                move = ai_result["move"]
                error = ai_result["error"]
                if ready:
                    ai_result["ready"] = False
                    ai_result["player_id"] = None
                    ai_result["move"] = None
                    ai_result["error"] = None

            if not ready:
                continue

            if error is not None:
                print(f"Agent {current_agent.player_id} prediction failed: {error}")
                continue

            # Ignore stale results if state changed (e.g., reset) while thinking.
            if result_player != game.current_player:
                continue

            if move is not None:
                row, col = move
                try:
                    game.make_move(row, col)
                except ValueError:
                    print(
                        f"Agent {current_agent.player_id} attempted an invalid move at {row}, {col}"
                    )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play (in headless mode)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="dqn",
        choices=["dqn", "alphazero", "minimax", "strategic", "random"],
        help="Opponent type for interactive mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Model/final.pt",
        help="Model path used by DQN/AlphaZero opponents",
    )
    parser.add_argument(
        "--az-simulations",
        type=int,
        default=20,
        help="MCTS simulations per move when using --opponent alphazero",
    )
    args = parser.parse_args()

    results = main(
        headless=args.headless,
        num_games=args.num_games,
        opponent=args.opponent,
        model_path=args.model_path,
        az_simulations=args.az_simulations,
    )
