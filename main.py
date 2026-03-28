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

# Define paths
DQN140_PATH = "Model/finaldqn140.pt"
DQN160_PATH = "Model/rohan_model_160_epochs_slim.pt"
# AZ_QUICK_PATH = "Model/alphazero_quick_final.pt"
AZ_PATH = "Model/alphazero_final.pt"

# Single place to configure default headless matchup.
HEADLESS_DEFAULTS = {
    "num_games": 2000,
    "board_size": 9,
    "agent1_type": "dqn",
    "agent1_model_path": DQN140_PATH,
    "agent1_az_simulations": 20,
    "agent2_type": "alphazero",
    "agent2_model_path": AZ_PATH,
    "agent2_az_simulations": 20,
}

HEADLESS_AGENT_CHOICES = [
    "dqn",
    "alphazero",
    "minimax",
    "strategic",
    "random",
]


def default_model_path_for_agent(agent_name):
    if agent_name == "dqn":
        return DQN140_PATH
    if agent_name == "alphazero":
        return AZ_PATH
    return None


def build_agent(
    agent_name,
    player_id,
    board_size,
    model_path=None,
    az_simulations=20,
):
    """Build any supported agent from CLI options."""
    if model_path is None:
        model_path = default_model_path_for_agent(agent_name)

    if agent_name == "dqn":
        if model_path is None:
            raise ValueError(
                "DQN requires a model path. Pass --model-path (interactive) or "
                "--agent1-model-path/--agent2-model-path (headless)."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DQN model file not found: '{model_path}'.")

        agent = DQNAgent(player_id=player_id, board_size=board_size)
        try:
            agent.load_model(model_path)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Failed to load DQN model from '{model_path}'. "
                "If this is an AlphaZero checkpoint, use agent type 'alphazero'."
            ) from exc
        agent.epsilon = 0.0
        return agent

    if agent_name == "alphazero":
        if model_path is None:
            raise ValueError(
                "AlphaZero requires a model path. Pass --model-path (interactive) "
                "or --agent1-model-path/--agent2-model-path (headless)."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"AlphaZero model file not found: '{model_path}'. "
                "Train first or pass a valid path."
            )

        agent = AlphaZeroAgent(
            player_id=player_id,
            board_size=board_size,
            num_simulations=az_simulations,
        )
        try:
            agent.load_model(model_path)
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Failed to load AlphaZero model from '{model_path}'. "
                "If this is a DQN checkpoint, use agent type 'dqn'."
            ) from exc
        return agent

    if agent_name == "minimax":
        return MinimaxAgent(player_id=player_id, board_size=board_size, skill_level=1)

    if agent_name == "strategic":
        return StrategicAgent(
            player_id=player_id, skill_level=1.0, board_size=board_size
        )

    if agent_name == "random":
        return RandomAgent(player_id=player_id)

    raise ValueError(f"Unsupported agent: {agent_name}")


def build_opponent(opponent_name, model_path, board_size, az_simulations):
    """Build opponent agent from CLI options."""
    return build_agent(
        agent_name=opponent_name,
        player_id=-1,
        board_size=board_size,
        model_path=model_path,
        az_simulations=az_simulations,
    )


def main(
    headless=False,
    num_games=HEADLESS_DEFAULTS["num_games"],
    board_size=HEADLESS_DEFAULTS["board_size"],
    opponent="dqn",
    model_path=None,
    az_simulations=20,
    agent1_type=HEADLESS_DEFAULTS["agent1_type"],
    agent1_model_path=HEADLESS_DEFAULTS["agent1_model_path"],
    agent1_az_simulations=HEADLESS_DEFAULTS["agent1_az_simulations"],
    agent2_type=HEADLESS_DEFAULTS["agent2_type"],
    agent2_model_path=HEADLESS_DEFAULTS["agent2_model_path"],
    agent2_az_simulations=HEADLESS_DEFAULTS["agent2_az_simulations"],
):
    if agent1_model_path is None:
        agent1_model_path = default_model_path_for_agent(agent1_type)
    if agent2_model_path is None:
        agent2_model_path = default_model_path_for_agent(agent2_type)

    if headless:
        agent1 = build_agent(
            agent_name=agent1_type,
            player_id=1,
            board_size=board_size,
            model_path=agent1_model_path,
            az_simulations=agent1_az_simulations,
        )
        agent2 = build_agent(
            agent_name=agent2_type,
            player_id=-1,
            board_size=board_size,
            model_path=agent2_model_path,
            az_simulations=agent2_az_simulations,
        )

        print(
            f"Headless eval: {agent1_type} vs {agent2_type} | "
            f"games={num_games} | board={board_size}"
        )

        # Run evaluation and get results dictionary.
        results = eval_agents(
            agent1,
            agent2,
            num_games=num_games,
            board_size=board_size,
        )
        return results

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
        default=HEADLESS_DEFAULTS["num_games"],
        help="Number of games to play (in headless mode)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="dqn",
        choices=HEADLESS_AGENT_CHOICES,
        help="Opponent type for interactive mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path used by DQN/AlphaZero opponents",
    )
    parser.add_argument(
        "--az-simulations",
        type=int,
        default=20,
        help="MCTS simulations per move when using --opponent alphazero",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=HEADLESS_DEFAULTS["board_size"],
        help="Board size for both interactive and headless modes",
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default=HEADLESS_DEFAULTS["agent1_type"],
        choices=HEADLESS_AGENT_CHOICES,
        help="Headless mode: first agent type",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default=HEADLESS_DEFAULTS["agent2_type"],
        choices=HEADLESS_AGENT_CHOICES,
        help="Headless mode: second agent type",
    )
    parser.add_argument(
        "--agent1-model-path",
        type=str,
        default=HEADLESS_DEFAULTS["agent1_model_path"],
        help="Headless mode: model path for agent1 if needed",
    )
    parser.add_argument(
        "--agent2-model-path",
        type=str,
        default=HEADLESS_DEFAULTS["agent2_model_path"],
        help="Headless mode: model path for agent2 if needed",
    )
    parser.add_argument(
        "--agent1-az-simulations",
        type=int,
        default=HEADLESS_DEFAULTS["agent1_az_simulations"],
        help="Headless mode: MCTS simulations per move for agent1 when AlphaZero",
    )
    parser.add_argument(
        "--agent2-az-simulations",
        type=int,
        default=HEADLESS_DEFAULTS["agent2_az_simulations"],
        help="Headless mode: MCTS simulations per move for agent2 when AlphaZero",
    )
    args = parser.parse_args()

    results = main(
        headless=args.headless,
        num_games=args.num_games,
        board_size=args.board_size,
        opponent=args.opponent,
        model_path=args.model_path,
        az_simulations=args.az_simulations,
        agent1_type=args.agent1,
        agent1_model_path=args.agent1_model_path,
        agent1_az_simulations=args.agent1_az_simulations,
        agent2_type=args.agent2,
        agent2_model_path=args.agent2_model_path,
        agent2_az_simulations=args.agent2_az_simulations,
    )
