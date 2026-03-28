# main.py
import sys
import pygame
import argparse

from agents.dqn_agent import DQNAgent
from agents.dqn_simple import DQNAgent as SimpleDQNAgent
from agents.alphazero_net import AlphaZeroNet
from agents.alphazero_mcts import AlphaZeroMCTS
from game.logic import GomokuLogic
from game.board import Board
from game.match import eval_agents

from agents.strategic_agent import StrategicAgent
from agents.base_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

import torch


def main(headless=False, num_games=100):
    if headless:
        agent1 = RandomAgent(player_id=1)  # Replace with whatever agents
        agent2 = RandomAgent(player_id=-1)

        # Run evaluation and get results dictionary
        results = eval_agents(agent1, agent2, num_games=num_games, board_size=9)
        return results

    # non-headless mode (PyGame)
    game = GomokuLogic(board_size=9)
    board = Board(game)

    player_1 = HumanAgent(player_id=1)

    # === CHOOSE YOUR OPPONENT ===

    # Option 1: Gen 3 — Dueling DQN with PER (best overall DQN agent)
    player_2 = DQNAgent(player_id=-1, board_size=9)
    player_2.load_model("submission_models/gen3_final.pt")
    player_2.epsilon = 0.0

    # Option 2: Gen 2 — Baseline DQN trained via curriculum (self-play + mixed opponents)
    # player_2 = SimpleDQNAgent(player_id=-1, board_size=9)
    # player_2.load_model("submission_models/gen2_best.pt")
    # player_2.epsilon = 0.0

    # Option 3: Gen 4 — AlphaZero (ResNet + MCTS, strongest agent)
    # _az_net = AlphaZeroNet(board_size=9)
    # _az_net.load_state_dict(torch.load("submission_models/gen4_alphazero.pt", map_location="cpu", weights_only=False))
    # _az_net.eval()
    # player_2 = AlphaZeroMCTS(net=_az_net, player_id=-1, num_simulations=400, board_size=9)

    # Option 4: Minimax agent - strong deterministic opponent
    # player_2 = MinimaxAgent(player_id=-1, board_size=9, skill_level=1)

    # Option 5: Strategic agent
    # player_2 = StrategicAgent(player_id=-1, skill_level=1.0, board_size=9)

    # Option 6: Random agent
    # player_2 = RandomAgent(player_id=-1)

    players = {1: player_1, -1: player_2}

    running = True

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

        # Handle AI Turn
        if board.game_started and not game.game_over and not current_agent.is_human:
            pygame.time.delay(300)

            move = current_agent.predict(game.board)

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
    args = parser.parse_args()

    results = main(headless=args.headless, num_games=args.num_games)
