# main.py
import sys
import pygame
import argparse
from game.logic import GomokuLogic
from game.board import Board
from game.match import eval_agents


from agents.base_agent import HumanAgent
from agents.random_agent import RandomAgent


def main(headless=False, num_games=100):
    if headless:
        agent1 = RandomAgent(player_id=1)  # Replace with whatever agents
        agent2 = RandomAgent(player_id=-1)

        # Run evaluation and get results dictionary
        results = eval_agents(agent1, agent2, num_games=num_games)
        return results

    # non-headless mode (PyGame)
    game = GomokuLogic(board_size=15)
    board = Board(game)

    player_1 = HumanAgent(player_id=1)
    player_2 = RandomAgent(player_id=-1)

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
