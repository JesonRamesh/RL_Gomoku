import sys
import pygame
from game.logic import GomokuLogic
from game.board import Board

from agents.base_agent import HumanAgent
from agents.random_agent import RandomAgent


def main():
    game = GomokuLogic(board_size=15)
    board = Board(game)

    player_1 = HumanAgent(player_id=1)
    player_2 = RandomAgent(player_id=-1)

    players = {
        1: player_1,
        -1: player_2
    }

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
                    print(f"Agent {current_agent.player_id} attempted an invalid move at {row}, {col}")

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
