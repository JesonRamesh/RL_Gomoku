import sys
import pygame
from game.logic import GomokuLogic
from game.board import Board

from agents.random_agent import RandomAgent


def main():
    game = GomokuLogic(board_size=15)
    board = Board(game)

    random_agent = RandomAgent(player=-1)

    running = True

    while running:
        board.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                board.mouse_click(pos)

        # Handle AI Turn
        if board.game_started and not game.game_over and game.current_player == random_agent.player:
            pygame.time.delay(300)

            move = random_agent.predict(game.board)

            if move is not None:
                row, col = move
                try:
                    game.make_move(row, col)
                except ValueError:
                    pass

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
