import sys
import pygame
from game.logic import GomokuLogic
from game.board import Board


def main():
    game = GomokuLogic(board_size=15)
    board = Board(game)

    running = True
    while running:
        board.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the game is not over, prompt the user before quitting
                if not game.game_over:
                    running = False
                    pygame.quit()
                    sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                board.mouse_click(pos)

                if game.game_over:
                    if board.quit_button.is_clicked(pos):
                        running = False
                        pygame.quit()
                        sys.exit()
                    if game.winner == 1:
                        print("Game Over! Winner: Player 1")
                    elif game.winner == -1:
                        print("Game Over! Winner: Player 2")
                    else:
                        print("Game Over! Draw!")

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
