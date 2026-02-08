import sys
import pygame
from game.logic import GomokuLogic
from game.board import Board

def main():
    game = GomokuLogic(board_size=15)
    view = Board(game)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                pos = pygame.mouse.get_pos()
                view.mouse_click(pos)
                if game.game_over:
                    print(f"Game Over! Winner: {game.winner}")

        view.draw()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
    
