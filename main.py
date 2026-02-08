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
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                pos = pygame.mouse.get_pos()
                board.mouse_click(pos)
                if game.game_over:
                    print(f"Game Over! Winner: {game.winner}")

        

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
    
