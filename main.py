import pygame
from board import Board



board = Board(15)

def main():


    running = True
    while running:

        board.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
    
