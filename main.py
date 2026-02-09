import sys
import pygame
from agents.random_agent import rand_agent
from game.logic import GomokuLogic
from game.board import Board


def main():
    game = GomokuLogic(board_size=15)
    board = Board(game)
    ran_bot = rand_agent(player_id=-1)


    running = True

    while running:
        board.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                
                pos = pygame.mouse.get_pos()
                board.mouse_click(pos)
                   
     
        # random bot move
        if not game.game_over and game.current_player == -1:  
            move = ran_bot.play_move(game.board)
         
            if move:
                try:
                    game.make_move(*move)   # pass in tuple coordinates from move
                except ValueError:
                    pass


        pygame.display.flip()


    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
