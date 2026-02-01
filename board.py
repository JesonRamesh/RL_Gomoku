import pygame
import sys

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_YELLOW = (255, 240, 210)  # 255, 250, 205

class Board:
    def __init__(self, grid_size):

        pygame.init()

        self.rows = grid_size
        self.cols = grid_size
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.Window_Width = 900
        self.Window_Height = 700
        self.screen = pygame.display.set_mode((self.Window_Width, self.Window_Height))
        self.clock = pygame.time.Clock()


    def draw(self):

        # to center the grid
        cell_pitch = 40
        cell_size = 38

        grid_width = self.cols * cell_pitch
        grid_height = self.rows * cell_pitch

        self.offset_x = (self.Window_Width - grid_width) // 2
        self.offset_y = (self.Window_Height - grid_height) // 2

        
        
        self.screen.fill(LIGHT_YELLOW)

        # black background for grid
        pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.offset_x-2, self.offset_y+28, grid_width+2, grid_height+2))
        
        # draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                self.colour = WHITE if self.grid[row][col] == 0 else BLACK
                pygame.draw.rect(self.screen, self.colour, ((col * cell_pitch)+self.offset_x, (row * cell_pitch)+self.offset_y+30, cell_size, cell_size) )

        pygame.display.flip()
        self.clock.tick(60)




