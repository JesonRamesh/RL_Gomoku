import pygame
import sys

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_YELLOW = (255, 240, 210)  # 255, 250, 205

class Board:
    def __init__(self, grid_size):

        self.rows = grid_size
        self.cols = grid_size
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.cell_pitch = 40
        self.cell_size = 38


        self.Window_Width = 900
        self.Window_Height = 700
        self.screen = pygame.display.set_mode((self.Window_Width, self.Window_Height))
        self.clock = pygame.time.Clock()


    def draw(self):

        # to center the grid
        
        grid_size_px = (self.rows - 1) * self.cell_pitch # grid size in pixels

        self.offset_x = (self.Window_Width - grid_size_px) // 2
        self.offset_y = (self.Window_Height - grid_size_px) // 2

        
        
        self.screen.fill(LIGHT_YELLOW)

        # draw veritcal lines
        for col in range(self.cols):
            x = self.offset_x + col*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (x, self.offset_y), (x, self.offset_y + grid_size_px), 2)
        # draw horizontal
        for row in range(self.rows):
            y = self.offset_y + row*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (self.offset_x, y), (self.offset_x + grid_size_px, y), 2)
        
        pygame.display.flip()
        self.clock.tick(60)




