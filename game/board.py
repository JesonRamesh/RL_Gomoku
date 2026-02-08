import pygame
import sys

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_YELLOW = (255, 240, 210)  # 255, 250, 205

class Board:
    def __init__(self, game_logic, window_width=900, window_height=700):

        pygame.init()
        self.game_logic = game_logic
        self.rows = game_logic.board_size
        self.cols = game_logic.board_size

        self.Window_Width = window_width
        self.Window_Height = window_height
        self.screen = pygame.display.set_mode((self.Window_Width, self.Window_Height))
        pygame.display.set_caption("Gomoku Game")
        self.clock = pygame.time.Clock()

        # center the grid
        self.cell_pitch = 40
        self.cell_size = 38
        self.grid_size_px = (self.rows - 1) * self.cell_pitch # grid size in pixels

        self.offset_x = (self.Window_Width - self.grid_size_px) // 2
        self.offset_y = (self.Window_Height - self.grid_size_px) // 2

        
    def draw_stones(self):
        self.stone_radius = 18
        # draw grid
        for row in range(self.rows):
            for col in range(self.cols):

                # draw the stones
                stone = self.game_logic.board[row, col]
                if stone != 0:
                    x = self.offset_x + col *self.cell_pitch
                    y = self.offset_y + row * self.cell_pitch

                    color = BLACK if stone == 1 else RED
                    pygame.draw.circle(self.screen, color, (x, y), self.stone_radius)


    def draw(self):
        self.screen.fill(LIGHT_YELLOW)

        # black background for grid
        grid_w = self.cols * self.cell_pitch
        grid_h = self.rows * self.cell_pitch
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(self.offset_x-2, self.offset_y-2, grid_w-10, grid_h-10))
        
        # draw veritcal lines
        for col in range(self.cols):
            x = self.offset_x + col*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (x, self.offset_y), (x, self.offset_y + self.grid_size_px), 2)
        # draw horizontal
        for row in range(self.rows):
            y = self.offset_y + row*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (self.offset_x, y), (self.offset_x + self.grid_size_px, y), 2)
        
        self.draw_stones()

        pygame.display.flip()
        self.clock.tick(60)

    def mouse_click(self, pos):
        mouse_x, mouse_y = pos
        col = round((mouse_x - self.offset_x) / self.cell_pitch)
        row = round((mouse_y - self.offset_y) / self.cell_pitch)
        if 0 <= row < self.rows and 0 <= col < self.cols:
        
            self.game_logic.make_move(row, col)
        return None
    
    



