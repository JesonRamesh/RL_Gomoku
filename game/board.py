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
        grid_width = self.cols * self.cell_pitch
        grid_height = self.rows * self.cell_pitch
        self.offset_x = (self.Window_Width - grid_width) // 2
        self.offset_y = (self.Window_Height - grid_height) // 2


    def draw(self):
        self.screen.fill(LIGHT_YELLOW)

        # black background for grid
        grid_w = self.cols * self.cell_pitch
        grid_h = self.rows * self.cell_pitch
        pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.offset_x-2, self.offset_y-2, grid_w+2, grid_h+2))
        
        # draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                x = self.offset_x + col * self.cell_pitch
                y = self.offset_y + row * self.cell_pitch
                pygame.draw.rect(self.screen, WHITE, (x, y, self.cell_size, self.cell_size))

                # draw the stones
                stone = self.game_logic.board[row, col]
                if stone != 0:
                    color = BLACK if stone == 1 else RED
                    pygame.draw.circle(self.screen, color, (x + self.cell_size // 2, y + self.cell_size // 2), self.cell_size // 2 - 2)

        pygame.display.flip()
        self.clock.tick(60)

    def get_cell_from_mouse_pos(self, pos):
        x, y = pos
        col = (x - self.offset_x) // self.cell_pitch
        row = (y - self.offset_y) // self.cell_pitch
        
        self.game_logic.make_move(row, col)
        return None



