import pygame
import sys
from .gomoku_env import Button

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 100)
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

        # define grid size
        self.cell_pitch = 40
        self.cell_size = 38
        self.grid_size_px = (self.rows - 1) * self.cell_pitch # grid size in pixels

        self.offset_x = (self.Window_Width - self.grid_size_px) // 2
        self.offset_y = (self.Window_Height - self.grid_size_px) // 2

        self.font = pygame.font.SysFont(None, 28)

        self.game_started = False

        self.start_button = Button((200, 15, 120, 40), "Start", colour=GREEN)
        self.reset_button = Button((400, 15, 120, 40), "Reset", colour=RED)

        
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
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(self.offset_x-2, self.offset_y-2, grid_w-35, grid_h-35))
        
        # draw veritcal lines
        for col in range(self.cols):
            x = self.offset_x + col*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (x, self.offset_y), (x, self.offset_y + self.grid_size_px), 2)
        # draw horizontal
        for row in range(self.rows):
            y = self.offset_y + row*self.cell_pitch
            pygame.draw.line(self.screen, BLACK, (self.offset_x, y), (self.offset_x + self.grid_size_px, y), 2)
        
        self.draw_stones()
        self.draw_ui()

        pygame.display.flip()
        self.clock.tick(60)

    def draw_ui(self):

        # Draw buttons

        self.start_button.draw(self.screen, self.font)
        self.reset_button.draw(self.screen, self.font)

        # Show turns

        if not self.game_started:
            status = "Ready To Play"
        elif self.game_logic.game_over:
            if self.game_logic.winner == 1:
                status = "Black wins!"
            elif self.game_logic.winner == -1:
                status = "Red wins!"
            
            else:
                status = "Draw"

        else:
            status = "Player 1" if self.game_logic.current_player == 1 else "Player 2"

        label = self.font.render(status, True, (0,0,0))
        self.screen.blit(label, (40, 30))

    def mouse_click(self, pos):
        # Buttons

        if self.start_button.is_clicked(pos):
            self.game_started = True
            return
        if self.reset_button.is_clicked(pos):
            self.game_logic.reset_game()
            self.game_started = False
            return
        
        # Board 
        if not self.game_started or self.game_logic.game_over:
            return
        
        mouse_x, mouse_y = pos
        col = round((mouse_x - self.offset_x) / self.cell_pitch)
        row = round((mouse_y - self.offset_y) / self.cell_pitch)
        if 0 <= row < self.rows and 0 <= col < self.cols:
        
            self.game_logic.make_move(row, col)
        return None
    
    



