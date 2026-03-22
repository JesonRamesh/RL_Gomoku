import pygame
import sys
from .gomoku_env import (Button, Overlays)
 
BLACK = (0,   0,   0)  
WHITE = (255, 255, 255)
RED = (200,  40,  40) 
GREEN = (0,   200, 100) 
TURQUOISE = (64,  224, 208)
LIGHT_YELLOW = (255, 240, 210) 
STONE_RED = (210,  50,  50) 


class Board:
    def __init__(self, game_logic, window_width=650, window_height=550):

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
        self.grid_size_px = (self.rows - 1) * self.cell_pitch  # grid size in pixels

        self.offset_x = (self.Window_Width - self.grid_size_px) // 2
        self.offset_y = (self.Window_Height - self.grid_size_px) // 2

        self.font = pygame.font.SysFont(None, 28)

        self.game_started = False

        self.stone_radius = 18
        self.start_button = Button((120, 25, 120, 40), "Start", colour=GREEN)
        self.reset_button = Button((260, 25, 120, 40), "Reset", colour=TURQUOISE)
        self.quit_button = Button((400, 25, 120, 40), "Quit", colour=RED)

        self.overlays = Overlays(
            window_width  = self.Window_Width,
            window_height = self.Window_Height,
            cell_pitch    = self.cell_pitch,
            stone_radius  = self.stone_radius,
            offset_x      = self.offset_x,
            offset_y      = self.offset_y,
            rows          = self.rows,
            cols          = self.cols,
        )
 
        # Track whether we've already signalled game-over to overlays
        self._game_over_signalled = False


    def find_winning_sequence(self):
        """Return the 5 (row, col) positions that form the winner's line."""
        if not self.game_logic.game_over or self.game_logic.winner == 0:
            return []
 
        board      = self.game_logic.board
        winner     = self.game_logic.winner
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
 
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row, col] != winner:
                    continue
                for dr, dc in directions:
                    seq = [(row, col)]
                    for i in range(1, 5):
                        r, c = row + dr * i, col + dc * i
                        if (0 <= r < self.rows and 0 <= c < self.cols
                                and board[r, c] == winner):
                            seq.append((r, c))
                        else:
                            break
                    if len(seq) >= 5:
                        return seq[:5]
        return []

    def draw_stones(self):
        
        # draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                # draw the stones
                stone = self.game_logic.board[row, col]
                if stone != 0:
                    x = self.offset_x + col * self.cell_pitch
                    y = self.offset_y + row * self.cell_pitch

                    color = BLACK if stone == 1 else STONE_RED
                    pygame.draw.circle(self.screen, color, (x, y), self.stone_radius)

                    # graphic inner highlights
                    hi_surf = pygame.Surface(
                    (self.stone_radius * 2, self.stone_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(hi_surf, (255, 255, 255, 45),
                                   (self.stone_radius - 4, self.stone_radius - 5),
                                   self.stone_radius // 2)
                    self.screen.blit(hi_surf, (x - self.stone_radius, y - self.stone_radius))

        
    def draw(self):

        self.overlays.tick()
        self.screen.fill(LIGHT_YELLOW)

        # white background for grid
        grid_w = self.cols * self.cell_pitch
        grid_h = self.rows * self.cell_pitch
        pygame.draw.rect(
            self.screen,
            WHITE,
            pygame.Rect(self.offset_x - 2, self.offset_y - 2, grid_w - 35, grid_h - 35),
        )

        # draw vertical lines
        for col in range(self.cols):
            x = self.offset_x + col * self.cell_pitch
            pygame.draw.line(
                self.screen,
                BLACK,
                (x, self.offset_y),
                (x, self.offset_y + self.grid_size_px),
                2,
            )
        # draw horizontal
        for row in range(self.rows):
            y = self.offset_y + row * self.cell_pitch
            pygame.draw.line(
                self.screen,
                BLACK,
                (self.offset_x, y),
                (self.offset_x + self.grid_size_px, y),
                2,
            )

        self.draw_stones()

        # Overlays
        if not self.game_started:
            self.overlays.draw_ready(self.screen)
 
        elif self.game_logic.game_over:
            # Signal the overlay once when the game first ends
            if not self._game_over_signalled:
                winning_seq = self.find_winning_sequence()
                self.overlays.on_game_over(winning_seq)
                self._game_over_signalled = True
 
            self.overlays.draw_winning_highlight(self.screen)
            self.overlays.draw_winner_panel(self.screen, self.game_logic.winner)
        
        self.draw_ui()

        pygame.display.flip()
        self.clock.tick(60)

    def draw_ui(self):

        # Draw buttons

        self.start_button.draw(self.screen, self.font)
        self.reset_button.draw(self.screen, self.font)

        if self.game_started:
            self.start_button.colour = (150, 150, 150)
        else:
            self.start_button.colour = GREEN

        self.quit_button.draw(self.screen, self.font)

        # Show turns
        if self.game_started and not self.game_logic.game_over:
            cy = self.Window_Height // 2
            left_cx  = self.offset_x // 2
            right_cx = self.offset_x + self.grid_size_px + (self.Window_Width - self.offset_x - self.grid_size_px) // 2
 
            ellipse_w, ellipse_h = 90, 44
 
            for player_id, cx, label_text, stone_col in (
                (1,  left_cx,  "Player 1", BLACK),
                (-1, right_cx, "Player 2", STONE_RED),
            ):
                active = self.game_logic.current_player == player_id
                ex = cx - ellipse_w // 2
                ey = cy - ellipse_h // 2
 
                # solid ellipse for player labels
                fill_col   = (*stone_col, 220) if active else (*stone_col, 60)
                border_col = (*stone_col, 255) if active else (*stone_col, 120)
 
                el_surf = pygame.Surface((ellipse_w, ellipse_h), pygame.SRCALPHA)
                pygame.draw.ellipse(el_surf, fill_col,   (0, 0, ellipse_w, ellipse_h))
                pygame.draw.ellipse(el_surf, border_col, (0, 0, ellipse_w, ellipse_h), 2)
                self.screen.blit(el_surf, (ex, ey))
 
                # Sheen blitted onto screen for 3D effect
                if active:
                    sheen_surf = pygame.Surface((ellipse_w, ellipse_h // 4), pygame.SRCALPHA)
                    pygame.draw.ellipse(sheen_surf, (255, 255, 255, 55),
                                        (0, 0, ellipse_w, ellipse_h // 4))
                    self.screen.blit(sheen_surf, (ex, ey + 3))
 
                # Label on top of everything 
                if active:
                    label = self.font.render(label_text, True, WHITE)
                    self.screen.blit(label, label.get_rect(center=(cx, cy)))
 

    def mouse_click(self, pos):
        
        # Buttons
        if self.start_button.is_clicked(pos):
            self.game_started = True
            self._game_over_signalled = False
            self.overlays.reset()
            return

        if self.reset_button.is_clicked(pos):
            self.game_logic.reset_game()
            self.game_started = False
            self._game_over_signalled = False
            self.overlays.reset()
            return
        if self.quit_button.is_clicked(pos):
            pygame.quit()
            sys.exit()
        
        # Board
        if not self.game_started or self.game_logic.game_over:
            return
        
        if self.game_logic.current_player != 1:
            return

        mouse_x, mouse_y = pos
        col = round((mouse_x - self.offset_x) / self.cell_pitch)
        row = round((mouse_y - self.offset_y) / self.cell_pitch)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            try:
                self.game_logic.make_move(row, col)
            except ValueError:
                pass
        return None
