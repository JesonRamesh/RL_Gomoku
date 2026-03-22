import pygame
import numpy as np
import math

# colour palette
WHITE = (255, 255, 255)
TURQUOISE = (64,  224, 208)
GOLD = (255, 200,  50)


class Button:
    def __init__(self, rect, text, colour=(0, 0, 0)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.colour = colour

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.colour, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        # highlight
        hi_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(
            hi_surf, (255, 255, 255, 45),
            (4, 2, self.rect.width - 8, self.rect.height // 2)
        )
        screen.blit(hi_surf, (self.rect.x, self.rect.y))

        label = font.render(self.text, True, (0, 0, 0))
        screen.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Overlays:
    
    """
    All overlay graphics for the Gomoku board.
 
    Board creates one instance and calls:
        overlays.tick()                        — every frame, before drawing
        overlays.draw_ready(screen)            — pre-game
        overlays.draw_winning_highlight(screen)— after game ends
        overlays.draw_winner_panel(screen)     — after short delay
        overlays.on_game_over(winning_seq)     — call once when game ends
        overlays.reset()                       — call on game reset
    """
 
    def __init__(self, window_width, window_height, cell_pitch,stone_radius, offset_x, offset_y, rows, cols):
        self.W  = window_width
        self.H  = window_height
        self.cell_pitch   = cell_pitch
        self.stone_radius = stone_radius
        self.offset_x     = offset_x
        self.offset_y     = offset_y
        self.rows = rows
        self.cols = cols
 
        # Fonts
        self.font_large  = pygame.font.SysFont(None, 56)
        self.font_small  = pygame.font.SysFont(None, 22)
 
        # Animation state
        self.frame            = 0
        self.win_frame        = 0
        self.winning_sequence = []   # list of (row, col)
 
    
 
    def tick(self):
        
        self.frame += 1
 
    def on_game_over(self, winning_sequence):
        # call once at the end
        self.winning_sequence = winning_sequence
        self.win_frame        = self.frame
 
    def reset(self):
        
        self.frame            = 0
        self.win_frame        = 0
        self.winning_sequence = []
 
    # ── public draw calls ───────────────────────────────────────────
 
    def draw_ready(self, screen):
        #'GOMOKU / Press Start to begin' panel with animated dots.
        cx, cy = self.W // 2, self.H // 2
 
        pulse = (math.sin(self.frame * 0.04) + 1) / 2
        title_colour = (
            int(220 + pulse * 35),
            int(180 + pulse * 20),
            int(40  + pulse * 10),
        )
 
        self._draw_panel(screen, cx, cy, 300, 112, (10, 10, 10, 185), GOLD)
        self._blit_centred(screen, "GOMOKU",                self.font_large, title_colour, cx, cy - 24)
        self._blit_centred(screen, "Press  Start  to  begin", self.font_small, WHITE,       cx, cy + 22)
 
        # Staggered blinking dots
        for i, dx in enumerate((-18, 0, 18)):
            dot_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            offset   = math.sin(self.frame * 0.08 + i * 1.0)
            a        = max(40, int((offset + 1) / 2 * 220))
            pygame.draw.circle(dot_surf, (*GOLD, a), (4, 4), 4)
            screen.blit(dot_surf, (cx + dx - 4, cy + 46))
 
    def draw_winning_highlight(self, screen):
        # "Gold rings on winning stones + line through
        if not self.winning_sequence:
            return
 
        t   = self.frame - self.win_frame
        pts = [self._grid_to_px(r, c) for r, c in self.winning_sequence]
 
        # Connecting line 
        if len(pts) >= 2:
            line_alpha = min(255, t * 6)
            ls = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            pygame.draw.line(ls, (*GOLD, line_alpha), pts[0], pts[-1], 5)
            screen.blit(ls, (0, 0))
 
        # Pulsing ring per stone, staggered reveal
        pulse       = (math.sin(self.frame * 0.12) + 1) / 2
        ring_radius = self.stone_radius + 5 + int(pulse * 4)
        ring_alpha  = int(160 + pulse * 95)
 
        for i, (row, col) in enumerate(self.winning_sequence):
            x, y         = self._grid_to_px(row, col)
            reveal_frame = i * 8
            if t < reveal_frame:
                continue
            alpha_scale = min(1.0, (t - reveal_frame) / 12)
 
            rs   = pygame.Surface((ring_radius * 2 + 8, ring_radius * 2 + 8), pygame.SRCALPHA)
            cx2  = cy2 = ring_radius + 4
            pygame.draw.circle(rs, (*GOLD,  int(ring_alpha * alpha_scale)), (cx2, cy2), ring_radius, 4)
            pygame.draw.circle(rs, (*WHITE, int(80 * alpha_scale)),          (cx2, cy2), self.stone_radius + 1, 2)
            screen.blit(rs, (x - ring_radius - 4, y - ring_radius - 4))
 
    def draw_winner_panel(self, screen, winner):
        # Winner announcement 
        delay = 70
        t     = self.frame - self.win_frame - delay
        if t < 0:
            return
 
        fade  = min(1.0, t / 25)
        alpha = int(fade * 220)
        cx, cy = self.W // 2, self.H // 2
 
        if winner == 1:
            label, bg, border = "BLACK  WINS",  (10,  10,  10, alpha), GOLD
        elif winner == -1:
            label, bg, border = "RED  WINS",    (150, 20,  20, alpha), GOLD
        else:
            label, bg, border = "IT'S A DRAW",  (40,  40,  80, alpha), TURQUOISE
 
        self._draw_panel(screen, cx, cy, 306, 112, bg, border)
 
        title_surf = self.font_large.render(label, True, WHITE)
        title_surf.set_alpha(int(fade * 255))
        screen.blit(title_surf, title_surf.get_rect(center=(cx, cy - 22)))
 
        sub_surf = self.font_small.render("Press  Reset  to  play  again", True, border)
        sub_surf.set_alpha(int(fade * 200))
        screen.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 22)))
 
        # Divider line
        if fade > 0.5:
            line_alpha = int((fade - 0.5) * 2 * 180)
            ls = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            pygame.draw.line(ls, (*border, line_alpha), (cx - 110, cy), (cx + 110, cy), 1)
            screen.blit(ls, (0, 0))
 
    
 
    def _grid_to_px(self, row, col):
        return (self.offset_x + col * self.cell_pitch,
                self.offset_y + row * self.cell_pitch)
 
    def _draw_panel(self, screen, cx, cy, w, h, bg_rgba, border_colour, radius=10):
        x, y = cx - w // 2, cy - h // 2
        surf  = pygame.Surface((w, h), pygame.SRCALPHA)
        r, g, b, a = bg_rgba
        pygame.draw.rect(surf, (r, g, b, a), (0, 0, w, h), border_radius=radius)
        screen.blit(surf, (x, y))
        pygame.draw.rect(screen, border_colour, pygame.Rect(x, y, w, h), 2, border_radius=radius)
 
    def _blit_centred(self, screen, text, font, colour, cx, cy):
        surf = font.render(text, True, colour)
        screen.blit(surf, surf.get_rect(center=(cx, cy)))

class GomokuEnv:
    """
    RL environment wrapper for Gomoku.
    Agents will use this to train headlessly (no visualisation - PyGame)
    """

    def __init__(self, game_logic, use_sparse_rewards=True):
        self.logic = game_logic
        self.use_sparse_rewards = use_sparse_rewards

    def reset(self):
        self.logic.reset_game()
        return np.copy(self.logic.board)

    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        row, col = action
        moving_player = self.logic.current_player

        try:
            self.logic.make_move(row, col)
        except ValueError:
            return np.copy(self.logic.board), -1.0, True, {"error": "Illegal Move"}

        # reward = 0.0
        done = self.logic.game_over

        if done:
            if self.logic.winner == moving_player:
                reward = 1.0
            elif self.logic.winner == 0:
                reward = 0.0
            else:
                reward = -1.0
        else:
            if self.use_sparse_rewards:
                reward = 0.0
            else:
                reward = 0.0
                reward += self._evaluate_threat_value(row, col, moving_player)

                # Reward for blocking opponent threats
                opponent_player = -moving_player
                blocking_reward = self._evaluate_blocking_move(row, col, opponent_player)
                reward += blocking_reward
            # # Reward for creating own threats
            # reward += self._evaluate_threat_value(row, col, moving_player)
            
            # # Penalty for allowing opponent threats
            # opponent_player = -moving_player
            # # opponent_threat = self._count_max_threat_on_board(opponent_player)
            
            # # if opponent_threat == 4:
            # #     reward -= 0.3
            # # elif opponent_threat == 3:
            # #     reward -= 0.08

            # blocking_reward = self._evaluate_blocking_move(row, col, opponent_player)
            # reward += blocking_reward
            
                reward = np.clip(reward, -1.0, 1.0)

        return np.copy(self.logic.board), reward, done, {}
    
    
    def _evaluate_threat_value(self, row, col, player):
        """Reward the agent for creating or blocking threats. This is a simple heuristic that gives small positive rewards for creating 2, 3, or 4 in a row, and small negative rewards for allowing the opponent to create threats."""
        board = self.logic.board
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        threat_value = 0
        for dr, dc in directions:
            count = 1
            # Keep count of consecutive pieces in both directions
            for i in range(1,5):
                r, c = row + dr*i, col + dc*i
                if 0 <= r < self.logic.board_size and 0 <= c < self.logic.board_size and board[r, c] == player:
                    count += 1
                else:
                    break

            for i in range(1,5):
                r, c = row - dr*i, col - dc*i
                if 0 <= r < self.logic.board_size and 0 <= c < self.logic.board_size and board[r, c] == player:
                    count += 1
                else:
                    break

            # Reward by threat level
            if count == 2:
                threat_value += 0.005
            elif count == 3:
                threat_value += 0.02
            elif count == 4:
                threat_value += 0.1

        return threat_value
    
    def _count_max_threat_on_board(self, player):
        """
        Returns the longest consecutive sequence for a player on the entire board.
        Used to penalize agent for allowing opponent threats.
        """
        board = self.logic.board
        max_count = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # horizontal, vertical, diagonal /, diagonal \

        for row in range(self.logic.board_size):
            for col in range(self.logic.board_size):
                if board[row, col] == player:
                    # Check all 4 directions from this stone
                    for dr, dc in directions:
                        count = 1
                        # Count consecutive stones in forward direction
                        r, c = row + dr, col + dc
                        while (0 <= r < self.logic.board_size and 
                            0 <= c < self.logic.board_size and 
                            board[r, c] == player):
                            count += 1
                            r += dr
                            c += dc
                        
                        # Update max if this sequence is longer
                        if count > max_count:
                            max_count = count

        return max_count
    
    def _evaluate_blocking_move(self, row, col, opponent_player):
        """
        Check if the move at (row, col) blocks an opponent threat.
        Returns positive reward if blocking a dangerous sequence.
        """
        board = self.logic.board
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        blocking_value = 0.0
        
        for dr, dc in directions:
            # Count opponent stones that would form a line through this position
            forward_count = 0
            for i in range(1, 5):
                r, c = row + dr*i, col + dc*i
                if (0 <= r < self.logic.board_size and 0 <= c < self.logic.board_size):
                    if board[r, c] == opponent_player:
                        forward_count += 1
                    else:
                        break
                else:
                    break
            
            backward_count = 0
            for i in range(1, 5):
                r, c = row - dr*i, col - dc*i
                if (0 <= r < self.logic.board_size and 0 <= c < self.logic.board_size):
                    if board[r, c] == opponent_player:
                        backward_count += 1
                    else:
                        break
                else:
                    break
            
            total_blocked = forward_count + backward_count
            
            # Reward blocking based on threat severity
            if total_blocked >= 4:
                blocking_value += 0.3  # Blocked a winning threat
            elif total_blocked == 3:
                blocking_value += 0.1  # Blocked 4-in-a-row setup
            elif total_blocked == 2:
                blocking_value += 0.02  # Blocked 3-in-a-row setup
        
        return blocking_value