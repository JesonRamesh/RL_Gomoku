import pygame
import numpy as np


class Button:
    def __init__(self, rect, text, colour=(0, 0, 0)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.colour = colour

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.colour, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        label = font.render(self.text, True, (0, 0, 0))
        screen.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


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