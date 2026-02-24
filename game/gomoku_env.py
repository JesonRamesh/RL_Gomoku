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

    def __init__(self, game_logic):
        self.logic = game_logic

    def reset(self):
        self.logic.reset_game()
        return np.copy(self.logic.board)

    def step(self, action):
        """
        Exectutes a move and returns (next state, reward, done, info)
        """
        row, col = action
        reward = 0
        done = False
        info = {}

        try:
            self.logic.make_move(row, col)
        except ValueError:
            # If the agent tries to make an illegal move, give a big negative reward and end the game immediately
            return np.copy(self.logic.board), -10, True, {"error": "Illegal Move"}

        if self.logic.game_over:
            done = True
            if self.logic.winner == self.logic.current_player:
                # The player who made the last move won
                reward = 10
            elif self.logic.winner == 0:
                # Draw
                reward = 0

        return np.copy(self.logic.board), reward, done, info
