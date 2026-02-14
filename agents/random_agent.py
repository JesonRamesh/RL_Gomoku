import random
import numpy as np

class RandomAgent:

    def __init__(self, player):
        """
        Player: 1 (Black) and -1 (Red)
        """
        self.player = player

    def predict(self, board_state):
        """
        Return a random valid move given the current board state.
        """
        # Find all coordinates where the board is empty
        empty_spots = np.where(board_state == 0)

        # Zip them together to get a list of (row,col)
        valid_moves = list(zip(empty_spots[0], empty_spots[1]))

        # If there are no valid moves, return None
        if not valid_moves:
            return None
        
        # Choose one random move
        return random.choice(valid_moves)