import numpy as np


class GomokuLogic:
    def __init__(self, board_size=15):
        self.board_size = board_size  # Initialize the game board as a 2D numpy array filled with zeros (empty cells)
        self.board = np.zeros(
            (board_size, board_size), dtype=int
        )  # Player 1 will be represented by 1, Player 2 by -1, and empty cells by 0
        self.current_player = 1  # Player 1 starts the game
        self.winner = None  # Variable to store the winner (1 for Player 1, -1 for Player 2, 0 for draw)
        self.game_over = False  # Flag to indicate if the game has ended

    def get_valid_moves(self):
        """
        Returns a list of valid moves (empty cells) on the board.
        """
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, row, col):
        """
        Place the current player's piece on the board at the specified row and column.
        """
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("Move out of bounds")
        if self.board[row, col] != 0 or self.game_over:
            raise ValueError("Invalid move")
        
        self.board[row, col] = self.current_player

        if self.check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # Draw
        else:
            self.current_player *= -1  # Switch player

        return True

    def check_win(self, row, col):
        """
        Check 4 directions (horizontal, vertical, diagonal) for a win condition (5 in a row) after the current player has made a move at (row, col).
        """
        player = self.board[row, col]
        # Check all 4 directions: horizontal, vertical, diagonal (top-left to bottom-right), diagonal (top-right to bottom-left)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dir_row, dir_col in directions:
            count = 1

            # Check in the positive direction
            for i in range(1, 5):
                r, c = row + dir_row * i, col + dir_col * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break

            # Check in the negative direction
            for i in range(1, 5):
                r, c = row - dir_row * i, col - dir_col * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break

            # If we have 5 or more in a row, the current player wins
            if count >= 5:
                return True
        return False

    def reset_game(self):
        """
        Reset the game to the initial state.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
