import numpy as np
from agents.base_agent import BaseAgent

class ThreateningAgent(BaseAgent):
    """
    An agent that tries to extend its longest sequence.
    Can be configured to play sub-optimally for curriculum learning.
    """
    
    def __init__(self, player_id, skill_level=1.0):
        """
        Args:
            player_id: 1 or -1
            skill_level: 0.0 to 1.0
                - 0.0 = completely random (like RandomAgent)
                - 0.5 = 50% smart moves, 50% random
                - 1.0 = always smart moves (hardest)
        """
        super().__init__(player_id)
        self.skill_level = skill_level
        
    def predict(self, board_state):
        """
        With probability = skill_level, play smart move.
        Otherwise, play randomly.
        """
        # Random move with probability (1 - skill_level)
        if np.random.random() > self.skill_level:
            valid_moves = list(zip(*np.where(board_state == 0)))
            return valid_moves[np.random.randint(len(valid_moves))] if valid_moves else None
        
        # Smart move with probability = skill_level
        board_size = board_state.shape[0]
        
        # Check for immediate win
        win_move = self._find_winning_move(board_state, self.player_id)
        if win_move:
            return win_move
        
        # Check for immediate block
        block_move = self._find_winning_move(board_state, -self.player_id)
        if block_move:
            return block_move
        
        # Extend longest sequence
        extend_move = self._find_best_extension(board_state, self.player_id)
        if extend_move:
            return extend_move
        
        # Random move as fallback
        valid_moves = list(zip(*np.where(board_state == 0)))
        return valid_moves[np.random.randint(len(valid_moves))] if valid_moves else None
    
    def _find_winning_move(self, board, player):
        """Find move that creates 5-in-a-row for player."""
        valid_moves = list(zip(*np.where(board == 0)))
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        for move in valid_moves:
            row, col = move
            for dr, dc in directions:
                count = 1
                # Count forward
                r, c = row + dr, col + dc
                while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                       board[r, c] == player):
                    count += 1
                    r += dr
                    c += dc
                
                # Count backward
                r, c = row - dr, col - dc
                while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                       board[r, c] == player):
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 5:
                    return move
        return None
    
    def _find_best_extension(self, board, player):
        """Find move that extends the longest sequence."""
        valid_moves = list(zip(*np.where(board == 0)))
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        best_move = None
        max_count = 0
        
        for move in valid_moves:
            row, col = move
            for dr, dc in directions:
                count = 1
                # Count forward
                r, c = row + dr, col + dc
                while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                       board[r, c] == player):
                    count += 1
                    r += dr
                    c += dc
                
                # Count backward
                r, c = row - dr, col - dc
                while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                       board[r, c] == player):
                    count += 1
                    r -= dr
                    c -= dc
                
                if count > max_count:
                    max_count = count
                    best_move = move
        
        return best_move