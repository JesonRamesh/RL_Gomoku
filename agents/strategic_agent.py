"""
Enhanced Threatening Agent with Strategic Offense

filepath: agents/strategic_agent.py
"""

import numpy as np
import random
from agents.base_agent import BaseAgent


class StrategicAgent(BaseAgent):
    """
    Agent with both offensive and defensive strategy.
    
    Unlike ThreateningAgent (only blocks), StrategicAgent:
    - Builds own threats (3-in-a-row patterns)
    - Blocks opponent threats
    - Uses varied opening patterns
    - Forces diverse board states
    """
    
    def __init__(self, player_id: int, skill_level: float = 0.5, board_size: int = 9):
        """
        Args:
            player_id: 1 or -1
            skill_level: 0.0 = random, 1.0 = expert
                - Controls how often agent plays strategically vs randomly
        """
        super().__init__(player_id)
        self.skill_level = skill_level
        self.board_size = board_size
        self.opening_patterns = self._generate_opening_patterns()
        self.games_played = 0
    
    def _generate_opening_patterns(self):
        """Generate diverse opening move sequences."""
        # center = self.board_size // 2
        # return [
        #     # Center openings
        #     [(center, center)],
        #     [(center-1, center), (center, center-1)],
        #     [(center, center-1), (center-1, center)],
            
        #     # Corner openings
        #     [(3, 3), (4, 4)],
        #     [(11, 11), (10, 10)],
        #     [(3, 11), (4, 10)],
        #     [(11, 3), (10, 4)],
            
        #     # Side openings
        #     [(center, 3), (center, 4)],
        #     [(3, center), (4, center)],
            
        #     # Diagonal patterns
        #     [(5, 5), (6, 6), (7, 7)],
        #     [(9, 9), (8, 8), (7, 7)],
        # ]
        center = self.board_size // 2  # Now 4 for 9×9
        # Update pattern coordinates:
        return [
            # Center openings
            [(center, center)],           # (4, 4)
            [(center-1, center), (center, center-1)],
            
            # Corner openings (adjust to 9×9)
            [(2, 2), (3, 3)],             # Was (3, 3), (4, 4)
            [(6, 6), (5, 5)],             # Was (11, 11), (10, 10)
            
            # Side openings
            [(center, 2), (center, 3)],   # Was (center, 3), (center, 4)
            [(2, center), (3, center)],
            
            # Diagonals
            [(3, 3), (4, 4), (5, 5)],     # Was (5, 5), (6, 6), (7, 7)
        ]
    
    def predict(self, board_state: np.ndarray) -> tuple:
        """
        Strategic decision-making.
        
        Priority:
        1. Win if possible
        2. Block opponent's win
        3. Extend own 3-in-a-row (offensive)
        4. Block opponent's 3-in-a-row (defensive)
        5. Follow opening pattern (early game)
        6. Random move (fallback)
        """
        # Decide: Strategic or random?
        if random.random() > self.skill_level:
            # Random move
            valid_moves = list(zip(*np.where(board_state == 0)))
            return random.choice(valid_moves) if valid_moves else None
        
        # Strategic play
        
        # 1. Win immediately
        win_move = self._find_winning_move(board_state, self.player_id)
        if win_move:
            return win_move
        
        # 2. Block opponent's win
        block_win = self._find_winning_move(board_state, -self.player_id)
        if block_win:
            return block_win
        
        # 3. Extend own 4-in-a-row
        extend_4 = self._find_extension(board_state, self.player_id, target_length=4)
        if extend_4:
            return extend_4
        
        # 4. Block opponent's 4-in-a-row
        block_4 = self._find_extension(board_state, -self.player_id, target_length=4)
        if block_4:
            return block_4
        
        # 5. Extend own 3-in-a-row
        extend_3 = self._find_extension(board_state, self.player_id, target_length=3)
        if extend_3:
            return extend_3
        
        # 6. Block opponent's 3-in-a-row
        block_3 = self._find_extension(board_state, -self.player_id, target_length=3)
        if block_3 and random.random() < 0.5:  # 50% chance to block 3s
            return block_3
        
        # 7. Opening pattern (if early game)
        move_count = np.sum(board_state != 0)
        if move_count < 6:
            pattern = self.opening_patterns[self.games_played % len(self.opening_patterns)]
            for move in pattern:
                if board_state[move] == 0:
                    return move
        
        # 8. Random fallback
        valid_moves = list(zip(*np.where(board_state == 0)))
        return random.choice(valid_moves) if valid_moves else None
    
    def _find_winning_move(self, board: np.ndarray, player: int) -> tuple:
        """Find move that creates 5-in-a-row."""
        valid_moves = list(zip(*np.where(board == 0)))
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
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
    
    def _find_extension(self, board: np.ndarray, player: int, target_length: int) -> tuple:
        """Find move that extends sequence to target_length."""
        valid_moves = list(zip(*np.where(board == 0)))
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
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
                
                if count >= target_length and count > max_count:
                    max_count = count
                    best_move = move
        
        return best_move if max_count >= target_length else None
    
    def reset_for_new_game(self):
        """Call before starting new game."""
        self.games_played += 1