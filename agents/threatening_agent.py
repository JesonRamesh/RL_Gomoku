"""
Threatening Agent for Curriculum Learning
"""

import numpy as np
import random
from agents.base_agent import BaseAgent


class ThreateningAgent(BaseAgent):
    """
    Agent that blocks opponent's critical threats at configurable rate.
    
    Strategy:
    1. Detect if opponent has 4-in-a-row (about to win)
    2. With probability = block_probability, block it
    3. Otherwise, play randomly
    
    This creates gentle defensive pressure without adding offensive complexity.
    Perfect for curriculum learning where we want to teach blocking gradually.
    
    Examples:
        Threatening-0.1: Blocks 10% of threats, random 90%
        Threatening-0.3: Blocks 30% of threats, random 70%
        Threatening-0.5: Blocks 50% of threats, random 50%
    """
    
    def __init__(self, player_id: int, block_probability: float = 0.0, board_size: int = 15):
        """
        Initialize Threatening Agent.
        
        Args:
            player_id: 1 or -1
            block_probability: Probability of blocking threats (0.0 to 1.0)
                - 0.0: Never blocks (same as RandomAgent)
                - 0.1: Blocks 10% of threats
                - 0.3: Blocks 30% of threats
                - 0.5: Blocks 50% of threats
                - 1.0: Always blocks threats
            board_size: Size of game board (default 15)
        """
        super().__init__(player_id)
        self.block_probability = block_probability
        self.board_size = board_size
        
        # Statistics tracking (for testing/debugging)
        self.stats = {
            'threats_detected': 0,
            'threats_blocked': 0,
            'random_moves': 0,
            'total_moves': 0
        }
    
    def predict(self, board_state: np.ndarray) -> tuple:
        """
        Make a move decision.
        
        Strategy:
        1. Find all critical threats (opponent 4-in-a-row)
        2. If found AND random() < block_probability: Block it
        3. Else: Random move
        
        Args:
            board_state: 2D numpy array (board_size x board_size)
                        Values: -1, 0, 1
        
        Returns:
            (row, col): Tuple representing the chosen move
        """
        self.stats['total_moves'] += 1
        
        # 1. Find critical threats (opponent about to win)
        threats = self._find_critical_threats(board_state)
        
        # 2. Decide: Block or random?
        if threats and random.random() < self.block_probability:
            # Block one of the threats
            self.stats['threats_detected'] += 1
            self.stats['threats_blocked'] += 1
            return random.choice(threats)
        else:
            # Play randomly
            if threats:
                self.stats['threats_detected'] += 1
            self.stats['random_moves'] += 1
            
            valid_moves = list(zip(*np.where(board_state == 0)))
            if not valid_moves:
                return None  # Board full
            return random.choice(valid_moves)
    
    def _find_critical_threats(self, board: np.ndarray) -> list:
        """
        Find all positions where opponent has 4-in-a-row (critical threat).
        
        This is a SIMPLE detector that only finds IMMEDIATE win threats.
        Does NOT find:
        - 3-in-a-row (not critical yet)
        - Double threats
        - Fork patterns
        
        We keep it simple because:
        1. Faster to compute
        2. Easier to test
        3. Sufficient for curriculum learning
        
        Args:
            board: 2D numpy array
        
        Returns:
            List of (row, col) tuples where blocking would prevent opponent win
        """
        threats = []
        opponent_id = -self.player_id
        
        # Check all four directions
        directions = [
            (0, 1),   # Horizontal →
            (1, 0),   # Vertical ↓
            (1, 1),   # Diagonal ↘
            (1, -1)   # Diagonal ↙
        ]
        
        for direction in directions:
            threats.extend(self._find_threats_in_direction(board, opponent_id, direction))
        
        # Remove duplicates (same position might be found in multiple scans)
        return list(set(threats))
    
    def _find_threats_in_direction(
        self, 
        board: np.ndarray, 
        opponent_id: int, 
        direction: tuple
    ) -> list:
        """
        Find all 4-in-a-row threats in one specific direction.
        
        Pattern: O O O O _ (4 opponent stones + 1 gap)
        
        Args:
            board: Game board
            opponent_id: -1 or 1
            direction: (dr, dc) tuple for direction vector
        
        Returns:
            List of (row, col) tuples where the gap is
        """
        threats = []
        dr, dc = direction
        rows, cols = board.shape
        
        # Scan all possible 5-stone windows in this direction
        for r in range(rows):
            for c in range(cols):
                # Check if we can fit a 5-stone window starting from (r, c)
                end_r = r + 4 * dr
                end_c = c + 4 * dc
                
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue  # Window goes out of bounds
                
                # Extract the 5-stone window
                window = []
                positions = []
                for i in range(5):
                    pos_r = r + i * dr
                    pos_c = c + i * dc
                    window.append(board[pos_r, pos_c])
                    positions.append((pos_r, pos_c))
                
                # Check if window has exactly 4 opponent stones and 1 gap
                opponent_count = window.count(opponent_id)
                gap_count = window.count(0)
                
                if opponent_count == 4 and gap_count == 1:
                    # Found a threat! Find the gap position
                    gap_index = window.index(0)
                    gap_position = positions[gap_index]
                    threats.append(gap_position)
        
        return threats
    
    def get_blocking_rate(self) -> float:
        """
        Get actual blocking rate from statistics.
        
        Returns:
            Percentage of detected threats that were blocked (0.0 to 1.0)
        """
        if self.stats['threats_detected'] == 0:
            return 0.0
        return self.stats['threats_blocked'] / self.stats['threats_detected']
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'threats_detected': 0,
            'threats_blocked': 0,
            'random_moves': 0,
            'total_moves': 0
        }
    
    def __repr__(self):
        """String representation for debugging."""
        return f"ThreateningAgent(player={self.player_id}, block_prob={self.block_probability})"


# ============================================================
# LEGACY COMPATIBILITY: Keep old ThreateningAgent as alias
# ============================================================

class ThreateningAgentLegacy(BaseAgent):
    """
    Original skill_level implementation (DEPRECATED).
    
    Use ThreateningAgent with block_probability instead.
    This is kept for backward compatibility with existing code.
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
        """With probability = skill_level, play smart move. Otherwise, random."""
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