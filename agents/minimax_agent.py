"""
Minimax Agent with Alpha-Beta Pruning for Gomoku

A strong deterministic agent that searches the game tree to find optimal moves.
Uses alpha-beta pruning for efficiency and a sophisticated evaluation function
that considers pattern threats, positional value, and sequence analysis.

Optimizations:
- Transposition table for caching evaluated positions
- Aggressive move pruning (only near existing pieces)
- Iterative deepening with time limit
- Move ordering for maximum alpha-beta cutoffs
- Early threat detection to skip deep search
"""

import numpy as np
from agents.base_agent import BaseAgent
from typing import Optional, Tuple, List, Dict
import time


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning for Gomoku.
    
    Features:
    - Alpha-beta pruning for efficient tree search
    - Transposition table for position caching
    - Iterative deepening with time limit (default 2s)
    - Sophisticated evaluation function
    - Aggressive move ordering for better pruning
    """
    
    # Pattern scores (tuned for Gomoku)
    SCORES = {
        'five': 100000000,
        'open_four': 10000000,
        'half_four': 500000,
        'open_three': 50000,
        'half_three': 5000,
        'open_two': 500,
        'half_two': 50,
        'center_bonus': 15,
    }
    
    def __init__(
        self, 
        player_id: int, 
        board_size: int = 9, 
        depth: int = 6,
        time_limit: float = 2.0,
        skill_level: float = 1.0
    ):
        """
        Initialize the Minimax agent.
        
        Args:
            player_id: 1 or -1
            board_size: Size of the board (default 9)
            depth: Maximum search depth (default 6 with iterative deepening)
            time_limit: Time limit in seconds per move (default 2.0)
            skill_level: 0.0-1.0, probability of playing optimal move (default 1.0)
                        At 0.5, plays randomly 50% of the time
                        At 0.0, always plays randomly (like RandomAgent)
        """
        super().__init__(player_id)
        self.board_size = board_size
        self.depth = depth
        self.time_limit = time_limit
        self.skill_level = skill_level
        self.nodes_searched = 0
        self.start_time = 0
        self.best_move_found = None
        self.timeout = False
        
        # Transposition table: board_hash -> (depth, score, flag, best_move)
        self.tt: Dict[int, Tuple[int, float, str, Optional[Tuple[int, int]]]] = {}
        self.tt_hits = 0
        
        # Precompute center distances for positional scoring
        center = board_size // 2
        self.center_distances = np.zeros((board_size, board_size))
        for r in range(board_size):
            for c in range(board_size):
                self.center_distances[r, c] = max(abs(r - center), abs(c - center))
        
        # Directions for line checking
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    def _hash_board(self, board: np.ndarray) -> int:
        """Create a hash of the board state for transposition table."""
        return hash(board.tobytes())
    
    def predict(self, board_state: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Select the best move using minimax with alpha-beta pruning.
        With skill_level < 1.0, occasionally plays randomly.
        """
        import random
        
        valid_moves = self._get_candidate_moves(board_state)
        if not valid_moves:
            return None
        
        # Skill check: play randomly with probability (1 - skill_level)
        if random.random() > self.skill_level:
            all_valid = list(zip(*np.where(board_state == 0)))
            return random.choice(all_valid) if all_valid else None
        
        self.nodes_searched = 0
        self.tt_hits = 0
        self.start_time = time.time()
        self.best_move_found = None
        self.timeout = False
        self.tt.clear()
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Check for immediate win
        for move in valid_moves:
            if self._is_winning_move(board_state, move, self.player_id):
                return move
        
        # Check for immediate block (opponent about to win)
        for move in valid_moves:
            if self._is_winning_move(board_state, move, -self.player_id):
                return move
        
        # Check for forced moves (opponent has open-4 threat)
        urgent_moves = self._find_urgent_moves(board_state)
        if urgent_moves:
            valid_moves = urgent_moves
        
        # Iterative deepening with time limit
        best_move = valid_moves[0]
        for current_depth in range(1, self.depth + 1):
            if time.time() - self.start_time > self.time_limit * 0.7:
                break
            
            move = self._search_at_depth(board_state, valid_moves, current_depth)
            if move and not self.timeout:
                best_move = move
            
            if self.timeout:
                break
        
        return best_move
    
    def _find_urgent_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Find moves that must be played to prevent losing."""
        urgent = []
        opponent = -self.player_id
        
        # Find all opponent open-4 threats (must block immediately)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != opponent:
                    continue
                for dr, dc in self.directions:
                    count, open_ends, gaps = self._analyze_line_detailed(board, r, c, dr, dc, opponent)
                    if count >= 4 and open_ends >= 1:
                        urgent.extend(gaps)
        
        return list(set(urgent))
    
    def _analyze_line_detailed(
        self, board: np.ndarray, r: int, c: int, dr: int, dc: int, player: int
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        """Analyze line and return (count, open_ends, gap_positions)."""
        count = 0
        gaps = []
        
        nr, nc = r, c
        while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
            if board[nr, nc] == player:
                count += 1
            elif board[nr, nc] == 0:
                gaps.append((nr, nc))
                break
            else:
                break
            nr += dr
            nc += dc
        
        open_ends = 0
        if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 0:
            open_ends += 1
        
        nr, nc = r - dr, c - dc
        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
            if board[nr, nc] == 0:
                open_ends += 1
                gaps.append((nr, nc))
        
        return count, open_ends, gaps
    
    def _search_at_depth(
        self, 
        board: np.ndarray, 
        valid_moves: List[Tuple[int, int]], 
        depth: int
    ) -> Optional[Tuple[int, int]]:
        """Perform minimax search at a specific depth."""
        best_move = valid_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        ordered_moves = self._order_moves(board, valid_moves, self.player_id)
        
        for move in ordered_moves:
            if time.time() - self.start_time > self.time_limit:
                self.timeout = True
                break
            
            new_board = board.copy()
            new_board[move] = self.player_id
            
            score = self._minimax(new_board, depth - 1, alpha, beta, False, move)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        self.best_move_found = best_move
        return best_move
    
    def _minimax(
        self, 
        board: np.ndarray, 
        depth: int, 
        alpha: float, 
        beta: float, 
        is_maximizing: bool,
        last_move: Tuple[int, int]
    ) -> float:
        """Minimax with alpha-beta pruning and transposition table."""
        self.nodes_searched += 1
        
        # Check time limit periodically
        if self.nodes_searched % 500 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.timeout = True
                return 0
        
        if self.timeout:
            return 0
        
        # Check for terminal state (win)
        last_player = -self.player_id if is_maximizing else self.player_id
        if self._check_win(board, last_move, last_player):
            if last_player == self.player_id:
                return self.SCORES['five'] + depth
            else:
                return -self.SCORES['five'] - depth
        
        # Depth limit - evaluate position
        if depth <= 0:
            return self._evaluate(board)
        
        # Transposition table lookup
        board_hash = self._hash_board(board)
        if board_hash in self.tt:
            tt_depth, tt_score, tt_flag, tt_move = self.tt[board_hash]
            if tt_depth >= depth:
                self.tt_hits += 1
                if tt_flag == 'exact':
                    return tt_score
                elif tt_flag == 'lower' and tt_score >= beta:
                    return tt_score
                elif tt_flag == 'upper' and tt_score <= alpha:
                    return tt_score
        
        valid_moves = self._get_candidate_moves(board)
        if not valid_moves:
            return 0
        
        # Limit moves at deeper depths for speed
        max_moves = 12 if depth >= 3 else 8 if depth >= 2 else 6
        current_player = self.player_id if is_maximizing else -self.player_id
        ordered_moves = self._order_moves(board, valid_moves, current_player)[:max_moves]
        
        best_move = ordered_moves[0]
        
        if is_maximizing:
            max_score = float('-inf')
            for move in ordered_moves:
                new_board = board.copy()
                new_board[move] = self.player_id
                score = self._minimax(new_board, depth - 1, alpha, beta, False, move)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            
            # Store in transposition table
            flag = 'exact' if max_score > alpha else 'lower'
            self.tt[board_hash] = (depth, max_score, flag, best_move)
            return max_score
        else:
            min_score = float('inf')
            for move in ordered_moves:
                new_board = board.copy()
                new_board[move] = -self.player_id
                score = self._minimax(new_board, depth - 1, alpha, beta, True, move)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            
            flag = 'exact' if min_score < beta else 'upper'
            self.tt[board_hash] = (depth, min_score, flag, best_move)
            return min_score
    
    def _evaluate(self, board: np.ndarray) -> float:
        """
        Evaluate the board position from the perspective of self.player_id.
        Fast evaluation focusing on key patterns.
        """
        my_score = self._evaluate_player_fast(board, self.player_id)
        opp_score = self._evaluate_player_fast(board, -self.player_id)
        
        return my_score - opp_score * 1.1
    
    def _evaluate_player_fast(self, board: np.ndarray, player: int) -> float:
        """Fast evaluation for a player using pattern scoring."""
        score = 0.0
        opponent = -player
        
        # Scan all lines on the board
        # Horizontal lines
        for r in range(self.board_size):
            score += self._score_line(board[r, :], player, opponent)
        
        # Vertical lines
        for c in range(self.board_size):
            score += self._score_line(board[:, c], player, opponent)
        
        # Diagonals (top-left to bottom-right)
        for k in range(-self.board_size + 1, self.board_size):
            diag = np.diag(board, k)
            if len(diag) >= 5:
                score += self._score_line(diag, player, opponent)
        
        # Anti-diagonals (top-right to bottom-left)
        flipped = np.fliplr(board)
        for k in range(-self.board_size + 1, self.board_size):
            diag = np.diag(flipped, k)
            if len(diag) >= 5:
                score += self._score_line(diag, player, opponent)
        
        # Positional bonus
        positions = np.where(board == player)
        for r, c in zip(*positions):
            score += (self.board_size // 2 - self.center_distances[r, c]) * 10
        
        return score
    
    def _score_line(self, line: np.ndarray, player: int, opponent: int) -> float:
        """Score a single line (row, column, or diagonal)."""
        score = 0.0
        n = len(line)
        
        i = 0
        while i < n:
            if line[i] == player:
                # Count consecutive player pieces
                count = 0
                start = i
                while i < n and line[i] == player:
                    count += 1
                    i += 1
                
                # Check open ends
                open_before = start > 0 and line[start - 1] == 0
                open_after = i < n and line[i] == 0
                open_ends = int(open_before) + int(open_after)
                
                # Score based on pattern
                if count >= 5:
                    score += self.SCORES['five']
                elif count == 4:
                    if open_ends == 2:
                        score += self.SCORES['open_four']
                    elif open_ends == 1:
                        score += self.SCORES['half_four']
                elif count == 3:
                    if open_ends == 2:
                        score += self.SCORES['open_three']
                    elif open_ends == 1:
                        score += self.SCORES['half_three']
                elif count == 2:
                    if open_ends == 2:
                        score += self.SCORES['open_two']
                    elif open_ends == 1:
                        score += self.SCORES['half_two']
            else:
                i += 1
        
        return score
    
    def _get_candidate_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get candidate moves - empty cells near existing pieces.
        
        Only considers moves within 1 cell of existing pieces for speed.
        """
        if np.all(board == 0):
            center = self.board_size // 2
            return [(center, center)]
        
        # Find all cells within distance 1 of any piece (8 neighbors)
        candidates = set()
        pieces = np.where(board != 0)
        
        for r, c in zip(*pieces):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.board_size and 
                        0 <= nc < self.board_size and 
                        board[nr, nc] == 0):
                        candidates.add((nr, nc))
        
        # If very few candidates, expand to distance 2
        if len(candidates) < 5:
            for r, c in zip(*pieces):
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.board_size and 
                            0 <= nc < self.board_size and 
                            board[nr, nc] == 0):
                            candidates.add((nr, nc))
        
        return list(candidates) if candidates else list(zip(*np.where(board == 0)))
    
    def _order_moves(
        self, 
        board: np.ndarray, 
        moves: List[Tuple[int, int]], 
        player: int
    ) -> List[Tuple[int, int]]:
        """
        Order moves for better alpha-beta pruning.
        Fast heuristic scoring without board copies.
        """
        move_scores = []
        opponent = -player
        
        for move in moves:
            r, c = move
            score = 0
            
            # Quick threat assessment by counting neighbors
            for dr, dc in self.directions:
                my_count = 0
                opp_count = 0
                open_before = False
                open_after = False
                
                # Count forward
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                        break
                    if board[nr, nc] == player:
                        my_count += 1
                    elif board[nr, nc] == opponent:
                        opp_count += 1
                        break
                    else:
                        open_after = True
                        break
                
                # Count backward
                for i in range(1, 5):
                    nr, nc = r - dr * i, c - dc * i
                    if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                        break
                    if board[nr, nc] == player:
                        my_count += 1
                    elif board[nr, nc] == opponent:
                        opp_count += 1
                        break
                    else:
                        open_before = True
                        break
                
                # Score based on what we'd create
                if my_count >= 4:
                    score += 100000  # Winning move
                elif my_count == 3 and (open_before or open_after):
                    score += 5000
                elif my_count == 2 and open_before and open_after:
                    score += 500
                elif my_count >= 1:
                    score += my_count * 50
                
                # Score based on blocking
                if opp_count >= 3:
                    score += 10000  # Block opponent win
                elif opp_count == 2:
                    score += 200
            
            # Center preference
            score += int((self.board_size // 2 - self.center_distances[r, c]) * 20)
            
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def _is_winning_move(
        self, 
        board: np.ndarray, 
        move: Tuple[int, int], 
        player: int
    ) -> bool:
        """Check if placing a piece at move would win for player."""
        test_board = board.copy()
        test_board[move] = player
        return self._check_win(test_board, move, player)
    
    def _check_win(
        self, 
        board: np.ndarray, 
        move: Tuple[int, int], 
        player: int
    ) -> bool:
        """Check if player has won after placing at move."""
        row, col = move
        
        for dr, dc in self.directions:
            count = 1
            
            # Count forward
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Count backward
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def get_stats(self) -> dict:
        """Return search statistics from the last move."""
        return {
            'nodes_searched': self.nodes_searched,
            'tt_hits': self.tt_hits,
            'time_elapsed': time.time() - self.start_time,
            'best_move': self.best_move_found
        }
    
    def set_time_limit(self, seconds: float):
        """Adjust the time limit per move."""
        self.time_limit = seconds
    
    def set_skill_level(self, skill: float):
        """Adjust skill level (0.0 = random, 1.0 = optimal)."""
        self.skill_level = max(0.0, min(1.0, skill))
    
    def __repr__(self):
        return f"MinimaxAgent(player={self.player_id}, skill={self.skill_level}, time={self.time_limit}s)"
