import numpy as np
from game.logic import GomokuLogic


class GomokuEnvShaped:
    """
    RL environment with carefully calibrated shaped rewards.
    """

    def __init__(self, game_logic):
        self.logic = game_logic
        self.board_size = game_logic.board_size
        self.prev_board = None

    def reset(self):
        self.logic.reset_game()
        self.prev_board = np.copy(self.logic.board)
        return np.copy(self.logic.board)

    def step(self, action):
        """Execute action with shaped rewards."""
        row, col = action
        moving_player = self.logic.current_player
        opponent = -moving_player
        
        # Store board state before move for threat analysis
        board_before = np.copy(self.logic.board)
        
        # Check opponent threats BEFORE our move
        opp_threats_before = self._analyze_threats(board_before, opponent)

        try:
            self.logic.make_move(row, col)
        except ValueError:
            return np.copy(self.logic.board), -1.0, True, {"error": "Illegal Move"}

        done = self.logic.game_over
        board_after = np.copy(self.logic.board)

        # === REWARD CALCULATION ===
        reward = 0.0
        
        if done:
            if self.logic.winner == moving_player:
                reward = 1.0
            elif self.logic.winner == 0:
                reward = 0.0
            else:
                reward = -1.0
        else:
            # SHAPED REWARDS (non-terminal states)
            
            # 1. BLOCKING REWARDS (most important for learning defense)
            block_reward = self._calc_blocking_reward(row, col, board_after, opponent, opp_threats_before)
            
            # 2. THREAT CREATION REWARDS  
            threat_reward = self._calc_threat_reward(row, col, board_after, moving_player)
            
            # 3. PENALTY for ignoring critical threats
            ignore_penalty = self._calc_ignore_penalty(row, col, board_before, opponent, opp_threats_before)
            
            # 4. POSITIONAL BONUS (center control, adjacency)
            positional_reward = self._calc_positional_reward(row, col, board_after, moving_player)
            
            reward = block_reward + threat_reward + ignore_penalty + positional_reward
            reward = np.clip(reward, -0.5, 0.5)

        self.prev_board = board_after
        return board_after, reward, done, {}

    def _analyze_threats(self, board, player):
        """
        Analyze all threats for a player.
        Returns dict with threat counts and positions.
        """
        threats = {
            'win_threats': [],      # Can win next move (4 in a row with open end)
            'four_threats': [],     # 4 in a row (blocked or open)
            'open_threes': [],      # Open 3 in a row
            'threes': [],           # Any 3 in a row
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Check each empty cell as potential threat completion point
        empty_cells = list(zip(*np.where(board == 0)))
        
        for pos in empty_cells:
            row, col = pos
            for dr, dc in directions:
                count, open_ends = self._count_line(board, row, col, dr, dc, player)
                
                if count >= 4:
                    threats['win_threats'].append(pos)
                elif count == 3 and open_ends >= 1:
                    threats['four_threats'].append(pos)
                elif count == 2 and open_ends == 2:
                    threats['open_threes'].append(pos)
                elif count == 2:
                    threats['threes'].append(pos)
        
        return threats
    
    def _count_line(self, board, row, col, dr, dc, player):
        """
        Count consecutive pieces in a line through (row, col) for player.
        Returns (count, open_ends).
        """
        count = 0
        open_ends = 0
        
        # Count forward
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == player:
                count += 1
                r += dr
                c += dc
            else:
                if board[r, c] == 0:
                    open_ends += 1
                break
        else:
            pass  # Hit edge
        
        # Count backward
        r, c = row - dr, col - dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            else:
                if board[r, c] == 0:
                    open_ends += 1
                break
        
        return count, open_ends

    def _calc_blocking_reward(self, row, col, board, opponent, opp_threats):
        """
        Reward for blocking opponent threats.
        """
        reward = 0.0
        pos = (row, col)
        
        # Huge reward for blocking a winning threat
        if pos in opp_threats['win_threats']:
            reward += 0.4
        
        # Good reward for blocking 4-threats
        if pos in opp_threats['four_threats']:
            reward += 0.2
        
        # Small reward for blocking open threes
        if pos in opp_threats['open_threes']:
            reward += 0.08
        
        return reward

    def _calc_threat_reward(self, row, col, board, player):
        """
        Reward for creating our own threats.
        """
        reward = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        threats_created = 0
        strong_threats = 0
        
        for dr, dc in directions:
            count, open_ends = self._count_line(board, row, col, dr, dc, player)
            # Note: count doesn't include the piece we just placed, so actual line = count + 1
            
            actual_count = count + 1  # Include the piece we placed
            
            if actual_count >= 4 and open_ends >= 1:
                strong_threats += 1  # Created a winning threat
                reward += 0.15
            elif actual_count == 3 and open_ends >= 2:
                strong_threats += 1  # Open four
                reward += 0.08
            elif actual_count == 3 and open_ends >= 1:
                threats_created += 1
                reward += 0.03
            elif actual_count == 2 and open_ends >= 2:
                reward += 0.01
        
        # Bonus for creating multiple threats (fork)
        if strong_threats >= 2:
            reward += 0.15  # Fork bonus
        
        return reward

    def _calc_ignore_penalty(self, row, col, board_before, opponent, opp_threats):
        """
        Penalty for NOT blocking critical threats.
        """
        penalty = 0.0
        pos = (row, col)
        
        # If opponent had winning threats and we didn't block any of them
        if opp_threats['win_threats']:
            if pos not in opp_threats['win_threats']:
                # We ignored a winning threat! Big penalty
                penalty -= 0.3
        
        # If opponent had 4-threats and we didn't block
        elif opp_threats['four_threats']:
            if pos not in opp_threats['four_threats']:
                # Check if there were multiple - if so, we might be okay
                if len(opp_threats['four_threats']) == 1:
                    penalty -= 0.1
        
        return penalty

    def _calc_positional_reward(self, row, col, board, player):
        """
        Small reward for good positional play.
        """
        reward = 0.0
        center = self.board_size // 2
        
        # Slight preference for center area
        dist_to_center = abs(row - center) + abs(col - center)
        if dist_to_center <= 2:
            reward += 0.005
        
        # Reward for playing adjacent to own pieces (connectivity)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if board[r, c] == player:
                        reward += 0.002
        
        return reward
