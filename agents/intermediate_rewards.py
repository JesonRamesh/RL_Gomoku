import numpy as np

class RewardLogic:
    def __init__(self, player, board, board_size=9):
        self.player = player
        self.opponent = self.player * -1
        self.board = board
        self.board_size = board_size
        

    def _scan_line(self, row, col, dr, dc):

        gap = False
        space = [0, 0]
        blocked = [False, False]
        count = 1  # anchor only — dir_count[0] always 0 due to threats() dedup

        # Forward scan (direction +1): count stones
        for i in range(1, 6):
            r, c = row + dr*i, col + dc*i
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                blocked[1] = True
                break
            cell = self.board[r, c]
            if cell == self.player:
                count += 1
            elif cell == 0:
            # Check for gap pattern X X _ X
                r2, c2 = row + dr*(i+1), col + dc*(i+1)
                if (0 <= r2 < self.board_size and 0 <= c2 < self.board_size
                    and self.board[r2, c2] == self.player):
                    gap = True
                # Count usable forward space
                for j in range(i, 7):
                    r3, c3 = row + dr*j, col + dc*j
                    if (0 <= r3 < self.board_size and 0 <= c3 < self.board_size
                        and self.board[r3, c3] != self.opponent):
                        space[1] += 1
                    else:
                        break
                break
            else:
                blocked[1] = True
                break

        # Backward scan (direction -1): open end + space check only, no counting
        for i in range(1, 6):
            r, c = row - dr*i, col - dc*i
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                blocked[0] = True
                break
            cell = self.board[r, c]
            if cell == 0:
                # Count usable backward space
                for j in range(i, 7):
                    r3, c3 = row - dr*j, col - dc*j
                    if (0 <= r3 < self.board_size and 0 <= c3 < self.board_size
                        and self.board[r3, c3] != self.opponent):
                        space[0] += 1
                    else:
                        break
                break
            elif cell == self.opponent:
                blocked[0] = True
                break
            # cell == self.player means we hit the previous stone,
            # but threats() dedup guarantees this never happens

        # Open end check: can we reach 5 through each end?
        open_end2end = [False, False]
        for dir_idx in range(2):
            if not blocked[dir_idx] and count + space[dir_idx] >= 5:
                open_end2end[dir_idx] = True

        open_ends = sum(open_end2end)
        return count, open_ends, gap



    def threats(self):
        totals = {
            "five": 0,
            "open_four":  0,
            "four":       0,
            "open_three": 0,
            "gap_four":   0,
            "three": 0
        }

        seen = set()  # ensure no duplicates
        
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r,c] != self.player:
                    continue
                for dr, dc in directions:

                    prev_r, prev_c = r - dr, c - dc
                    # start sequence from first stone to avoid duplicates
                    if (0<= prev_r < self.board_size and 0<= prev_c < self.board_size and self.board[prev_r, prev_c] == self.player):
                        continue

                    # start scanning
                    count, open_ends, gap = self._scan_line(r,c, dr, dc)

                    # for safety
                    seq_key = (r, c, dr, dc)

                    if seq_key in seen:
                        continue

                    seen.add(seq_key)

                    if count == 5:
                        totals["five"] += 1
                    elif count == 4:
                        if open_ends == 2:
                            totals["open_four"] += 1
                        elif open_ends == 1:
                            totals["four"] +=1 
                    elif count == 3:
                        if open_ends == 2:
                            totals["open_three"] += 1
                        elif open_ends == 1:
                            totals["three"] += 1
                        

                    if gap and count >= 3:
                        totals["gap_four"] += 1

        return totals
    
    def _proximity_reward(self, board_before, board_after):

        # Find where the new stone was placed
        diff = board_after - board_before
        placed = np.argwhere(diff == self.player)
        if len(placed) == 0:
            return 0.0

        r, c = placed[0]
        bonus = 0.0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board_after[nr, nc] == self.player:
                        dist = max(abs(dr), abs(dc))
                        bonus += 3 if dist == 1 else 0.5   # adjacent > nearby
        return min(bonus, 3.0)   # cap it
    
    def _centre_control_reward(self, board_before, board_after):
        
        # reward for playing near the center at the start
        # Only apply in the opening phase
        moves_played = int(np.count_nonzero(board_after))
        if moves_played > 15:
            return 0.0

        diff = board_after - board_before
        placed = np.argwhere(diff == self.player)
        if len(placed) == 0:
            return 0.0

        r, c = placed[0]
        centre = (self.board_size - 1) / 2  # 4.0 for 9x9

        # Chebyshev distance (max of row/col distance) from centre
        distance = max(abs(r - centre), abs(c - centre))

        # Bonus decays with distance, scaled down as game progresses
        opening_progress = moves_played / 15  # 0.0 (start) -> 1.0 (move 15)
        decay = 1.0 - opening_progress        # reward fades as opening ends

        bonus = 1 * (centre - distance) * decay
        return max(bonus, 0.0)  # no penalty for edge plays, just no bonus




    def rewards(self, board_before, board_after):

        reward =0 

        # offensive, make strategical moves
        self.board = board_before
        agent_before = self.threats()

        self.board = board_after
        agent_after = self.threats()

        # count openings created
        new_open_fours = agent_after["open_four"] - agent_before["open_four"]  # best one: garuanteed win
        new_fours       = agent_after["four"]        - agent_before["four"]   # force opponents move - good
        new_open_threes = agent_after["open_three"]  - agent_before["open_three"]   # good opening
        new_gap_fours   = agent_after["gap_four"]    - agent_before["gap_four"]     # missed links - good
        new_fives = agent_after["five"] - agent_before["five"]
        

        if new_open_fours > 0:
            reward += 4.5 * new_open_fours    # 6 
        if new_fours > 0:
            reward += 4.5* new_fours          # 5
        
        # Fork: single move created simultaneous threats
        total_new_threats = new_open_fours + new_fours + new_open_threes + new_gap_fours
        if total_new_threats >= 2:
            reward += 4                    # 4

        if new_open_threes > 0:
            reward += 3.5 *new_open_threes  # 3.5

        if new_gap_fours > 0:
            reward += 2.0 * new_gap_fours

        # bonus reward for sequential moves towards a line of 5
        if new_fours > 0 and agent_before["open_three"] > 0:
            reward += 2.5

        # Closed three -> four (weaker progression but still valid)
        if new_fours > 0 and agent_before["three"] > 0:
            reward += 2.0

      
        # but reward the setup: had a four and now created open four (near-win pressure)
        if new_open_fours > 0 and (agent_before["four"] > 0 or agent_before["gap_four"] > 0):
            reward += 2.5

        if new_fives > 0 and (agent_before["four"] > 0 or agent_before["open_four"] > 0):
            reward+=4
        
        # Defensive - block opponent
        self.player, self.opponent = self.opponent, self.player
        
        self.board = board_before
        opp_before = self.threats()

        self.board = board_after
        opp_after = self.threats()

        # change back to original perspective
        self.player, self.opponent = self.opponent, self.player
        self.board = board_after

        blocked_open_fours  = opp_before["open_four"]  - opp_after["open_four"]
        blocked_fours       = opp_before["four"]        - opp_after["four"]
        blocked_open_threes = opp_before["open_three"]  - opp_after["open_three"]
        blocked_gap_fours   = opp_before["gap_four"]    - opp_after["gap_four"]
        blocked_threes = opp_before["three"] - opp_after["three"]

        if blocked_open_fours > 0:
            reward += 1.0 * blocked_open_fours   # 1 

        if blocked_fours > 0:
            reward += 4 * blocked_fours        # 3.5

        if blocked_open_threes > 0:
            reward += 4.5 * blocked_open_threes   # 5

        if blocked_gap_fours > 0:
            reward += 3 * blocked_gap_fours      # 2

        if blocked_fours > 0 and opp_before["open_three"] > 0:
            reward += 3.5   # stopped an active escalation, very good

        # Blocked a gap_four that was building from an existing three
        if blocked_gap_fours > 0 and opp_before["three"] > 0:
            reward += 2.0

        if blocked_threes > 0:
            reward += 3

        # Opponent was escalating four->open_four and you stopped it  
        if blocked_open_fours > 0 and opp_before["four"] > 0:
            reward += 2.5

        reward += self._proximity_reward(board_before, board_after)
        reward += self._centre_control_reward(board_before, board_after)
        return reward















        







