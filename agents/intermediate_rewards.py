

class RewardLogic:
    def __init__(self, player, board, board_size=9):
        self.player = player
        self.opponent = self.player * -1
        self.board = board
        self.board_size = board_size
        

    def _scan_line(self, row, col, dr, dc):

        gap = False    # gap connecting two lines of same stones
        open_ends = 0  # open fours or open threes 
        count = 1      # consecutive same player stones (including anchor stone)

        open_end2end = [False, False]


        for dir, ind in [(-1, 0), (1,1)]:  # to scan both direction in a single axis
        

            for i in range(1, 5):
                r, c = row + dr*dir*i, col + dc*dir*i   
                if not ((0<= r < self.board_size) and (0<= c < self.board_size)):   # break scan in this direction if end of board
                    break

                cell = self.board[r,c]
                if cell == self.player:  
                    count+=1 
                elif cell == 0: 
                    r2, c2 = row + dr * dir * (i + 1), col + dc * dir * (i + 1)
                    if (0 <= r2 < self.board_size and 0 <= c2 < self.board_size
                        and self.board[r2, c2] == self.player):
                        gap = True  # check for XX_X 

                    # check how many spaces there are and if there's a chance of making 5 in a row
                    space = 1
                    for j in range(i + 1, 6):
                        r3, c3 = row + dr * dir * j, col + dc * dir * j
                        if (0 <= r3 < self.board_size and 0 <= c3 < self.board_size
                            and self.board[r3, c3] != self.opponent):  
                            space += 1
                        else:
                            break
                
                    if (count + space) >= 5:
                        open_end2end[ind] = True
                    break
                else:
                    break  # blocked by opponent

        open_ends = sum(open_end2end)
        return count, open_ends, gap
    

    def threats(self):
        totals = {
            "open_four":  0,
            "four":       0,
            "open_three": 0,
            "gap_four":   0,
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

                    if count == 4:
                        if open_ends == 2:
                            totals["open_four"] += 1
                        elif open_ends == 1:
                            totals["four"] +=1 
                    elif count == 3:
                        if open_ends == 2:
                            totals["open_three"] += 1
                        

                    if gap:
                        totals["gap_four"] += 1

        return totals
    
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

        if new_open_fours > 0:
            reward += 6 * new_open_fours
        if new_fours > 0:
            reward += 5* new_fours 
        
        # Fork: single move created simultaneous threats
        total_new_threats = new_open_fours + new_fours + new_open_threes + new_gap_fours
        if total_new_threats >= 2:
            reward += 4

        if new_open_threes > 0:
            reward += 3.5 *new_open_threes

        if new_gap_fours > 0:
            reward += 2.0 * new_gap_fours

        # bonus reward for sequential moves towards a line of 5
        if new_open_fours > 0 and agent_before["open_three"] > 0:
            reward += 1
        
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

        if blocked_open_fours > 0:
            reward += 1.0 * blocked_open_fours   # 1 

        if blocked_fours > 0:
            reward += 3.5 * blocked_fours        # 3.5

        if blocked_open_threes > 0:
            reward += 5 * blocked_open_threes   # 5

        if blocked_gap_fours > 0:
            reward += 2 * blocked_gap_fours      # 2

        return reward















        







