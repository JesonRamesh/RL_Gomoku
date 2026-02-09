import random


class rand_agent:
    def __init__(self, player_id):
        self.player_id = player_id

    def play_move(self, board):


        # make list of possible moves
        empty_spaces = []
        

        for r in range(len(board)):
            for c in range(len(board)):
                if board[r][c] == 0:
                    empty_spaces.append((r,c))
            
        if not empty_spaces:
            return None
        
        return random.choice(empty_spaces)
