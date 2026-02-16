class BaseAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.is_human = False

    def predict(self, board_state):
        raise NotImplementedError("This agent has not been implemented yet")
    
class HumanAgent(BaseAgent):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.is_human = True

    def predict(self, board_state):
        return None