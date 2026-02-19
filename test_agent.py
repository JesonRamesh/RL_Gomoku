import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from agents.dqn_jeson import DQNAgent
from agents.random_agent import RandomAgent
from game.match import eval_agents

# Load your trained agent
agent = DQNAgent(player_id=1, board_size=15)
agent.load_model("models_sparse/dqn_sparse_final.pt")
agent.epsilon = 0.0  # Pure exploitation

# Test against random
random_opponent = RandomAgent(player_id=-1)

results = eval_agents(agent, random_opponent, num_games=100, board_size=15)
# Should show ~95-100% win rate