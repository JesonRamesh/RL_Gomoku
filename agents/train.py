from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.vin_agent import RLAgent
import numpy as np
from agents.random_agent import RandomAgent
from game.match import eval_agents
import matplotlib.pyplot as plt

import os

env = GomokuEnv(GomokuLogic(board_size=7))

agent = RLAgent(player_id=1, board_size=7)

frozen_refresh_freq = 600
eval_freq = 200
num_games = 200
max_episodes = 10000

episodes = []
winrates = []

# load existing trained model

if os.path.exists("gomoku_best_model.pth"):
    print("Loading existing model...")
    agent.load("gomoku_best_model.pth")
else:
    print("No saved model found. Training from scratch.")

frozen_opponent = agent.get_frozen_copy()
frozen_opponent.player_id = -1

best_winrate = 0


for episode in range(max_episodes):
    state = env.reset()
    done = False
    agent_last_state = None
    agent_last_action = None
    
    trajectory = []  # store (state, action, reward, current_player)

    while not done:
        current_player = env.logic.current_player

        if current_player == 1:
            # current player's turn
            agent.player_id = 1
            action = agent.predict(state)
            agent_last_state = state
            agent_last_action = action
        else:
            # frozen agent's turn
            frozen_opponent.player_id = -1
            action = frozen_opponent.predict(state)

        row, col = action
        # stop illegal moves
        if state[row, col] != 0:
            legal = list(zip(*np.where(state == 0)))
            if not legal:
                break
            action = legal[np.random.randint(len(legal))]

        next_state, reward, done, info = env.step(action)

        if current_player == 1:
            trajectory.append((state, action, reward))
        elif done and reward == 10 and agent_last_state is not None:
            trajectory.append((agent_last_state, agent_last_action, -10))

        state = next_state

    # compute discounted returns and update once per step
    agent.learn(trajectory, episode=episode, max_episodes=max_episodes)
   

    # refresh frozen agent
    if episode % frozen_refresh_freq == 0 and episode > 0:
        frozen_opponent = agent.get_frozen_copy()
        frozen_opponent.player_id = -1
        print(f"Episode {episode}: Updated frozen opponent.")

    # Evaluate against random agent
    if episode % eval_freq == 0:

        print(f"Avg trajectory length: {len(trajectory)}, rewards: {[r for _,_,r in trajectory]}")
        random_agent = RandomAgent(player_id=-1)
        results = eval_agents(agent, random_agent, num_games=num_games, board_size=7)
        winrate = results["agent1_wins"] / num_games
        print(f"Episode: {episode}, winrate vs random: {winrate:.3f}")

        if winrate > best_winrate:
            best_winrate = winrate
            agent.save("gomoku_best_model.pth")
            print(f"  -> New best model saved ({winrate:.3f})")

        # lists for plotting winrates 

        episodes.append(episode)
        winrates.append(winrate)

# plotting

plt.plot(episodes, winrates, linestyle='-', color='red', marker='o')

plt.xlabel("eval episode")
plt.ylabel("winrate against random")
plt.title("PLotting evaluated winrates against corresponding episodes")

plt.show()
