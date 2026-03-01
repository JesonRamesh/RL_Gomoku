from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.vin_agent import RLAgent
from agents.intermediate_rewards import RewardLogic 
import numpy as np
from agents.random_agent import RandomAgent
from game.match import eval_agents
import matplotlib.pyplot as plt

import os

env = GomokuEnv(GomokuLogic(board_size=9))

agent = RLAgent(player_id=1, board_size=9)
random_ag = RandomAgent(player_id=-1)

frozen_refresh_freq = 1000
eval_freq = 200
num_games = 200
max_episodes = 10000

episodes = []
winrates = []
board_size=9 

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
    episode_shaped_total = 0
    use_random = np.random.random() < 0.2
    
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
            if use_random:   # 20% of episodes use random opponent
                action = random_ag.predict(state)
            else:
                
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
            if not done:
                rl = RewardLogic(player=1, board=state, board_size=board_size)
                shaped = rl.rewards(board_before=state, board_after=next_state)
                reward += shaped
                episode_shaped_total += shaped
                

            agent.learn_td(state, action, reward, next_state, done,
                episode=episode, max_episodes=max_episodes)

            
        elif done and reward == 30 and agent_last_state is not None:
            agent.learn_td(agent_last_state, agent_last_action, -30, next_state, True, episode=episode, max_episodes=max_episodes)
        state = next_state

    
    if episode % eval_freq == 0:
        print(f"Episode {episode} | total shaped reward this episode: {episode_shaped_total:.2f}")
    
   

    # refresh frozen agent
    if episode % frozen_refresh_freq == 0 and episode > 0:
        frozen_opponent = agent.get_frozen_copy()
        frozen_opponent.player_id = -1
        print(f"Episode {episode}: Updated frozen opponent.")

    # Evaluate against random agent
    if episode % eval_freq == 0:

        random_agent = RandomAgent(player_id=-1)
        agent_copy = agent.get_frozen_copy()
        results = eval_agents(agent, random_agent, num_games=num_games, board_size=9)
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
