from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.vin_agent import RLAgent
from agents.intermediate_rewards import RewardLogic 
import numpy as np
from agents.random_agent import RandomAgent
from game.match import eval_agents
import matplotlib.pyplot as plt
from agents.human_train import human_train

import os

episodes = []
winrates = []

def train():

    env = GomokuEnv(GomokuLogic(board_size=9))

    agent = RLAgent(player_id=1, board_size=9)
    random_ag = RandomAgent(player_id=-1)

    frozen_refresh_freq = 500
    eval_freq = 200
    num_games = 200
    max_episodes = 10000

    
    board_size=9 

    # load existing trained mod
    LOAD_MODEL = "gomoku_best_reward_model.pth"  # change this to whichever you want

    if os.path.exists(LOAD_MODEL):
        print(f"Loading {LOAD_MODEL}...")
        agent.load(LOAD_MODEL)
    else:
        print("No saved model found. Training from scratch.")

    frozen_opponent = agent.get_frozen_copy()
    frozen_opponent.player_id = -1

    best_winrate = 0
    best_reward = 11
    eval_shaped_rewards = []  # track reward per eval episode

    def scale_reward(shaped, terminal=100.0, max_shaped_fraction=0.3):
        
            cap = terminal * max_shaped_fraction  # 15.0 — 30% of terminal
            if shaped > 0:
                # compress positives: grows fast early, flattens near cap
                scaled = cap * (1 - np.exp(-shaped / cap))
            else:
                scaled = shaped  # leave negatives unscaled
            return scaled

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        agent_last_state = None
        agent_last_action = None
        episode_shaped_total = 0
        use_random = np.random.random() < 0.4
        
        while not done:
            current_player = env.logic.current_player

            if current_player == 1:

                if agent_last_state is not None and not done:
                    rl_prev = RewardLogic(player=1, board=agent_last_state, board_size=board_size)
                    threats_then = rl_prev.threats()
                    had_winning_threat = threats_then["four"] > 0 or threats_then["open_four"] > 0

                    if had_winning_threat:
                        rl_now = RewardLogic(player=1, board=state, board_size=board_size)
                        threats_now = rl_now.threats()
                        threat_was_blocked = (
                            threats_now["four"] < threats_then["four"] or
                            threats_now["open_four"] < threats_then["open_four"]
                        )
                        if threat_was_blocked:
                        # penalise the PREVIOUS action that failed to complete
                            penalty = scale_reward(-6.0, terminal=100.0)  # stays -6 since negatives are unscaled
                            agent.learn_td(agent_last_state, agent_last_action, penalty, state, False,
                            episode=episode, max_episodes=max_episodes)

    
                
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
                    shaped = scale_reward(shaped, terminal=100.0)
                    reward += shaped
                    episode_shaped_total += shaped
                    

                agent.learn_td(state, action, reward, next_state, done,
                    episode=episode, max_episodes=max_episodes)

                
            elif done and reward == 100 and agent_last_state is not None:
                agent.learn_td(agent_last_state, agent_last_action, -100, next_state, True, episode=episode, max_episodes=max_episodes)
            state = next_state

        

        
        if episode % eval_freq == 0 and episode > 0:
            print(f"Episode {episode} | total shaped reward this episode: {episode_shaped_total:.2f}")
        
    

        # refresh frozen agent
        if episode % frozen_refresh_freq == 0 and episode > 0:
            frozen_opponent = agent.get_frozen_copy()
            frozen_opponent.player_id = -1
            print(f"Episode {episode}: Updated frozen opponent.")

        # Evaluate against random agent
        if episode % eval_freq == 0 and episode > 0:

            random_agent = RandomAgent(player_id=-1)
            # agent_copy = agent.get_frozen_copy()
            results = eval_agents(agent, random_agent, num_games=num_games, board_size=9)
            agent.player_id = 1
            winrate = results["agent1_wins"] / num_games
            print(f"Episode: {episode}, winrate vs random: {winrate:.3f}")

            # Track rewards over eval windows
            eval_shaped_rewards.append(episode_shaped_total)
            avg_reward = np.mean(eval_shaped_rewards[-5:])  # rolling average over last 5 evals
            
            if len(eval_shaped_rewards) >= 5:
                recent = eval_shaped_rewards[-5:]
                reward_std = np.std(recent)
                reward_plateaued = reward_std < 2.0  # stable if variance is low
            else:
                reward_plateaued = False

            # save by winrate
            if winrate > best_winrate:
                best_winrate = winrate
                agent.save("gomoku_best_winrate_model.pth")
                print(f"  -> New best winrate model saved ({winrate:.3f})")

            # save by reward
            if avg_reward > best_reward and winrate >= best_winrate - 0.1:
                best_reward = avg_reward
                agent.save("gomoku_best_reward_model.pth")
                print(f"  -> New best reward model saved (avg shaped: {avg_reward:.2f})")
            
            elif winrate < best_winrate - 0.15:  # dropped more than 15%
                print(f"  -> Winrate dropped significantly, reloading best model...")
                agent.load("gomoku_best_winrate_model.pth")
                agent.player_id = 1
            
            if reward_plateaued and winrate >= best_winrate - 0.05:
                agent.save("gomoku_plateaued_model.pth")
                print(f"  -> Plateaued model saved (std: {reward_std:.2f}, avg: {avg_reward:.2f}, winrate: {winrate:.3f})")

            # lists for plotting winrates 

            episodes.append(episode)
            winrates.append(winrate)

            




# replace bottom of train.py with:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--humantrain", action="store_true", help="Train interactively against human")
    parser.add_argument("--model", type=str, default="gomoku_reward_model.pth",
                        help="Model path to load for human training")
    args = parser.parse_args()

    if args.humantrain:
        human_train(board_size=9, model_path=args.model)
    else:
        train()

# plotting

plt.plot(episodes, winrates, linestyle='-', color='red', marker='o')

plt.xlabel("eval episode")
plt.ylabel("winrate against random")
plt.title("PLotting evaluated winrates against corresponding episodes")

plt.show()