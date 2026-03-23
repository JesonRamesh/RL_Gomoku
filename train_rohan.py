"""
Training script for DQN Rohan Agent with shaped rewards.

Key improvements:
1. Shaped rewards that teach blocking and threat creation
2. Enhanced DQN with dueling architecture
3. Prioritized experience replay
4. Adaptive curriculum
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from collections import deque

from game.logic import GomokuLogic
from game.gomoku_env_shaped import GomokuEnvShaped
from game.gomoku_env import GomokuEnv
from agents.dqn_rohan import DQNAgentRohan
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from agents.minimax_agent import MinimaxAgent


def _save_policy_weights(agent, path):
    """Save only inference-relevant policy weights."""
    torch.save(agent.q_network.state_dict(), path)


def _load_policy_weights(agent, path):
    """Load either legacy full checkpoints or weights-only state dicts."""
    payload = torch.load(path, map_location=agent.device, weights_only=False)

    if isinstance(payload, dict) and "q_network_state_dict" in payload:
        state_dict = payload["q_network_state_dict"]
    else:
        state_dict = payload

    agent.q_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(agent.q_network.state_dict())


def evaluate_agent(agent, opponent, board_size, num_games=30):
    """Evaluate agent with sparse rewards (real game performance)."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    wins = 0
    for g in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)  # Sparse for eval
        
        if g % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
        else:
            agent.player_id = -1
            opponent.player_id = 1
        
        state = env.reset()
        done = False
        
        while not done:
            if env.logic.current_player == agent.player_id:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    wins += 1
            else:
                action = opponent.predict(state)
                state, _, done, _ = env.step(action)
    
    agent.epsilon = original_epsilon
    return (wins / num_games) * 100


def train_rohan(
    total_episodes=80000,
    batch_size=64,
    train_frequency=4,
    board_size=9,
    save_dir="models_rohan",
    load_from=None,
):
    """
    Train DQN Rohan agent with shaped rewards and adaptive curriculum.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize agent
    agent = DQNAgentRohan(player_id=1, board_size=board_size)
    
    if load_from:
        _load_policy_weights(agent, load_from)
        print(f"Loaded policy weights from {load_from}")
        agent.epsilon = 0.15
    else:
        agent.epsilon = 1.0
    
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.99995
    
    # Opponents
    random_opp = RandomAgent(player_id=-1)
    strategic_opp = StrategicAgent(player_id=-1, skill_level=0.5, board_size=board_size)
    minimax_opp = MinimaxAgent(player_id=-1, board_size=board_size, time_limit=0.05, skill_level=0.5)
    
    # Difficulty progression
    difficulty_levels = [
        ("Random", random_opp, 0),
        ("Strat-0.2", lambda: setattr(strategic_opp, 'skill_level', 0.2) or strategic_opp, 1),
        ("Strat-0.4", lambda: setattr(strategic_opp, 'skill_level', 0.4) or strategic_opp, 2),
        ("Strat-0.5", lambda: setattr(strategic_opp, 'skill_level', 0.5) or strategic_opp, 3),
        ("Strat-0.6", lambda: setattr(strategic_opp, 'skill_level', 0.6) or strategic_opp, 4),
        ("Strat-0.7", lambda: setattr(strategic_opp, 'skill_level', 0.7) or strategic_opp, 5),
        ("Strat-0.8", lambda: setattr(strategic_opp, 'skill_level', 0.8) or strategic_opp, 6),
        ("MM-0.3", lambda: minimax_opp.set_skill_level(0.3) or minimax_opp, 7),
        ("MM-0.5", lambda: minimax_opp.set_skill_level(0.5) or minimax_opp, 8),
        ("MM-0.6", lambda: minimax_opp.set_skill_level(0.6) or minimax_opp, 9),
        ("MM-0.7", lambda: minimax_opp.set_skill_level(0.7) or minimax_opp, 10),
    ]
    
    current_level = 0
    max_level = 0
    recent_wins = deque(maxlen=100)
    games_at_level = 0
    
    PROMOTE_THRESHOLD = 0.60
    DEMOTE_THRESHOLD = 0.25
    MIN_GAMES = 150
    
    # Tracking
    all_rewards = []
    all_losses = []
    level_history = []
    eval_points = []
    eval_random = []
    eval_strategic = []
    eval_minimax = []
    
    step_count = 0
    
    print("=" * 70)
    print("ROHAN DQN TRAINING - Shaped Rewards + Adaptive Curriculum")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Using shaped rewards for blocking and threat creation")
    print(f"Device: {agent.device}")
    print("=" * 70)
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        # Get current opponent
        level_name, opp_getter, level_idx = difficulty_levels[current_level]
        if callable(opp_getter):
            opponent = opp_getter()
        else:
            opponent = opp_getter
        
        # Mix in easier opponents to prevent catastrophic forgetting
        mix_random = np.random.random() < max(0.15, 0.3 - current_level * 0.02)
        if mix_random and current_level > 0:
            opponent = random_opp
            is_curriculum_game = False
        else:
            is_curriculum_game = True
        
        # Alternate first player
        if episode % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
            agent_first = True
        else:
            agent.player_id = -1
            opponent.player_id = 1
            agent_first = False
        
        # Use SHAPED rewards environment for training
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnvShaped(game_logic)
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        if not agent_first:
            opp_action = opponent.predict(state)
            state, _, done, _ = env.step(opp_action)
        
        while not done:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    all_losses.append(loss)
            
            if done:
                break
            
            state = next_state
            opp_action = opponent.predict(state)
            next_state, _, done, _ = env.step(opp_action)
            state = next_state
        
        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        
        # Track wins for curriculum
        if is_curriculum_game:
            won = episode_reward > 0.5  # Won the game
            recent_wins.append(1 if won else 0)
            games_at_level += 1
        
        level_history.append(current_level)
        
        # Check promotion/demotion
        if len(recent_wins) >= 50 and games_at_level >= MIN_GAMES:
            win_rate = np.mean(recent_wins)
            
            if win_rate >= PROMOTE_THRESHOLD and current_level < len(difficulty_levels) - 1:
                current_level += 1
                games_at_level = 0
                recent_wins.clear()
                
                if current_level > max_level:
                    max_level = current_level
                    _save_policy_weights(agent, os.path.join(save_dir, f"level_{current_level}.pt"))
                
                print(f"\n>>> PROMOTED to Level {current_level}: {difficulty_levels[current_level][0]} "
                      f"(win rate: {win_rate:.1%}) <<<\n")
            
            elif win_rate < DEMOTE_THRESHOLD and current_level > 0:
                current_level -= 1
                games_at_level = 0
                recent_wins.clear()
                print(f"\n>>> DEMOTED to Level {current_level}: {difficulty_levels[current_level][0]} "
                      f"(win rate: {win_rate:.1%}) <<<\n")
        
        # Logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-500:])
            avg_loss = np.mean(all_losses[-1000:]) if all_losses else 0
            win_rate = np.mean(recent_wins) if recent_wins else 0
            elapsed = (time.time() - start_time) / 60
            
            print(f"Ep {episode+1:>6}/{total_episodes} | "
                  f"Lv {current_level}: {level_name:10} | "
                  f"WR: {win_rate:.1%} | "
                  f"Rew: {avg_reward:>5.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"{elapsed:.0f}m")
        
        # Full evaluation
        if (episode + 1) % 2000 == 0:
            print("\n--- Evaluation (using sparse rewards) ---")
            
            wr_rand = evaluate_agent(agent, RandomAgent(-1), board_size, 30)
            wr_strat = evaluate_agent(agent, StrategicAgent(-1, 0.5, board_size), board_size, 25)
            wr_mm = evaluate_agent(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.05, skill_level=0.5), board_size, 20)
            
            eval_points.append(episode + 1)
            eval_random.append(wr_rand)
            eval_strategic.append(wr_strat)
            eval_minimax.append(wr_mm)
            
            print(f"vs Random: {wr_rand:.0f}% | vs Strat-0.5: {wr_strat:.0f}% | vs MM-0.5: {wr_mm:.0f}%")
            print(f"Max level: {max_level} ({difficulty_levels[max_level][0]})")
            print("-" * 45 + "\n")
            
            _save_policy_weights(agent, os.path.join(save_dir, "checkpoint.pt"))
    
    # Final save
    _save_policy_weights(agent, os.path.join(save_dir, "final.pt"))
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    agent.epsilon = 0.0
    
    final_rand = evaluate_agent(agent, RandomAgent(-1), board_size, 100)
    final_strat03 = evaluate_agent(agent, StrategicAgent(-1, 0.3, board_size), board_size, 50)
    final_strat05 = evaluate_agent(agent, StrategicAgent(-1, 0.5, board_size), board_size, 50)
    final_strat07 = evaluate_agent(agent, StrategicAgent(-1, 0.7, board_size), board_size, 50)
    final_mm03 = evaluate_agent(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.3), board_size, 30)
    final_mm05 = evaluate_agent(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.5), board_size, 30)
    final_mm07 = evaluate_agent(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.7), board_size, 30)
    
    print(f"\nTraining time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Max level reached: {max_level} ({difficulty_levels[max_level][0]})")
    print(f"\nFinal Results:")
    print(f"  vs Random:       {final_rand:.1f}%")
    print(f"  vs Strat-0.3:    {final_strat03:.1f}%")
    print(f"  vs Strat-0.5:    {final_strat05:.1f}%")
    print(f"  vs Strat-0.7:    {final_strat07:.1f}%")
    print(f"  vs MM-0.3:       {final_mm03:.1f}%")
    print(f"  vs MM-0.5:       {final_mm05:.1f}%")
    print(f"  vs MM-0.7:       {final_mm07:.1f}%")
    print("=" * 70)
    
    # Plot
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(eval_points, eval_random, 'go-', linewidth=2, markersize=4)
    plt.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('vs Random')
    plt.ylim([0, 105])
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(eval_points, eval_strategic, 'b^-', linewidth=2, markersize=4, label='vs Strat-0.5')
    plt.plot(eval_points, eval_minimax, 'mp-', linewidth=2, markersize=4, label='vs MM-0.5')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('vs Stronger Opponents')
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(level_history, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Difficulty Level')
    plt.title(f'Curriculum (Max: {max_level})')
    plt.ylim([0, len(difficulty_levels)])
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    labels = ['Rand', 'St03', 'St05', 'St07', 'MM03', 'MM05', 'MM07']
    values = [final_rand, final_strat03, final_strat05, final_strat07, final_mm03, final_mm05, final_mm07]
    colors = ['green', 'lightblue', 'blue', 'darkblue', 'plum', 'purple', 'darkviolet']
    plt.bar(labels, values, color=colors, alpha=0.7)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.ylabel('Win Rate (%)')
    plt.title('Final Results')
    plt.ylim([0, 105])
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"\nPlot saved to {save_dir}/training_curves.png")
    plt.close()
    
    return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Rohan DQN Agent")
    parser.add_argument("--episodes", type=int, default=80000,
                        help="Total episodes (default 80000)")
    parser.add_argument("--load-from", type=str, default=None,
                        help="Path to existing model to continue from")
    parser.add_argument("--save-dir", type=str, default="models_rohan",
                        help="Directory to save models")
    args = parser.parse_args()
    
    train_rohan(
        total_episodes=args.episodes,
        save_dir=args.save_dir,
        load_from=args.load_from,
    )
