"""
Adaptive Curriculum Training

Key insight: DQN needs to WIN frequently to learn.
This script only increases difficulty when the agent maintains high win rate.

Designed for long training runs (6+ hours, 50k+ episodes).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import deque

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple_jeson import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from agents.minimax_agent import MinimaxAgent


def train_adaptive(
    total_episodes=60000,
    batch_size=32,
    train_frequency=4,
    board_size=9,
    save_dir="models_adaptive",
    load_from=None,
    start_level=0,
):
    """
    Adaptive curriculum that increases difficulty only when agent is winning.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize agent
    agent = DQNAgent(player_id=1, board_size=board_size)
    
    if load_from:
        agent.load_model(load_from)
        print(f"Loaded model from {load_from}")
        agent.epsilon = 0.15
    else:
        agent.epsilon = 1.0
    
    agent.epsilon_end = 0.02
    agent.epsilon_decay = 0.99995  # Very slow decay for long training
    
    # Opponents at different difficulties
    random_opp = RandomAgent(player_id=-1)
    strategic_opp = StrategicAgent(player_id=-1, skill_level=0.1, board_size=board_size)
    minimax_opp = MinimaxAgent(player_id=-1, board_size=board_size, time_limit=0.05, skill_level=0.3)
    
    # Difficulty levels: (opponent_type, skill, name)
    # opponent_type: 0=random, 1=strategic, 2=minimax
    difficulty_levels = [
        (0, 0.0, "Random"),
        (1, 0.1, "Strat-0.1"),  # 90% random
        (1, 0.2, "Strat-0.2"),  # 80% random
        (1, 0.3, "Strat-0.3"),
        (1, 0.4, "Strat-0.4"),
        (1, 0.5, "Strat-0.5"),
        (1, 0.6, "Strat-0.6"),
        (1, 0.7, "Strat-0.7"),
        (1, 0.8, "Strat-0.8"),
        (1, 0.9, "Strat-0.9"),
        (2, 0.3, "MM-0.3"),     # Minimax with 70% random
        (2, 0.4, "MM-0.4"),
        (2, 0.5, "MM-0.5"),
        (2, 0.6, "MM-0.6"),
        (2, 0.7, "MM-0.7"),
    ]
    
    current_level = start_level
    max_level_reached = start_level
    
    # Tracking
    recent_wins = deque(maxlen=100)  # Track last 100 games at current level
    all_rewards = []
    all_losses = []
    level_history = []
    eval_points = []
    eval_random = []
    eval_strategic = []
    eval_minimax = []
    
    # Thresholds - more aggressive advancement
    PROMOTE_THRESHOLD = 0.55  # Need 55% win rate to increase difficulty
    DEMOTE_THRESHOLD = 0.20  # Drop back if win rate falls below 20%
    MIN_GAMES_AT_LEVEL = 150  # Play at least 150 games before considering promotion
    
    games_at_current_level = 0
    step_count = 0
    best_level = 0
    
    print("=" * 70)
    print("ADAPTIVE CURRICULUM TRAINING")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Difficulty levels: {len(difficulty_levels)}")
    print(f"Promote threshold: {PROMOTE_THRESHOLD*100:.0f}% win rate")
    print(f"Device: {agent.device}")
    print("=" * 70)
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        # Get current difficulty
        opp_type, skill, level_name = difficulty_levels[current_level]
        
        # Select opponent
        if opp_type == 0:
            opponent = random_opp
        elif opp_type == 1:
            strategic_opp.skill_level = skill
            opponent = strategic_opp
        else:
            minimax_opp.set_skill_level(skill)
            opponent = minimax_opp
        
        # Mix in some random games to prevent forgetting (20%)
        if current_level > 0 and np.random.random() < 0.2:
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
        
        # Play game
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
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
        
        # Track wins for curriculum games only
        if is_curriculum_game:
            won = episode_reward > 0
            recent_wins.append(1 if won else 0)
            games_at_current_level += 1
        
        level_history.append(current_level)
        
        # Check for promotion/demotion
        if len(recent_wins) >= 50 and games_at_current_level >= MIN_GAMES_AT_LEVEL:
            win_rate = np.mean(recent_wins)
            
            # Promote
            if win_rate >= PROMOTE_THRESHOLD and current_level < len(difficulty_levels) - 1:
                current_level += 1
                games_at_current_level = 0
                recent_wins.clear()
                
                if current_level > max_level_reached:
                    max_level_reached = current_level
                    agent.save_model(os.path.join(save_dir, f"level_{current_level}.pt"))
                
                print(f"\n>>> PROMOTED to Level {current_level}: {difficulty_levels[current_level][2]} "
                      f"(win rate was {win_rate:.1%}) <<<\n")
            
            # Demote (but not below level 1)
            elif win_rate < DEMOTE_THRESHOLD and current_level > 1:
                current_level -= 1
                games_at_current_level = 0
                recent_wins.clear()
                print(f"\n>>> DEMOTED to Level {current_level}: {difficulty_levels[current_level][2]} "
                      f"(win rate was {win_rate:.1%}) <<<\n")
        
        # Log progress
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-500:])
            avg_loss = np.mean(all_losses[-1000:]) if all_losses else 0
            win_rate = np.mean(recent_wins) if recent_wins else 0
            elapsed = (time.time() - start_time) / 60
            
            print(f"Ep {episode+1:>6}/{total_episodes} | "
                  f"Level {current_level}: {level_name:10} | "
                  f"WR: {win_rate:.1%} | "
                  f"Reward: {avg_reward:>5.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.0f}m")
        
        # Full evaluation every 2000 episodes
        if (episode + 1) % 2000 == 0:
            print("\n--- Evaluation ---")
            
            # Quick evaluations
            agent.epsilon = 0.0
            
            wr_rand = evaluate_quick(agent, RandomAgent(-1), board_size, 30)
            wr_strat = evaluate_quick(agent, StrategicAgent(-1, 0.5, board_size), board_size, 20)
            wr_mm = evaluate_quick(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.05, skill_level=0.5), board_size, 15)
            
            agent.epsilon = max(0.02, agent.epsilon)
            
            eval_points.append(episode + 1)
            eval_random.append(wr_rand)
            eval_strategic.append(wr_strat)
            eval_minimax.append(wr_mm)
            
            print(f"vs Random: {wr_rand:.0f}% | vs Strat-0.5: {wr_strat:.0f}% | vs MM-0.5: {wr_mm:.0f}%")
            print(f"Max level reached: {max_level_reached} ({difficulty_levels[max_level_reached][2]})")
            print("------------------\n")
            
            # Save best
            if current_level > best_level:
                best_level = current_level
                agent.save_model(os.path.join(save_dir, "best.pt"))
    
    # Final save
    agent.save_model(os.path.join(save_dir, "final.pt"))
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    agent.epsilon = 0.0
    
    final_rand = evaluate_quick(agent, RandomAgent(-1), board_size, 100)
    final_strat03 = evaluate_quick(agent, StrategicAgent(-1, 0.3, board_size), board_size, 50)
    final_strat05 = evaluate_quick(agent, StrategicAgent(-1, 0.5, board_size), board_size, 50)
    final_strat07 = evaluate_quick(agent, StrategicAgent(-1, 0.7, board_size), board_size, 50)
    final_mm03 = evaluate_quick(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.3), board_size, 30)
    final_mm05 = evaluate_quick(agent, MinimaxAgent(-1, board_size=board_size, time_limit=0.1, skill_level=0.5), board_size, 30)
    
    print(f"\nTraining time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Max level reached: {max_level_reached} ({difficulty_levels[max_level_reached][2]})")
    print(f"\nFinal Results:")
    print(f"  vs Random:       {final_rand:.1f}%")
    print(f"  vs Strat-0.3:    {final_strat03:.1f}%")
    print(f"  vs Strat-0.5:    {final_strat05:.1f}%")
    print(f"  vs Strat-0.7:    {final_strat07:.1f}%")
    print(f"  vs MM-0.3:       {final_mm03:.1f}%")
    print(f"  vs MM-0.5:       {final_mm05:.1f}%")
    print("=" * 70)
    
    # Plot
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(eval_points, eval_random, 'go-', linewidth=2, markersize=4, label='vs Random')
    plt.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate vs Random')
    plt.ylim([0, 105])
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(eval_points, eval_strategic, 'b^-', linewidth=2, markersize=4, label='vs Strat-0.5')
    plt.plot(eval_points, eval_minimax, 'mp-', linewidth=2, markersize=4, label='vs MM-0.5')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate vs Stronger Opponents')
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(level_history, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Difficulty Level')
    plt.title(f'Curriculum Progression\n(Max: {max_level_reached})')
    plt.ylim([0, len(difficulty_levels)])
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    labels = ['Rand', 'St03', 'St05', 'St07', 'MM03', 'MM05']
    values = [final_rand, final_strat03, final_strat05, final_strat07, final_mm03, final_mm05]
    colors = ['green', 'lightblue', 'blue', 'darkblue', 'plum', 'purple']
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


def evaluate_quick(agent, opponent, board_size, num_games):
    """Quick evaluation."""
    wins = 0
    for g in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        
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
    
    return (wins / num_games) * 100


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Curriculum Training")
    parser.add_argument("--episodes", type=int, default=60000,
                        help="Total episodes (default 60000, ~6 hours)")
    parser.add_argument("--load-from", type=str, default=None,
                        help="Path to existing model to continue from")
    parser.add_argument("--save-dir", type=str, default="models_adaptive",
                        help="Directory to save models")
    parser.add_argument("--start-level", type=int, default=0,
                        help="Starting difficulty level (0-14, default 0)")
    args = parser.parse_args()
    
    train_adaptive(
        total_episodes=args.episodes,
        save_dir=args.save_dir,
        load_from=args.load_from,
        start_level=args.start_level,
    )
