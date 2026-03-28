import numpy as np
import matplotlib.pyplot as plt
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple import DQNAgent
from agents.random_agent import RandomAgent
import time
import os

def train_dqn(num_episodes=10000, batch_size=32, train_frequency=4, save_frequency=500, 
              eval_frequency=100, eval_games=50, board_size=9, save_dir="models_baseline"):
    """
    Train DQN agent vs RandomAgent with proper epsilon decay.
    """

    os.makedirs(save_dir, exist_ok=True)

    game_logic = GomokuLogic(board_size=board_size)
    env = GomokuEnv(game_logic, use_sparse_rewards=True)

    dqn_agent = DQNAgent(
        player_id=1, 
        board_size=board_size, 
        learning_rate=0.0001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,  # ← SLOWER decay
        buffer_capacity=50000,
        target_update_frequency=500
    )

    opponent = RandomAgent(player_id=-1)

    episode_rewards = []
    episode_lengths = []
    losses = []
    win_rates = []
    epsilon_history = []

    print("=" * 60)
    print("🎯 BASELINE Training: Residual Architecture vs RandomAgent")
    print("=" * 60)
    print(f"Device: {dqn_agent.device}")
    print(f"Rewards: SPARSE (Win: +1, Loss: -1, Ongoing: 0)")
    print(f"Opponent: RandomAgent ONLY")
    print(f"Goal: Stable 98-100% win rate")
    print(f"Total Episodes: {num_episodes}")
    print(f"Epsilon decay: 0.9995 (reaches 0.1 at ~4600 episodes)")
    print("=" * 60)

    start_time = time.time()
    step_count = 0
    best_win_rate = 0
    stable_98_achieved = False

    for episode in range(num_episodes):
        # Alternate starting player
        if episode % 2 == 0:
            dqn_agent.player_id = 1
            opponent.player_id = -1
            dqn_first = True
        else:
            dqn_agent.player_id = -1
            opponent.player_id = 1
            dqn_first = False

        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        if not dqn_first:
            opponent_action = opponent.predict(state)
            state, _, done, _ = env.step(opponent_action)

        while not done:
            action = dqn_agent.predict(state)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            step_count += 1

            dqn_agent.store_experience(state, action, reward, next_state, done)

            if step_count % train_frequency == 0:
                loss = dqn_agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

            if done:
                break

            state = next_state
            opponent_action = opponent.predict(state)
            next_state, _, done, _ = env.step(opponent_action)
            state = next_state

        dqn_agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(dqn_agent.epsilon)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Length: {avg_length:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {dqn_agent.epsilon:.3f} | "
                  f"Buffer: {len(dqn_agent.replay_buffer)}")

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            win_rate = evaluate_agent(dqn_agent, opponent, board_size, eval_games)
            win_rates.append(win_rate)
            
            print(f"\n{'='*60}")
            print(f"📊 Evaluation at Episode {episode + 1}")
            print(f"Win Rate vs Random: {win_rate:.1f}%")
            print(f"Best So Far: {max(best_win_rate, win_rate):.1f}%")
            
            # Check stability over last 3 evaluations
            if len(win_rates) >= 3:
                recent_avg = np.mean(win_rates[-3:])
                print(f"Last 3 Evals Avg: {recent_avg:.1f}%")
            
            print(f"{'='*60}\n")
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model_path = os.path.join(save_dir, "dqn_baseline_best.pt")
                dqn_agent.save_model(best_model_path)
                print(f"💾 New best model saved: {win_rate:.1f}%\n")
            
            # Check for STABLE 98%+ (5 consecutive evaluations)
            if len(win_rates) >= 5 and not stable_98_achieved:
                last_5_avg = np.mean(win_rates[-5:])
                last_5_min = min(win_rates[-5:])
                if last_5_avg >= 98.0 and last_5_min >= 95.0:
                    stable_98_achieved = True
                    print(f"\n🏆 STABLE SUCCESS ACHIEVED!")
                    print(f"   Last 5 evals average: {last_5_avg:.1f}%")
                    print(f"   Minimum in last 5: {last_5_min:.1f}%")
                    print(f"   Episode: {episode + 1}")
                    print(f"   Epsilon: {dqn_agent.epsilon:.3f}")
                    print(f"   Continuing training for robustness...\n")

        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"dqn_baseline_ep{episode + 1}.pt")
            dqn_agent.save_model(checkpoint_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Best Win Rate: {best_win_rate:.1f}%")
    print(f"Final Win Rate: {win_rates[-1]:.1f}%")
    
    if len(win_rates) >= 5:
        final_avg = np.mean(win_rates[-5:])
        print(f"Last 5 Evals Avg: {final_avg:.1f}%")
    
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print(f"Total Steps: {step_count}")
    print(f"Stable 98%+ Achieved: {'Yes' if stable_98_achieved else 'No'}")
    print("=" * 60)

    final_model_path = os.path.join(save_dir, "dqn_baseline_final.pt")
    dqn_agent.save_model(final_model_path)

    plot_training_metrics(episode_rewards, losses, epsilon_history, win_rates, 
                         eval_frequency, save_dir)

    return dqn_agent

# Add this function after the train_dqn() function (around line 140)

def train_dqn_continued(agent, start_episode, num_episodes, batch_size=32, 
                        train_frequency=4, save_frequency=500, eval_frequency=100, 
                        eval_games=50, board_size=9, save_dir="models_baseline_9x9"):
    """Continue training from checkpoint."""
    
    game_logic = GomokuLogic(board_size=board_size)
    env = GomokuEnv(game_logic, use_sparse_rewards=True)
    opponent = RandomAgent(player_id=-1)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    win_rates = []
    epsilon_history = []
    
    print("=" * 60)
    print("🔄 CONTINUING BASELINE TRAINING")
    print("=" * 60)
    print(f"Starting from episode {start_episode}")
    print(f"Training {num_episodes} more episodes")
    print(f"Target epsilon: {agent.epsilon_end:.3f}")
    print("=" * 60)
    
    start_time = time.time()
    step_count = 0
    best_win_rate = 92.0  # Your previous best
    
    for episode in range(num_episodes):
        episode_num = start_episode + episode
        
        # Alternate starting player
        if episode % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
            dqn_first = True
        else:
            agent.player_id = -1
            opponent.player_id = 1
            dqn_first = False
        
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        if not dqn_first:
            opponent_action = opponent.predict(state)
            state, _, done, _ = env.step(opponent_action)
        
        while not done:
            action = agent.predict(state)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if step_count % train_frequency == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)
            
            if done:
                break
            
            state = next_state
            opponent_action = opponent.predict(state)
            next_state, _, done, _ = env.step(opponent_action)
            state = next_state
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(agent.epsilon)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode_num + 1}/50000 | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Length: {avg_length:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            win_rate = evaluate_agent(agent, opponent, board_size, eval_games)
            win_rates.append(win_rate)
            
            print(f"\n{'='*60}")
            print(f"📊 Evaluation at Episode {episode_num + 1}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Best: {max(best_win_rate, win_rate):.1f}%")
            
            if len(win_rates) >= 3:
                recent_avg = np.mean(win_rates[-3:])
                print(f"Last 3 Avg: {recent_avg:.1f}%")
            
            print(f"{'='*60}\n")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = os.path.join(save_dir, "dqn_baseline_best.pt")
                agent.save_model(best_path)
                print(f"💾 New best: {win_rate:.1f}%\n")
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"dqn_baseline_ep{episode_num + 1}.pt")
            agent.save_model(checkpoint_path)
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Extension Training Complete!")
    print(f"Best Win Rate: {best_win_rate:.1f}%")
    print(f"Final Win Rate: {win_rates[-1]:.1f}%")
    
    if len(win_rates) >= 5:
        final_avg = np.mean(win_rates[-5:])
        print(f"Last 5 Avg: {final_avg:.1f}%")
    
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print("=" * 60)
    
    final_path = os.path.join(save_dir, "dqn_baseline_final_50k.pt")
    agent.save_model(final_path)
    
    return agent


def evaluate_agent(agent, opponent, board_size, num_games=50):
    """Evaluate the DQN agent against the opponent."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins = 0

    for game in range(num_games):
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic)

        if game % 2 == 0:
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


def plot_training_metrics(episode_rewards, losses, epsilon_history, win_rates, 
                          eval_frequency, save_dir):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Raw')
    if len(episode_rewards) > 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(smoothed, label='Smoothed (100ep)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards (Sparse)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training loss
    axes[0, 1].plot(losses, alpha=0.3)
    if len(losses) > 100:
        smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
        axes[0, 1].plot(smoothed)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True)

    # Epsilon decay
    axes[1, 0].plot(epsilon_history)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate')
    axes[1, 0].grid(True)

    # Win rates
    eval_episodes = [i * eval_frequency for i in range(1, len(win_rates) + 1)]
    axes[1, 1].plot(eval_episodes, win_rates, marker='o', color='green', linewidth=2)
    axes[1, 1].axhline(y=98, color='red', linestyle='--', label='Target (98%)', linewidth=2)
    axes[1, 1].axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Win Rate vs RandomAgent')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'baseline_training_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"📈 Training curves saved to {plot_path}")
    plt.close()

# 15x15 CONFIG (for reference, not used in this script)
# if __name__ == "__main__":
#     trained_agent = train_dqn(
#         num_episodes=6000,      # Reduced from 10k (epsilon floor at ~4600)
#         batch_size=32,
#         train_frequency=4,
#         save_frequency=500,
#         eval_frequency=100,
#         eval_games=50,
#         board_size=9,
#         save_dir="models_baseline"
#     )

# if __name__ == "__main__":
#     print("\n" + "="*70)
#     print("9x9 GOMOKU BASELINE TRAINING")
#     print("="*70)
#     print("Board Size: 9x9")
#     print("Opponent: RandomAgent")
#     print("Target: 98%+ win rate")
#     print("="*70 + "\n")
    
#     input("Press Enter to start training...")
    
#     trained_agent = train_dqn(
#         num_episodes=6000,
#         batch_size=32,
#         train_frequency=4,
#         save_frequency=500,
#         eval_frequency=100,
#         eval_games=50,
#         board_size=9,  # ← 9×9 board
#         save_dir="models_baseline_9x9"  # ← Separate folder
#     )
    
#     print("\n" + "="*70)
#     print("BASELINE TRAINING COMPLETE")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 9x9 GOMOKU BASELINE TRAINING - EXTENSION")
    print("="*70)
    print("Board Size: 9x9")
    print("Opponent: RandomAgent")
    print("Target: 98%+ win rate (stable)")
    print("Episodes: 10000 (continuing from 30000)")
    print("="*70 + "\n")
    
    input("Press Enter to continue training...")
    
    # Load existing model and continue
    from agents.dqn_simple import DQNAgent
    
    agent = DQNAgent(player_id=1, board_size=9)
    agent.load_model("models_baseline_9x9/dqn_baseline_final_30k.pt")
    
    # LOWER epsilon for less exploration
    agent.epsilon = 0.03  # Start from lower epsilon
    agent.epsilon_end = 0.01  # Even lower floor
    agent.epsilon_decay = 0.9995
    
    print(f"📥 Loaded checkpoint from episode 30000")
    print(f"   Current ε: {agent.epsilon:.3f}")
    print(f"   Target ε: {agent.epsilon_end:.3f}")
    print(f"   Replay buffer: {len(agent.replay_buffer)} experiences\n")
    
    trained_agent = train_dqn_continued(
        agent=agent,  # Pass loaded agent
        start_episode=8000,  # Continue from 30000
        num_episodes=42000,  # Train 10000 more (total 50000)
        batch_size=32,
        train_frequency=4,
        save_frequency=500,
        eval_frequency=100,
        eval_games=50,
        board_size=9,
        save_dir="models_baseline_9x9"
    )