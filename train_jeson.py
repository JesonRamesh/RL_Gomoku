import numpy as np
import matplotlib.pyplot as plt
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_jeson import DQNAgent
from agents.random_agent import RandomAgent
from agents.threatening_agent import ThreateningAgent
import time
import os

def train_dqn(num_episodes=10000, batch_size=32, train_frequency=4, save_frequency=500, eval_frequency=100, eval_games=50, board_size=15, save_dir="models"):
    """
    Train a DQN agent to play Gomoku.

    Args:
        num_episodes (int): Total number of training episodes.
        batch_size (int): Batch size for training the DQN.
        train_frequency (int): Train the DQN every N steps.
        save_frequency (int): Save the model every N episodes.
        eval_frequency (int): Evaluate the agent every N episodes.
        eval_games (int): Number of games to play during evaluation.
        board_size (int): Size of the Gomoku board.
        save_dir (str): Directory to save model checkpoints and training logs.
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize game logic and environment
    game_logic = GomokuLogic(board_size=board_size)
    env = GomokuEnv(game_logic)

    # Initialize DQN agent (FIXED parameter name)
    dqn_agent = DQNAgent(
        player_id=1, 
        board_size=board_size, 
        learning_rate=0.00005, 
        gamma=0.90, 
        epsilon_start=1.0, 
        epsilon_end=0.15, 
        epsilon_decay=0.9995, 
        buffer_capacity=50000, 
        target_update_frequency=1000 
    )

    # Initialize random opponent
    # opponent = RandomAgent(player_id=-1)
    opponent = ThreateningAgent(player_id=-1)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    win_rates = []
    epsilon_history = []

    print("=" * 60)
    print("Starting DQN Training for Gomoku")
    print("=" * 60)
    print(f"Device: {dqn_agent.device}")
    print(f"Total Episodes: {num_episodes}")
    print("=" * 60)

    start_time = time.time()
    step_count = 0

    for episode in range(num_episodes):
        # Alternate starting player each episode
        if episode % 2 == 0:
            dqn_agent.player_id = 1
            opponent.player_id = -1
            dqn_first = True
        else:
            dqn_agent.player_id = -1
            opponent.player_id = 1
            dqn_first = False

        # Reset environment and get initial state
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        if not dqn_first:
            # Opponent makes the first move (FIXED method name)
            opponent_action = opponent.predict(state)
            state, _, done, _ = env.step(opponent_action)

        if episode % 2000 == 0 and episode > 0:
            print(f"\n🔄 Clearing replay buffer at episode {episode} to remove stale experiences...")
            dqn_agent.replay_buffer.buffer.clear()
            print(f"   Buffer cleared. Starting fresh experience collection.\n")

        while not done:
            # DQN agent selects action
            action = dqn_agent.predict(state)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            step_count += 1

            # Store experience (REMOVED duplicate .push() call)
            dqn_agent.store_experience(state, action, reward, next_state, done)

            # Train the DQN agent every train_frequency steps (FIXED method name)
            if step_count % train_frequency == 0:
                loss = dqn_agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)

            if done:
                break

            # Opponent's turn (FIXED method name)
            state = next_state
            opponent_action = opponent.predict(state)
            next_state, opponent_reward, done, info = env.step(opponent_action)

            # # Store opponent's move from DQN perspective
            # dqn_agent.store_experience(state, opponent_action, -opponent_reward, next_state, done)

            state = next_state

        # Decay epsilon after each episode
        dqn_agent.decay_epsilon()

        # Track metrics
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
            print(f"{'='*60}\n")

        # Save model checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"dqn_ep{episode + 1}.pt")
            dqn_agent.save_model(checkpoint_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print(f"Total Steps: {step_count}")
    print("=" * 60)

    # Save final model
    final_model_path = os.path.join(save_dir, "dqn_gomoku_final.pt")
    dqn_agent.save_model(final_model_path)

    # Plot training metrics
    plot_training_metrics(episode_rewards, losses, epsilon_history, win_rates, eval_frequency, save_dir)

    return dqn_agent


def evaluate_agent(agent, opponent, board_size, num_games=50):
    """Evaluate the DQN agent against the opponent and return the win rate."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Disable exploration for evaluation

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        # Create fresh environment for each game
        game_logic = GomokuLogic(board_size=board_size)
        env = GomokuEnv(game_logic)

        # Alternate starting player each game
        if game % 2 == 0:
            agent.player_id = 1
            opponent.player_id = -1
            agent_first = True
        else:
            agent.player_id = -1
            opponent.player_id = 1
            agent_first = False

        state = env.reset()
        done = False

        # FIXED: Game loop structure
        while not done:
            if env.logic.current_player == agent.player_id:
                # Agent's turn
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                
                if done:
                    if reward > 0:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                    else:
                        losses += 1
            else:
                # Opponent's turn
                action = opponent.predict(state)
                state, reward, done, _ = env.step(action)
                
                if done:
                    if reward > 0:
                        losses += 1
                    elif reward == 0:
                        draws += 1
                    else:
                        wins += 1

    # Restore original epsilon
    agent.epsilon = original_epsilon

    win_rate = (wins / num_games) * 100
    return win_rate


def plot_training_metrics(episode_rewards, losses, epsilon_history, win_rates, eval_frequency, save_dir):
    """Plot training metrics and save the figures."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot episode rewards   
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Raw')
    if len(episode_rewards) > 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(smoothed, label='Smoothed (100 ep)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot training loss
    axes[0, 1].plot(losses, alpha=0.3)
    if len(losses) > 100:
        smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
        axes[0, 1].plot(smoothed)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True)

    # Plot epsilon decay
    axes[1, 0].plot(epsilon_history)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].grid(True)

    # Plot win rates (FIXED: 0-100 range for percentage)
    eval_episodes = [i * eval_frequency for i in range(1, len(win_rates) + 1)]
    axes[1, 1].plot(eval_episodes, win_rates, marker='o')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Win Rate vs Random Agent')
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0, 100])  # FIXED: was [0, 1]
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"📈 Training curves saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Train the agent
    trained_agent = train_dqn(
        num_episodes=2000,
        batch_size=32,
        train_frequency=4,
        save_frequency=500,
        eval_frequency=100,
        eval_games=20,
        board_size=15
    )