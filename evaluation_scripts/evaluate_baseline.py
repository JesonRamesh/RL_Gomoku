"""
Baseline Model Evaluation Script
---------------------------------
Evaluates the trained DQN agent against multiple opponents:
1. RandomAgent (should win 95%+)
2. ThreateningAgent at various skill levels (0.1, 0.2, 0.3, 0.5)

Results saved to CSV with detailed statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tqdm import tqdm

from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple import DQNAgent
from agents.random_agent import RandomAgent
from agents.threatening_agent import ThreateningAgent


class GameEvaluator:
    """Evaluates agent performance against different opponents."""
    
    def __init__(self, agent, board_size=15):
        self.agent = agent
        self.board_size = board_size
        self.agent.epsilon = 0.0  # Disable exploration for evaluation
        
    def play_game(self, opponent, agent_plays_first=True):
        """
        Play a single game and return detailed results.
        
        Returns:
            dict: Game statistics including winner, length, final board state
        """
        game_logic = GomokuLogic(board_size=self.board_size)
        env = GomokuEnv(game_logic)
        
        # Set player IDs
        if agent_plays_first:
            self.agent.player_id = 1
            opponent.player_id = -1
        else:
            self.agent.player_id = -1
            opponent.player_id = 1
        
        state = env.reset()
        move_count = 0
        move_history = []
        done = False
        
        while not done:
            current_player = env.logic.current_player
            
            if current_player == self.agent.player_id:
                # Agent's turn
                action = self.agent.predict(state)
                player_type = "agent"
            else:
                # Opponent's turn
                action = opponent.predict(state)
                player_type = "opponent"
            
            if action is None:
                # No valid moves (shouldn't happen in Gomoku)
                break
            
            move_history.append({
                'move_num': move_count,
                'player': current_player,
                'player_type': player_type,
                'action': action
            })
            
            state, reward, done, info = env.step(action)
            move_count += 1
            
            # Safety check for infinite games
            if move_count > self.board_size * self.board_size:
                break
        
        # Determine result
        winner = env.logic.winner
        if winner == self.agent.player_id:
            result = "win"
        elif winner == 0:
            result = "draw"
        else:
            result = "loss"
        
        return {
            'result': result,
            'move_count': move_count,
            'winner': winner,
            'agent_player_id': self.agent.player_id,
            'agent_first': agent_plays_first,
            'move_history': move_history
        }
    
    def evaluate_opponent(self, opponent_name, opponent, num_games=100, verbose=True):
        """
        Evaluate agent against a specific opponent.
        
        Args:
            opponent_name (str): Name of opponent for reporting
            opponent: Opponent agent instance
            num_games (int): Number of games to play
            verbose (bool): Print progress
        
        Returns:
            dict: Detailed statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating vs {opponent_name}")
            print(f"{'='*60}")
        
        results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'move_counts': [],
            'wins_as_first': 0,
            'wins_as_second': 0,
            'games_as_first': 0,
            'games_as_second': 0
        }
        
        # Play games with progress bar
        iterator = tqdm(range(num_games), desc=f"vs {opponent_name}") if verbose else range(num_games)
        
        for game_num in iterator:
            # Alternate who plays first
            agent_first = (game_num % 2 == 0)
            
            game_result = self.play_game(opponent, agent_plays_first=agent_first)
            
            # Update statistics
            results['move_counts'].append(game_result['move_count'])
            
            if agent_first:
                results['games_as_first'] += 1
            else:
                results['games_as_second'] += 1
            
            if game_result['result'] == 'win':
                results['wins'] += 1
                if agent_first:
                    results['wins_as_first'] += 1
                else:
                    results['wins_as_second'] += 1
            elif game_result['result'] == 'loss':
                results['losses'] += 1
            else:
                results['draws'] += 1
        
        # Calculate percentages
        results['win_rate'] = (results['wins'] / num_games) * 100
        results['loss_rate'] = (results['losses'] / num_games) * 100
        results['draw_rate'] = (results['draws'] / num_games) * 100
        
        if results['games_as_first'] > 0:
            results['win_rate_as_first'] = (results['wins_as_first'] / results['games_as_first']) * 100
        else:
            results['win_rate_as_first'] = 0
            
        if results['games_as_second'] > 0:
            results['win_rate_as_second'] = (results['wins_as_second'] / results['games_as_second']) * 100
        else:
            results['win_rate_as_second'] = 0
        
        results['avg_move_count'] = np.mean(results['move_counts'])
        results['std_move_count'] = np.std(results['move_counts'])
        results['min_move_count'] = np.min(results['move_counts'])
        results['max_move_count'] = np.max(results['move_counts'])
        
        if verbose:
            print(f"\n📊 Results vs {opponent_name}:")
            print(f"  Win Rate:    {results['win_rate']:.1f}% ({results['wins']}/{num_games})")
            print(f"  Loss Rate:   {results['loss_rate']:.1f}% ({results['losses']}/{num_games})")
            print(f"  Draw Rate:   {results['draw_rate']:.1f}% ({results['draws']}/{num_games})")
            print(f"  As First:    {results['win_rate_as_first']:.1f}% ({results['wins_as_first']}/{results['games_as_first']})")
            print(f"  As Second:   {results['win_rate_as_second']:.1f}% ({results['wins_as_second']}/{results['games_as_second']})")
            print(f"  Avg Moves:   {results['avg_move_count']:.1f} ± {results['std_move_count']:.1f}")
            print(f"  Move Range:  [{results['min_move_count']}, {results['max_move_count']}]")
        
        return results


def run_full_evaluation(model_path, num_games_per_opponent=100, save_dir="evaluation_results"):
    """
    Run complete evaluation suite against all opponents.
    
    Args:
        model_path (str): Path to trained model checkpoint
        num_games_per_opponent (int): Games to play against each opponent
        save_dir (str): Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("🎯 BASELINE MODEL EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Games per opponent: {num_games_per_opponent}")
    print(f"Timestamp: {timestamp}")
    print("="*60)
    
    # Load agent
    print("\n📦 Loading trained agent...")
    agent = DQNAgent(
        player_id=1,
        board_size=15,
        learning_rate=0.0001,
        gamma=0.95,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_capacity=50000,
        target_update_frequency=500
    )
    agent.load_model(model_path)
    print(f"✅ Model loaded successfully (epsilon: {agent.epsilon})")
    
    # Create evaluator
    evaluator = GameEvaluator(agent, board_size=15)
    
    # Define opponents
    opponents = [
        ("RandomAgent", RandomAgent(player_id=-1)),
        ("Threatening-0.1", ThreateningAgent(player_id=-1, skill_level=0.1)),
        ("Threatening-0.2", ThreateningAgent(player_id=-1, skill_level=0.2)),
        ("Threatening-0.3", ThreateningAgent(player_id=-1, skill_level=0.3)),
        ("Threatening-0.5", ThreateningAgent(player_id=-1, skill_level=0.5)),
    ]
    
    # Evaluate against each opponent
    all_results = []
    
    for opponent_name, opponent in opponents:
        results = evaluator.evaluate_opponent(
            opponent_name, 
            opponent, 
            num_games=num_games_per_opponent,
            verbose=True
        )
        
        # Add metadata
        results['opponent_name'] = opponent_name
        results['timestamp'] = timestamp
        results['model_path'] = model_path
        results['num_games'] = num_games_per_opponent
        
        all_results.append(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        'opponent_name', 'win_rate', 'loss_rate', 'draw_rate',
        'wins', 'losses', 'draws', 'num_games',
        'win_rate_as_first', 'win_rate_as_second',
        'wins_as_first', 'wins_as_second',
        'games_as_first', 'games_as_second',
        'avg_move_count', 'std_move_count', 'min_move_count', 'max_move_count',
        'timestamp', 'model_path'
    ]
    df = df[column_order]
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f"evaluation_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Create summary report
    create_summary_report(df, save_dir, timestamp)
    
    # Create visualizations
    create_visualizations(df, save_dir, timestamp)
    
    return df


def create_summary_report(df, save_dir, timestamp):
    """Create a text summary report."""
    report_path = os.path.join(save_dir, f"summary_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BASELINE MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {df['model_path'].iloc[0]}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"\n{row['opponent_name']}:\n")
            f.write(f"  Win Rate:     {row['win_rate']:6.1f}% ({row['wins']}/{row['num_games']})\n")
            f.write(f"  Loss Rate:    {row['loss_rate']:6.1f}% ({row['losses']}/{row['num_games']})\n")
            f.write(f"  Draw Rate:    {row['draw_rate']:6.1f}% ({row['draws']}/{row['num_games']})\n")
            f.write(f"  As First:     {row['win_rate_as_first']:6.1f}%\n")
            f.write(f"  As Second:    {row['win_rate_as_second']:6.1f}%\n")
            f.write(f"  Avg Moves:    {row['avg_move_count']:6.1f} ± {row['std_move_count']:.1f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("-"*60 + "\n")
        
        # Analyze performance patterns
        random_win_rate = df[df['opponent_name'] == 'RandomAgent']['win_rate'].values[0]
        threatening_03_win_rate = df[df['opponent_name'] == 'Threatening-0.3']['win_rate'].values[0]
        
        f.write(f"\n1. Baseline Performance:\n")
        f.write(f"   - vs Random: {random_win_rate:.1f}%\n")
        if random_win_rate >= 95:
            f.write(f"   ✅ EXCELLENT: Baseline goal achieved (>95%)\n")
        elif random_win_rate >= 90:
            f.write(f"   ⚠️  GOOD: Close to baseline goal (90-95%)\n")
        else:
            f.write(f"   ❌ NEEDS WORK: Below baseline goal (<90%)\n")
        
        f.write(f"\n2. Defensive Capabilities:\n")
        f.write(f"   - vs Threatening(0.3): {threatening_03_win_rate:.1f}%\n")
        if threatening_03_win_rate >= 50:
            f.write(f"   ✅ Agent has learned some defense\n")
        elif threatening_03_win_rate >= 20:
            f.write(f"   ⚠️  Agent has minimal defense (expected for baseline)\n")
        else:
            f.write(f"   ❌ Agent has no defense (expected for pure offensive training)\n")
        
        f.write(f"\n3. Performance Degradation:\n")
        threat_levels = ['Threatening-0.1', 'Threatening-0.2', 'Threatening-0.3', 'Threatening-0.5']
        for threat in threat_levels:
            wr = df[df['opponent_name'] == threat]['win_rate'].values[0]
            f.write(f"   - {threat}: {wr:.1f}%\n")
        
        f.write(f"\n4. First vs Second Player:\n")
        for _, row in df.iterrows():
            diff = row['win_rate_as_first'] - row['win_rate_as_second']
            f.write(f"   - {row['opponent_name']}: {diff:+.1f}% advantage as first\n")
    
    print(f"📄 Summary report saved to: {report_path}")


def create_visualizations(df, save_dir, timestamp):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Win rates by opponent
    ax1 = axes[0, 0]
    opponents = df['opponent_name']
    win_rates = df['win_rate']
    
    bars = ax1.bar(range(len(opponents)), win_rates, color=['green' if wr >= 95 else 'orange' if wr >= 50 else 'red' for wr in win_rates])
    ax1.axhline(y=95, color='green', linestyle='--', label='Target (95%)', linewidth=2)
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Opponent', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Win Rate vs Different Opponents', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(opponents)))
    ax1.set_xticklabels(opponents, rotation=45, ha='right')
    ax1.set_ylim([0, 105])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. First vs Second player performance
    ax2 = axes[0, 1]
    x = np.arange(len(opponents))
    width = 0.35
    
    first_rates = df['win_rate_as_first']
    second_rates = df['win_rate_as_second']
    
    ax2.bar(x - width/2, first_rates, width, label='As First Player', color='skyblue')
    ax2.bar(x + width/2, second_rates, width, label='As Second Player', color='lightcoral')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Opponent', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title('Win Rate: First vs Second Player', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(opponents, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Average move count
    ax3 = axes[1, 0]
    move_counts = df['avg_move_count']
    move_stds = df['std_move_count']
    
    ax3.bar(range(len(opponents)), move_counts, yerr=move_stds, capsize=5, color='mediumpurple', alpha=0.7)
    ax3.set_xlabel('Opponent', fontsize=12)
    ax3.set_ylabel('Average Moves per Game', fontsize=12)
    ax3.set_title('Game Length by Opponent', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(opponents)))
    ax3.set_xticklabels(opponents, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Win/Loss/Draw distribution
    ax4 = axes[1, 1]
    categories = ['Wins', 'Losses', 'Draws']
    colors = ['green', 'red', 'gray']
    
    for i, opponent in enumerate(opponents):
        row = df[df['opponent_name'] == opponent].iloc[0]
        values = [row['wins'], row['losses'], row['draws']]
        total = sum(values)
        percentages = [v/total*100 for v in values]
        
        bottom = 0
        for j, (val, pct, color) in enumerate(zip(values, percentages, colors)):
            ax4.barh(i, pct, left=bottom, color=color, alpha=0.7)
            if pct > 5:  # Only show label if segment is large enough
                ax4.text(bottom + pct/2, i, f'{val}', ha='center', va='center', fontweight='bold')
            bottom += pct
    
    ax4.set_yticks(range(len(opponents)))
    ax4.set_yticklabels(opponents)
    ax4.set_xlabel('Percentage (%)', fontsize=12)
    ax4.set_title('Win/Loss/Draw Distribution', fontsize=14, fontweight='bold')
    ax4.legend(categories, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"evaluation_plots_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models_baseline/dqn_baseline_best.pt"  # Change to your model path
    NUM_GAMES = 100  # Games per opponent (increase to 200 for more confidence)
    SAVE_DIR = "evaluation_results"
    
    # Run evaluation
    results_df = run_full_evaluation(
        model_path=MODEL_PATH,
        num_games_per_opponent=NUM_GAMES,
        save_dir=SAVE_DIR
    )
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved in '{SAVE_DIR}/' directory:")
    print("  - CSV file with detailed statistics")
    print("  - Text summary report")
    print("  - Visualization plots")
    print("\nNext steps:")
    print("  1. Review win rate vs RandomAgent (should be 95%+)")
    print("  2. Check win rate vs Threatening agents (identifies weaknesses)")
    print("  3. Use insights to design Phase 2 defensive training")
    print("="*60)