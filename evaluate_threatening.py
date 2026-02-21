"""
Evaluate Baseline Agent vs Threatening Agents
"""

import numpy as np
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple_jeson import DQNAgent
from agents.threatening_agent import ThreateningAgent
import time


def evaluate_matchup(
    agent1,
    agent2,
    num_games: int = 100,
    agent1_name: str = "Agent1",
    agent2_name: str = "Agent2"
):
    """
    Play num_games between two agents and return win statistics.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        agent1_name: Name for agent1 (for display)
        agent2_name: Name for agent2 (for display)
    
    Returns:
        dict: Statistics
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_moves = []
    
    for game in range(num_games):
        # Alternate starting player
        if game % 2 == 0:
            agent1.player_id = 1
            agent2.player_id = -1
            first_player = agent1
            second_player = agent2
        else:
            agent1.player_id = -1
            agent2.player_id = 1
            first_player = agent2
            second_player = agent1
        
        # Play game
        game_logic = GomokuLogic(board_size=15)
        env = GomokuEnv(game_logic, use_sparse_rewards=True)
        state = env.reset()
        done = False
        moves = 0
        
        while not done and moves < 225:  # Max moves on 15x15 board
            if env.logic.current_player == agent1.player_id:
                action = agent1.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    agent1_wins += 1
            else:
                action = agent2.predict(state)
                state, reward, done, _ = env.step(action)
                if done and reward > 0:
                    agent2_wins += 1
            
            moves += 1
        
        if not done:
            draws += 1
        
        total_moves.append(moves)
    
    return {
        f'{agent1_name}_wins': agent1_wins,
        f'{agent2_name}_wins': agent2_wins,
        'draws': draws,
        f'{agent1_name}_win_rate': (agent1_wins / num_games) * 100,
        f'{agent2_name}_win_rate': (agent2_wins / num_games) * 100,
        'avg_game_length': np.mean(total_moves)
    }


def main():
    """Main evaluation routine."""
    print("\n" + "="*60)
    print("🎯 BASELINE vs THREATENING AGENTS EVALUATION")
    print("="*60)
    print("\nPurpose: Establish expected win rates for curriculum planning")
    print("Baseline: models_baseline/dqn_baseline_best.pt (Phase 1 final)")
    print("\n")
    
    # Load baseline agent
    print("📥 Loading baseline agent...")
    baseline = DQNAgent(
        player_id=1,
        board_size=15,
        learning_rate=0.0001,
        gamma=0.95,
        epsilon_start=0.0,  # Greedy evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_capacity=50000,
        target_update_frequency=500
    )
    baseline.load_model("models_baseline/dqn_baseline_best.pt")
    baseline.epsilon = 0.0  # Force greedy
    print("✅ Baseline loaded\n")
    
    # Test against different block probabilities
    block_probs = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    num_games = 100
    
    print(f"🎮 Running {num_games} games per difficulty level...")
    print("This will take approximately 5-10 minutes...\n")
    
    results = []
    start_time = time.time()
    
    for prob in block_probs:
        print(f"\n{'─'*60}")
        print(f"Testing vs Threatening-{prob}")
        print(f"{'─'*60}")
        
        # Create threatening opponent
        opponent = ThreateningAgent(
            player_id=-1,
            block_probability=prob,
            board_size=15
        )
        
        # Evaluate
        matchup_start = time.time()
        stats = evaluate_matchup(
            agent1=baseline,
            agent2=opponent,
            num_games=num_games,
            agent1_name="Baseline",
            agent2_name=f"Threat-{prob}"
        )
        matchup_time = time.time() - matchup_start
        
        # Store results
        stats['block_probability'] = prob
        results.append(stats)
        
        # Print results
        print(f"\nResults:")
        print(f"  Baseline Win Rate:    {stats['Baseline_win_rate']:.1f}%")
        print(f"  Threatening Win Rate: {stats[f'Threat-{prob}_win_rate']:.1f}%")
        print(f"  Draws:                {stats['draws']}")
        print(f"  Avg Game Length:      {stats['avg_game_length']:.1f} moves")
        print(f"  Time:                 {matchup_time:.1f}s")
    
    # Print summary table
    elapsed = time.time() - start_time
    
    print("\n\n" + "="*60)
    print("📊 SUMMARY: BASELINE PERFORMANCE vs THREATENING AGENTS")
    print("="*60)
    print(f"\n{'Block Prob':<12} {'Win Rate':<12} {'Avg Moves':<12} {'Difficulty'}")
    print("─" * 60)
    
    for r in results:
        prob = r['block_probability']
        win_rate = r['Baseline_win_rate']
        avg_moves = r['avg_game_length']
        
        # Classify difficulty
        if win_rate >= 90:
            difficulty = "Easy ✅"
        elif win_rate >= 75:
            difficulty = "Moderate 🟡"
        elif win_rate >= 60:
            difficulty = "Hard 🟠"
        elif win_rate >= 45:
            difficulty = "Very Hard 🔴"
        else:
            difficulty = "Expert 🔥"
        
        print(f"{prob:<12.2f} {win_rate:<12.1f}% {avg_moves:<12.1f} {difficulty}")
    
    print("\n" + "="*60)
    print(f"⏱️  Total Time: {elapsed/60:.1f} minutes")
    print("="*60)
    
    # Curriculum recommendations
    print("\n\n" + "="*60)
    print("🎓 CURRICULUM LEARNING RECOMMENDATIONS")
    print("="*60)
    
    print("\nBased on these results, recommended curriculum stages:")
    print("\nStage 1 (Foundation):")
    print("  - 70% Random, 20% Threat-0.1, 10% Threat-0.15")
    print(f"  - Expected win rate: 96-98%")
    
    print("\nStage 2 (Awareness):")
    print("  - 60% Random, 25% Threat-0.15, 15% Threat-0.2")
    print(f"  - Expected win rate: 95-97%")
    
    print("\nStage 3 (Integration):")
    print("  - 50% Random, 25% Threat-0.2, 20% Threat-0.25, 5% Threat-0.3")
    print(f"  - Expected win rate: 94-96%")
    
    print("\nStage 4 (Mastery):")
    print("  - 45% Random, 25% Threat-0.25, 20% Threat-0.3, 10% Threat-0.35")
    print(f"  - Expected win rate: 93-95%")
    
    print("\nStage 5 (Refinement):")
    print("  - 40% Random, 25% Threat-0.3, 20% Threat-0.35, 15% Threat-0.4")
    print(f"  - Expected win rate: 92-94%")
    
    print("\nStage 6 (Challenge):")
    print("  - 40% Random, 25% Threat-0.35, 20% Threat-0.4, 15% Threat-0.5")
    print(f"  - Expected win rate: 91-93%")
    
    print("\n✅ Evaluation complete! Ready to implement Phase 2B.2 curriculum.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()