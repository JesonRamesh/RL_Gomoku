import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from agents.dqn_simple_jeson import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent
from game.match import eval_agents

BOARD_SIZE = 9
NUM_GAMES = 200

# -------------------------------------------------------
# Load agents
# -------------------------------------------------------

baseline = DQNAgent(player_id=1, board_size=BOARD_SIZE)
baseline.load_model("models_baseline_9x9/dqn_baseline_final_20k.pt")
baseline.epsilon = 0.0

phase4 = DQNAgent(player_id=1, board_size=BOARD_SIZE)
phase4.load_model("models_phase4_v2/phase4_best_strategic.pt")
phase4.epsilon = 0.0

random_opp     = RandomAgent(player_id=-1)
strategic_03   = StrategicAgent(player_id=-1, skill_level=0.3, board_size=BOARD_SIZE)
strategic_05   = StrategicAgent(player_id=-1, skill_level=0.5, board_size=BOARD_SIZE)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------

print("=" * 50)
print(f"Evaluation over {NUM_GAMES} games each")
print("=" * 50)

print("\n--- Baseline vs RandomAgent ---")
r = eval_agents(baseline, random_opp, num_games=NUM_GAMES, board_size=BOARD_SIZE)
baseline_vs_random = r["agent1_wins"] / NUM_GAMES * 100

print("\n--- Phase 4 v2 vs RandomAgent ---")
r = eval_agents(phase4, random_opp, num_games=NUM_GAMES, board_size=BOARD_SIZE)
phase4_vs_random = r["agent1_wins"] / NUM_GAMES * 100

print("\n--- Baseline vs StrategicAgent-0.3 ---")
r = eval_agents(baseline, strategic_03, num_games=NUM_GAMES, board_size=BOARD_SIZE)
baseline_vs_s03 = r["agent1_wins"] / NUM_GAMES * 100

print("\n--- Phase 4 v2 vs StrategicAgent-0.3 ---")
r = eval_agents(phase4, strategic_03, num_games=NUM_GAMES, board_size=BOARD_SIZE)
phase4_vs_s03 = r["agent1_wins"] / NUM_GAMES * 100

print("\n--- Baseline vs StrategicAgent-0.5 ---")
r = eval_agents(baseline, strategic_05, num_games=NUM_GAMES, board_size=BOARD_SIZE)
baseline_vs_s05 = r["agent1_wins"] / NUM_GAMES * 100

print("\n--- Phase 4 v2 vs StrategicAgent-0.5 ---")
r = eval_agents(phase4, strategic_05, num_games=NUM_GAMES, board_size=BOARD_SIZE)
phase4_vs_s05 = r["agent1_wins"] / NUM_GAMES * 100

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"{'Opponent':<25} {'Baseline':>10} {'Phase 4 v2':>12} {'Change':>8}")
print("-" * 55)
print(f"{'vs RandomAgent':<25} {baseline_vs_random:>9.1f}% {phase4_vs_random:>11.1f}% {phase4_vs_random - baseline_vs_random:>+7.1f}%")
print(f"{'vs StrategicAgent-0.3':<25} {baseline_vs_s03:>9.1f}% {phase4_vs_s03:>11.1f}% {phase4_vs_s03 - baseline_vs_s03:>+7.1f}%")
print(f"{'vs StrategicAgent-0.5':<25} {baseline_vs_s05:>9.1f}% {phase4_vs_s05:>11.1f}% {phase4_vs_s05 - baseline_vs_s05:>+7.1f}%")
print("=" * 50)
