"""
Quick script to evaluate all saved Phase 5 checkpoints and find the best one.
Run this before deciding next steps.
"""
import os
from game.logic import GomokuLogic
from game.gomoku_env import GomokuEnv
from agents.dqn_simple_jeson import DQNAgent
from agents.random_agent import RandomAgent
from agents.strategic_agent import StrategicAgent


def evaluate_model(path, board_size=9, num_games=200):
    if not os.path.exists(path):
        return None
    agent = DQNAgent(player_id=1, board_size=board_size)
    agent.load_model(path)
    agent.epsilon = 0.0

    results = {}
    for skill, label in [(None, "random"), (0.3, "s03"), (0.5, "s05")]:
        wins = 0
        for g in range(num_games):
            logic = GomokuLogic(board_size=board_size)
            env = GomokuEnv(logic, use_sparse_rewards=True)
            opponent = RandomAgent(player_id=-1) if skill is None else \
                       StrategicAgent(player_id=-1, skill_level=skill, board_size=board_size)
            agent.player_id = 1 if g % 2 == 0 else -1
            opponent.player_id = -agent.player_id
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
        results[label] = wins / num_games * 100
    return results


if __name__ == "__main__":
    checkpoints = [
        ("Phase 4 v2 best (baseline)",   "models_phase4_v2/phase4_best_strategic.pt"),
        ("Phase 5 best vs Random",        "models_phase5/phase5_best_random.pt"),
        ("Phase 5 best vs S-0.3",         "models_phase5/phase5_best_strategic03.pt"),
        ("Phase 5 best vs S-0.5",         "models_phase5/phase5_best_strategic05.pt"),
        ("Phase 5 ep500  (test run)",     "models_phase5_test/phase5_best_strategic05.pt"),
        ("Phase 5 final  (collapsed)",    "models_phase5/phase5_final.pt"),
    ]

    print(f"\n{'Model':<40} {'vs Random':>10} {'vs S-0.3':>10} {'vs S-0.5':>10}")
    print("-" * 75)
    for label, path in checkpoints:
        r = evaluate_model(path)
        if r is None:
            print(f"{label:<40} {'(not found)':>10}")
        else:
            print(f"{label:<40} {r['random']:>9.1f}% {r['s03']:>9.1f}% {r['s05']:>9.1f}%")
    print()
