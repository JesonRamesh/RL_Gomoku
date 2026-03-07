import pygame
import numpy as np
import os
from game.logic import GomokuLogic
from game.board import Board
from agents.vin_agent import RLAgent
from agents.base_agent import HumanAgent
from agents.intermediate_rewards import RewardLogic


def scale_reward(shaped, terminal=100.0, max_shaped_fraction=0.3):
    cap = terminal * max_shaped_fraction
    if shaped > 0:
        scaled = cap * (1 - np.exp(-shaped / cap))
    else:
        scaled = shaped
    return scaled


def human_train(board_size=9, model_path="gomoku_best_winrate_model.pth"):

    rl_agent = RLAgent(player_id=-1, board_size=board_size)

    # track existing winrate so we only save improvements
    session_wins = 0
    session_games = 0
    best_winrate = 0.0

    if os.path.exists(model_path):
        print(f"Loading {model_path}...")
        rl_agent.load(model_path)
    else:
        print("No saved model found, starting fresh.")

    game = GomokuLogic(board_size=board_size)
    board = Board(game)

    human = HumanAgent(player_id=1)
    players = {1: human, -1: rl_agent}

    agent_last_state = None
    agent_last_action = None
    human_last_state = None   # track human's last state to detect threats building
    episode = 0
    running = True

    print(f"Starting human training. Model saves to {model_path} only on improvement.")

    while running:
        board.draw()
        current_agent = players[game.current_player]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                # capture state BEFORE human move
                if (board.game_started and not game.game_over
                        and current_agent.is_human):
                    human_last_state = np.copy(game.board)

                board.mouse_click(pos)

                # AFTER human move — give agent a defensive signal
                if (board.game_started and not game.game_over
                        and human_last_state is not None
                        
                        and current_agent.is_human):

                    state_after_human = np.copy(game.board)

                    # check if human just escalated a threat the agent should have prevented
                    rl_h = RewardLogic(player=1, board=human_last_state, board_size=board_size)
                    human_threats_before = rl_h.threats()

                    rl_h2 = RewardLogic(player=1, board=state_after_human, board_size=board_size)
                    human_threats_after = rl_h2.threats()

                    new_human_fours = (human_threats_after["four"] - human_threats_before["four"] +
                                       human_threats_after["open_four"] - human_threats_before["open_four"])
                    new_human_threes = (human_threats_after["open_three"] - human_threats_before["open_three"])

                    # human just created a dangerous threat the agent failed to prevent
                    if new_human_fours > 0 and agent_last_state is not None:
                        print("  Human escalated to four — agent failed to prevent, penalising")
                        rl_agent.learn_td(agent_last_state, agent_last_action,
                            -5.0, state_after_human, False,
                            episode=episode, max_episodes=10000)

                    elif new_human_threes > 0 and agent_last_state is not None:
                        # softer penalty for allowing a three
                        rl_agent.learn_td(agent_last_state, agent_last_action,
                            -2.0, state_after_human, False,
                            episode=episode, max_episodes=10000)

        # AI turn
        if board.game_started and not game.game_over and not current_agent.is_human:
            pygame.time.delay(300)
            state_before = np.copy(game.board)
            move = rl_agent.predict(game.board)

            if move is not None:
                row, col = move
                try:
                    game.make_move(row, col)
                    state_after = np.copy(game.board)

                    if game.game_over:
                        if game.winner == -1:
                            rl_agent.learn_td(state_before, move, 100, state_after, True,
                                episode=episode, max_episodes=10000)
                            print(f"  Game {episode}: Agent won!")
                    else:
                        # missed win penalty — had a four last turn, now it's gone
                        if agent_last_state is not None:
                            rl_prev = RewardLogic(player=-1, board=agent_last_state, board_size=board_size)
                            threats_then = rl_prev.threats()
                            had_winning_threat = (threats_then["four"] > 0
                                                  or threats_then["open_four"] > 0)

                            if had_winning_threat:
                                rl_now = RewardLogic(player=-1, board=state_before, board_size=board_size)
                                threats_now = rl_now.threats()
                                threat_was_blocked = (
                                    threats_now["four"] < threats_then["four"] or
                                    threats_now["open_four"] < threats_then["open_four"]
                                )
                                if threat_was_blocked:
                                    print("  Agent missed a winning move — penalising")
                                    rl_agent.learn_td(agent_last_state, agent_last_action,
                                        -6.0, state_before, False,
                                        episode=episode, max_episodes=10000)

                        # shaped reward
                        rl = RewardLogic(player=-1, board=state_before, board_size=board_size)
                        shaped = scale_reward(
                            rl.rewards(board_before=state_before, board_after=state_after)
                        )
                        rl_agent.learn_td(state_before, move, shaped, state_after, False,
                            episode=episode, max_episodes=10000)

                    agent_last_state = state_before
                    agent_last_action = move

                except ValueError:
                    pass

        # game over
        if game.game_over and board.game_started:
            session_games += 1

            if game.winner == -1:
                session_wins += 1
                print(f"  Game {episode}: Agent won!")
            elif game.winner == 1:
                if agent_last_state is not None:
                    rl_agent.learn_td(agent_last_state, agent_last_action, -100,
                        np.copy(game.board), True, episode=episode, max_episodes=10000)
                print(f"  Game {episode}: Human won — agent penalised")
            
            # only save if winrate has improved over the session
            if session_games >= 3:   # wait for at least 3 games before judging
                current_winrate = session_wins / session_games
                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    rl_agent.save(model_path)
                    print(f"  -> Model saved (session winrate: {current_winrate:.2f}) -> {model_path}")
                else:
                    print(f"  -> No save (session winrate: {current_winrate:.2f} <= best: {best_winrate:.2f})")

            pygame.time.delay(1500)
            episode += 1
            game.reset_game()
            board.game_started = False
            agent_last_state = None
            agent_last_action = None
            human_last_state = None

    pygame.quit()