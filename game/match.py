from game.logic import GomokuLogic


def eval_agents(agent1, agent2, num_games=100, board_size=15):
    """
    Run agent evaluations headlessly (no visualisation - PyGame) and return the results as a dictionary.

    :param agent1: First agent
    :param agent2: Second agent duh
    :param num_games: Number of games
    :param board_size: Size of Gomoku board

    :return: Dictionary with results:
    {
        "agent1_wins": int,
        "agent2_wins": int,
        "draws": int
    }
    """

    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    for game_num in range(num_games):
        # Alternate first move for fariness
        if game_num % 2 == 0:
            p1, p2 = agent1, agent2
            p1.player_id = 1
            p2.player_id = -1
        else:
            p1, p2 = agent2, agent1
            p1.player_id = 1
            p2.player_id = -1
        game = GomokuLogic(board_size=board_size)
        players = {1: p1, -1: p2}  # initialise players dict

        # play a game
        while not game.game_over:
            current_agent = players[game.current_player]

            if current_agent.is_human:
                raise ValueError("eval_agents is for clankers only!")

            move = current_agent.predict(game.board)

            if move is None:
                break

            row, col = move
            try:
                game.make_move(row, col)
            except ValueError:
                # if illegal move then agent loses immediately (could change this behaviour)
                print(
                    f"Agent {current_agent.player_id} attempted an invalid move at {row}, {col}"
                )
                game.game_over = True
                game.winner = current_agent.player_id * -1  # other player wins
                break
        # update results
        winner = game.winner if game.game_over else 0

        # Map winner back to original agents
        if game_num % 2 == 0:
            # agent1 was player 1
            if winner == 1:
                results["agent1_wins"] += 1
            elif winner == -1:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
        else:
            # agent2 was player 1 (swapped)
            if winner == 1:
                results["agent2_wins"] += 1
            elif winner == -1:
                results["agent1_wins"] += 1
            else:
                results["draws"] += 1

    # Results
    total = results["agent1_wins"] + results["agent2_wins"] + results["draws"]
    print(f"\nResults after {num_games} games:")

    print(
        f"Agent 1 wins: {results['agent1_wins']} ({results['agent1_wins'] / total * 100:.1f}%)"
    )
    print(
        f"Agent 2 wins: {results['agent2_wins']} ({results['agent2_wins'] / total * 100:.1f}%)"
    )
    print(f"Draws: {results['draws']} ({results['draws'] / total * 100:.1f}%)")

    return results
