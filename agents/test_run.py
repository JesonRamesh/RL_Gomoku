def test_run(env, agent1, agent2):
    board_state = env.reset()
    done = False
    current = agent1

    while not done:
        action = current.predict(board_state)
        board_state, _, done, info = env.step(action)
        # Switch to the other agent for the next turn
        current = agent2 if current is agent1 else agent1

        return info
