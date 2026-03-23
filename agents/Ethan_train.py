"""
Train Ethan_Agent with DQN in the Gomoku environment.

This file should contain:
1) Environment creation
2) Replay buffer
3) Online/target networks
4) Epsilon schedule
5) Optimization loop
6) Checkpoint saving
"""

# ...existing code...
# Pseudocode-level structure:


def train():
    """
    Run DQN training loop.

    Per episode:
        - reset env
        - while not done:
            - choose action via epsilon-greedy (agent.predict or trainer policy)
            - step env
            - store transition (s, a, r, s', done) in replay buffer
            - sample mini-batch when buffer is ready
            - compute TD target with target network
            - optimize online network
            - periodically update target network
        - log metrics and save checkpoints periodically
    """


def optimize_step():
    """
    Sample batch from replay buffer and apply one DQN gradient step.

    Must include legal-action masking when computing max_a' Q_target(s', a')
    so target uses only valid next actions.
    """


if __name__ == "__main__":
    train()
