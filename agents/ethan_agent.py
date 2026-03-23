from agents.base_agent import BaseAgent
import random
import numpy as np
import torch
import torch.nn as nn

EPSILON = 0.1  # Exploration rate for epsilon-greedy strategy


class Ethan_Agent(nn.Module, BaseAgent):
    def __init__(
        self,
        player__id,
        n_observations=225,
        n_actions=225,
        hidden_dim=128,
        epsilon=EPSILON,
        device=None,
    ):
        BaseAgent.__init__(self, player__id)
        nn.Module.__init__(self)

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.device = device or torch.device("cpu")
        self.board_state = None

        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        """
        Run a forward pass of the Q-network and return Q-values for all actions.
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

    def _encode_state(self, board_state):
        """
        Convert environment board representation into DQN input tensor.

        Expected behavior:
            - Accept board as array-like (or environment-specific object that can
              be converted to array).
            - Encode from this player's perspective:
                own stones -> +1
                opponent stones -> -1
                empty cells -> 0
            - Flatten to size `n_observations`.
            - Return tensor on `self.device`, shape [1, n_observations].

        Args:
            board_state: Current board snapshot from environment.

        Returns:
            torch.Tensor: Encoded single-state batch for network input.

        Raises:
            ValueError: If flattened board size does not match `n_observations`.
        """
        q_state = board_state.flatten()
        if q_state.size != self.n_observations:
            raise ValueError(
                f"Encoded state size {q_state.size} does not match expected {self.n_observations}"
            )
        return (
            torch.tensor(q_state, dtype=torch.float32, device=self.device).unsqueeze(
                q_state.dim()
            ),
        )

    def _get_legal_action_indices(self, board_state):
        """
        Get currently legal moves as flattened action indices.

        Behavior:
            - Read legal actions from environment state if available
              (e.g., `get_legal_actions()` or `legal_actions`).
            - If legal actions are (row, col), convert to flat index.
            - If legal actions are already flat indices, validate and return.
            - Optionally fallback to scanning empty cells if state is an array.

        Args:
            board_state: Current environment state/board.

        Returns:
            list[int]: Legal action indices in [0, n_actions).

        Notes:
            - This function is central for masking illegal actions during action
              selection and training target computation.
        """

    def _normalize_legal_actions(self, legal_actions):
        """
        Normalize mixed legal-action formats into flat integer indices.

        Supported input formats:
            - [(row, col), ...]
            - [flat_idx, flat_idx, ...]

        Args:
            legal_actions (list): Actions in one of the supported formats.

        Returns:
            list[int]: Flattened action indices compatible with Q-value vector.
        """

    def _index_to_move(self, action_idx):
        """
        Convert flattened action index into board coordinates.

        Args:
            action_idx (int): Flat action in [0, n_actions).

        Returns:
            tuple[int, int]: (row, col) move for environment `step`/`play` call.
        """

    def predict(self, board_state):
        """
        Choose next move using epsilon-greedy policy over legal actions only.

        Decision flow:
            1) Get legal actions.
            2) With probability epsilon: sample random legal action.
            3) Otherwise:
               - encode state,
               - compute Q-values with network,
               - mask illegal actions to -inf,
               - select argmax over legal actions.
            4) Convert selected flat action to (row, col).

        Args:
            board_state: Current board snapshot.

        Returns:
            tuple[int, int] | None:
                - (row, col) for selected legal move.
                - None if no legal action exists (terminal/full board).

        Important:
            - This is inference-time action selection.
            - During training, epsilon may be scheduled externally.
        """
