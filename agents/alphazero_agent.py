import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent

Move = Tuple[int, int]


class ResidualBlock(nn.Module):
    """A standard 2-layer residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroPolicyValueNet(nn.Module):
    """Policy-value network used by AlphaZeroAgent."""

    def __init__(self, board_size: int = 9, channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


class _MCTSNode:
    """Internal MCTS node for AlphaZeroAgent."""

    def __init__(
        self,
        board: Optional[np.ndarray],
        to_play: int,
        parent: Optional["_MCTSNode"] = None,
        prior: float = 0.0,
        action_from_parent: Optional[int] = None,
        last_move: Optional[Move] = None,
    ):
        self.board = board
        self.to_play = to_play
        self.parent = parent
        self.prior = prior
        self.action_from_parent = action_from_parent
        self.last_move = last_move

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, _MCTSNode] = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroAgent(BaseAgent):
    """
    AlphaZero-style Gomoku agent.

    Uses a policy-value network plus PUCT MCTS at inference time.
    """

    def __init__(
        self,
        player_id: int,
        board_size: int = 9,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 0.0,
        channels: int = 128,
        num_res_blocks: int = 6,
    ):
        super().__init__(player_id)
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.network = AlphaZeroPolicyValueNet(
            board_size=board_size,
            channels=channels,
            num_blocks=num_res_blocks,
        ).to(self.device)
        self.network.eval()

        print(f"AlphaZeroAgent using device: {self.device}")

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Support both direct state_dict and wrapped checkpoints.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.network.load_state_dict(state_dict)
        self.network.eval()
        print("AlphaZero model loaded successfully.")

    def predict(self, board_state: np.ndarray) -> Optional[Move]:
        """BaseAgent entrypoint: return one move (row, col)."""
        move, _ = self.self_play_move(board_state, add_dirichlet_noise=False)
        return move

    def self_play_move(
        self, board_state: np.ndarray, add_dirichlet_noise: bool = True
    ) -> Tuple[Optional[Move], np.ndarray]:
        """
        Return a move and the MCTS visit distribution over all actions.

        This is useful for AlphaZero training targets:
        - policy target = normalized visit counts
        - value target = game outcome from each state's player perspective
        """
        valid_moves = self._valid_moves(board_state)
        if not valid_moves:
            return None, np.zeros(self.action_size, dtype=np.float32)

        root = _MCTSNode(board_state.copy(), to_play=self.player_id)
        self._expand_node(root, add_dirichlet_noise=add_dirichlet_noise)

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        visit_counts = np.zeros(self.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if visit_counts.sum() <= 0:
            move = valid_moves[np.random.randint(len(valid_moves))]
            return move, visit_counts

        if self.temperature <= 0.0:
            action = int(np.argmax(visit_counts))
        else:
            tempered = np.power(visit_counts, 1.0 / self.temperature)
            probs = tempered / (tempered.sum() + 1e-12)
            action = int(np.random.choice(self.action_size, p=probs))

        return self._action_to_move(action), visit_counts / visit_counts.sum()

    def _run_simulation(self, root: _MCTSNode) -> None:
        node = root
        search_path = [node]

        # Selection
        while node.children:
            action, node = self._select_child(node)
            _ = action

            # Lazily materialize child board only when traversed.
            if node.board is None:
                assert node.parent is not None
                assert node.parent.board is not None
                assert node.action_from_parent is not None
                parent_board = node.parent.board
                move = self._action_to_move(node.action_from_parent)
                next_board = parent_board.copy()
                next_board[move] = node.parent.to_play
                node.board = next_board

            search_path.append(node)

        assert node.board is not None
        winner = self._get_winner(node.board, node.last_move)

        # Expansion + Evaluation
        if winner is None:
            value = self._expand_node(node, add_dirichlet_noise=False)
        elif winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == node.to_play else -1.0

        # Backpropagation: flip sign each ply.
        for visited in reversed(search_path):
            visited.visit_count += 1
            visited.value_sum += value
            value = -value

    def _expand_node(self, node: _MCTSNode, add_dirichlet_noise: bool) -> float:
        assert node.board is not None
        valid_moves = self._valid_moves(node.board)
        if not valid_moves:
            return 0.0

        policy_logits, value = self._policy_value(node.board, node.to_play)

        mask = np.full(self.action_size, -1e9, dtype=np.float32)
        for move in valid_moves:
            mask[self._move_to_action(move)] = 0.0

        logits = policy_logits + mask
        probs = self._softmax(logits)

        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
            for i, move in enumerate(valid_moves):
                action = self._move_to_action(move)
                probs[action] = (1.0 - self.dirichlet_epsilon) * probs[
                    action
                ] + self.dirichlet_epsilon * noise[i]

        # Normalize over valid actions only (safety for numeric stability).
        valid_actions = [self._move_to_action(mv) for mv in valid_moves]
        valid_prob_sum = float(np.sum(probs[valid_actions]))
        if valid_prob_sum <= 1e-12:
            uniform = 1.0 / len(valid_actions)
            for action in valid_actions:
                probs[action] = uniform
        else:
            for action in valid_actions:
                probs[action] /= valid_prob_sum

        for move in valid_moves:
            action = self._move_to_action(move)
            node.children[action] = _MCTSNode(
                board=None,
                to_play=-node.to_play,
                parent=node,
                prior=float(probs[action]),
                action_from_parent=action,
                last_move=move,
            )

        return float(value)

    def _select_child(self, node: _MCTSNode) -> Tuple[int, _MCTSNode]:
        sqrt_parent_visits = math.sqrt(max(1, node.visit_count))

        best_action = -1
        best_child: Optional[_MCTSNode] = None
        best_score = -float("inf")

        for action, child in node.children.items():
            prior_score = (
                self.c_puct
                * child.prior
                * (sqrt_parent_visits / (1 + child.visit_count))
            )
            # child.q_value is from child's player perspective; negate to score
            # from the current node's player perspective.
            score = -child.q_value + prior_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None
        return best_action, best_child

    def _policy_value(
        self, board: np.ndarray, current_player: int
    ) -> Tuple[np.ndarray, float]:
        state = self._encode_board(board, current_player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

        policy_np = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
        value_scalar = float(value.squeeze(0).item())
        return policy_np, value_scalar

    def _encode_board(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """Encode board from current_player perspective into 3 planes."""
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (board == current_player).astype(np.float32)
        state[1] = (board == -current_player).astype(np.float32)
        state[2] = np.full(
            (self.board_size, self.board_size), current_player, dtype=np.float32
        )
        return state

    def _valid_moves(self, board: np.ndarray) -> List[Move]:
        return list(zip(*np.where(board == 0)))

    def _move_to_action(self, move: Move) -> int:
        return move[0] * self.board_size + move[1]

    def _action_to_move(self, action: int) -> Move:
        return action // self.board_size, action % self.board_size

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def _get_winner(
        self, board: np.ndarray, last_move: Optional[Move]
    ) -> Optional[int]:
        """
        Return winner on the given board.

        Returns:
        - 1 or -1 if that player has 5 in a row
        - 0 if draw (board full and no winner)
        - None if game is not terminal
        """
        if last_move is None:
            if not np.any(board == 0):
                return 0
            return None

        row, col = last_move
        player = board[row, col]
        if player == 0:
            return None

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return int(player)

        if not np.any(board == 0):
            return 0

        return None
