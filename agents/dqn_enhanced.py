"""
DQN Agent with improved architecture for strategic play.
Based on dqn_simple.py with enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from agents.base_agent import BaseAgent


class DQNetwork(nn.Module):
    """
    Deeper CNN with residual connections for better pattern recognition.
    """

    def __init__(self, board_size=9):
        super(DQNetwork, self).__init__()
        self.board_size = board_size

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks for deeper pattern recognition
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Separate value and advantage streams (Dueling DQN)
        self.fc_value = nn.Linear(128 * board_size * board_size, 256)
        self.fc_adv = nn.Linear(128 * board_size * board_size, 256)

        self.value_out = nn.Linear(256, 1)
        self.adv_out = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Conv blocks with residual
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Second residual block
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        # Dueling architecture
        value = F.relu(self.fc_value(x))
        value = self.value_out(value)

        adv = F.relu(self.fc_adv(x))
        adv = self.adv_out(adv)

        # Combine value and advantage
        q_values = value + (adv - adv.mean(dim=1, keepdim=True))

        return q_values


def preprocess_board(board, current_player):
    """
    Preprocess the board state into a 3-channel tensor.
    """
    board_size = board.shape[0]
    state = np.zeros((3, board_size, board_size), dtype=np.float32)

    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == -current_player).astype(np.float32)
    state[2] = np.ones((board_size, board_size), dtype=np.float32) * current_player

    return state


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better learning from important experiences."""

    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, player_id):
        max_priority = max(self.priorities) if self.priorities else 1.0

        experience = (state, action, reward, next_state, done, player_id)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones, player_ids = zip(*samples)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            player_ids,
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Enhanced DQN Agent with:
    - Dueling architecture
    - Prioritized replay
    - Double DQN
    """

    def __init__(
        self,
        player_id,
        board_size=9,
        learning_rate=5e-5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        buffer_capacity=100000,
        target_update_frequency=500,
    ):
        super().__init__(player_id)
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.steps = 0

        # Initialize networks
        self.q_network = DQNetwork(board_size)
        self.target_network = DQNetwork(board_size)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.beta = 0.4  # For importance sampling
        self.beta_increment = 0.001

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"DQNAgent using device: {self.device}")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def predict(self, board_state):
        """Selects action using epsilon-greedy RL policy with tactical checks."""
        valid_moves = list(zip(*np.where(board_state == 0)))

        if len(valid_moves) == 0:
            return None

        # Always check for immediate wins or blocks
        urgent_move = self._find_urgent_move(board_state, valid_moves)
        if urgent_move and random.random() > 0.1:
            return urgent_move

        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            return self._get_best_action(board_state, valid_moves)

    def _find_urgent_move(self, board, valid_moves):
        """Find winning move, block opponent's win, or create/block strong threats."""
        for move in valid_moves:
            if self._is_winning_move(board, move, self.player_id):
                return move

        for move in valid_moves:
            if self._is_winning_move(board, move, -self.player_id):
                return move

        for move in valid_moves:
            if self._creates_open_four(board, move, self.player_id):
                return move

        for move in valid_moves:
            if self._creates_open_four(board, move, -self.player_id):
                return move

        fork_move = self._find_fork_move(board, valid_moves, self.player_id)
        if fork_move:
            return fork_move

        fork_move = self._find_fork_move(board, valid_moves, -self.player_id)
        if fork_move:
            return fork_move

        return None

    def _is_winning_move(self, board, move, player):
        """Check if a move wins the game."""
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def _creates_open_four(self, board, move, player):
        """Check if move creates an open four."""
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            open_ends = 0
            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == 0
            ):
                open_ends += 1
            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r -= dr
                c -= dc
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == 0
            ):
                open_ends += 1
            if count >= 4 and open_ends >= 1:
                return True
        return False

    def _find_fork_move(self, board, valid_moves, player):
        """Find move that creates multiple open threes."""
        for move in valid_moves:
            open_threes = 0
            row, col = move
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                count = 1
                open_ends = 0
                r, c = row + dr, col + dc
                while (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == player
                ):
                    count += 1
                    r += dr
                    c += dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == 0
                ):
                    open_ends += 1
                r, c = row - dr, col - dc
                while (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == player
                ):
                    count += 1
                    r -= dr
                    c -= dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and board[r, c] == 0
                ):
                    open_ends += 1
                if count == 3 and open_ends == 2:
                    open_threes += 1
            if open_threes >= 2:
                return move
        return None

    def _get_best_action(self, board_state, valid_moves):
        """Gets the best action based on Q-values."""
        state_tensor = (
            torch.FloatTensor(preprocess_board(board_state, self.player_id))
            .unsqueeze(0)
            .to(self.device)
        )

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()
        self.q_network.train()

        q_values_np = q_values.cpu().numpy().reshape(self.board_size, self.board_size)
        q_values_np[board_state != 0] = -float("inf")

        best_move = np.unravel_index(np.argmax(q_values_np), q_values_np.shape)
        return best_move

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)
        self.beta = min(1.0, self.beta + self.beta_increment)

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones, player_ids, indices, weights = (
            self.replay_buffer.sample(batch_size, self.beta)
        )

        states_list = [preprocess_board(s, pid) for s, pid in zip(states, player_ids)]
        next_states_list = [
            preprocess_board(ns, pid) for ns, pid in zip(next_states, player_ids)
        ]

        states_tensor = torch.FloatTensor(np.array(states_list)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states_list)).to(
            self.device
        )

        action_indices = torch.LongTensor(
            [a[0] * self.board_size + a[1] for a in actions]
        ).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        current_q_values = (
            self.q_network(states_tensor)
            .gather(1, action_indices.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            next_actions = self.q_network(next_states_tensor).max(1)[1]
            next_q_values = (
                self.target_network(next_states_tensor)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards_tensor + self.gamma * next_q_values * (
                1 - dones_tensor
            )

        # TD errors for priority update
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Weighted loss
        loss = (
            weights_tensor
            * F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done, self.player_id)

    def save_model(self, path):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "beta": self.beta,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        if "beta" in checkpoint:
            self.beta = checkpoint["beta"]
        print("Model loaded successfully.")


# Alias for backward compatibility.
DQNAgentEnhanced = DQNAgent
