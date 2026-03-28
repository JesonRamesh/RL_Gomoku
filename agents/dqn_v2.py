import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from agents.base_agent import BaseAgent

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """A residual block with two convolutional layers and a skip connection."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass through the residual block."""
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # Skip connection
        return F.relu(x)

class DQNetwork(nn.Module):
    
    def __init__(self, board_size=15, num_res_blocks=5):
        """
        Convolutional Neural Network for DQN agent.
        
        Architecture:
        - Input: (batch_size, 3, board_size, board_size) - 3 channels for current player, opponent, and empty
        - Conv1: 64 filters, kernel size 3, padding 1, ReLU
        - Conv2: 128 filters, kernel size 3, padding 1, ReLU
        - Conv3: 128 filters, kernel size 3, padding 1, ReLU
        - FC1: 512 units, ReLU
        - FC2: board_size * board_size units (Q-values for each action)
        """
        super().__init__()
        # self.board_size = board_size
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(512, board_size * board_size)
        self.conv_input = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_res_blocks)])
        self.conv_policy = nn.Conv2d(128, 32, kernel_size=1)
        self.fc_policy = nn.Linear(32 * board_size * board_size, board_size * board_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, board_size, board_size)

        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, board_size * board_size)
        """
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.bn4(self.fc1(x)))
        # q_values = self.fc2(x)
        # return q_values
        x = F.relu(self.conv_input(x))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.conv_policy(x))
        x = x.view(x.size(0), -1)
        q_values = self.fc_policy(x)
        return q_values
    
def preprocess_board(board, current_player):
    """
    Preprocess the board state into a 3-channel tensor for the DQN.

    Args:
        board (np.ndarray): The current board state, shape (board_size, board_size)
        current_player (int): The current player (1 or -1)

    Returns:
        torch.Tensor: Preprocessed board state, shape (1, 3, board_size, board_size)
        Channel 0: Current player's piece
        Channel 1: Opponent's piece
        Channel 2: Empty spaces
    """

    # Create 3 channels
    player_pieces = (board == current_player).astype(np.float32)
    opponent_pieces = (board == -current_player).astype(np.float32)
    empty_spaces = (board == 0).astype(np.float32)

    # Stack channels and add batch dimension
    processed_board = np.stack([player_pieces, opponent_pieces, empty_spaces], axis=0)
    return torch.FloatTensor(processed_board).unsqueeze(0)  # Shape: (1, 3, board_size, board_size)  


class ReplayBuffer:
    def __init__(self, capacity=100000):
        """Randomly samples past experiences for training the DQN."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, player_id):
        """Adds a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done, player_id))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, player_ids = zip(*batch)
        return states, actions, rewards, next_states, dones, player_ids

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)
    
class DQNAgent(BaseAgent):
    def __init__(self, player_id, board_size=15, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, buffer_capacity=100000, target_update_frequency=1000):
        super().__init__(player_id)
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.steps = 0  # To track when to update target network

        # Intialize DQN
        self.q_network = DQNetwork(board_size)
        self.target_network = DQNetwork(board_size)

        # Initlialize optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # Copy weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()


    def predict(self, board_state):
        """Selects an action using epsilon-greedy policy."""
        # Get valid moves
        valid_moves = list(zip(*np.where(board_state == 0)))

        if len(valid_moves) == 0:
            return None  # No valid moves available
        
        # Explore with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            # Exploit: choose action with highest Q-value
            return self._get_best_action(board_state, valid_moves)
        
    def _get_best_action(self, board_state, valid_moves):
        """Gets the best action based on Q-values for valid moves."""
        state_tensor = preprocess_board(board_state, self.player_id).to(self.device)
        
        # self.q_network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            q_values = self.q_network(state_tensor).view(self.board_size, self.board_size)
        # self.q_network.train()  # Set back to train mode
        
        q_values = q_values.cpu().numpy()
        masked_q_values = np.full((self.board_size, self.board_size), -np.inf)
        for move in valid_moves:
            masked_q_values[move] = q_values[move]
        best_action_idx = np.argmax(masked_q_values)
        return (best_action_idx // self.board_size, best_action_idx % self.board_size)
    
    def decay_epsilon(self):
        """Decays the exploration rate epsilon after each episode."""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def train_step(self, batch_size):
        """Performs a training step using a batch of experiences from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None # Not enough samples to train

        states, actions, rewards, next_states, dones, player_ids = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        state_tensors = torch.cat([preprocess_board(s, player_id) for s, player_id in zip(states, player_ids)]).to(self.device)
        next_state_tensors = torch.cat([preprocess_board(s, player_id) for s, player_id in zip(next_states, player_ids)]).to(self.device)
        
        # Convert actions to indices
        action_indices = [a[0] * self.board_size + a[1] for a in actions]
        action_indices = torch.LongTensor(action_indices).to(self.device)

        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values and next Q-values
        current_q_values = self.q_network(state_tensors).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            #next_actions = self.q_network(next_state_tensors).argmax(dim=1, keepdim=True)

            next_online_q = self.q_network(next_state_tensors)
            best_actions = next_online_q.max(dim=1)[1]  # Get indices of best actions from online network

            next_q_values = self.target_network(next_state_tensors).gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values using the Bellman equation
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print("Target network updated at step:", self.steps)

        return loss.item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, self.player_id)

    def save_model(self, path):
        """Saves the model weights to a file."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            'steps': self.steps
        }, path)

    def load_model(self, path):
        """Loads the model weights from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        print("Model loaded successfully.")