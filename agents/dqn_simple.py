import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from agents.base_agent import BaseAgent


class DQNetwork(nn.Module):
    """
    Simple CNN architecture (proven to achieve 100% win rate).
    """
    def __init__(self, board_size=9):
        super(DQNetwork, self).__init__()
        self.board_size = board_size
        
        # 3 convolutional layers with batch norm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        return self.fc2(x)


def preprocess_board(board, current_player):
    """
    Preprocess the board state into a 3-channel tensor for the DQN.
    
    Channel 0: Current player's pieces
    Channel 1: Opponent's pieces
    Channel 2: Current player ID (constant)
    """
    board_size = board.shape[0]
    state = np.zeros((3, board_size, board_size), dtype=np.float32)
    
    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == -current_player).astype(np.float32)
    state[2] = np.ones((board_size, board_size), dtype=np.float32) * current_player
    
    return state  # Return numpy array


class ReplayBuffer:
    def __init__(self, capacity=100000):
        """Replay buffer that stores player_id with each experience."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, player_id):
        """Store experience with the player_id at time of action."""
        self.buffer.append((state, action, reward, next_state, done, player_id))

    def sample(self, batch_size):
        """Sample experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, player_ids = zip(*batch)
        return states, actions, rewards, next_states, dones, player_ids

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(self, player_id, board_size=9, learning_rate=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, 
                 buffer_capacity=100000, target_update_frequency=1000):
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

        # Initialize optimizer
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

        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def predict(self, board_state):
        """Selects an action using epsilon-greedy policy."""
        valid_moves = list(zip(*np.where(board_state == 0)))

        if len(valid_moves) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            return self._get_best_action(board_state, valid_moves)
        
    def _get_best_action(self, board_state, valid_moves):
        """Gets the best action based on Q-values for valid moves."""
        state_tensor = torch.FloatTensor(
            preprocess_board(board_state, self.player_id)
        ).unsqueeze(0).to(self.device)
        
        self.q_network.eval()  # Set to eval mode
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()
        self.q_network.train()  # Back to train mode
        
        # Mask invalid moves
        q_values_np = q_values.cpu().numpy().reshape(self.board_size, self.board_size)
        q_values_np[board_state != 0] = -float('inf')
        
        # Select best valid move
        best_move = np.unravel_index(np.argmax(q_values_np), q_values_np.shape)
        return best_move
    
    def decay_epsilon(self):
        """Decays the exploration rate epsilon after each episode."""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def train_step(self, batch_size):
        """Performs a training step using a batch of experiences from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones, player_ids = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states_list = [preprocess_board(s, pid) for s, pid in zip(states, player_ids)]
        next_states_list = [preprocess_board(ns, pid) for ns, pid in zip(next_states, player_ids)]
        
        states_tensor = torch.FloatTensor(np.array(states_list)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states_list)).to(self.device)
        
        # Convert actions to indices
        action_indices = torch.LongTensor([a[0] * self.board_size + a[1] for a in actions]).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Get next Q-values using Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states_tensor).max(1)[1]
            next_q_values = self.target_network(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.steps}")

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "q_network_state_dict" in checkpoint:
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            if "target_network_state_dict" in checkpoint:
                self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
            if "steps" in checkpoint:
                self.steps = checkpoint["steps"]
        else:
            # Raw state_dict (weights-only save)
            self.q_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)
        print("Model loaded successfully.")