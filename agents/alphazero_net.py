import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Single residual block: conv→BN→ReLU→conv→BN + skip→ReLU."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """
    Combined policy + value network.

    Input:   (batch, 3, board_size, board_size)
    Outputs:
        policy: (batch, board_size²) — softmax move probabilities
        value:  (batch, 1)           — tanh win estimate in [-1, +1]
    """

    def __init__(self, board_size: int = 9, num_filters: int = 128, num_res_blocks: int = 6):
        super().__init__()
        self.board_size = board_size

        # Input block: maps 3 input channels to num_filters feature maps
        self.input_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head: 1x1 conv reduces channels before FC to 81 logits
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head: 1x1 conv → two FC layers → scalar
        self.value_conv  = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        # Shared body
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))

        return p, v

    def save_weights(self, path: str):
        """Save state_dict only (~7.5 MB). No optimizer or target network."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: torch.device):
        """Load state_dict from path and move network to device."""
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)


def preprocess_board(board: np.ndarray, current_player: int) -> np.ndarray:
    board_size = board.shape[0]
    state = np.zeros((3, board_size, board_size), dtype=np.float32)
    state[0] = (board == current_player).astype(np.float32)
    state[1] = (board == -current_player).astype(np.float32)
    state[2] = np.full((board_size, board_size), current_player, dtype=np.float32)
    return state
