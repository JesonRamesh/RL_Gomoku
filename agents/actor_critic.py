import torch 
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size

        # CNN
        # Input: (batch, 1, board_size, board_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)   # 8 groups, 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)

        self.flatten = nn.Flatten()

        # Policy head 
        self.policy_h = nn.Linear(64*board_size*board_size, board_size*board_size)

        # Value Head
        self.val1 = nn.Linear(64*board_size*board_size, 128)
        self.val2 = nn.Linear(128,1)

    def forward(self, x, legal_mask):

        # x shape: (batch, 1, board_size, board_size)

        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        flat = self.flatten(x)


        # Policy Head
        logits = self.policy_h(flat)

        # Masking 

        logits = logits.masked_fill(legal_mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)

        # Vlaue Head
        V = F.relu(self.val1(flat))
        value = self.val2(V)

        return probs, value
        
    
    
    

