# models/dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()

        # First fully connected layer (input → 64 hidden units)
        self.fc1 = nn.Linear(input_size, 64)

        # Second hidden layer (64 → 64)
        self.fc2 = nn.Linear(64, 64)

        # Output layer (64 → 3 actions: Buy, Sell, Hold)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation after first layer
        x = F.relu(self.fc2(x))  # Activation after second layer
        return self.out(x)       # No activation here — outputs raw Q-values
