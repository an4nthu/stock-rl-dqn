import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from stock_env import StockTradingEnv
import torch
from models.dqn import DQN

# Load environment
data = pd.read_csv('../data/preprocessed_data.csv')
env = StockTradingEnv(data)
state = env.reset()

# Input size (length of state)
input_size = len(state)

print("Initial state vector:", state)
print("Input size (len of state):", len(state))

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model instantiation
model = DQN(input_size).to(device)
print(model)
