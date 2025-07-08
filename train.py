import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# ==== 설정 ====
BATCH_SIZE = 32
LEARNING_RATE = 0.001
AMOUNT_OF_RESIDUAL_BLOCKS = 5
CONVOLUTION_FILTERS = 64
INPUT_SHAPE = (11, 11, 11)
POLICY_OUTPUT_SIZE = 200
MODEL_FOLDER = "models"
LOSS_PLOTS_FOLDER = "plots"

class YinshDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        state, action, value = self.data[idx]
        state = torch.tensor(state, dtype=torch.float32)
        action_vec = torch.zeros(POLICY_OUTPUT_SIZE)
        action_vec[hash(action) % POLICY_OUTPUT_SIZE] = 1.0
        value = torch.tensor([value], dtype=torch.float32)
        return state, action_vec, value

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()

    def train(self, data):
        dataset = YinshDataset(data)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        history = []
        for epoch in tqdm(range(2 * max(5, len(dataset) // BATCH_SIZE)), desc="Training batches"):
            for states, policy_targets, value_targets in dataloader:
                states = states.to(torch.device("cpu"))
                policy_targets = policy_targets.to(torch.device("cpu"))
                value_targets = value_targets.to(torch.device("cpu"))
                self.optimizer.zero_grad()
                pred_policy, pred_value = self.model(states)
                loss_policy = self.policy_loss_fn(pred_policy, policy_targets)
                loss_value = self.value_loss_fn(pred_value, value_targets)
                loss = 0.5 * loss_policy + 0.5 * loss_value
                loss.backward()
                self.optimizer.step()
                history.append({
                    "total_loss": loss.item(),
                    "policy_loss": loss_policy.item(),
                    "value_loss": loss_value.item()
                })
        return history

    def plot_loss(self, history):
        df = pd.DataFrame(history)
        plt.plot(df['total_loss'], label='total_loss')
        plt.plot(df['policy_loss'], label='policy_loss')
        plt.plot(df['value_loss'], label='value_loss')
        plt.legend()
        plt.title(f"Loss over time (lr={LEARNING_RATE})")
        os.makedirs(LOSS_PLOTS_FOLDER, exist_ok=True)
        name = f"loss-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        plt.savefig(os.path.join(LOSS_PLOTS_FOLDER, name))
        print(" Loss plot saved.")

    def save_model(self):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        path = os.path.join(MODEL_FOLDER, f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
        torch.save(self.model.state_dict(), path)
        print(f" Model saved to {path}")
