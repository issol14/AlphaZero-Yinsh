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
import config
from model import YinshNet
from agent import Agent
from yinshEnv import YinshEnv
from game import Game
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format=' %(message)s')


class YinshDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action_probs, value = self.data[idx]
        
        # 상태를 tensor로 변환
        if isinstance(state, dict):
            # dict에서 board 정보 추출
            board_env = state.get('board')
            if hasattr(board_env, '_state_to_input_array'):
                # YinshEnv 객체에서 배열 생성
                board = board_env._state_to_input_array().squeeze()
            else:
                board = np.zeros(config.INPUT_SHAPE)
        else:
            board = state
            
        state_tensor = torch.tensor(board, dtype=torch.float32)
        
        # 액션 확률을 벡터로 변환
        action_vec = torch.zeros(config.OUTPUT_SHAPE[0])
        if isinstance(action_probs, dict):
            for action_str, prob in action_probs.items():
                # 간단한 해싱으로 액션을 인덱스로 변환
                action_idx = hash(action_str) % config.OUTPUT_SHAPE[0]
                action_vec[action_idx] = prob
        else:
            action_vec = torch.tensor(action_probs, dtype=torch.float32)
        
        value_tensor = torch.tensor([value], dtype=torch.float32)
        return state_tensor, action_vec, value_tensor


class Trainer:
    def __init__(self, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if model is None:
            self.model = YinshNet()
        else:
            self.model = model
            
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()

    def train(self, data, epochs=None):
        """
        Train the model on the provided data
        """
        if not data:
            print("No data provided for training")
            return []
            
        dataset = YinshDataset(data)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        self.model.train()

        if epochs is None:
            epochs = max(5, len(dataset) // config.BATCH_SIZE)
        
        history = []
        print(f"Training for {epochs} epochs with {len(dataset)} samples")
        
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            epoch_losses = []
            for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                self.optimizer.zero_grad()
                pred_policy, pred_value = self.model(states)
                
                # Policy loss (KL divergence)
                loss_policy = self.policy_loss_fn(pred_policy, policy_targets)
                
                # Value loss (MSE)
                loss_value = self.value_loss_fn(pred_value, value_targets)
                
                # Combined loss
                total_loss = 0.5 * loss_policy + 0.5 * loss_value

                total_loss.backward()
                self.optimizer.step()
                
                batch_losses = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_loss": total_loss.item(),
                    "policy_loss": loss_policy.item(),
                    "value_loss": loss_value.item()
                }
                epoch_losses.append(batch_losses)
                history.append(batch_losses)
                
            # Print epoch summary
            avg_total_loss = np.mean([x["total_loss"] for x in epoch_losses])
            avg_policy_loss = np.mean([x["policy_loss"] for x in epoch_losses])
            avg_value_loss = np.mean([x["value_loss"] for x in epoch_losses])
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}")
                
        return history

    def plot_loss(self, history):
        """Plot training losses"""
        if not history:
            print("No history to plot")
            return
            
        df = pd.DataFrame(history)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(df['total_loss'])
        plt.title('Total Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(df['policy_loss'])
        plt.title('Policy Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(df['value_loss'])
        plt.title('Value Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        os.makedirs(config.LOSS_PLOTS_FOLDER, exist_ok=True)
        name = f"loss-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        plt.savefig(os.path.join(config.LOSS_PLOTS_FOLDER, name))
        print(f"Loss plot saved to {os.path.join(config.LOSS_PLOTS_FOLDER, name)}")
        plt.show()

    def save_model(self, path=None):
        """Save the trained model"""
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        if path is None:
            path = os.path.join(config.MODEL_FOLDER, f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        return path


def generate_self_play_data(n_games=10, simulations_per_move=100):
    """
    Generate training data through self-play
    """
    print(f"Generating self-play data with {n_games} games")
    
    # Create two agents
    agent1 = Agent()
    agent2 = Agent()
    
    all_data = []
    
    for game_idx in range(n_games):
        print(f"Playing game {game_idx + 1}/{n_games}")
        
        # Create game environment
        env = YinshEnv()
        game = Game(env, agent1, agent2)
        
        # Play the game and collect data
        try:
            # Play one game with reduced simulations for faster training data generation
            original_sims = config.SIMULATIONS_PER_MOVE
            config.SIMULATIONS_PER_MOVE = simulations_per_move
            
            result = game.play_one_game(stochastic=True)
            
            # Add game data to training data
            if game.memory and len(game.memory) > 0 and len(game.memory[-1]) > 0:
                all_data.extend(game.memory[-1])
                print(f"Game {game_idx + 1} completed. Result: {result}. "
                      f"Generated {len(game.memory[-1])} training samples")
            else:
                print(f"Game {game_idx + 1} generated no data")
                
            # Restore original simulation count
            config.SIMULATIONS_PER_MOVE = original_sims
            
        except Exception as e:
            print(f"Error in game {game_idx + 1}: {e}")
            continue
    
    print(f"Generated {len(all_data)} total training samples from {n_games} games")
    return all_data


def train_from_self_play(n_games=5, training_epochs=10):
    """
    Complete training pipeline: generate data and train model
    """
    print("Starting self-play training pipeline")
    
    # Generate training data through self-play
    training_data = generate_self_play_data(n_games=n_games, simulations_per_move=50)
    
    if not training_data:
        print("No training data generated. Cannot train model.")
        return None
    
    # Train the model
    trainer = Trainer()
    print(f"Training model on {len(training_data)} samples")
    
    history = trainer.train(training_data, epochs=training_epochs)
    
    # Plot training results
    trainer.plot_loss(history)
    
    # Save the trained model
    model_path = trainer.save_model()
    
    print("Training completed successfully!")
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Yinsh AI")
    parser.add_argument("--games", type=int, default=5, help="Number of self-play games")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--test-model", action="store_true", help="Test model creation only")
    
    args = parser.parse_args()
    
    if args.test_model:
        # Test model creation
        print("Testing model creation...")
        model = YinshNet()
        print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        test_input = torch.randn(1, *config.INPUT_SHAPE)
        with torch.no_grad():
            policy, value = model(test_input)
            print(f"Forward pass successful. Policy shape: {policy.shape}, Value shape: {value.shape}")
    else:
        # Full training pipeline
        train_from_self_play(n_games=args.games, training_epochs=args.epochs) 