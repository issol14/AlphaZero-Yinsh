import argparse
import os
import time
from typing import Tuple
import numpy as np
from yinshEnv import YinshEnv
import config
import tensorflow as tf
from keras.models import Model
from keras.models import load_model, save_model
from matplotlib import pyplot as plt
import pandas as pd
import uuid
import utils
from tqdm import tqdm
from datetime import datetime


class Trainer:
    def __init__(self, model: Model):
        self.model = model
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self, data):
        if self.batch_size > len(data):
            return data
        else:
            np.random.shuffle(data)
            return data[:self.batch_size]

    def split_Xy(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into X (input) and y (output) for training
        """
        # Game state to input format
        X = []
        y_probs = []
        y_value = []
        
        for position in data:
            # Convert game state to neural network input
            game_state = position[0]
            board = game_state.get('board', np.zeros((config.BOARD_SIZE, config.BOARD_SIZE, 4)))
            current_player = game_state.get('current_player', 0)
            game_phase = game_state.get('game_phase', 'placement')
            
            # Convert to input format
            input_state = YinshEnv.state_to_input(board, current_player, game_phase)
            X.append(input_state[0])  # Remove batch dimension
            
            # Convert move probabilities to output vector
            moves = utils.moves_to_output_vector(position[1], game_state)
            y_probs.append(moves)
            
            # Game result (winner)
            y_value.append(position[2])
        
        return np.array(X), (np.array(y_probs), np.array(y_value))

    def train_batch(self, X, y_probs, y_value):
        return self.model.train_on_batch(x=X, y={
                "policy_head": y_probs,
                "value_head": y_value
            }, return_dict=True)

    def train_all_data(self, data):
        """
        Train the model on all given data.
        """
        history = []
        np.random.shuffle(data)
        print("Splitting data into features and targets...")
        X, y = self.split_Xy(data)
        print("Training batches...")
        
        for part in tqdm(range(len(X)//self.batch_size)):
            start = part * self.batch_size
            end = start + self.batch_size
            losses = self.train_batch(X[start:end], y[0][start:end], y[1][start:end])
            history.append(losses)
        return history

    def train_random_batches(self, data):
        """
        Train the model on random batches of data
        """
        history = []
        X, (y_probs, y_value) = self.split_Xy(data)
        
        num_batches = max(5, len(data) // self.batch_size) * 2
        
        for _ in tqdm(range(num_batches)):
            indexes = np.random.choice(len(data), size=min(self.batch_size, len(data)), replace=True)
            # only select X values with these indexes
            X_batch = X[indexes]
            y_probs_batch = y_probs[indexes]
            y_value_batch = y_value[indexes]
            
            losses = self.train_batch(X_batch, y_probs_batch, y_value_batch)
            history.append(losses)
        return history

    def plot_loss(self, history):
        df = pd.DataFrame(history)
        df[['loss', 'policy_head_loss', 'value_head_loss']] = df[['loss', 'policy_head_loss', 'value_head_loss']].apply(pd.to_numeric, errors='coerce')
        
        total_loss = df[['loss']].values
        policy_loss = df[['policy_head_loss']].values
        value_loss = df[['value_head_loss']].values
        
        plt.figure(figsize=(10, 6))
        plt.plot(total_loss, label='Total Loss')
        plt.plot(policy_loss, label='Policy Loss')
        plt.plot(value_loss, label='Value Loss')
        plt.legend()
        plt.title(f"Yinsh Training Loss over Time\nLearning rate: {config.LEARNING_RATE}")
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        # Create plots folder if it doesn't exist
        os.makedirs(config.LOSS_PLOTS_FOLDER, exist_ok=True)
        plt.savefig(f"{config.LOSS_PLOTS_FOLDER}/loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")
        plt.close()

    def save_model(self):
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        path = f"{config.MODEL_FOLDER}/model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.h5"
        save_model(self.model, path)
        print(f"Model trained and saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Yinsh model')
    parser.add_argument('--model', type=str, help='The model to train')
    parser.add_argument('--data-folder', type=str, help='The data folder to train on')
    args = parser.parse_args()
    args = vars(args)

    if not args["model"]:
        print("Error: --model argument is required")
        exit(1)
    
    if not args["data_folder"]:
        print("Error: --data-folder argument is required")
        exit(1)

    # Load the model
    try:
        model = load_model(args["model"])
        print(f"Loaded model from {args['model']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
        from rlmodelbuilder import RLModelBuilder
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
    
    trainer = Trainer(model=model)

    # Load training data
    folder = args['data_folder']
    if not os.path.exists(folder):
        print(f"Error: Data folder {folder} does not exist")
        exit(1)
    
    files = os.listdir(folder)
    data = []
    print(f"Loading all games from {folder}...")
    
    for file in files:
        if file.endswith('.npy'):
            try:
                game_data = np.load(f"{folder}/{file}", allow_pickle=True)
                data.append(game_data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    if not data:
        print("No valid training data found!")
        exit(1)
    
    data = np.concatenate(data)
    
    # Analyze data
    print(f"Total positions: {len(data)}")
    winners = [pos[2] for pos in data if pos[2] is not None]
    if winners:
        print(f"Player 1 wins: {len([w for w in winners if w > 0])}")
        print(f"Player 2 wins: {len([w for w in winners if w < 0])}")
        print(f"Draws: {len([w for w in winners if w == 0])}")
    
    # Train the model
    print(f"Training with {len(data)} positions")
    history = trainer.train_random_batches(data)
    
    # Plot and save results
    trainer.plot_loss(history)
    trainer.save_model()
    
    print("Training completed!") 