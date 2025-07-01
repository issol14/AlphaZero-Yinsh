import logging
import config
import time
import utils
from tqdm import tqdm
from mcts import MCTS
import numpy as np
import torch
import os
from dotenv import load_dotenv
from model import YinshNet

load_dotenv()


class Agent:
    def __init__(self, model_path=None):
        """
        An agent that can play Yinsh moves.
        Uses PyTorch model predictions.
        It holds an MCTS object that is used to run MCTS simulations to build a tree.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if model_path is not None and os.path.exists(model_path):
            logging.info(f"Loading PyTorch model from {model_path}")
            self.model = YinshNet()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            logging.info("Creating new model")
            self.model = YinshNet()
            self.model.to(self.device)

        # Don't initialize MCTS here - it will be created fresh for each move
        self.mcts = None
        
    def run_simulations(self, n: int = 1):
        """
        Run n simulations of the MCTS algorithm. This function gets called every move.
        """
        if self.mcts is None:
            raise ValueError("MCTS not initialized. Call create_mcts first.")
        print(f"Running {n} simulations...")
        self.mcts.run_simulations(n)

    def save_model(self, timestamped: bool = False):
        """
        Save the current model to a file
        """
        if self.model is None:
            logging.warning("No model to save")
            return
            
        if not os.path.exists(config.MODEL_FOLDER):
            os.makedirs(config.MODEL_FOLDER)
            
        if timestamped:
            filename = f"{config.MODEL_FOLDER}/model-{int(time.time())}.pt"
        else:
            filename = f"{config.MODEL_FOLDER}/model.pt"
            
        torch.save(self.model.state_dict(), filename)
        logging.info(f"Model saved to {filename}")

    def predict(self, data):
        """
        Predict using the PyTorch model
        """
        if self.model is None:
            # Return random predictions if no model
            batch_size = data.shape[0]
            p = np.random.random((batch_size, config.OUTPUT_SHAPE[0]))
            v = np.random.random((batch_size, 1)) * 2 - 1  # Random value between -1 and 1
            return p[0], v[0][0]
        
        # Convert numpy to tensor
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            policy, value = self.model(data)
            
            # Convert back to numpy
            policy = torch.exp(policy)  # Convert log_softmax back to probabilities
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            
            return policy[0], value[0][0] 