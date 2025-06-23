import logging
import config
from keras.models import Model
import time
import utils
from tqdm import tqdm
from mcts import MCTS
import numpy as np

import os
from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, model_path=None, state=None):
        """
        An agent that can play Yinsh moves.
        Uses local model predictions only (no server).
        It holds an MCTS object that is used to run MCTS simulations to build a tree.
        """
        self.local_predictions = True
        
        if model_path is not None:
            logging.info("Loading local model")
            from tensorflow.python.ops.numpy_ops import np_config
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            np_config.enable_numpy_behavior()
        else:
            logging.info("No model path provided, will build new model")
            self.model = None

        # Initialize default state if none provided
        if state is None:
            state = {
                'board': np.zeros((config.BOARD_SIZE, config.BOARD_SIZE, 4)),
                'current_player': 0,
                'game_phase': 'placement',
                'rings_placed': [0, 0],
                'rings_removed': [0, 0],
                'move_count': 0
            }

        self.mcts = MCTS(self, state=state)
        

    def build_model(self) -> Model:
        """
        Build a new model based on the configuration in config.py
        """
        from rlmodelbuilder import RLModelBuilder
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model

    def run_simulations(self, n: int = 1):
        """
        Run n simulations of the MCTS algorithm. This function gets called every move.
        """
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
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    def predict(self, data):
        """
        Predict using the local model
        """
        if self.model is None:
            # Return random predictions if no model
            batch_size = data.shape[0]
            p = np.random.random((batch_size, config.OUTPUT_SHAPE[0]))
            v = np.random.random((batch_size, 1)) * 2 - 1  # Random value between -1 and 1
            return p[0], v[0][0]
        
        # Use tf.function for optimization
        import local_prediction
        p, v = local_prediction.predict_local(self.model, data)
        return p.numpy(), v[0][0] 