import logging
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
import utils
from tqdm import tqdm
from mcts import MCTS
import json
import numpy as np
import chess

import os
from dotenv import load_dotenv

load_dotenv()


class Agent:
    def __init__(
        self, local_predictions: bool = True, model_path=None, state=chess.STARTING_FEN
    ):
        """
        An agent is an object that can play chess moves on the environment.
        It uses a local model to make predictions and holds an MCTS object
        that is used to run MCTS simulations to build a tree.
        """
        if model_path is not None:
            logging.info("Using local predictions")
            from tensorflow.python.ops.numpy_ops import np_config
            import tensorflow as tf
            from tensorflow.keras.models import load_model

            self.model = load_model(model_path)
            np_config.enable_numpy_behavior()
        else:
            logging.info("Building new model")
            self.model = self.build_model()

        self.mcts = MCTS(self, state=state)

    def build_model(self) -> Model:
        """
        Build a new model based on the configuration in config.py
        """
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
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    def predict(self, data):
        """
        Predict using the local model
        """
        # use tf.function
        import local_prediction

        p, v = local_prediction.predict_local(self.model, data)
        return p.numpy(), v[0][0]
