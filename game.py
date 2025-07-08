# game.py
from yinshEnv import YinshEnv
from neural_agent import Agent #mcts에서 사용된 agent
import numpy as np
import uuid
import os
import config
import logging

class Game:
    def __init__(self, env: YinshEnv, white: Agent, black: Agent):
        self.env = env
        self.white = white
        self.black = black
        self.memory = []

    def reset(self):
        self.env.reset()

    def play_one_game(self, stochastic: bool = True) -> int:
        self.reset()
        self.memory.append([])
        while not self.env.done:
            current_player = self.white if self.env.current_player == 1 else self.black
            action = current_player.select_action(self.env, stochastic=stochastic)
            if action is None:
                logging.warning("No valid action. Game ends.")
                break
            self.save_to_memory(self.env.get_state_string(), current_player.mcts.root.edges)
            self.env.step(action)
        winner = self.env.get_winner()
        for i in range(len(self.memory[-1])):
            self.memory[-1][i] = (self.memory[-1][i][0], self.memory[-1][i][1], winner)
        self.save_game("yinsh_game")
        return winner

    def save_to_memory(self, state_string: str, edges: list) -> None:
        if not edges:
            return
        sum_visits = sum(e.N for e in edges)
        search_probs = {str(e.action): e.N / sum_visits for e in edges}
        self.memory[-1].append((state_string, search_probs, None))

    def save_game(self, name: str = "yinsh_game") -> None:
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        path = os.path.join(config.MEMORY_DIR, f"{game_id}.npy")
        np.save(path, self.memory[-1])
        logging.info(f"🎮 Game saved to {path}")
