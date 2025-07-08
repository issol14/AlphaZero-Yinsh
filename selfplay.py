import time
import numpy as np
from typing import List, Tuple, Dict
from yinsh_mcts.mcts import YinshMCTS
from yinsh_mcts.game_state import GameState
from yinsh_env import YinshEnv, YinshAction, Color
import config

def self_play_game(mcts: YinshMCTS, initial_state: GameState, temperature: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    data = []
    env = YinshEnv()
    env.load_from_state_string(initial_state.get_state_string())

    while not env.done:
        root_node = mcts.search(env)
        action_probs = root_node.get_action_probs(temperature)
        policy_vector = np.zeros(config.policy_output_size, dtype=np.float32)
        for action_str, prob in action_probs.items():
            action = mcts._string_to_action(action_str)
            index = hash(action) % config.policy_output_size
            policy_vector[index] = prob
        state_tensor = env._state_to_input_array().squeeze()
        data.append((state_tensor, policy_vector, env.current_player))
        best_action = mcts.get_best_action(root_node, temperature)
        if best_action is None:
            break
        env.step(best_action)

    winner = env.get_winner()
    finalized_data = []
    for state_tensor, policy_vector, player in data:
        result = 1.0 if winner == player else -1.0 if winner != 0 else 0.0
        finalized_data.append((state_tensor, policy_vector, result))
    return finalized_data
