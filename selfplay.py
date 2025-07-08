{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3787e0d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import numpy as np\n",
    "from typing import List, Tuple, Dict\n",
    "from yinsh_mcts.mcts import YinshMCTS\n",
    "from yinsh_mcts.game_state import GameState\n",
    "from yinsh_env import YinshEnv, YinshAction, Color\n",
    "import config\n",
    "\n",
    "\n",
    "def self_play_game(mcts: YinshMCTS, initial_state: GameState, temperature: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, float]]:\n",
    "    \"\"\"\n",
    "    Play one self-play game using MCTS and collect training data.\n",
    "\n",
    "    Args:\n",
    "        mcts: Monte Carlo Tree Search object\n",
    "        initial_state: Starting GameState\n",
    "        temperature: Softmax temperature\n",
    "\n",
    "    Returns:\n",
    "        A list of (state_tensor, policy_vector, value) tuples\n",
    "    \"\"\"\n",
    "    data = []\n",
    "    env = YinshEnv()\n",
    "    env.load_from_state_string(initial_state.get_state_string())\n",
    "\n",
    "    while not env.done:\n",
    "        # MCTS 탐색\n",
    "        root_node = mcts.search(env)\n",
    "        action_probs = root_node.get_action_probs(temperature)\n",
    "\n",
    "        # 정책 벡터 생성\n",
    "        policy_vector = np.zeros(config.policy_output_size, dtype=np.float32)\n",
    "        for action_str, prob in action_probs.items():\n",
    "            action = mcts._string_to_action(action_str)\n",
    "            index = hash(action) % config.policy_output_size\n",
    "            policy_vector[index] = prob\n",
    "\n",
    "        # 상태 텐서 저장\n",
    "        state_tensor = env._state_to_input_array().squeeze()  # (C, H, W)\n",
    "        data.append((state_tensor, policy_vector, env.current_player))\n",
    "\n",
    "        # 액션 선택 및 실행\n",
    "        best_action = mcts.get_best_action(root_node, temperature)\n",
    "        if best_action is None:\n",
    "            break\n",
    "        env.step(best_action)\n",
    "\n",
    "    # 게임 종료 후 결과 반영\n",
    "    winner = env.get_winner()\n",
    "    finalized_data = []\n",
    "    for state_tensor, policy_vector, player in data:\n",
    "        result = 1.0 if winner == player else -1.0 if winner != 0 else 0.0\n",
    "        finalized_data.append((state_tensor, policy_vector, result))\n",
    "\n",
    "    return finalized_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76a038a7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
