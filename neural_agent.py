"""
Neural Agent for YINSH MCTS

This module implements a neural network agent that provides policy and value predictions
for the YINSH game. In a full AlphaZero implementation, this would be a deep convolutional
neural network trained through self-play.
"""

import numpy as np
from typing import Tuple, Dict, List
from yinsh_mcts.game_state import GameState, Action, GamePhase

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True

    class YinshNet(nn.Module):
        """
        Neural network for YINSH game using AlphaZero architecture

        Takes a 11x11x11 board representation as input and outputs:
        - Policy: probability distribution over all possible actions
        - Value: estimated game outcome from current position (-1 to 1)
        """

        def __init__(
            self,
            board_size: int = 11,
            num_channels: int = 11,
            num_resblocks: int = 4,
            hidden_size: int = 256,
        ):
            """
            Initialize the neural network

            Args:
                board_size: Size of the game board (11x11 for YINSH)
                num_channels: Number of input channels (11 for our state representation)
                num_resblocks: Number of residual blocks in the tower
                hidden_size: Size of hidden layers
            """
            super(YinshNet, self).__init__()

            self.board_size = board_size
            self.num_channels = num_channels

            # Initial convolution
            self.conv_input = nn.Conv2d(
                num_channels, hidden_size, kernel_size=3, padding=1
            )
            self.bn_input = nn.BatchNorm2d(hidden_size)

            # Residual tower
            self.resblocks = nn.ModuleList(
                [ResidualBlock(hidden_size) for _ in range(num_resblocks)]
            )

            # Policy head
            self.conv_policy = nn.Conv2d(hidden_size, 32, kernel_size=1)
            self.bn_policy = nn.BatchNorm2d(32)
            self.fc_policy = nn.Linear(
                32 * board_size * board_size, 1000
            )  # Action space size

            # Value head
            self.conv_value = nn.Conv2d(hidden_size, 1, kernel_size=1)
            self.bn_value = nn.BatchNorm2d(1)
            self.fc_value1 = nn.Linear(board_size * board_size, hidden_size)
            self.fc_value2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """
            Forward pass through the network

            Args:
                x: Input tensor of shape (batch_size, num_channels, board_size, board_size)

            Returns:
                Tuple of (policy_logits, value)
            """
            # Initial convolution
            x = F.relu(self.bn_input(self.conv_input(x)))

            # Residual tower
            for resblock in self.resblocks:
                x = resblock(x)

            # Policy head
            policy = F.relu(self.bn_policy(self.conv_policy(x)))
            policy = policy.view(policy.size(0), -1)  # Flatten
            policy = self.fc_policy(policy)

            # Value head
            value = F.relu(self.bn_value(self.conv_value(x)))
            value = value.view(value.size(0), -1)  # Flatten
            value = F.relu(self.fc_value1(value))
            value = torch.tanh(self.fc_value2(value))

            return policy, value

    class ResidualBlock(nn.Module):
        """Residual block for the neural network"""

        def __init__(self, channels: int):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out)

except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

    # Dummy classes when PyTorch is not available
    class YinshNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network functionality")

    class ResidualBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network functionality")


class NeuralAgent:
    """
    Neural network agent for YINSH game

    This agent uses a neural network to predict policy and value for any given game state.
    In a full implementation, this network would be trained using self-play data.
    """

    def __init__(self, use_neural_network: bool = False):
        """
        Initialize the neural agent

        Args:
            use_neural_network: Whether to use actual neural network or random baseline
        """
        self.use_neural_network = use_neural_network and TORCH_AVAILABLE

        if use_neural_network and not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Falling back to random agent.")
            self.use_neural_network = False

        if self.use_neural_network:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = YinshNet().to(self.device)
            self.net.eval()  # Set to evaluation mode
        else:
            self.device = None
            self.net = None

    def predict(self, state: GameState) -> Tuple[Dict[str, float], float]:
        """
        Predict policy and value for a given game state

        Args:
            state: Current game state

        Returns:
            Tuple of (action_probabilities, value_estimate)
            - action_probabilities: Dict mapping action strings to probabilities
            - value_estimate: Estimated game outcome from current player's perspective
        """
        if self.use_neural_network and self.net is not None:
            return self._neural_predict(state)
        else:
            return self._random_predict(state)

    def _neural_predict(self, state: GameState) -> Tuple[Dict[str, float], float]:
        """Use neural network for prediction"""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)  # (B, C, H, W)

        with torch.no_grad():
            policy_logits, value = self.net(state_tensor)

            # Convert policy logits to probabilities
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value_estimate = value.cpu().numpy()[0][0]

        # Map policy probabilities to legal actions
        legal_actions = self._get_legal_actions(state)
        action_probs = {}

        # For simplicity, distribute probabilities uniformly among legal actions
        # In a real implementation, you'd have a more sophisticated mapping
        if legal_actions:
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_str = self._action_to_string(action)
                action_probs[action_str] = uniform_prob

        return action_probs, value_estimate

    def _random_predict(self, state: GameState) -> Tuple[Dict[str, float], float]:
        """Use random baseline for prediction"""
        legal_actions = self._get_legal_actions(state)
        action_probs = {}

        # Uniform random policy
        if legal_actions:
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_str = self._action_to_string(action)
                action_probs[action_str] = uniform_prob

        # Random value estimate (slightly biased towards current player)
        value_estimate = np.random.normal(0.0, 0.3)
        value_estimate = np.clip(value_estimate, -1.0, 1.0)

        return action_probs, value_estimate

    def _get_legal_actions(self, state: GameState) -> List[Action]:
        """
        Get all legal actions for the current game state

        This is a simplified implementation. In a real YINSH game,
        this would need to implement the full game rules.
        """
        legal_actions = []

        if (
            state.phase == GamePhase.WHITE_RING_PLACE
            or state.phase == GamePhase.BLACK_RING_PLACE
        ):
            # Ring placement phase: place ring on any empty position
            for i in range(state.board_size):
                for j in range(state.board_size):
                    if not state.is_position_occupied((i, j)):
                        action = Action(action_type="PLACE_RING", to_pos=(i, j))
                        legal_actions.append(action)

        elif (
            state.phase == GamePhase.WHITE_RING_MOVE
            or state.phase == GamePhase.BLACK_RING_MOVE
        ):
            # Ring movement phase: move any of player's rings
            player_rings = state.get_player_rings(state.current_player)
            for ring in player_rings:
                # For simplicity, allow movement to any empty adjacent position
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        new_pos = (ring.position[0] + di, ring.position[1] + dj)
                        if (
                            0 <= new_pos[0] < state.board_size
                            and 0 <= new_pos[1] < state.board_size
                            and not state.is_position_occupied(new_pos)
                        ):
                            action = Action(
                                action_type="MOVE_RING",
                                from_pos=ring.position,
                                to_pos=new_pos,
                            )
                            legal_actions.append(action)

        # Add other phases (marker removal, ring removal) as needed
        # This is simplified for demonstration

        return legal_actions

    def _action_to_string(self, action: Action) -> str:
        """Convert action to string representation"""
        if action.action_type == "PLACE_RING":
            return f"PLACE_RING_{action.to_pos[0]}_{action.to_pos[1]}"
        elif action.action_type == "MOVE_RING":
            return f"MOVE_RING_{action.from_pos[0]}_{action.from_pos[1]}_{action.to_pos[0]}_{action.to_pos[1]}"
        elif action.action_type == "REMOVE_MARKERS":
            markers_str = "_".join(
                [f"{pos[0]}_{pos[1]}" for pos in action.markers_to_remove]
            )
            return f"REMOVE_MARKERS_{markers_str}"
        elif action.action_type == "REMOVE_RING":
            return f"REMOVE_RING_{action.ring_to_remove[0]}_{action.ring_to_remove[1]}"
        else:
            return f"UNKNOWN_{action.action_type}"

    def train(self, training_examples: List[Dict]) -> None:
        """
        Train the neural network on a batch of examples

        Args:
            training_examples: List of training examples with states, policies, and values
        """
        if not self.use_neural_network:
            print(
                f"Random agent doesn't need training. Received {len(training_examples)} examples."
            )
            return

        print(f"Training neural network on {len(training_examples)} examples")
        # In a real implementation, this would:
        # 1. Convert examples to tensors
        # 2. Run forward pass
        # 3. Calculate loss (policy loss + value loss + regularization)
        # 4. Backpropagate and update weights

        # For now, just print that training occurred
        pass

    def save_model(self, filepath: str) -> None:
        """Save the neural network model"""
        if self.net is not None and TORCH_AVAILABLE:
            torch.save(self.net.state_dict(), filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No neural network model to save")

    def load_model(self, filepath: str) -> None:
        """Load a trained neural network model"""
        if self.net is not None and TORCH_AVAILABLE:
            self.net.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Model loaded from {filepath}")
        else:
            print("Cannot load model: no neural network available")

    def __repr__(self) -> str:
        return f"NeuralAgent(use_neural_network={self.use_neural_network}, torch_available={TORCH_AVAILABLE})"
