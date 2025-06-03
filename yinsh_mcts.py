"""
YINSH MCTS Implementation

This module implements the main MCTS algorithm for the YINSH game using AlphaZero-style
neural network guidance. The implementation follows the four key phases:
1. Selection (using PUCT)
2. Expansion (with neural network priors)
3. Evaluation (using neural network value function)
4. Backpropagation (updating statistics)
"""

import time
import random
from typing import Optional, Dict, List, Tuple
from yinsh_mcts.mcts_node import MCTSNode
from yinsh_mcts.neural_agent import NeuralAgent
from yinsh_mcts.game_state import GameState, Action, GamePhase


class YinshGameRules:
    """
    Simplified YINSH game rules for demonstration

    In a full implementation, this would contain all the complex rules of YINSH
    including ring movement, marker placement, line detection, etc.
    """

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """
        Apply an action to a game state and return the new state

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            New game state after applying the action
        """
        new_state = state.copy()

        if action.action_type == "PLACE_RING":
            # Place a ring on the board
            from yinsh_mcts.game_state import Ring

            ring = Ring(action.to_pos, new_state.current_player)
            new_state.rings.append(ring)

            # Update phase and player
            if new_state.phase == GamePhase.WHITE_RING_PLACE:
                new_state.phase = GamePhase.BLACK_RING_PLACE
                new_state.current_player = "black"
            elif new_state.phase == GamePhase.BLACK_RING_PLACE:
                # Check if both players have placed all rings
                white_rings = len([r for r in new_state.rings if r.color == "white"])
                black_rings = len([r for r in new_state.rings if r.color == "black"])

                if white_rings >= 5 and black_rings >= 5:
                    # Move to ring movement phase
                    new_state.phase = GamePhase.WHITE_RING_MOVE
                    new_state.current_player = "white"
                else:
                    new_state.phase = GamePhase.WHITE_RING_PLACE
                    new_state.current_player = "white"

        elif action.action_type == "MOVE_RING":
            # Move a ring and place markers along the path
            # Find and move the ring
            for i, ring in enumerate(new_state.rings):
                if (
                    ring.position == action.from_pos
                    and ring.color == new_state.current_player
                ):
                    new_state.rings[i].position = action.to_pos
                    break

            # In a real implementation, place markers along the movement path
            # and flip existing markers

            # Switch to opponent
            if new_state.current_player == "white":
                new_state.current_player = "black"
                new_state.phase = GamePhase.BLACK_RING_MOVE
            else:
                new_state.current_player = "white"
                new_state.phase = GamePhase.WHITE_RING_MOVE

        # Add logic for marker removal and ring removal phases
        # This is simplified for demonstration

        return new_state

    def is_terminal(self, state: GameState) -> bool:
        """Check if the game is in a terminal state"""
        return state.is_terminal()

    def get_winner(self, state: GameState) -> Optional[str]:
        """Get the winner of the game if it's terminal"""
        return state.get_winner()


class YinshMCTS:
    """
    Monte Carlo Tree Search implementation for YINSH game

    Uses AlphaZero-style MCTS with neural network guidance for action selection
    and position evaluation.
    """

    def __init__(
        self,
        neural_agent: NeuralAgent,
        c_puct: float = 1.0,
        simulation_timeout: float = 1.0,
        max_simulations: int = 1000,
    ):
        """
        Initialize YINSH MCTS

        Args:
            neural_agent: Neural network agent for policy and value predictions
            c_puct: Exploration constant for PUCT algorithm
            simulation_timeout: Maximum time to spend on simulations
            max_simulations: Maximum number of simulations to run
        """
        self.neural_agent = neural_agent
        self.c_puct = c_puct
        self.simulation_timeout = simulation_timeout
        self.max_simulations = max_simulations
        self.game_rules = YinshGameRules()

        # Statistics for analysis
        self.total_simulations = 0
        self.total_time = 0.0

    def search(
        self, root_state: GameState, num_simulations: Optional[int] = None
    ) -> MCTSNode:
        """
        Perform MCTS search from the given root state

        Args:
            root_state: Starting game state for the search
            num_simulations: Number of simulations to run (overrides default)

        Returns:
            Root node of the search tree with statistics
        """
        start_time = time.time()
        simulations = num_simulations or self.max_simulations

        # Create root node
        root = MCTSNode(root_state)

        # Run simulations
        for i in range(simulations):
            # Check timeout
            if time.time() - start_time > self.simulation_timeout:
                print(f"MCTS timeout after {i} simulations")
                break

            # Run one simulation
            self._simulate(root)
            self.total_simulations += 1

        self.total_time += time.time() - start_time
        return root

    def _simulate(self, root: MCTSNode) -> None:
        """
        Run one MCTS simulation starting from the root

        This implements the four phases of MCTS:
        1. Selection: Traverse tree using PUCT to find leaf
        2. Expansion: Add children to leaf node
        3. Evaluation: Get value estimate from neural network
        4. Backpropagation: Update statistics up the tree
        """
        # Phase 1: Selection - find the leaf node to expand
        path = []
        current = root

        while not current.is_leaf() and not current.is_terminal():
            current = current.select_child(self.c_puct)
            path.append(current)

        # Phase 2: Expansion - add children if not terminal
        if not current.is_terminal():
            self._expand(current)

            # If expansion created children, select one for evaluation
            if current.children:
                current = current.select_child(self.c_puct)
                path.append(current)

        # Phase 3: Evaluation - get value estimate
        if current.is_terminal():
            # Terminal node: use actual game result
            winner = self.game_rules.get_winner(current.state)
            if winner is None:
                value = 0.0  # Draw
            elif winner == current.state.current_player:
                value = 1.0  # Win for current player
            else:
                value = -1.0  # Loss for current player
        else:
            # Non-terminal node: use neural network evaluation
            _, value = self.neural_agent.predict(current.state)

        # Phase 4: Backpropagation - update statistics
        current.backup(value)

    def _expand(self, node: MCTSNode) -> None:
        """
        Expand a node by adding all legal actions as children

        Args:
            node: Node to expand
        """
        if node.is_expanded:
            return

        # Get policy predictions from neural network
        action_probs, _ = self.neural_agent.predict(node.state)

        # Convert to action objects with probabilities
        action_prob_pairs = {}
        for action_str, prob in action_probs.items():
            action = self._string_to_action(action_str)
            action_prob_pairs[action_str] = (action, prob)

        # Expand the node
        node.expand(action_prob_pairs, self.game_rules)

    def get_best_action(
        self, root: MCTSNode, temperature: float = 0.0
    ) -> Optional[Action]:
        """
        Get the best action from the root node based on visit counts

        Args:
            root: Root node of the search tree
            temperature: Temperature for action selection
                        0.0 = deterministic (most visited)
                        >0.0 = stochastic (proportional to visits)

        Returns:
            Best action to take, or None if no actions available
        """
        if not root.children:
            return None

        action_probs = root.get_action_probs(temperature)

        if temperature == 0.0:
            # Deterministic: pick most visited action
            best_action_str = root.get_best_action()
            if best_action_str:
                return root.children[best_action_str].action_taken
        else:
            # Stochastic: sample based on probabilities
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            chosen_action_str = random.choices(actions, weights=probs)[0]
            return root.children[chosen_action_str].action_taken

        return None

    def get_move_probabilities(self, root: MCTSNode) -> Dict[str, float]:
        """
        Get action probabilities based on visit counts

        Args:
            root: Root node of the search tree

        Returns:
            Dictionary mapping action strings to probabilities
        """
        return root.get_action_probs(temperature=1.0)

    def get_principal_variation(
        self, root: MCTSNode, max_depth: int = 10
    ) -> List[Action]:
        """
        Get the principal variation (most likely sequence of moves)

        Args:
            root: Root node of the search tree
            max_depth: Maximum depth to traverse

        Returns:
            List of actions in the principal variation
        """
        return root.get_pv(max_depth)

    def _string_to_action(self, action_str: str) -> Action:
        """
        Convert string representation back to Action object

        Args:
            action_str: String representation of action

        Returns:
            Action object
        """
        parts = action_str.split("_")

        if parts[0] == "PLACE" and parts[1] == "RING":
            return Action(
                action_type="PLACE_RING", to_pos=(int(parts[2]), int(parts[3]))
            )
        elif parts[0] == "MOVE" and parts[1] == "RING":
            return Action(
                action_type="MOVE_RING",
                from_pos=(int(parts[2]), int(parts[3])),
                to_pos=(int(parts[4]), int(parts[5])),
            )
        # Add other action types as needed

        # Fallback
        return Action(action_type="UNKNOWN")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the MCTS performance

        Returns:
            Dictionary containing performance statistics
        """
        return {
            "total_simulations": self.total_simulations,
            "total_time": self.total_time,
            "simulations_per_second": self.total_simulations
            / max(self.total_time, 1e-6),
            "c_puct": self.c_puct,
            "max_simulations": self.max_simulations,
            "simulation_timeout": self.simulation_timeout,
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics"""
        self.total_simulations = 0
        self.total_time = 0.0

    def __repr__(self) -> str:
        return f"YinshMCTS(c_puct={self.c_puct}, max_sims={self.max_simulations})"
