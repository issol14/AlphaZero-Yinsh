"""
MCTS Node Implementation for YINSH

This module implements the MCTS node structure with PUCT (Polynomial Upper Confidence Trees)
algorithm as used in AlphaZero. Each node represents a game state and maintains statistics
for action selection and backpropagation.
"""

import math
import numpy as np
from typing import Dict, Optional, List, Tuple
from yinsh_mcts.game_state import GameState, Action, PlayerColor


class MCTSNode:
    """
    A node in the MCTS tree representing a game state

    Uses PUCT (Polynomial Upper Confidence Trees) for action selection,
    which combines the UCB1 formula with prior probabilities from a neural network.
    """

    def __init__(
        self,
        state: GameState,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[Action] = None,
        prior_prob: float = 0.0,
    ):
        """
        Initialize a new MCTS node

        Args:
            state: The game state this node represents
            parent: Parent node in the tree (None for root)
            action_taken: The action that led to this state
            prior_prob: Prior probability from neural network policy
        """
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob

        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[str, MCTSNode] = {}

        # Track if this node has been expanded
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state"""
        return self.state.is_terminal()

    def get_value(self) -> float:
        """Get the average value of this node"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def get_puct_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate PUCT score for action selection

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant that controls the balance between
                   exploitation and exploration

        Returns:
            PUCT score for this node
        """
        if self.visits == 0:
            # Unvisited nodes get infinite score to ensure exploration
            return float("inf")

        # Q(s,a): Average value of this node
        q_value = self.get_value()

        # Exploration term with prior probability
        if self.parent is not None:
            exploration = (
                c_puct
                * self.prior_prob
                * math.sqrt(self.parent.visits)
                / (1 + self.visits)
            )
        else:
            exploration = 0.0

        return q_value + exploration

    def select_child(self, c_puct: float = 1.0) -> "MCTSNode":
        """
        Select the child with the highest PUCT score

        Args:
            c_puct: Exploration constant

        Returns:
            Child node with highest PUCT score
        """
        if not self.children:
            raise ValueError("Cannot select child from leaf node")

        best_child = None
        best_score = float("-inf")

        for child in self.children.values():
            score = child.get_puct_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, action_probs: Dict[str, Tuple[Action, float]], game_rules) -> None:
        """
        Expand this node by adding children for all legal actions

        Args:
            action_probs: Dictionary mapping action strings to (Action, probability) pairs
            game_rules: Game rules object to apply actions and get next states
        """
        if self.is_expanded:
            return

        for action_str, (action, prob) in action_probs.items():
            # Apply action to get next state
            next_state = game_rules.apply_action(self.state, action)

            # Create child node
            child = MCTSNode(
                state=next_state, parent=self, action_taken=action, prior_prob=prob
            )

            self.children[action_str] = child

        self.is_expanded = True

    def backup(self, value: float) -> None:
        """
        Backpropagate value up the tree

        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        self.visits += 1
        self.value_sum += value

        if self.parent is not None:
            # Flip value for opponent
            self.parent.backup(-value)

    def get_action_probs(self, temperature: float = 1.0) -> Dict[str, float]:
        """
        Get action probabilities based on visit counts

        Args:
            temperature: Temperature parameter for exploration
                        - temp = 0: deterministic (pick most visited)
                        - temp = 1: proportional to visit counts
                        - temp > 1: more exploration

        Returns:
            Dictionary mapping action strings to probabilities
        """
        if not self.children:
            return {}

        if temperature == 0:
            # Deterministic: pick action with most visits
            best_action = max(
                self.children.keys(), key=lambda a: self.children[a].visits
            )
            return {best_action: 1.0}

        # Calculate probabilities based on visit counts
        visits = np.array([child.visits for child in self.children.values()])

        if temperature == 1.0:
            # Standard proportional selection
            probs = (
                visits / visits.sum()
                if visits.sum() > 0
                else np.ones(len(visits)) / len(visits)
            )
        else:
            # Apply temperature
            log_visits = np.log(visits + 1e-8)  # Add small epsilon to avoid log(0)
            scaled_visits = np.exp(log_visits / temperature)
            probs = scaled_visits / scaled_visits.sum()

        # Return as dictionary
        return {action: prob for action, prob in zip(self.children.keys(), probs)}

    def get_best_action(self) -> Optional[str]:
        """
        Get the action with the highest visit count

        Returns:
            Action string of most visited child, or None if no children
        """
        if not self.children:
            return None

        return max(self.children.keys(), key=lambda a: self.children[a].visits)

    def get_pv(self, max_depth: int = 10) -> List[Action]:
        """
        Get the principal variation (most visited path) from this node

        Args:
            max_depth: Maximum depth to traverse

        Returns:
            List of actions in the principal variation
        """
        pv = []
        current = self

        for _ in range(max_depth):
            if not current.children or current.is_terminal():
                break

            # Get most visited child
            best_action = current.get_best_action()
            if best_action is None:
                break

            best_child = current.children[best_action]
            pv.append(best_child.action_taken)
            current = best_child

        return pv

    def get_statistics(self) -> Dict:
        """
        Get detailed statistics about this node for debugging/analysis

        Returns:
            Dictionary containing node statistics
        """
        stats = {
            "visits": self.visits,
            "value": self.get_value(),
            "prior_prob": self.prior_prob,
            "is_expanded": self.is_expanded,
            "is_terminal": self.is_terminal(),
            "num_children": len(self.children),
            "state": str(self.state),
        }

        if self.children:
            child_stats = {}
            for action, child in self.children.items():
                child_stats[action] = {
                    "visits": child.visits,
                    "value": child.get_value(),
                    "puct_score": child.get_puct_score(),
                    "prior_prob": child.prior_prob,
                }
            stats["children"] = child_stats

        return stats

    def __repr__(self) -> str:
        return f"MCTSNode(visits={self.visits}, value={self.get_value():.3f}, children={len(self.children)})"
