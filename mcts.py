from yinshEnv import YinshEnv
from node import Node
from edge import Edge
import numpy as np
import time
from tqdm import tqdm
import utils
import threading

# graphing mcts
from graphviz import Digraph

import config
# output vector mapping
from mapper import Mapping, YinshMove

import logging


class MCTS:
    def __init__(self, agent: "Agent", state: dict = None, stochastic=False):
        """
        An object of the MCTS class represents a tree that can be built using 
        the Monte Carlo Tree Search algorithm for Yinsh. The tree consists of nodes and edges.
        The root node represents the current state of the game.

        Hundreds of simulations are run to build the tree.
        """
        if state is None:
            # Initialize default Yinsh game state
            state = {
                'board': np.zeros((config.BOARD_SIZE, config.BOARD_SIZE, 4)),
                'current_player': 0,
                'game_phase': 'placement',
                'rings_placed': [0, 0],
                'rings_removed': [0, 0],
                'move_count': 0
            }
        
        self.root = Node(state=state)
        self.game_path: list[Edge] = []
        self.cur_env: YinshEnv = None
        self.agent = agent
        self.stochastic = stochastic

    def run_simulations(self, n: int) -> None:
        """
        Run n simulations from the root node.
        1) select child
        2) expand and evaluate
        3) backpropagate
        """
        for _ in tqdm(range(n)):
            self.game_path = []

            # traverse the tree by selecting edges with max Q+U
            # leaf is root on first iteration
            leaf = self.select_child(self.root)

            # expand the leaf node
            leaf.N += 1
            leaf = self.expand(leaf)

            # backpropagate the result
            leaf = self.backpropagate(leaf, leaf.value)

    def select_child(self, node: Node) -> Node:
        """
        Traverse the tree from the given node, by selecting actions with the maximum Q+U.

        If the node has not been visited yet, return the node. That is the new leaf node.
        If this is the first simulation, the leaf node is the root node.
        """
        # traverse the tree by selecting nodes until a leaf node is reached
        while not node.is_leaf():
            if not len(node.edges):
                # if the node is terminal, return the node
                return node
            noise = [1 for _ in range(len(node.edges))]
            if self.stochastic and node == self.root:
                noise = np.random.dirichlet([config.DIRICHLET_NOISE]*len(node.edges))
            best_edge = None
            best_score = -np.inf                
            for i, edge in enumerate(node.edges):
                if edge.upper_confidence_bound(noise[i]) > best_score:
                    best_score = edge.upper_confidence_bound(noise[i])
                    best_edge = edge

            if best_edge is None:
                # this should never happen
                raise Exception("No edge found")
        
            # get that actions's new node
            node = best_edge.output_node
            self.game_path.append(best_edge)
        return node

    def probabilities_to_actions(self, probabilities: list, game_state: dict) -> dict:
        """
        Map the output vector probabilities to Yinsh moves. 
        Returns a dictionary of moves and their probabilities.

        The output vector is a list of probabilities for every possible action
        """
        # TODO: Implement Yinsh probability to action mapping
        probabilities = probabilities.reshape(-1)
        actions = {}

        # Get valid moves for current state
        env = YinshEnv()
        # Set environment to current state
        # valid_moves = env.get_valid_moves()
        
        # For now, return placeholder
        # In actual implementation, map probabilities to valid Yinsh moves
        for i, prob in enumerate(probabilities):
            move = Mapping.action_index_to_move(i)
            actions[str(move)] = prob

        return actions

    def expand(self, leaf: Node) -> Node:
        """
        Expand the leaf node by adding all possible moves to the leaf node.
        This will generate new edges and nodes.
        Return the leaf node
        """
        logging.debug("Expanding...")

        # Check if game is over
        if leaf.is_game_over():
            # Calculate terminal value
            rings_removed = leaf.state.get('rings_removed', [0, 0])
            if rings_removed[0] >= config.MARKERS_TO_WIN:
                leaf.value = 1
            elif rings_removed[1] >= config.MARKERS_TO_WIN:
                leaf.value = -1
            else:
                leaf.value = 0
            return leaf

        # Get all possible moves
        env = YinshEnv()
        # Set env state from leaf.state
        possible_actions = env.get_valid_moves()

        if not len(possible_actions):
            # No valid moves, estimate value
            leaf.value = YinshEnv.estimate_winner(
                leaf.state.get('board'), 
                leaf.state.get('rings_removed', [0, 0])
            )
            return leaf

        # predict p and v using neural network
        input_state = YinshEnv.state_to_input(
            leaf.state.get('board'),
            leaf.state.get('current_player', 0),
            leaf.state.get('game_phase', 'placement')
        )
        p, v = self.agent.predict(input_state)

        # map probabilities to moves
        actions = self.probabilities_to_actions(p, leaf.state)

        logging.debug(f"Model predictions: {p}")
        logging.debug(f"Value of state: {v}")

        leaf.value = v

        # create a child node for every action
        for action in possible_actions:
            # make the move and get the new state
            new_state = leaf.step(action)
            # add a new child node with the new state, the action taken and its prior probability
            action_prob = actions.get(str(action), 0.01)  # small default probability
            leaf.add_child(Node(new_state), action, action_prob)
        
        return leaf

    def backpropagate(self, end_node: Node, value: float) -> Node:
        """
        The backpropagation step will update the values of the nodes 
        in the traversed path from the given leaf node up to the root node.
        """
        logging.debug("Backpropagation...")

        for edge in self.game_path:
            edge.input_node.N += 1
            edge.N += 1
            edge.W += value
        return end_node

    def plot_node(self, dot: Digraph, node: Node):
        """
        Recursive function to plot nodes.
        """
        dot.node(f"{id(node)}", f"N={node.N}")
        for edge in node.edges:
            dot.edge(str(id(edge.input_node)), str(id(edge.output_node)), 
                    label=str(edge.action))
            dot = self.plot_node(dot, edge.output_node)
        return dot

    def plot_tree(self, save_path: str = "tests/mcts_tree.gv") -> None:
        """
        Plot the MCTS tree using graphviz.
        """
        logging.debug("Plotting tree...")
        # tree plotting
        dot = Digraph(comment='Yinsh MCTS Tree')
        logging.info(f"# of nodes in tree: {len(self.root.get_all_children())}")

        # recursively plot the tree
        dot = self.plot_node(dot, self.root)
        dot.save(save_path) 