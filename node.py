from edge import Edge


class Node:
    def __init__(self, state: dict):
        """
        A node is a state inside the MCTS tree for Yinsh.
        """
        self.state = state  # Yinsh game state (board + metadata)
        self.current_player = state.get('current_player', 0)
        # the edges connected to this node
        self.edges: list[Edge] = []
        # the visit count for this node
        self.N = 0
        # the value of this node
        self.value = 0

    def __eq__(self, node: object) -> bool:
        """
        Check if two nodes are equal.
        Two nodes are equal if the state is the same
        """
        if isinstance(node, Node):
            return self.state == node.state
        else:
            return NotImplemented

    def step(self, action) -> dict:
        """
        Take a step in the game, returns new state
        """
        # TODO: Implement Yinsh move execution
        # This should apply the action to the current state
        # and return the new game state
        
        new_state = self.state.copy()
        # Apply move logic here
        # new_state = apply_move(self.state, action)
        
        return new_state

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        # TODO: Implement Yinsh game over check
        # Check if any player has won (removed 3 rings)
        rings_removed = self.state.get('rings_removed', [0, 0])
        return rings_removed[0] >= 3 or rings_removed[1] >= 3

    def is_leaf(self) -> bool:
        """
        Check if the current node is a leaf node.
        """
        return self.N == 0

    def add_child(self, child, action, prior: float) -> Edge:
        """
        Add a child node to the current node.

        Returns the created edge between the nodes
        """
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge

    def get_all_children(self):
        """
        Get all children of the current node and their children, recursively
        """
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_all_children())
        return children

    def get_edge(self, action) -> Edge:
        """
        Get the edge between the current node and the child node with the given action.
        """
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None 