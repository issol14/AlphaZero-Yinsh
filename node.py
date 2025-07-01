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
        # Get the board environment from current state
        board_env = self.state.get('board')
        
        if hasattr(board_env, 'step'):
            # Make a copy of the environment or state
            from yinshEnv import YinshEnv
            new_env = YinshEnv()
            new_env.load_from_state_string(board_env.get_state_string())
            
            # Apply the action
            new_env.step(action)
            
            # Return new state
            new_state = {
                'board': new_env,
                'current_player': new_env.current_player,
                'game_phase': new_env.game_phase,
                'rings_placed': new_env.rings_placed.copy(),
                'rings_removed': new_env.rings_removed.copy(),
                'move_count': new_env.move_count
            }
            return new_state
        else:
            # Fallback - return current state
            return self.state

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        # Check the board environment directly
        board_env = self.state.get('board')
        if hasattr(board_env, 'is_game_over'):
            return board_env.is_game_over()
        
        # Fallback: check rings removed
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