import numpy as np
from PIL import Image
import time
from mapper import Mapping, YinshMove
import config
from node import Node


def save_input_state_to_imgs(input_state: np.ndarray, path: str):
    """
    Save an input state to images
    """
    start_time = time.time()
    # convert booleans to integers
    input_state = np.array(input_state) * np.uint8(255)
    # pad input_state with grey values
    input_state = np.pad(input_state, ((0, 0), (1, 1), (1, 1)),
                         'constant', constant_values=128)

    full_array = np.concatenate(input_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)),
                        'constant', constant_values=128)
    img = Image.fromarray(full_array)
    img.save(f"{path}/full.png")
    print(f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")


def save_output_state_to_imgs(output_state: np.ndarray, path: str, name: str = "full"):
    """
    Save an output state to images
    """
    start_time = time.time()
    # pad output_state with grey values
    output_state = np.pad(output_state.astype(float) * 255, ((0, 0), (1, 1), (1, 1)), 
                         'constant', constant_values=128)
    full_array = np.concatenate(output_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)), 'constant', constant_values=128)
    img = Image.fromarray(full_array.astype(np.uint8))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(f"{path}/{name}.png")
    print(f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")


def time_function(func):
    """
    Decorator to time a function
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def moves_to_output_vector(moves: dict, game_state: dict) -> np.ndarray:
    """
    Convert a dictionary of moves to a vector of probabilities for Yinsh
    """
    # TODO: Implement Yinsh moves to output vector conversion
    vector = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
    
    for move_str, probability in moves.items():
        # Convert move string back to YinshMove object
        # This is a placeholder - actual implementation needed
        try:
            # For now, just distribute probabilities randomly
            # In actual implementation, this should properly map Yinsh moves
            action_index = hash(move_str) % config.OUTPUT_SHAPE[0]
            vector[action_index] = probability
        except:
            continue
    
    return vector


def move_to_action_index(move: YinshMove) -> int:
    """
    Convert a Yinsh move to action index
    """
    return Mapping.move_to_action_index(move)


def get_height_of_tree(node: Node) -> int:
    """
    Calculate the height of MCTS tree
    """
    if node is None:
        return 0

    h = 0
    for edge in node.edges:
        h = max(h, get_height_of_tree(edge.output_node))
    return h + 1


def format_game_state(state: dict) -> str:
    """
    Format game state for display
    """
    # TODO: Implement proper Yinsh game state formatting
    return f"Player: {state.get('current_player', 0)}, Phase: {state.get('game_phase', 'unknown')}"


def validate_yinsh_move(move: YinshMove, game_state: dict) -> bool:
    """
    Validate if a Yinsh move is legal in the given game state
    """
    # TODO: Implement Yinsh move validation
    # This should check:
    # 1. If it's a valid position on the hex board
    # 2. If the move is legal according to Yinsh rules
    # 3. If it's the correct phase of the game
    return True  # Placeholder


def calculate_hex_distance(pos1: tuple, pos2: tuple) -> int:
    """
    Calculate distance between two positions on hexagonal grid
    """
    return Mapping.hex_distance(pos1, pos2)


if __name__ == "__main__":
    # Test functions
    from yinshEnv import YinshEnv
    from agent import Agent
    
    # Test basic functionality
    env = YinshEnv()
    print(f"Environment initialized: {env}")
    
    # Test state formatting
    test_state = {
        'board': np.zeros((config.BOARD_SIZE, config.BOARD_SIZE, 4)),
        'current_player': 0,
        'game_phase': 'placement',
        'rings_placed': [0, 0],
        'rings_removed': [0, 0],
        'move_count': 0
    }
    
    print(f"Formatted state: {format_game_state(test_state)}")
    
    # Test if basic imports work
    try:
        agent = Agent()
        print("Agent created successfully")
    except Exception as e:
        print(f"Error creating agent: {e}") 