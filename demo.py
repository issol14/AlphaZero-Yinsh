"""
YINSH MCTS Demo

This script demonstrates how to use the YINSH MCTS implementation with AlphaZero-style
neural network guidance. It shows a simple game scenario and analyzes the MCTS search results.
"""

import numpy as np
import time
from yinsh_mcts import YinshMCTS, NeuralAgent, GameState, GamePhase, Action


def print_board(state: GameState):
    """Print a simple visualization of the game board"""
    print("\n" + "=" * 50)
    print(f"Game Phase: {state.phase.name}")
    print(f"Current Player: {state.current_player}")
    print(
        f"Removed Rings - White: {state.removed_rings['white']}, Black: {state.removed_rings['black']}"
    )

    # Create board visualization
    board = [["." for _ in range(11)] for _ in range(11)]

    # Place rings
    for ring in state.rings:
        x, y = ring.position
        if 0 <= x < 11 and 0 <= y < 11:
            board[x][y] = "W" if ring.color == "white" else "B"

    # Place markers
    for marker in state.markers:
        x, y = marker.position
        if 0 <= x < 11 and 0 <= y < 11:
            board[x][y] = "w" if marker.color == "white" else "b"

    # Print board
    print("\nBoard:")
    print("   " + " ".join([str(i) for i in range(11)]))
    for i, row in enumerate(board):
        print(f"{i:2} " + " ".join(row))

    print("\nLegend: W/B = White/Black Rings, w/b = White/Black Markers, . = Empty")
    print("=" * 50)


def run_mcts_analysis(state: GameState, agent: NeuralAgent, num_simulations: int = 100):
    """Run MCTS analysis on a given game state"""
    print(f"\n🔍 Running MCTS analysis with {num_simulations} simulations...")

    # Create MCTS instance
    mcts = YinshMCTS(
        neural_agent=agent,
        c_puct=1.4,  # Standard AlphaZero exploration constant
        max_simulations=num_simulations,
        simulation_timeout=5.0,
    )

    # Perform search
    start_time = time.time()
    root = mcts.search(state)
    search_time = time.time() - start_time

    # Get search results
    best_action = mcts.get_best_action(root, temperature=0.0)
    move_probs = mcts.get_move_probabilities(root)
    pv = mcts.get_principal_variation(root, max_depth=5)
    stats = mcts.get_statistics()

    # Print results
    print(f"\n📊 MCTS Search Results:")
    print(f"Search time: {search_time:.3f} seconds")
    print(f"Total simulations: {root.visits}")
    print(f"Root value estimate: {root.get_value():.3f}")
    print(f"Simulations per second: {stats['simulations_per_second']:.1f}")

    if best_action:
        print(f"\n🎯 Best Action: {best_action.action_type}")
        if best_action.to_pos:
            print(f"   To position: {best_action.to_pos}")
        if best_action.from_pos:
            print(f"   From position: {best_action.from_pos}")

    print(f"\n📈 Top 5 Move Probabilities:")
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (action_str, prob) in enumerate(sorted_moves[:5]):
        visits = root.children[action_str].visits if action_str in root.children else 0
        value = (
            root.children[action_str].get_value()
            if action_str in root.children
            else 0.0
        )
        print(
            f"   {i+1}. {action_str}: {prob:.3f} (visits: {visits}, value: {value:.3f})"
        )

    print(f"\n🎲 Principal Variation (next {len(pv)} moves):")
    for i, action in enumerate(pv):
        print(f"   {i+1}. {action.action_type}", end="")
        if action.to_pos:
            print(f" to {action.to_pos}", end="")
        if action.from_pos:
            print(f" from {action.from_pos}", end="")
        print()

    return root, best_action


def demo_ring_placement():
    """Demonstrate MCTS during ring placement phase"""
    print("\n🎮 DEMO 1: Ring Placement Phase")
    print("=" * 60)

    # Create initial game state
    state = GameState()

    # Add a couple of rings to make it more interesting
    from yinsh_mcts.game_state import Ring

    state.rings.append(Ring((3, 3), "white"))
    state.rings.append(Ring((7, 7), "black"))
    state.phase = GamePhase.WHITE_RING_PLACE
    state.current_player = "white"

    print_board(state)

    # Create neural agent (using random baseline for demo)
    agent = NeuralAgent(use_neural_network=False)

    # Run MCTS analysis
    root, best_action = run_mcts_analysis(state, agent, num_simulations=200)

    return root, best_action


def demo_ring_movement():
    """Demonstrate MCTS during ring movement phase"""
    print("\n🎮 DEMO 2: Ring Movement Phase")
    print("=" * 60)

    # Create game state with rings placed
    state = GameState()
    from yinsh_mcts.game_state import Ring, Marker

    # Place some rings
    state.rings.extend(
        [
            Ring((2, 2), "white"),
            Ring((4, 4), "white"),
            Ring((6, 6), "white"),
            Ring((1, 8), "black"),
            Ring((5, 5), "black"),
            Ring((8, 2), "black"),
        ]
    )

    # Add some markers
    state.markers.extend(
        [
            Marker((3, 3), "white"),
            Marker((7, 7), "black"),
            Marker((2, 5), "white"),
        ]
    )

    state.phase = GamePhase.WHITE_RING_MOVE
    state.current_player = "white"

    print_board(state)

    # Create neural agent
    agent = NeuralAgent(use_neural_network=False)

    # Run MCTS analysis
    root, best_action = run_mcts_analysis(state, agent, num_simulations=150)

    return root, best_action


def demo_neural_vs_random():
    """Compare neural network agent vs random agent (if PyTorch is available)"""
    print("\n🎮 DEMO 3: Neural Network vs Random Agent Comparison")
    print("=" * 60)

    state = GameState()

    try:
        # Try to use neural network agent
        neural_agent = NeuralAgent(use_neural_network=True)
        print("✅ Neural network agent created successfully")
    except Exception as e:
        print(f"❌ Could not create neural network agent: {e}")
        print("📝 Using random agent as fallback")
        neural_agent = NeuralAgent(use_neural_network=False)

    random_agent = NeuralAgent(use_neural_network=False)

    print_board(state)

    # Compare both agents on the same position
    print("\n🧠 Neural Agent Analysis:")
    neural_root, neural_action = run_mcts_analysis(
        state, neural_agent, num_simulations=100
    )

    print("\n🎲 Random Agent Analysis:")
    random_root, random_action = run_mcts_analysis(
        state, random_agent, num_simulations=100
    )

    # Compare results
    print(f"\n⚖️  Comparison:")
    print(f"Neural agent value estimate: {neural_root.get_value():.3f}")
    print(f"Random agent value estimate: {random_root.get_value():.3f}")

    if neural_action and random_action:
        print(f"Neural agent preferred action: {neural_action.action_type}")
        print(f"Random agent preferred action: {random_action.action_type}")

        if neural_action.to_pos and random_action.to_pos:
            print(f"Neural agent target: {neural_action.to_pos}")
            print(f"Random agent target: {random_action.to_pos}")


def demo_state_encoding():
    """Demonstrate the neural network state encoding"""
    print("\n🎮 DEMO 4: Neural Network State Encoding")
    print("=" * 60)

    # Create a complex game state
    state = GameState()
    from yinsh_mcts.game_state import Ring, Marker

    state.rings.extend(
        [
            Ring((2, 2), "white"),
            Ring((4, 4), "white"),
            Ring((6, 6), "black"),
            Ring((8, 8), "black"),
        ]
    )

    state.markers.extend(
        [
            Marker((3, 3), "white"),
            Marker((5, 5), "black"),
            Marker((1, 1), "white"),
        ]
    )

    state.phase = GamePhase.WHITE_RING_MOVE
    state.current_player = "white"
    state.removed_rings = {"white": 1, "black": 0}

    print_board(state)

    # Get tensor representation
    tensor = state.to_tensor()
    print(f"\n🔢 State Tensor Shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")

    # Analyze each channel
    channel_names = [
        "White Rings",
        "Black Rings",
        "White Markers",
        "Black Markers",
        "Valid Positions",
        "White Removable",
        "Black Removable",
        "Game Phase",
        "Current Player",
        "White Removed",
        "Black Removed",
    ]

    print(f"\n📊 Channel Analysis:")
    for i, name in enumerate(channel_names):
        channel = tensor[:, :, i]
        unique_vals = np.unique(channel)
        non_zero = np.count_nonzero(channel)
        print(
            f"   Channel {i:2d} ({name:15s}): {non_zero:3d} non-zero values, range: {unique_vals}"
        )

    # Show some channel details
    print(f"\n🔍 Detailed Channel Examples:")
    print(f"White Rings (Channel 0) - positions with value 1.0:")
    white_ring_positions = np.where(tensor[:, :, 0] == 1.0)
    for x, y in zip(white_ring_positions[0], white_ring_positions[1]):
        print(f"   ({x}, {y})")

    print(f"Current Phase (Channel 7) - constant value: {tensor[0, 0, 7]:.3f}")
    print(f"Current Player (Channel 8) - constant value: {tensor[0, 0, 8]:.3f}")


def main():
    """Run all demos"""
    print("🎯 YINSH MCTS with AlphaZero-style Neural Networks")
    print("=" * 60)
    print("This demo showcases Monte Carlo Tree Search for the YINSH board game")
    print("using neural network guidance similar to AlphaZero.")
    print()
    print("Features demonstrated:")
    print("• PUCT algorithm for action selection")
    print("• Neural network policy and value predictions")
    print("• Multi-channel state representation (11×11×11 tensor)")
    print("• MCTS statistics and analysis")
    print("• Principal variation extraction")

    try:
        # Run demos
        demo_ring_placement()
        demo_ring_movement()
        demo_neural_vs_random()
        demo_state_encoding()

        print("\n✅ All demos completed successfully!")
        print("\n📝 Next Steps:")
        print("• Implement full YINSH game rules")
        print("• Train neural network with self-play")
        print("• Add more sophisticated action encoding")
        print("• Implement evaluation against other agents")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
