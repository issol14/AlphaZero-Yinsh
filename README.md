# YINSH MCTS with AlphaZero-style Neural Networks

A Monte Carlo Tree Search (MCTS) implementation for the YINSH board game using AlphaZero-style neural network guidance.

## 🎯 Overview

This project implements a sophisticated MCTS algorithm for the YINSH game, incorporating:

- **PUCT Algorithm**: Polynomial Upper Confidence Trees for optimal action selection
- **Neural Network Guidance**: Policy and value predictions using deep learning
- **Multi-channel State Representation**: 11×11×11 tensor encoding of game states
- **AlphaZero Architecture**: Residual network design with policy and value heads

## 🏗️ Architecture

### Core Components

1. **GameState** (`game_state.py`): Complete game state representation
2. **MCTSNode** (`mcts_node.py`): MCTS tree node with PUCT scoring
3. **NeuralAgent** (`neural_agent.py`): Neural network for policy/value prediction
4. **YinshMCTS** (`yinsh_mcts.py`): Main MCTS algorithm implementation

### State Representation

The game state is encoded as an 11×11×11 tensor with the following channels:

| Channel | Description | Values |
|---------|-------------|--------|
| 0 | White rings | 1 where white ring exists, 0 elsewhere |
| 1 | Black rings | 1 where black ring exists, 0 elsewhere |
| 2 | White markers | 1 where white marker exists, 0 elsewhere |
| 3 | Black markers | 1 where black marker exists, 0 elsewhere |
| 4 | Valid positions | 1 where moves are possible, 0 elsewhere |
| 5 | White removable markers | 1 where white can remove markers |
| 6 | Black removable markers | 1 where black can remove markers |
| 7 | Game phase | Normalized phase value (0-7) |
| 8 | Current player | +1 for white, -1 for black |
| 9 | White removed rings | Normalized count (0-3) |
| 10 | Black removed rings | Normalized count (0-3) |

### MCTS Algorithm

The implementation follows the standard MCTS phases:

1. **Selection**: Use PUCT to traverse tree to leaf node
2. **Expansion**: Add children using neural network priors
3. **Evaluation**: Get value estimate from neural network
4. **Backpropagation**: Update statistics up the tree

#### PUCT Formula

```
PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)`: Average value of action a from state s
- `P(s,a)`: Prior probability from neural network
- `N(s)`: Visit count of parent node
- `N(s,a)`: Visit count of child node
- `c_puct`: Exploration constant (typically 1.0-2.0)

## 🚀 Quick Start

### Installation

```bash
# Clone or download the yinsh_mcts folder
cd yinsh_mcts

# Install dependencies
pip install numpy torch  # For neural network support
pip install numpy        # For basic functionality only
```

### Basic Usage

```python
from yinsh_mcts import YinshMCTS, NeuralAgent, GameState

# Create a game state
state = GameState()

# Create neural agent (random baseline for demo)
agent = NeuralAgent(use_neural_network=False)

# Create MCTS instance
mcts = YinshMCTS(
    neural_agent=agent,
    c_puct=1.4,
    max_simulations=1000
)

# Perform search
root = mcts.search(state, num_simulations=500)

# Get best action
best_action = mcts.get_best_action(root, temperature=0.0)
print(f"Best action: {best_action}")

# Get move probabilities
move_probs = mcts.get_move_probabilities(root)
print(f"Move probabilities: {move_probs}")
```

### Running the Demo

```bash
python demo.py
```

The demo showcases:
- Ring placement phase analysis
- Ring movement phase analysis  
- Neural network vs random agent comparison
- State tensor encoding visualization

## 🧠 Neural Network Architecture

### Network Structure

```
Input: 11×11×11 tensor
   ↓
Initial Conv (3×3, 256 filters)
   ↓
Residual Blocks (4 blocks)
   ↓
     ├─ Policy Head → 1000 action probabilities
     └─ Value Head → Single value estimate [-1, +1]
```

### Training Process (Future Work)

1. **Self-play**: Generate games using current network
2. **Data Collection**: Store (state, policy, value) examples
3. **Network Training**: Minimize combined loss:
   ```
   Loss = (z - V(s))² - Σ π(a|s) log P(a|s) + λ||θ||²
   ```
4. **Model Evaluation**: Test against previous best model
5. **Model Update**: Replace if win rate > 55%

## 🎮 Game Phases

The YINSH game progresses through several phases:

1. **Ring Placement**: Players alternate placing rings (5 each)
2. **Ring Movement**: Players move rings and place markers
3. **Marker Removal**: Remove 5-in-a-row markers (when formed)
4. **Ring Removal**: Remove a ring after marker removal
5. **Game End**: First to remove 3 rings wins

## 📊 Performance Analysis

### MCTS Statistics

The implementation tracks various performance metrics:

```python
stats = mcts.get_statistics()
print(f"Simulations per second: {stats['simulations_per_second']}")
print(f"Total search time: {stats['total_time']}")
```

### Node Analysis

```python
node_stats = root.get_statistics()
print(f"Root visits: {node_stats['visits']}")
print(f"Root value: {node_stats['value']}")
print(f"Number of children: {node_stats['num_children']}")
```

## 🔧 Configuration

### MCTS Parameters

- `c_puct`: Exploration constant (default: 1.0)
  - Higher values → more exploration
  - Lower values → more exploitation
  
- `max_simulations`: Maximum MCTS simulations (default: 1000)

- `simulation_timeout`: Time limit in seconds (default: 1.0)

### Neural Network Parameters

- `num_resblocks`: Number of residual blocks (default: 4)
- `hidden_size`: Channel size for convolutions (default: 256)
- `board_size`: Game board dimensions (default: 11)

## 🚧 Current Limitations

This is a demonstration/research implementation with several simplifications:

1. **Simplified Game Rules**: Not all YINSH rules are implemented
2. **Action Encoding**: Basic string-based action representation
3. **Neural Network**: Placeholder implementation (not trained)
4. **Move Generation**: Simplified legal move detection

## 🔮 Future Enhancements

### High Priority
- [ ] Complete YINSH rule implementation
- [ ] Train neural network with self-play
- [ ] Sophisticated action encoding/decoding
- [ ] Performance optimizations

### Medium Priority
- [ ] Multi-GPU training support
- [ ] Game tree visualization
- [ ] Opening book integration
- [ ] Time management improvements

### Low Priority
- [ ] Web interface for human play
- [ ] Tournament system
- [ ] Statistical analysis tools
- [ ] Model compression techniques

## 📚 References

1. **AlphaZero Paper**: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
2. **MCTS Survey**: "A Survey of Monte Carlo Tree Search Methods"
3. **YINSH Rules**: Official game rules by Kris Burm

## 🤝 Contributing

This is a research/educational project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Kris Burm for creating the YINSH game
- DeepMind for the AlphaZero algorithm
- PyTorch team for the neural network framework 