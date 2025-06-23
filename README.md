# AlphaYinsh - Reinforcement Learning for Yinsh

A reinforcement learning implementation for the abstract strategy board game Yinsh, based on AlphaZero methodology. This project is adapted from a chess AI implementation to work with Yinsh game mechanics.

## Project Structure

```
AlphaYinsh/
├── config.py              # Configuration parameters
├── yinshEnv.py            # Yinsh game environment
├── node.py                # MCTS tree nodes
├── edge.py                # MCTS tree edges
├── mcts.py                # Monte Carlo Tree Search implementation
├── mapper.py              # Move mapping utilities
├── agent.py               # AI agent implementation
├── local_prediction.py    # TensorFlow prediction helpers
├── rlmodelbuilder.py      # Neural network architecture
├── game.py                # Game logic and self-play
├── utils.py               # Utility functions
├── main.py                # Human vs AI gameplay
├── selfplay.py            # Self-play training
├── train.py               # Model training
├── evaluate.py            # Model evaluation
├── test.py                # Testing and benchmarking
└── README.md              # This file
```

## Key Components

### Game Environment (`yinshEnv.py`)
- Handles Yinsh game state representation
- Board state encoding for neural network input
- Move validation and execution
- Win condition checking

### MCTS Implementation (`mcts.py`, `node.py`, `edge.py`)
- Monte Carlo Tree Search algorithm
- Upper Confidence Bound (UCB) for move selection
- Tree expansion and backpropagation
- Dirichlet noise for exploration

### Neural Network (`rlmodelbuilder.py`)
- Residual convolutional neural network
- Policy head for move probability prediction
- Value head for position evaluation
- Based on AlphaZero architecture

### Training Pipeline
- **Self-play** (`selfplay.py`): Generate training data
- **Training** (`train.py`): Train neural network on self-play data
- **Evaluation** (`evaluate.py`): Compare model versions

## Usage

### Prerequisites
```bash
pip install tensorflow numpy matplotlib pandas tqdm pillow graphviz python-dotenv
```

### Create Initial Model
```bash
python rlmodelbuilder.py --model-folder models --model-name initial_model
```

### Run Self-Play
```bash
python selfplay.py --model models/initial_model.h5
```

### Train Model
```bash
python train.py --model models/initial_model.h5 --data-folder memory
```

### Human vs AI
```bash
python main.py --player 1 --model models/trained_model.h5
```

### Evaluate Models
```bash
python evaluate.py models/model1.h5 models/model2.h5 10
```

### Run Tests
```bash
python test.py
```

## Configuration

Edit `config.py` to modify:
- MCTS simulation count
- Neural network architecture
- Training parameters
- Board representation

## Implementation Status

### ✅ Completed
- Basic project structure
- MCTS algorithm framework
- Neural network architecture
- Training pipeline
- Evaluation framework

### 🚧 TODO (Implementation Details)
- **Yinsh Game Rules**: Complete implementation of:
  - Ring placement mechanics
  - Ring movement with marker placement
  - Line formation detection
  - Ring removal when lines are formed
  - Hexagonal board geometry
  
- **Move Representation**: 
  - Proper encoding of Yinsh moves to neural network outputs
  - Action space design for ring placement/movement
  - Marker removal action encoding

- **Board State Encoding**:
  - Hexagonal coordinate system
  - Efficient representation of rings and markers
  - Game phase tracking (placement vs movement)

- **GUI**: Visual interface for human play

## Yinsh Game Overview

Yinsh is an abstract strategy game where:
1. **Setup**: Players place 5 rings each on a hexagonal board
2. **Gameplay**: Move rings, placing markers along the path
3. **Lines**: Form lines of 5 markers to remove a ring
4. **Victory**: First player to remove 3 of their own rings wins

## Architecture Notes

This implementation follows the AlphaZero approach:
- **Self-play**: AI plays against itself to generate training data
- **Neural Network**: Predicts move probabilities and position values
- **MCTS**: Uses neural network to guide tree search
- **Iterative Improvement**: Train → Self-play → Train cycle

## Development Notes

The current implementation provides a complete framework but requires Yinsh-specific game logic to be implemented in the TODO sections marked throughout the code. The structure closely follows the original chess implementation to maintain consistency and allow for easy adaptation.

Key areas needing implementation:
1. `yinshEnv.py`: Complete game rules and move validation
2. `mapper.py`: Proper move encoding/decoding
3. `mcts.py`: Yinsh-specific move generation
4. `utils.py`: Move conversion utilities

## License

This project is adapted from a chess reinforcement learning implementation for educational purposes. 