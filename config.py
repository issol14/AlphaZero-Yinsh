# config file: includes parameters for the model and the mcts tree
import os
from dotenv import load_dotenv
load_dotenv()

# ============= MCTS =============
SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 400))

# exploration parameters 
C_base = 20000
C_init = 2

DIRICHLET_NOISE = 0.3

# limit the amount of moves played in a game
MAX_GAME_MOVES = 200

# ============= YINSH GAME PARAMETERS =============
BOARD_SIZE = 11  # Yinsh board is 11x11 hex grid
RINGS_PER_PLAYER = 5
MARKERS_TO_WIN = 3  # Need to remove 3 rings to win

# ============= NEURAL NETWORK INPUTS =============
# Yinsh game state representation
# 2 players, rings, markers, possible moves on hex board
n = BOARD_SIZE  # board size
# Input planes: rings for each player, markers for each player, valid positions, turn indicator
amount_of_input_planes = (2 + 2 + 1 + 1)  # rings, markers, valid_positions, turn
INPUT_SHAPE = (n, n, amount_of_input_planes)

# ============= NEURAL NETWORK OUTPUTS =============
# Yinsh move representation
# Ring placement (5 positions per player) + Ring movement + Marker placement
# For simplicity, using position-based action space
max_positions = n * n
# Actions: place ring, move ring, remove markers
OUTPUT_SHAPE = (max_positions * 3,)  # Fixed: removed extra dimension

# ============= NEURAL NETWORK PARAMETERS =============
LEARNING_RATE = 0.001  # Fixed: much lower learning rate for stable training
# filters for the convolutional layers
CONVOLUTION_FILTERS = 256
# amount of hidden residual layers
AMOUNT_OF_RESIDUAL_BLOCKS = 19

# where to save the model
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", './models')

# ============= TRAINING PARAMETERS =============
BATCH_SIZE = 32  # Fixed: smaller batch size for limited data
LOSS_PLOTS_FOLDER = "./plots"

# ============= MEMORY CONFIGURATION =============
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
MAX_REPLAY_MEMORY = 1000000 