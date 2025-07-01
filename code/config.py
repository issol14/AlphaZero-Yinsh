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
MAX_PUZZLE_MOVES = 4
MAX_GAME_MOVES = 200

# ============= NEURAL NETWORK INPUTS (CHESS) =============
# 2 players, 6 pieces, 8x8 board
n = 8  # board size
# non boolean values: pieces for every player + the square for en passant
# boolean values: side to move, castling rights for every side and every player, is repitition
amount_of_input_planes =  (2*6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (n, n, amount_of_input_planes)

# ============= NEURAL NETWORK INPUTS (YINSH) =============
# YINSH: 11x11 board with 85 valid positions
yinsh_board_size = 11
# Input planes for YINSH:
# - 1 plane: turn information
# - 5 planes: game phase (placement, main, marker_remove, ring_remove, ended)
# - 2 planes: rings removed count (white, black)
# - 4 planes: board state (white rings, black rings, white markers, black markers)
# - 1 plane: valid positions mask
yinsh_input_planes = 1 + 5 + 2 + 4 + 1  # 13 planes total
YINSH_INPUT_SHAPE = (yinsh_board_size, yinsh_board_size, yinsh_input_planes)

# ============= NEURAL NETWORK OUTPUTS (CHESS) =============
# the model has 2 outputs: policy and value
# ouput_shape[0] should be the number of possible moves
#       * 8x8 board: 8*8=64 possible actions
#       * 56 possible queen-like moves (horizontal/vertical/diagonal)
#       * 8 possible knight moves (every direction)
#       * 9 possible underpromotions
#   total values: 8*8*(56+8+9) = 4672
# ouput_shape[1] should be 1: a scalar value (v)
# 73 planes for chess:
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes + knight_planes + underpromotion_planes
# the output shape for the vector
OUTPUT_SHAPE = (8*8*amount_of_planes, 1)

# ============= NEURAL NETWORK OUTPUTS (YINSH) =============
# YINSH move possibilities:
# - Ring placement: 85 positions (action 0-84)
# - Ring movement: 85*85 = 7225 positions (action 85-7309)  
# - Marker removal: 1000 estimated combinations (action 7310-8309)
# - Ring removal: 85 positions (same as placement, distinguished by game phase)
# Total action space: 8310 actions
YINSH_ACTION_SPACE_SIZE = 8310
YINSH_OUTPUT_SHAPE = (YINSH_ACTION_SPACE_SIZE, 1)

# ============= NEURAL NETWORK PARAMETERS =============
# change if necessary. AZ started with 0.2 and then dropped three times to 0.02, 0.002 and 0.0002
LEARNING_RATE = 0.2
# filters for the convolutional layers (AZ: 256)
CONVOLUTION_FILTERS = 256
# amount of hidden residual layers
# According to the AlphaGo Zero paper:
#    "For the larger run (40 block, 40 days), MCTS search parameters were re-optimised using the neural network trained in the smaller run (20 block, 3 days)."
# ==> First train a small NN, then optimize longer with a larger NN.
AMOUNT_OF_RESIDUAL_BLOCKS = 19

# where to save the model
MODEL_FOLDER = os.environ.get("MODEL_FOLDER" ,'./models')

# ============= TRAINING PARAMETERS =============
BATCH_SIZE = 64
LOSS_PLOTS_FOLDER="./plots"

# ============= MEMORY CONFIGURATION =============
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
MAX_REPLAY_MEMORY = 1000000

# ============= SOCKET CONFIGURATION =============
SOCKET_BUFFER_SIZE = 8192 
# # config file: includes parameters for the model and the mcts tree
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # ============= MCTS =============
# SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 400))

# # exploration parameters 
# C_base = 20000
# C_init = 2

# DIRICHLET_NOISE = 0.3

# # limit the amount of moves played in a game
# MAX_PUZZLE_MOVES = 4
# MAX_GAME_MOVES = 200

# # ============= NEURAL NETWORK INPUTS (CHESS) =============
# # 2 players, 6 pieces, 8x8 board
# n = 8  # board size
# # non boolean values: pieces for every player + the square for en passant
# # boolean values: side to move, castling rights for every side and every player, is repitition
# amount_of_input_planes =  (2*6 + 1) + (1 + 4 + 1)
# INPUT_SHAPE = (n, n, amount_of_input_planes)

# # ============= NEURAL NETWORK INPUTS (YINSH) =============
# # YINSH: 11x11 board with 85 valid positions
# yinsh_board_size = 11
# # Input planes for YINSH:
# # - 1 plane: turn information
# # - 5 planes: game phase (placement, main, marker_remove, ring_remove, ended)
# # - 2 planes: rings removed count (white, black)
# # - 4 planes: board state (white rings, black rings, white markers, black markers)
# # - 1 plane: valid positions mask
# yinsh_input_planes = 1 + 5 + 2 + 4 + 1  # 13 planes total
# YINSH_INPUT_SHAPE = (yinsh_board_size, yinsh_board_size, yinsh_input_planes)

# # ============= NEURAL NETWORK OUTPUTS (CHESS) =============
# # the model has 2 outputs: policy and value
# # ouput_shape[0] should be the number of possible moves
# #       * 8x8 board: 8*8=64 possible actions
# #       * 56 possible queen-like moves (horizontal/vertical/diagonal)
# #       * 8 possible knight moves (every direction)
# #       * 9 possible underpromotions
# #   total values: 8*8*(56+8+9) = 4672
# # ouput_shape[1] should be 1: a scalar value (v)
# # 73 planes for chess:
# queen_planes = 56
# knight_planes = 8
# underpromotion_planes = 9
# amount_of_planes = queen_planes + knight_planes + underpromotion_planes
# # the output shape for the vector
# OUTPUT_SHAPE = (8*8*amount_of_planes, 1)


# # ============= NEURAL NETWORK PARAMETERS =============
# # change if necessary. AZ started with 0.2 and then dropped three times to 0.02, 0.002 and 0.0002
# LEARNING_RATE = 0.2
# # filters for the convolutional layers (AZ: 256)
# CONVOLUTION_FILTERS = 256
# # amount of hidden residual layers
# # According to the AlphaGo Zero paper:
# #    "For the larger run (40 block, 40 days), MCTS search parameters were re-optimised using the neural network trained in the smaller run (20 block, 3 days)."
# # ==> First train a small NN, then optimize longer with a larger NN.
# AMOUNT_OF_RESIDUAL_BLOCKS = 19

# # where to save the model
# MODEL_FOLDER = os.environ.get("MODEL_FOLDER" ,'./models')

# # ============= TRAINING PARAMETERS =============
# BATCH_SIZE = 64
# LOSS_PLOTS_FOLDER="./plots"

# # ============= MEMORY CONFIGURATION =============
# MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
# MAX_REPLAY_MEMORY = 1000000

# # ============= SOCKET CONFIGURATION =============
# SOCKET_BUFFER_SIZE = 8192