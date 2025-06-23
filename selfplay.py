import argparse
import logging
import os
import time
from agent import Agent
from yinshEnv import YinshEnv
from game import Game
import config
import numpy as np

# set logging config
logging.basicConfig(level=logging.INFO, format=' %(message)s')


def setup(model_path=None) -> Game:
    """
    Setup function to set up a Yinsh game for self-play
    """
    # set different random seeds for each process
    import socket
    number = int.from_bytes(socket.gethostname().encode(), 'little')
    number *= os.getpid() if os.getpid() != 0 else 1
    number *= int(time.time())
    number %= 123456789
    
    np.random.seed(number)
    print(f"========== > Setup. Test Random number: {np.random.randint(0, 123456789)}")

    # create environment and game
    env = YinshEnv()

    # create agents
    if model_path is None:
        model_path = os.path.join(config.MODEL_FOLDER, "model.h5")
    
    # Check if model exists
    if os.path.exists(model_path):
        player1 = Agent(model_path=model_path)
        player2 = Agent(model_path=model_path)
    else:
        logging.warning(f"Model not found at {model_path}, creating agents without models")
        player1 = Agent()
        player2 = Agent()

    return Game(env=env, player1=player1, player2=player2)


def self_play(model_path=None):
    """
    Continuously play games against itself for Yinsh
    """
    game = setup(model_path=model_path)

    # play games continuously
    game_count = 0
    while True:
        game_count += 1
        print(f"\n{'='*50}")
        print(f"Starting Self-Play Game #{game_count}")
        print(f"{'='*50}")
        
        try:
            result = game.play_one_game(stochastic=True)
            print(f"Game #{game_count} completed. Result: {result}")
        except KeyboardInterrupt:
            print("\nStopping self-play...")
            break
        except Exception as e:
            print(f"Error in game #{game_count}: {e}")
            logging.error(f"Error in self-play: {e}")
            # Continue with next game
            game = setup(model_path=model_path)  # Reset game on error


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Run Yinsh self-play')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model file (if not specified, uses default path)')
    args = parser.parse_args()
    args = vars(args)

    model_path = args.get('model')
    if model_path and not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist")
    
    print("Starting Yinsh self-play...")
    print(f"Using model: {model_path if model_path else 'default path'}")
    
    try:
        self_play(model_path)
    except KeyboardInterrupt:
        print("\nSelf-play interrupted by user")
    except Exception as e:
        print(f"Self-play failed with error: {e}")
        raise 