import random
import threading
import time
import numpy as np
from yinshEnv import YinshEnv
from game import Game
from agent import Agent
import argparse
import logging

logging.basicConfig(level=logging.INFO, format=" %(message)s")
logging.disable(logging.WARN)


class Main:
    def __init__(self, player: bool, model_path: str = None):
        self.player = player  # True if human plays first, False if AI plays first
        
        # create an agent for the opponent
        self.opponent = Agent(model_path=model_path)

        if self.player:
            self.game = Game(YinshEnv(), None, self.opponent)
        else:
            self.game = Game(YinshEnv(), self.opponent, None)

        print("*" * 50)
        print(f"You play as Player {'1' if self.player else '2'}!")
        print("*" * 50)

        # previous moves (for the opponent's MCTS)
        self.previous_moves = (None, None)

        # Simple text-based interface for now
        # TODO: Implement GUI for Yinsh
        self.play_game()

    def play_game(self):
        self.game.reset()
        winner = None
        
        while winner is None:
            print(f"\n{self.game.env}")
            print(f"Current player: {self.game.current_player}")
            
            if self.player == (self.game.current_player == 0):
                # Human player's turn
                self.get_player_move()
            else:
                # AI player's turn
                self.opponent_move()
            
            # check if the game is over
            if self.game.env.is_game_over():
                # get the winner
                winner = self.game.env.get_winner()
                print(f"\nGame Over!")
                if winner == 1:
                    print("Player 1 wins!")
                elif winner == -1:
                    print("Player 2 wins!")
                else:
                    print("Draw!")

    def get_player_move(self):
        """
        Get move input from human player
        """
        print("\nYour turn!")
        print("Enter your move (format: action_type from_pos to_pos)")
        print("Example: PLACE_RING 5,5")
        print("Example: MOVE_RING 5,5 6,6")
        
        while True:
            try:
                move_input = input("Enter move: ").strip()
                if move_input.lower() == 'quit':
                    print("Quitting game...")
                    return
                
                # TODO: Parse human input to YinshMove
                # For now, just make a random valid move
                valid_moves = self.game.env.get_valid_moves()
                if valid_moves:
                    move = random.choice(valid_moves)
                    success = self.game.env.make_move(move)
                    if success:
                        self.game.current_player = 1 - self.game.current_player
                        self.game.env.current_player = self.game.current_player
                        break
                    else:
                        print("Invalid move, try again.")
                else:
                    print("No valid moves available!")
                    break
                    
            except (ValueError, IndexError) as e:
                print(f"Invalid input format: {e}")
                print("Please try again.")

    def opponent_move(self):
        """
        Get move from AI opponent
        """
        print("\nAI is thinking...")
        
        # Create game state for MCTS
        game_state = {
            'board': self.game.env.board.copy(),
            'current_player': self.game.env.current_player,
            'game_phase': self.game.env.game_phase,
            'rings_placed': self.game.env.rings_placed.copy(),
            'rings_removed': self.game.env.rings_removed.copy(),
            'move_count': self.game.env.move_count
        }
        
        # Update opponent's MCTS with current state
        from mcts import MCTS
        self.opponent.mcts = MCTS(self.opponent, state=game_state, stochastic=False)
        
        # Run simulations
        self.opponent.run_simulations(n=100)  # Fewer simulations for faster play
        
        moves = self.opponent.mcts.root.edges
        
        if moves:
            # Choose best move (deterministic)
            best_move = max(moves, key=lambda x: x.N)
            success = self.game.env.make_move(best_move.action)
            
            if success:
                print(f"AI played: {best_move.action}")
                self.game.current_player = 1 - self.game.current_player
                self.game.env.current_player = self.game.current_player
            else:
                print("AI move failed!")
        else:
            print("AI has no valid moves!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=str, default=None, choices=('1', '2'), 
                       help="Whether to play as player 1 or 2. No argument means random.")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to the model to use for AI opponent.")
    args = parser.parse_args()
    args = vars(args)

    model_path = args["model"]
    
    if args['player']:
        player = args['player'] == '1'
    else:
        player = np.random.choice([True, False])
        print(f"Randomly assigned as Player {'1' if player else '2'}")

    m = Main(player, model_path) 