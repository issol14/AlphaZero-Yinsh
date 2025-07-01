import os
import time
from yinshEnv import YinshEnv
from agent import Agent
import utils
import logging
import config
from edge import Edge
from mcts import MCTS
import uuid
import pandas as pd
import numpy as np


class Game:
    def __init__(self, env: YinshEnv, player1: Agent, player2: Agent):
        """
        The Game class is used to play Yinsh games between two agents.
        """
        self.env = env
        self.player1 = player1  # Player 0
        self.player2 = player2  # Player 1

        self.memory = []
        self.reset()

    def reset(self):
        self.env.reset()
        # Convert env's current_player (Color.WHITE=1, Color.BLACK=-1) to Game's player index (0, 1)
        self.current_player = 0 if self.env.current_player == 1 else 1

    @staticmethod
    def get_winner(result: int) -> int:
        """
        Convert game result to winner format
        """
        return result  # 1 for player1 win, -1 for player2 win, 0 for draw

    @utils.time_function
    def play_one_game(self, stochastic: bool = True) -> int:
        """
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        """
        # reset everything
        self.reset()
        # add a new memory entry
        self.memory.append([])
        # show the board
        logging.info(f"\n{self.env}")
        
        # counter to check amount of moves played. if above limit, estimate winner
        counter, previous_edges, full_game = 0, (None, None), True
        
        while not self.env.is_game_over():
            # Get current agent before playing move
            current_agent = self.player1 if self.current_player == 0 else self.player2
            
            # play one move (previous move is used for updating the MCTS tree)
            previous_edges = self.play_move(stochastic=stochastic, previous_moves=previous_edges)
            logging.info(f"\n{self.env}")
            
            # Log the value from the agent that just played (before current_player changed)
            if current_agent.mcts is not None:
                logging.info(f"Value according to previous player: {current_agent.mcts.root.value}")

            # end if the game drags on too long
            counter += 1
            if counter > config.MAX_GAME_MOVES:
                # estimate the winner based on game state
                winner = YinshEnv.estimate_winner(self.env.board, self.env.rings_removed)
                logging.info(f"Game over by move limit ({config.MAX_GAME_MOVES}). Result: {winner}")
                full_game = False
                break
        
        if full_game:
            # get the winner based on the result of the game
            winner = self.env.get_winner()
            logging.info(f"Game over. Result: {winner}")
        
        # save game result to memory for all positions
        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        # TODO: Implement Yinsh game notation/logging
        logging.info(f"Game completed in {counter} moves")

        # save memory to file
        self.save_game(name="game", full_game=full_game)

        return winner

    def play_move(self, stochastic: bool = True, previous_moves: tuple[Edge, Edge] = (None, None), save_moves=True) -> tuple:
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        """
        # get current player agent
        current_agent = self.player1 if self.current_player == 0 else self.player2
        
        # Create game state for MCTS
        game_state = {
            'board': self.env,  # Pass the environment directly
            'current_player': self.env.current_player,
            'game_phase': self.env.game_phase,
            'rings_placed': self.env.rings_placed.copy(),
            'rings_removed': self.env.rings_removed.copy(),
            'move_count': self.env.move_count
        }

        # Always create new MCTS tree with current game state to ensure fresh start
        current_agent.mcts = MCTS(current_agent, state=game_state, stochastic=stochastic)
        logging.debug(f"Created new MCTS tree. Current state: {game_state['board'].turn_state}")
        
        # play n simulations from the root node
        current_agent.run_simulations(n=config.SIMULATIONS_PER_MOVE)

        moves = current_agent.mcts.root.edges

        if save_moves:
            self.save_to_memory(game_state, moves)

        if not moves:
            logging.warning("No valid moves found!")
            return previous_moves

        sum_move_visits = sum(e.N for e in moves)
        probs = [e.N / sum_move_visits for e in moves]
        
        if stochastic:
            # choose a move based on a probability distribution
            best_move = np.random.choice(moves, p=probs)
        else:
            # choose a move based on the highest N
            best_move = moves[np.argmax(probs)]

        # play the move
        logging.info(f"Player {self.current_player} played move: {best_move.action}")
        success = self.env.make_move(best_move.action)
        
        if not success:
            logging.warning("Move failed!")
            return previous_moves
        
        # Update Game's current_player to match env's current_player (don't force switch)
        # Let YinshEnv handle turn switching internally
        self.current_player = 0 if self.env.current_player == 1 else 1  # Convert Color to player index

        # return the previous move and the new move
        return (previous_moves[1], best_move)

    def save_to_memory(self, state: dict, moves: list) -> None:
        """
        Append the current state and move probabilities to the internal memory.
        """
        if not moves:
            return
            
        sum_move_visits = sum(e.N for e in moves)
        # create dictionary of moves and their probabilities
        search_probabilities = {
            str(e.action): e.N / sum_move_visits for e in moves
        }
        # winner gets added after game is over
        self.memory[-1].append((state, search_probabilities, None))

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        # create necessary directories
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        
        # the game id consist of game + datetime
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        
        if full_game:
            # if the game result was not estimated, save the game id to a separate file
            with open("full_games.txt", "a") as f:
                f.write(f"{game_id}.npy\n")
        
        np.save(os.path.join(config.MEMORY_DIR, game_id), self.memory[-1])
        logging.info(f"Game saved to {os.path.join(config.MEMORY_DIR, game_id)}.npy")
        logging.info(f"Memory size: {len(self.memory)}") 