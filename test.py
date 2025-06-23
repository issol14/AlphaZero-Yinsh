from agent import Agent
from yinshEnv import YinshEnv
from game import Game
import utils
import logging
import numpy as np
import selfplay
import config


class Test:
    """
    Class to do unit tests and benchmark functions for optimization of Yinsh AI.
    """
    def __init__(self):
        pass

    @utils.time_function
    def run_state_to_input_test(self):
        """
        Test the state to input conversion
        """
        env = YinshEnv()
        
        # Create a test state
        test_state = {
            'board': np.random.randint(0, 2, (config.BOARD_SIZE, config.BOARD_SIZE, 4)),
            'current_player': 0,
            'game_phase': 'placement',
            'rings_placed': [2, 1],
            'rings_removed': [0, 0],
            'move_count': 5
        }
        
        # test input_state conversion
        input_state = YinshEnv.state_to_input(
            test_state['board'], 
            test_state['current_player'], 
            test_state['game_phase']
        )
        
        print(f"Input state shape: {input_state.shape}")
        print(f"Expected shape: {(1, *config.INPUT_SHAPE)}")
        
        # Save visualization if possible
        try:
            input_state_reshaped = np.reshape(input_state, config.INPUT_SHAPE)
            utils.save_input_state_to_imgs(input_state_reshaped, 'tests/input_planes')
            print("Input state visualization saved")
        except Exception as e:
            print(f"Could not save visualization: {e}")

    @utils.time_function
    def test_mcts_tree(self, n: int):
        """
        Test MCTS tree building and analysis
        """
        game = selfplay.setup()
        
        print(f"Running {n} MCTS simulations...")
        game.player1.run_simulations(n)

        # get height of tree
        print(f"Tree height: {utils.get_height_of_tree(game.player1.mcts.root)}")
        print(f"Root node visits: {game.player1.mcts.root.N}")
        print(f"Number of edges from root: {len(game.player1.mcts.root.edges)}")

        # plot tree if graphviz is available
        try:
            game.player1.mcts.plot_tree(f"tests/mcts_tree_{n}_nodes.gv")
            print(f"MCTS tree visualization saved")
        except Exception as e:
            print(f"Could not save tree visualization: {e}")
        
    @utils.time_function
    def test_game_flow(self, moves: int = 10):
        """
        Test basic game flow for a few moves
        """
        print(f"Testing game flow for {moves} moves...")
        
        game = selfplay.setup()
        
        for i in range(moves):
            print(f"\n--- Move {i+1} ---")
            print(f"Current player: {game.env.current_player}")
            print(f"Game phase: {game.env.game_phase}")
            
            # Get current agent
            current_agent = game.player1 if game.env.current_player == 0 else game.player2
            
            # Run fewer simulations for faster testing
            current_agent.run_simulations(n=50)
            
            moves_available = current_agent.mcts.root.edges
            print(f"Available moves: {len(moves_available)}")
            
            if moves_available:
                # Make a move
                best_move = max(moves_available, key=lambda x: x.N)
                success = game.env.make_move(best_move.action)
                
                if success:
                    print(f"Move played: {best_move.action}")
                    game.env.current_player = 1 - game.env.current_player
                else:
                    print("Move failed!")
                    break
            else:
                print("No moves available!")
                break
            
            if game.env.is_game_over():
                print("Game over!")
                break

    @utils.time_function  
    def test_agent_prediction(self):
        """
        Test agent prediction functionality
        """
        print("Testing agent prediction...")
        
        # Create agent
        agent = Agent()
        
        # Create test input
        test_input = np.random.choice([True, False], size=(1, *config.INPUT_SHAPE))
        
        # Test prediction
        try:
            p, v = agent.predict(test_input)
            print(f"Policy output shape: {np.array(p).shape}")
            print(f"Value output: {v}")
            print(f"Policy sum: {np.sum(p)}")
            print("Agent prediction test successful!")
        except Exception as e:
            print(f"Agent prediction test failed: {e}")

    @utils.time_function
    def test_environment(self):
        """
        Test basic environment functionality
        """
        print("Testing Yinsh environment...")
        
        env = YinshEnv()
        print(f"Initial state: {env}")
        print(f"Board shape: {env.board.shape}")
        print(f"Is game over: {env.is_game_over()}")
        
        # Test valid moves
        valid_moves = env.get_valid_moves()
        print(f"Number of valid moves: {len(valid_moves)}")
        
        # Test winner estimation
        winner = YinshEnv.estimate_winner(env.board, env.rings_removed)
        print(f"Estimated winner: {winner}")


if __name__ == "__main__":
    test = Test()
    
    print("="*50)
    print("Running Yinsh AI Tests")
    print("="*50)
    
    # Run tests
    try:
        test.test_environment()
        print("\n" + "-"*30)
        
        test.test_agent_prediction()
        print("\n" + "-"*30)
        
        test.run_state_to_input_test()
        print("\n" + "-"*30)
        
        test.test_game_flow(5)
        print("\n" + "-"*30)
        
        test.test_mcts_tree(20)
        print("\n" + "-"*30)
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 