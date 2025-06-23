from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from agent import Agent
from yinshEnv import YinshEnv
from game import Game
from mcts import MCTS
import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Global variables to store game state and agents
current_games = {}  # Dictionary to store multiple game sessions
default_agent = None


def initialize_agent(model_path=None):
    """Initialize the AI agent"""
    global default_agent
    try:
        default_agent = Agent(model_path=model_path)
        logger.info(f"Agent initialized with model: {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        default_agent = Agent()  # Initialize without model
        logger.info("Agent initialized without model (random play)")


def game_state_to_json(env: YinshEnv):
    """Convert game state to JSON-serializable format"""
    return {
        'board': env.board.tolist(),
        'current_player': env.current_player,
        'game_phase': env.game_phase,
        'rings_placed': env.rings_placed,
        'rings_removed': env.rings_removed,
        'move_count': env.move_count,
        'is_game_over': env.is_game_over(),
        'winner': env.get_winner() if env.is_game_over() else None
    }


def json_to_game_state(data: dict) -> dict:
    """Convert JSON data to game state format"""
    return {
        'board': np.array(data['board']),
        'current_player': data['current_player'],
        'game_phase': data['game_phase'],
        'rings_placed': data['rings_placed'],
        'rings_removed': data['rings_removed'],
        'move_count': data['move_count']
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent_loaded': default_agent is not None,
        'version': '1.0.0'
    })


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game session"""
    try:
        data = request.get_json() or {}
        game_id = data.get('game_id', 'default')
        
        # Create new environment
        env = YinshEnv()
        
        # Store game session
        current_games[game_id] = {
            'env': env,
            'agent': default_agent
        }
        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'game_state': game_state_to_json(env)
        })
    
    except Exception as e:
        logger.error(f"Error creating new game: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/game_state/<game_id>', methods=['GET'])
def get_game_state(game_id):
    """Get current game state"""
    try:
        if game_id not in current_games:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 404
        
        env = current_games[game_id]['env']
        return jsonify({
            'success': True,
            'game_state': game_state_to_json(env)
        })
    
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move (either human or AI)"""
    try:
        data = request.get_json()
        game_id = data.get('game_id', 'default')
        
        if game_id not in current_games:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 404
        
        env = current_games[game_id]['env']
        
        # Handle human move
        if 'move' in data:
            # TODO: Parse and validate human move
            # For now, just acknowledge the move
            move_data = data['move']
            logger.info(f"Human move received: {move_data}")
            
            # Apply move to environment (placeholder)
            success = env.make_move(move_data)
            
            return jsonify({
                'success': success,
                'game_state': game_state_to_json(env),
                'move_valid': success
            })
        
        return jsonify({
            'success': False,
            'error': 'No move data provided'
        }), 400
    
    except Exception as e:
        logger.error(f"Error making move: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Get AI move for current position"""
    try:
        data = request.get_json() or {}
        game_id = data.get('game_id', 'default')
        simulations = data.get('simulations', 100)
        
        if game_id not in current_games:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 404
        
        game_session = current_games[game_id]
        env = game_session['env']
        agent = game_session['agent']
        
        if not agent:
            return jsonify({
                'success': False,
                'error': 'AI agent not available'
            }), 500
        
        # Create game state for MCTS
        game_state = {
            'board': env.board.copy(),
            'current_player': env.current_player,
            'game_phase': env.game_phase,
            'rings_placed': env.rings_placed.copy(),
            'rings_removed': env.rings_removed.copy(),
            'move_count': env.move_count
        }
        
        # Run MCTS to get best move
        agent.mcts = MCTS(agent, state=game_state, stochastic=False)
        agent.run_simulations(n=simulations)
        
        moves = agent.mcts.root.edges
        
        if moves:
            # Get best move
            best_move = max(moves, key=lambda x: x.N)
            
            # Apply move to environment
            success = env.make_move(best_move.action)
            
            if success:
                return jsonify({
                    'success': True,
                    'ai_move': str(best_move.action),
                    'confidence': best_move.N / sum(e.N for e in moves),
                    'game_state': game_state_to_json(env),
                    'move_info': {
                        'visits': best_move.N,
                        'value': best_move.W / best_move.N if best_move.N > 0 else 0,
                        'prior': best_move.P
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'AI move failed to apply'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'No valid moves available'
            }), 500
    
    except Exception as e:
        logger.error(f"Error getting AI move: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/update_game_state', methods=['POST'])
def update_game_state():
    """Update game state from web app"""
    try:
        data = request.get_json()
        game_id = data.get('game_id', 'default')
        new_state = data.get('game_state')
        
        if not new_state:
            return jsonify({
                'success': False,
                'error': 'No game state provided'
            }), 400
        
        # Create/update game session
        env = YinshEnv()
        
        # Apply state to environment
        env.board = np.array(new_state['board'])
        env.current_player = new_state['current_player']
        env.game_phase = new_state['game_phase']
        env.rings_placed = new_state['rings_placed']
        env.rings_removed = new_state['rings_removed']
        env.move_count = new_state['move_count']
        
        current_games[game_id] = {
            'env': env,
            'agent': default_agent
        }
        
        return jsonify({
            'success': True,
            'game_state': game_state_to_json(env)
        })
    
    except Exception as e:
        logger.error(f"Error updating game state: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/get_valid_moves', methods=['POST'])
def get_valid_moves():
    """Get valid moves for current position"""
    try:
        data = request.get_json() or {}
        game_id = data.get('game_id', 'default')
        
        if game_id not in current_games:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 404
        
        env = current_games[game_id]['env']
        valid_moves = env.get_valid_moves()
        
        # Convert moves to JSON-serializable format
        moves_json = [str(move) for move in valid_moves]
        
        return jsonify({
            'success': True,
            'valid_moves': moves_json,
            'count': len(valid_moves)
        })
    
    except Exception as e:
        logger.error(f"Error getting valid moves: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze_position', methods=['POST'])
def analyze_position():
    """Analyze current position and return evaluation"""
    try:
        data = request.get_json() or {}
        game_id = data.get('game_id', 'default')
        depth = data.get('depth', 50)
        
        if game_id not in current_games:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 404
        
        game_session = current_games[game_id]
        env = game_session['env']
        agent = game_session['agent']
        
        if not agent:
            return jsonify({
                'success': False,
                'error': 'AI agent not available'
            }), 500
        
        # Create game state for analysis
        game_state = {
            'board': env.board.copy(),
            'current_player': env.current_player,
            'game_phase': env.game_phase,
            'rings_placed': env.rings_placed.copy(),
            'rings_removed': env.rings_removed.copy(),
            'move_count': env.move_count
        }
        
        # Run shorter MCTS for analysis
        agent.mcts = MCTS(agent, state=game_state, stochastic=False)
        agent.run_simulations(n=depth)
        
        moves = agent.mcts.root.edges
        
        # Get top moves with their evaluations
        top_moves = []
        if moves:
            sorted_moves = sorted(moves, key=lambda x: x.N, reverse=True)[:5]
            total_visits = sum(e.N for e in moves)
            
            for move in sorted_moves:
                top_moves.append({
                    'move': str(move.action),
                    'visits': move.N,
                    'win_rate': move.W / move.N if move.N > 0 else 0,
                    'probability': move.N / total_visits,
                    'prior': move.P
                })
        
        return jsonify({
            'success': True,
            'position_evaluation': agent.mcts.root.value,
            'total_simulations': sum(e.N for e in moves),
            'top_moves': top_moves
        })
    
    except Exception as e:
        logger.error(f"Error analyzing position: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Yinsh AI Web Server')
    parser.add_argument('--model', type=str, default=None, help='Path to AI model')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize the AI agent
    initialize_agent(args.model)
    
    print(f"Starting Yinsh AI Web Server...")
    print(f"Server will run on http://{args.host}:{args.port}")
    print(f"API endpoints available at /api/")
    
    app.run(host=args.host, port=args.port, debug=args.debug) 