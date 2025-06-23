/**
 * YinshAI Web Client
 * JavaScript client for communicating with the Yinsh AI server
 */

class YinshAIClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
        this.gameId = 'default';
    }

    /**
     * Check if the AI server is healthy
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', error: error.message };
        }
    }

    /**
     * Start a new game
     */
    async newGame(gameId = null) {
        try {
            if (gameId) this.gameId = gameId;
            
            const response = await fetch(`${this.baseUrl}/api/new_game`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId
                })
            });
            
            const result = await response.json();
            if (result.success) {
                this.gameId = result.game_id;
            }
            return result;
        } catch (error) {
            console.error('Failed to start new game:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get current game state
     */
    async getGameState() {
        try {
            const response = await fetch(`${this.baseUrl}/api/game_state/${this.gameId}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get game state:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Make a human move
     */
    async makeMove(move) {
        try {
            const response = await fetch(`${this.baseUrl}/api/make_move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    move: move
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Failed to make move:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get AI move
     */
    async getAIMove(simulations = 100) {
        try {
            const response = await fetch(`${this.baseUrl}/api/ai_move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    simulations: simulations
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Failed to get AI move:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Update game state from web app
     */
    async updateGameState(gameState) {
        try {
            const response = await fetch(`${this.baseUrl}/api/update_game_state`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    game_state: gameState
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Failed to update game state:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get valid moves for current position
     */
    async getValidMoves() {
        try {
            const response = await fetch(`${this.baseUrl}/api/get_valid_moves`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Failed to get valid moves:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Analyze current position
     */
    async analyzePosition(depth = 50) {
        try {
            const response = await fetch(`${this.baseUrl}/api/analyze_position`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    depth: depth
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Failed to analyze position:', error);
            return { success: false, error: error.message };
        }
    }
}

// Usage example
async function exampleUsage() {
    const yinshAI = new YinshAIClient('http://localhost:5000');
    
    // Check if server is running
    const health = await yinshAI.checkHealth();
    console.log('Server health:', health);
    
    if (health.status === 'healthy') {
        // Start new game
        const newGame = await yinshAI.newGame();
        console.log('New game:', newGame);
        
        if (newGame.success) {
            // Get current game state
            const gameState = await yinshAI.getGameState();
            console.log('Game state:', gameState);
            
            // Get valid moves
            const validMoves = await yinshAI.getValidMoves();
            console.log('Valid moves:', validMoves);
            
            // Get AI move
            const aiMove = await yinshAI.getAIMove(100);
            console.log('AI move:', aiMove);
            
            // Analyze position
            const analysis = await yinshAI.analyzePosition(50);
            console.log('Position analysis:', analysis);
        }
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = YinshAIClient;
}

// Make available globally in browser
if (typeof window !== 'undefined') {
    window.YinshAIClient = YinshAIClient;
} 