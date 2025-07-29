#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based Human vs AI YINSH Game Interface
==========================================

Flask를 사용한 웹 기반 게임 인터페이스입니다.
"""

import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import uuid

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshAgent, YinshModel, Color, YinshAction, config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yinsh-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 게임 세션 저장
game_sessions = {}

class WebGameSession:
    """웹 게임 세션"""
    
    def __init__(self, session_id: str, model_path: str = None, use_mcts: bool = True):
        self.session_id = session_id
        self.env = YinshEnv()
        self.ai_agent = YinshAgent(
            model_path=model_path,
            use_mcts=use_mcts,
            device="cpu"
        )
        self.move_count = 0
        self.game_started = False
        self.winner = None
    
    def get_game_state(self):
        """게임 상태를 JSON으로 반환"""
        board_state = []
        for i in range(11):
            row = []
            for j in range(11):
                pos = (i, j)
                cell = {
                    'valid': self.env.is_valid_position(pos),
                    'piece': None
                }
                
                if pos in self.env.ring_positions[Color.WHITE]:
                    cell['piece'] = 'white_ring'
                elif pos in self.env.ring_positions[Color.BLACK]:
                    cell['piece'] = 'black_ring'
                elif pos in self.env.marker_positions[Color.WHITE]:
                    cell['piece'] = 'white_marker'
                elif pos in self.env.marker_positions[Color.BLACK]:
                    cell['piece'] = 'black_marker'
                
                row.append(cell)
            board_state.append(row)
        
        valid_actions = []
        for action in self.env.get_valid_actions():
            if action.action_type == "PLACE_RING":
                valid_actions.append({
                    'type': 'place',
                    'pos': action.to_pos
                })
            elif action.action_type == "MOVE_RING":
                valid_actions.append({
                    'type': 'move',
                    'from': action.from_pos,
                    'to': action.to_pos
                })
        
        return {
            'board': board_state,
            'current_player': self.env.current_player.name,
            'phase': self.env.phase,
            'move_count': self.move_count,
            'game_over': self.env.is_game_over(),
            'winner': self.winner.name if self.winner else None,
            'valid_actions': valid_actions,
            'white_rings': len(self.env.ring_positions[Color.WHITE]),
            'black_rings': len(self.env.ring_positions[Color.BLACK]),
            'white_removed': self.env.rings_removed[Color.WHITE],
            'black_removed': self.env.rings_removed[Color.BLACK]
        }
    
    def make_human_move(self, action_data):
        """사람의 액션 실행"""
        try:
            if action_data['type'] == 'place':
                action = YinshAction("PLACE_RING", to_pos=tuple(action_data['pos']))
            elif action_data['type'] == 'move':
                action = YinshAction("MOVE_RING", 
                                   from_pos=tuple(action_data['from']),
                                   to_pos=tuple(action_data['to']))
            else:
                return False, "Invalid action type"
            
            if self.env.step(action):
                self.move_count += 1
                return True, None
            else:
                return False, "Invalid move"
        except Exception as e:
            return False, str(e)
    
    def make_ai_move(self):
        """AI의 액션 실행"""
        try:
            start_time = time.time()
            action, action_info = self.ai_agent.select_action(
                self.env, 
                temperature=0.1,
                add_noise=False
            )
            thinking_time = time.time() - start_time
            
            success = self.env.step(action)
            if success:
                self.move_count += 1
                
                # AI 액션 정보
                ai_action = {
                    'type': action.action_type.lower().replace('_', ''),
                    'thinking_time': thinking_time,
                    'method': action_info.get('method', 'unknown')
                }
                
                if action.action_type == "PLACE_RING":
                    ai_action['pos'] = action.to_pos
                elif action.action_type == "MOVE_RING":
                    ai_action['from'] = action.from_pos
                    ai_action['to'] = action.to_pos
                
                return True, ai_action
            else:
                return False, "AI made invalid move"
        except Exception as e:
            return False, str(e)
    
    def reset_game(self):
        """게임 리셋"""
        self.env.reset()
        self.move_count = 0
        self.game_started = False
        self.winner = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/game')
def game():
    """게임 페이지"""
    return render_template('game.html')

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # 게임 세션 생성
    game_sessions[session_id] = WebGameSession(session_id)
    
    emit('connected', {'session_id': session_id})
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    session_id = session.get('session_id')
    if session_id in game_sessions:
        del game_sessions[session_id]
    print(f"Client disconnected: {session_id}")

@socketio.on('start_game')
def handle_start_game(data):
    """게임 시작"""
    session_id = session.get('session_id')
    if session_id not in game_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    game_session = game_sessions[session_id]
    game_session.reset_game()
    game_session.game_started = True
    
    # 게임 상태 전송
    emit('game_state', game_session.get_game_state())

@socketio.on('make_move')
def handle_make_move(data):
    """사람의 액션 처리"""
    session_id = session.get('session_id')
    if session_id not in game_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    game_session = game_sessions[session_id]
    
    # 사람의 턴인지 확인
    if game_session.env.current_player != Color.WHITE:
        emit('error', {'message': 'Not your turn'})
        return
    
    # 사람의 액션 실행
    success, error = game_session.make_human_move(data)
    if not success:
        emit('error', {'message': error})
        return
    
    # 게임 상태 전송
    game_state = game_session.get_game_state()
    emit('game_state', game_state)
    
    # 게임이 끝났는지 확인
    if game_session.env.is_game_over():
        game_session.winner = game_session.env.get_winner()
        emit('game_over', {'winner': game_session.winner.name if game_session.winner else None})
        return
    
    # AI의 턴
    if game_session.env.current_player == Color.BLACK:
        # AI가 생각 중임을 알림
        emit('ai_thinking', {'message': 'AI is thinking...'})
        
        # AI 액션 실행
        success, ai_action = game_session.make_ai_move()
        if not success:
            emit('error', {'message': ai_action})
            return
        
        # AI 액션 정보 전송
        emit('ai_move', ai_action)
        
        # 게임 상태 전송
        game_state = game_session.get_game_state()
        emit('game_state', game_state)
        
        # 게임이 끝났는지 확인
        if game_session.env.is_game_over():
            game_session.winner = game_session.env.get_winner()
            emit('game_over', {'winner': game_session.winner.name if game_session.winner else None})

@socketio.on('reset_game')
def handle_reset_game():
    """게임 리셋"""
    session_id = session.get('session_id')
    if session_id not in game_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    game_session = game_sessions[session_id]
    game_session.reset_game()
    
    emit('game_reset', {'message': 'Game reset'})

if __name__ == '__main__':
    # 템플릿 폴더 생성
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    print("🌐 Starting YINSH Web Interface...")
    print("📝 Please create HTML templates in the 'templates' folder")
    print("🎨 Please create CSS/JS files in the 'static' folder")
    print("🔗 Access the game at: http://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 