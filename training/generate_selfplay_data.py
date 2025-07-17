"""
YINSH AlphaZero 셀프플레이 데이터 생성 스크립트
"""

import multiprocessing as mp
import time
import random
from pathlib import Path
import sys
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import signal

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from yinsh.game import YinshGame
from yinsh.agent import YinshAgent, RandomAgent
from yinsh.env import YinshEnv, Color

# 로컬 임포트
try:
    import training.data_config as config
    from training.data_utils import DataProcessor, DatasetBuilder, save_selfplay_games
except ImportError:
    # 직접 실행하는 경우
    import data_config as config
    from data_utils import DataProcessor, DatasetBuilder, save_selfplay_games

class SelfPlayDataGenerator:
    """셀프플레이 데이터 생성 클래스"""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.data_processor = DataProcessor()
        self.dataset_builder = DatasetBuilder()
        
    def create_agent(self, agent_type: str, use_mcts: bool = True) -> Any:
        """에이전트를 생성합니다."""
        if agent_type == 'random':
            return RandomAgent()
        elif agent_type == 'ai':
            return YinshAgent(
                device=self.device, 
                use_mcts=use_mcts
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def play_single_game(self, white_type: str, black_type: str, game_id: int) -> Dict[str, Any]:
        """단일 게임을 실행하고 결과를 반환합니다."""
        try:
            # 에이전트 생성
            white_agent = self.create_agent(white_type, use_mcts=(white_type == 'ai'))
            black_agent = self.create_agent(black_type, use_mcts=(black_type == 'ai'))
            
            # 게임 생성
            game = YinshGame(white_agent, black_agent)
            
            # 게임 실행 (move history 수집 활성화)
            start_time = time.time()
            
            # 게임 진행 상태 추적
            moves_history = []
            
            while not game.env.is_game_over():
                # 현재 상태 저장 (기본 상태 텐서 사용)
                current_state = game.env.get_state_tensor()
                current_player = game.env.current_player
                
                # 현재 에이전트 선택
                if current_player == Color.WHITE:
                    agent = white_agent
                    agent_type = white_type
                else:
                    agent = black_agent
                    agent_type = black_type
                
                # 온도 계산 (게임 진행도에 따라)
                temperature = config.get_temperature(
                    game.moves_played, 
                    config.MAX_GAME_LENGTH
                )
                
                # 액션 선택
                if hasattr(agent, 'select_action'):
                    action_result = agent.select_action(game.env, temperature=temperature)
                    if isinstance(action_result, tuple):
                        action, action_info = action_result
                    else:
                        action = action_result
                        action_info = {}
                else:
                    # RandomAgent의 경우
                    valid_actions = game.env.get_valid_actions()
                    action = random.choice(valid_actions)
                    action_info = {'method': 'random'}
                
                # 무브 히스토리에 추가
                move_data = {
                    'move_number': game.moves_played,
                    'player': current_player,
                    'state': current_state,
                    'action': action,
                    'agent_type': agent_type,
                    'temperature': temperature
                }
                
                # MCTS 정보 추가 (있는 경우)
                if 'mcts_stats' in action_info:
                    mcts_stats = action_info['mcts_stats']
                    if 'action_probabilities' in mcts_stats:
                        move_data['mcts_probs'] = mcts_stats['action_probabilities']
                
                moves_history.append(move_data)
                
                # 액션 실행
                game.env.step(action)
                game.moves_played += 1
                
                # 최대 길이 제한
                if game.moves_played >= config.MAX_GAME_LENGTH:
                    break
            
            # 게임 결과 계산
            duration = time.time() - start_time
            winner = game.env.get_winner()
            
            if winner == Color.WHITE:
                result = 1.0
            elif winner == Color.BLACK:
                result = -1.0
            else:
                result = 0.0
            
            game_data = {
                'game_id': game_id,
                'white_agent': white_type,
                'black_agent': black_type,
                'winner': winner,
                'result': result,
                'moves_history': moves_history,
                'game_length': game.moves_played,
                'duration': duration,
                'final_state': {'phase': game.env.phase, 'winner': winner}
            }
            
            return game_data
            
        except Exception as e:
            print(f"❌ Error in game {game_id}: {str(e)}")
            return None
    
    def generate_games_batch(self, batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """게임 배치를 생성합니다."""
        batch_size = batch_config['batch_size']
        combinations = batch_config['combinations']
        start_id = batch_config['start_id']
        
        games = []
        game_id = start_id
        
        for combo_name, count in combinations.items():
            if combo_name == 'random_vs_random':
                white_type, black_type = 'random', 'random'
            elif combo_name == 'ai_vs_random':
                white_type, black_type = 'ai', 'random'
            elif combo_name == 'random_vs_ai':
                white_type, black_type = 'random', 'ai'
            else:
                continue
            
            for _ in range(count):
                game_data = self.play_single_game(white_type, black_type, game_id)
                if game_data:
                    games.append(game_data)
                game_id += 1
        
        return games

def generate_batch_configs(total_games: int, num_processes: int) -> List[Dict[str, Any]]:
    """프로세스별 배치 설정을 생성합니다."""
    games_per_process = total_games // num_processes
    
    # 에이전트 조합별 게임 수 계산
    combination_counts = {}
    for combo_name, ratio in config.AGENT_COMBINATIONS.items():
        combination_counts[combo_name] = int(games_per_process * ratio)
    
    batch_configs = []
    for i in range(num_processes):
        start_id = i * games_per_process
        batch_config = {
            'batch_size': games_per_process,
            'combinations': combination_counts.copy(),
            'start_id': start_id,
            'process_id': i
        }
        batch_configs.append(batch_config)
    
    return batch_configs

def worker_process(batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """워커 프로세스에서 실행되는 함수"""
    process_id = batch_config['process_id']
    
    try:
        # 프로세스별로 다른 시드 설정
        random.seed(42 + process_id)
        np.random.seed(42 + process_id)
        
        print(f"🔄 Process {process_id} starting...")
        
        # 데이터 생성기 생성
        generator = SelfPlayDataGenerator(use_cuda=True)
        
        # 게임 생성
        games = generator.generate_games_batch(batch_config)
        
        print(f"✅ Process {process_id} completed: {len(games)} games")
        return games
        
    except Exception as e:
        print(f"❌ Process {process_id} failed: {str(e)}")
        return []

def main():
    """메인 실행 함수"""
    print("🚀 YINSH AlphaZero 셀프플레이 데이터 생성 시작")
    print(f"📊 총 게임 수: {config.TOTAL_GAMES}")
    print(f"⚡ 프로세스 수: {config.NUM_PROCESSES}")
    print(f"🎯 에이전트 조합: {config.AGENT_COMBINATIONS}")
    
    # 설정 검증
    config.validate_config()
    
    # 배치 설정 생성
    batch_configs = generate_batch_configs(config.TOTAL_GAMES, config.NUM_PROCESSES)
    
    start_time = time.time()
    
    # 멀티프로세싱으로 게임 생성
    try:
        with mp.Pool(processes=config.NUM_PROCESSES) as pool:
            print(f"🎮 {config.NUM_PROCESSES}개 프로세스로 게임 실행 중...")
            
            # 진행상황 표시를 위한 tqdm 사용
            results = []
            for result in tqdm(
                pool.imap(worker_process, batch_configs),
                total=len(batch_configs),
                desc="Generating games"
            ):
                results.append(result)
    
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        return
    except Exception as e:
        print(f"❌ 멀티프로세싱 오류: {str(e)}")
        return
    
    # 결과 수집
    all_games = []
    for games_batch in results:
        all_games.extend(games_batch)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 데이터 생성 완료!")
    print(f"📊 생성된 게임 수: {len(all_games)}")
    print(f"⏱️ 소요 시간: {elapsed_time:.1f}초")
    print(f"📈 게임 생성 속도: {len(all_games)/elapsed_time:.1f} 게임/초")
    
    # 게임 데이터 저장
    if all_games:
        save_selfplay_games(all_games, config.SELFPLAY_DATA_FILE)
        
        # 간단한 통계 출력
        print_game_statistics(all_games)
        
        # 학습용 데이터셋 생성
        print("\n🔄 학습용 데이터셋 생성 중...")
        create_training_dataset(all_games)
    
    print(f"\n✅ 4단계 완료: 초기 셀프플레이 데이터 생성 성공!")

def print_game_statistics(games: List[Dict[str, Any]]):
    """게임 통계를 출력합니다."""
    if not games:
        return
    
    # 에이전트별 통계
    agent_stats = {}
    game_lengths = []
    durations = []
    
    for game in games:
        combo = f"{game['white_agent']}_vs_{game['black_agent']}"
        if combo not in agent_stats:
            agent_stats[combo] = {'count': 0, 'white_wins': 0, 'black_wins': 0, 'draws': 0}
        
        agent_stats[combo]['count'] += 1
        
        if game['winner'] == Color.WHITE:
            agent_stats[combo]['white_wins'] += 1
        elif game['winner'] == Color.BLACK:
            agent_stats[combo]['black_wins'] += 1
        else:
            agent_stats[combo]['draws'] += 1
        
        game_lengths.append(game['game_length'])
        durations.append(game['duration'])
    
    print(f"\n📈 게임 통계:")
    for combo, stats in agent_stats.items():
        total = stats['count']
        white_pct = stats['white_wins'] / total * 100
        black_pct = stats['black_wins'] / total * 100
        draw_pct = stats['draws'] / total * 100
        print(f"  {combo}: {total}게임 (W:{white_pct:.1f}% B:{black_pct:.1f}% D:{draw_pct:.1f}%)")
    
    print(f"\n📊 게임 길이: 평균 {np.mean(game_lengths):.1f}턴 (최소 {min(game_lengths)}, 최대 {max(game_lengths)})")
    print(f"⏱️ 게임 시간: 평균 {np.mean(durations):.2f}초")

def create_training_dataset(games: List[Dict[str, Any]]):
    """게임 데이터로부터 학습용 데이터셋을 생성합니다."""
    try:
        processor = DataProcessor()
        builder = DatasetBuilder()
        
        # 게임 데이터 처리
        game_data_list = []
        for game_dict in tqdm(games, desc="Processing games"):
            try:
                game_data = processor.process_game_result(game_dict)
                game_data_list.append(game_data)
            except Exception as e:
                print(f"⚠️ Game processing error: {str(e)}")
                continue
        
        # 학습용 데이터셋 구축
        dataset = builder.build_training_dataset(game_data_list)
        
        # 데이터셋 저장
        builder.save_dataset(dataset, config.PROCESSED_DATA_FILE)
        
        # 통계 출력
        from training.data_utils import analyze_dataset
        stats = analyze_dataset(dataset)
        
        print(f"📊 학습 데이터셋 통계:")
        print(f"  총 샘플 수: {stats['total_samples']:,}")
        print(f"  상태 형태: {stats['state_shape']}")
        print(f"  가치 분포: 평균 {stats['value_stats']['mean']:.3f} ± {stats['value_stats']['std']:.3f}")
        print(f"  정책 엔트로피: {stats['policy_entropy']:.3f}")
        
    except Exception as e:
        print(f"❌ 데이터셋 생성 오류: {str(e)}")

if __name__ == "__main__":
    # CUDA 멀티프로세싱 설정
    mp.set_start_method('spawn', force=True)
    
    main() 