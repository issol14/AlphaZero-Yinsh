"""
YINSH AlphaZero 셀프플레이 데이터 분석 스크립트
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import os
from typing import List, Dict, Any
# import seaborn as sns  # 선택적 의존성

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 로컬 임포트
try:
    import training.data_config as config
    from training.data_utils import load_selfplay_games, DatasetBuilder, analyze_dataset
except ImportError:
    # 직접 실행하는 경우
    import data_config as config
    from data_utils import load_selfplay_games, DatasetBuilder, analyze_dataset

# 한글 폰트 설정 (시각화용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """셀프플레이 데이터 분석 클래스"""
    
    def __init__(self):
        self.games = []
        self.dataset = None
        self.stats = {}
    
    def load_data(self):
        """데이터를 로드합니다."""
        print("📂 데이터 로딩 중...")
        
        # 셀프플레이 게임 로드
        if config.SELFPLAY_DATA_FILE.exists():
            self.games = load_selfplay_games(config.SELFPLAY_DATA_FILE)
            print(f"✅ {len(self.games)}개 게임 로드됨")
        else:
            print("⚠️ 셀프플레이 데이터 파일이 없습니다")
            return False
        
        # 처리된 데이터셋 로드
        if config.PROCESSED_DATA_FILE.exists():
            builder = DatasetBuilder()
            self.dataset = builder.load_dataset(config.PROCESSED_DATA_FILE)
            self.stats = analyze_dataset(self.dataset)
            print(f"✅ 학습 데이터셋 로드됨: {self.stats['total_samples']:,} 샘플")
        else:
            print("⚠️ 처리된 데이터셋 파일이 없습니다")
        
        return True
    
    def analyze_games(self) -> Dict[str, Any]:
        """게임 데이터를 분석합니다."""
        if not self.games:
            return {}
        
        analysis = {
            'total_games': len(self.games),
            'game_lengths': [],
            'durations': [],
            'agent_combinations': {},
            'win_rates': {},
            'phase_distribution': {},
            'move_distribution': {}
        }
        
        for game in self.games:
            # 기본 통계
            analysis['game_lengths'].append(game.get('game_length', 0))
            analysis['durations'].append(game.get('duration', 0))
            
            # 에이전트 조합
            combo = f"{game.get('white_agent', 'unknown')}_vs_{game.get('black_agent', 'unknown')}"
            if combo not in analysis['agent_combinations']:
                analysis['agent_combinations'][combo] = 0
            analysis['agent_combinations'][combo] += 1
            
            # 승률 분석
            winner = game.get('winner')
            if combo not in analysis['win_rates']:
                analysis['win_rates'][combo] = {'white': 0, 'black': 0, 'draw': 0, 'total': 0}
            
            analysis['win_rates'][combo]['total'] += 1
            
            if winner and hasattr(winner, 'name'):
                if winner.name == 'WHITE':
                    analysis['win_rates'][combo]['white'] += 1
                elif winner.name == 'BLACK':
                    analysis['win_rates'][combo]['black'] += 1
                else:
                    analysis['win_rates'][combo]['draw'] += 1
            else:
                analysis['win_rates'][combo]['draw'] += 1
            
            # 최종 단계 분석
            final_state = game.get('final_state', {})
            phase = str(final_state.get('phase', 'unknown'))
            if phase not in analysis['phase_distribution']:
                analysis['phase_distribution'][phase] = 0
            analysis['phase_distribution'][phase] += 1
        
        # 통계 계산
        if analysis['game_lengths']:
            analysis['avg_game_length'] = np.mean(analysis['game_lengths'])
            analysis['std_game_length'] = np.std(analysis['game_lengths'])
            analysis['min_game_length'] = min(analysis['game_lengths'])
            analysis['max_game_length'] = max(analysis['game_lengths'])
        
        if analysis['durations']:
            analysis['avg_duration'] = np.mean(analysis['durations'])
            analysis['total_duration'] = sum(analysis['durations'])
        
        return analysis
    
    def create_visualizations(self, analysis: Dict[str, Any]):
        """분석 결과를 시각화합니다."""
        if not analysis:
            return
        
        # 플롯 디렉토리 생성
        plots_dir = Path("training/plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 게임 길이 분포
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.hist(analysis['game_lengths'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Game Length Distribution')
        plt.xlabel('Game Length (moves)')
        plt.ylabel('Frequency')
        plt.axvline(analysis['avg_game_length'], color='red', linestyle='--', 
                   label=f'Mean: {analysis["avg_game_length"]:.1f}')
        plt.legend()
        
        # 2. 게임 시간 분포
        plt.subplot(2, 3, 2)
        plt.hist(analysis['durations'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Game Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(analysis['avg_duration'], color='red', linestyle='--',
                   label=f'Mean: {analysis["avg_duration"]:.2f}s')
        plt.legend()
        
        # 3. 에이전트 조합별 게임 수
        plt.subplot(2, 3, 3)
        combos = list(analysis['agent_combinations'].keys())
        counts = list(analysis['agent_combinations'].values())
        plt.bar(combos, counts, alpha=0.7)
        plt.title('Games by Agent Combination')
        plt.xlabel('Agent Combination')
        plt.ylabel('Number of Games')
        plt.xticks(rotation=45)
        
        # 4. 승률 분석
        plt.subplot(2, 3, 4)
        win_data = []
        labels = []
        
        for combo, stats in analysis['win_rates'].items():
            total = stats['total']
            if total > 0:
                white_pct = stats['white'] / total * 100
                black_pct = stats['black'] / total * 100
                draw_pct = stats['draw'] / total * 100
                
                win_data.append([white_pct, black_pct, draw_pct])
                labels.append(combo)
        
        if win_data:
            win_data = np.array(win_data)
            width = 0.6
            x = np.arange(len(labels))
            
            plt.bar(x, win_data[:, 0], width, label='White wins', alpha=0.7)
            plt.bar(x, win_data[:, 1], width, bottom=win_data[:, 0], label='Black wins', alpha=0.7)
            plt.bar(x, win_data[:, 2], width, bottom=win_data[:, 0] + win_data[:, 1], 
                   label='Draws', alpha=0.7)
            
            plt.title('Win Rate by Agent Combination')
            plt.xlabel('Agent Combination')
            plt.ylabel('Win Rate (%)')
            plt.xticks(x, labels, rotation=45)
            plt.legend()
        
        # 5. 최종 게임 단계 분포
        plt.subplot(2, 3, 5)
        phases = list(analysis['phase_distribution'].keys())
        phase_counts = list(analysis['phase_distribution'].values())
        plt.pie(phase_counts, labels=phases, autopct='%1.1f%%')
        plt.title('Final Game Phase Distribution')
        
        # 6. 데이터셋 통계 (있는 경우)
        plt.subplot(2, 3, 6)
        if self.dataset is not None and self.stats:
            value_targets = self.dataset['value_targets'].flatten()
            plt.hist(value_targets, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Value Target Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.axvline(self.stats['value_stats']['mean'], color='red', linestyle='--',
                       label=f'Mean: {self.stats["value_stats"]["mean"]:.3f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No dataset available', ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Dataset Statistics')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'selfplay_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 시각화 저장됨: {plots_dir / 'selfplay_analysis.png'}")
    
    def generate_report(self, analysis: Dict[str, Any]):
        """분석 리포트를 생성합니다."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'game_analysis': analysis,
            'dataset_stats': self.stats if self.stats else {},
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # JSON으로 저장
        with open(config.STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 분석 리포트 저장됨: {config.STATS_FILE}")
        
        # 콘솔에 요약 출력
        self._print_summary(analysis)
        
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """분석 결과를 바탕으로 권장사항을 생성합니다."""
        recommendations = []
        
        if not analysis:
            return recommendations
        
        # 게임 길이 분석
        if analysis.get('avg_game_length', 0) < 30:
            recommendations.append(
                "게임이 너무 짧습니다. MCTS 시뮬레이션 수를 늘리거나 더 다양한 에이전트를 사용해보세요."
            )
        elif analysis.get('avg_game_length', 0) > 150:
            recommendations.append(
                "게임이 너무 깁니다. 조기 종료 조건을 검토하거나 에이전트 성능을 개선해보세요."
            )
        
        # 승률 분석
        for combo, stats in analysis.get('win_rates', {}).items():
            total = stats.get('total', 0)
            if total > 0:
                white_rate = stats.get('white', 0) / total
                black_rate = stats.get('black', 0) / total
                
                if abs(white_rate - black_rate) > 0.2:
                    recommendations.append(
                        f"{combo}에서 승률 불균형이 있습니다 (White: {white_rate:.1%}, Black: {black_rate:.1%})"
                    )
        
        # 데이터셋 크기 확인
        if self.stats and self.stats.get('total_samples', 0) < 10000:
            recommendations.append(
                "학습 샘플이 부족합니다. 더 많은 게임을 생성하는 것을 권장합니다."
            )
        
        # 정책 다양성 확인
        if self.stats and self.stats.get('policy_entropy', 0) < 2.0:
            recommendations.append(
                "정책 다양성이 낮습니다. 온도 설정을 조정하거나 탐험을 늘려보세요."
            )
        
        return recommendations
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """분석 요약을 콘솔에 출력합니다."""
        print("\n" + "="*60)
        print("📊 YINSH 셀프플레이 데이터 분석 요약")
        print("="*60)
        
        if analysis:
            print(f"🎮 총 게임 수: {analysis['total_games']:,}")
            print(f"📏 평균 게임 길이: {analysis.get('avg_game_length', 0):.1f} ± {analysis.get('std_game_length', 0):.1f} 턴")
            print(f"⏱️ 평균 게임 시간: {analysis.get('avg_duration', 0):.2f}초")
            print(f"⏰ 총 수집 시간: {analysis.get('total_duration', 0)/3600:.1f}시간")
            
            print(f"\n🤖 에이전트 조합별 게임 수:")
            for combo, count in analysis['agent_combinations'].items():
                print(f"  {combo}: {count}게임")
            
            print(f"\n🏆 승률 통계:")
            for combo, stats in analysis['win_rates'].items():
                total = stats['total']
                if total > 0:
                    white_pct = stats['white'] / total * 100
                    black_pct = stats['black'] / total * 100
                    draw_pct = stats['draw'] / total * 100
                    print(f"  {combo}: W{white_pct:.1f}% B{black_pct:.1f}% D{draw_pct:.1f}%")
        
        if self.stats:
            print(f"\n📈 학습 데이터셋:")
            print(f"  총 샘플 수: {self.stats['total_samples']:,}")
            print(f"  상태 형태: {self.stats['state_shape']}")
            print(f"  가치 분포: {self.stats['value_stats']['mean']:.3f} ± {self.stats['value_stats']['std']:.3f}")
            print(f"  정책 엔트로피: {self.stats['policy_entropy']:.3f}")
        
        print("="*60)

def main():
    """메인 실행 함수"""
    print("🔍 YINSH 셀프플레이 데이터 분석 시작")
    
    analyzer = DataAnalyzer()
    
    # 데이터 로드
    if not analyzer.load_data():
        print("❌ 데이터 로드 실패")
        return
    
    # 게임 분석
    print("\n📊 게임 데이터 분석 중...")
    analysis = analyzer.analyze_games()
    
    # 시각화 생성
    print("\n📈 시각화 생성 중...")
    analyzer.create_visualizations(analysis)
    
    # 리포트 생성
    print("\n📄 분석 리포트 생성 중...")
    report = analyzer.generate_report(analysis)
    
    # 권장사항 출력
    if report['recommendations']:
        print(f"\n💡 권장사항:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n✅ 데이터 분석 완료!")

if __name__ == "__main__":
    main() 