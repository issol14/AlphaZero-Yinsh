#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Optimization Script for AlphaZero Pipeline
===============================================

파이프라인 실행 중 메모리 사용량을 최적화하는 스크립트
"""

import os
import sys
import gc
import torch
import psutil
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MemoryOptimizer:
    """메모리 최적화 클래스"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self):
        """현재 메모리 사용량 반환 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def optimize_memory(self):
        """메모리 최적화 수행"""
        print(f"🧠 메모리 최적화 시작 (현재: {self.get_memory_usage():.1f}MB)")
        
        # 1. Python 가비지 컬렉션
        gc.collect()
        
        # 2. PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   ✅ GPU 캐시 정리 완료")
        
        # 3. 메모리 사용량 확인
        final_memory = self.get_memory_usage()
        saved_memory = self.initial_memory - final_memory
        
        print(f"   📊 메모리 최적화 완료:")
        print(f"      초기: {self.initial_memory:.1f}MB")
        print(f"      최종: {final_memory:.1f}MB")
        print(f"      절약: {saved_memory:.1f}MB")
        
        return saved_memory
    
    def monitor_memory(self, threshold_mb=1000):
        """메모리 사용량 모니터링"""
        current_memory = self.get_memory_usage()
        
        if current_memory > threshold_mb:
            print(f"⚠️ 메모리 사용량 경고: {current_memory:.1f}MB > {threshold_mb}MB")
            self.optimize_memory()
            return True
        return False


def optimize_between_iterations():
    """Iteration 간 메모리 최적화"""
    optimizer = MemoryOptimizer()
    
    print("🔄 Iteration 간 메모리 최적화 수행...")
    saved_memory = optimizer.optimize_memory()
    
    if saved_memory > 100:
        print(f"✅ {saved_memory:.1f}MB 메모리 절약 완료")
    else:
        print(f"ℹ️ 메모리 사용량 정상 ({saved_memory:.1f}MB 절약)")


def optimize_before_training():
    """훈련 전 메모리 최적화"""
    optimizer = MemoryOptimizer()
    
    print("🎯 훈련 전 메모리 최적화 수행...")
    optimizer.optimize_memory()
    
    # GPU 메모리 설정
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리 80% 사용
        print(f"   ✅ GPU 메모리 사용량 제한: 80%")


def optimize_before_evaluation():
    """평가 전 메모리 최적화"""
    optimizer = MemoryOptimizer()
    
    print("🏆 평가 전 메모리 최적화 수행...")
    optimizer.optimize_memory()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaZero 메모리 최적화")
    parser.add_argument("--mode", choices=["between", "training", "evaluation"], 
                       default="between", help="최적화 모드")
    
    args = parser.parse_args()
    
    if args.mode == "between":
        optimize_between_iterations()
    elif args.mode == "training":
        optimize_before_training()
    elif args.mode == "evaluation":
        optimize_before_evaluation()