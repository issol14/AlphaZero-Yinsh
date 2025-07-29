#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Compatibility Test Script
=============================

selfplay.py와 train.py 간의 데이터 호환성을 테스트하는 스크립트
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import create_training_batch


def test_data_folder(data_folder: str):
    """데이터 폴더 테스트"""
    print(f"🔍 데이터 폴더 테스트: {data_folder}")
    print("=" * 60)
    
    # 폴더 존재 확인
    if not os.path.exists(data_folder):
        print(f"❌ 폴더가 존재하지 않습니다: {data_folder}")
        return False
    
    # 파일 목록 확인
    files = [f for f in os.listdir(data_folder) if f.endswith('.pt')]
    print(f"📂 발견된 .pt 파일: {len(files)}개")
    
    for file in files:
        print(f"   📄 {file}")
    
    if not files:
        print("❌ 훈련 데이터 파일이 없습니다!")
        return False
    
    # 데이터 로딩 테스트
    print(f"\n📖 데이터 로딩 테스트...")
    states, policies, values = create_training_batch(data_folder, batch_size=32)
    
    if states is None:
        print("❌ 데이터 로딩 실패!")
        return False
    
    print(f"✅ 데이터 로딩 성공!")
    print(f"   States: {states.shape}")
    print(f"   Policies: {policies.shape}")
    print(f"   Values: {values.shape}")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 호환성 테스트")
    parser.add_argument("--data-folder", type=str, required=True, 
                       help="테스트할 데이터 폴더")
    
    args = parser.parse_args()
    
    success = test_data_folder(args.data_folder)
    
    if success:
        print("\n✅ 데이터 호환성 테스트 통과!")
    else:
        print("\n❌ 데이터 호환성 테스트 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main()