#!/usr/bin/env python3
"""
HYDRA-Onco 전처리 데이터 로드 및 확인 스크립트
"""

import numpy as np
import json
import pandas as pd

def load_triplet_data():
    """전처리된 삼중항 데이터 로드"""

    print("="*80)
    print("✅ HYDRA-Onco 삼중항 데이터 로드")
    print("="*80)

    # 1️⃣ 데이터 로드
    print("\n[1] 데이터 로드 중...")
    data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)

    X = data['omics']                    # 특성 (110216, 2213)
    y = data['ic50']                     # IC50 값 (110216,)
    cell_ids = data['triplet_df_cells']  # 세포주 ID
    drug_ids = data['triplet_df_drugs']  # 약물 ID

    print(f"✓ 특성 (X): {X.shape}")
    print(f"✓ 타겟 (y): {y.shape}")
    print(f"✓ 세포주 ID: {len(cell_ids)}개")
    print(f"✓ 약물 ID: {len(drug_ids)}개")

    # 2️⃣ 메타데이터 로드
    print("\n[2] 메타데이터 로드 중...")
    with open('preprocessed_triplet/metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"✓ 특성명: {len(metadata['feature_names'])}개")
    print(f"✓ 약물명: {len(metadata['drug_names'])}개")
    print(f"✓ 세포주명: {len(metadata['cell_names'])}개")

    return X, y, metadata, cell_ids, drug_ids


def get_split_indices(metadata, split_type='random'):
    """분할 인덱스 가져오기

    Args:
        metadata: 메타데이터 dict
        split_type: 'random', 'drug_blind', 'cell_blind' 중 하나

    Returns:
        train_idx, val_idx, test_idx
    """

    split_name = {
        'random': 'random_split',
        'drug_blind': 'drug_blind_split',
        'cell_blind': 'cell_blind_split'
    }.get(split_type, 'random_split')

    split = metadata['split_info'][split_name]

    return (
        np.array(split['train_idx']),
        np.array(split['val_idx']),
        np.array(split['test_idx'])
    )


def show_data_info(X, y, metadata):
    """데이터 정보 출력"""

    print("\n[3] 데이터 샘플 확인")
    print("\n특성 데이터 (처음 5개 샘플, 처음 5개 특성):")
    print(X[:5, :5])

    print("\nIC50 값 (처음 10개):")
    print(y[:10])

    # 4️⃣ 분할 정보
    print("\n[4] 분할 정보")
    split_info = metadata['split_info']

    print("\n📍 Random Split:")
    train_idx = split_info['random_split']['train_idx']
    val_idx = split_info['random_split']['val_idx']
    test_idx = split_info['random_split']['test_idx']
    print(f"  Train: {len(train_idx):,}개")
    print(f"  Val:   {len(val_idx):,}개")
    print(f"  Test:  {len(test_idx):,}개")

    print("\n📍 Drug Blind Split:")
    train_idx2 = split_info['drug_blind_split']['train_idx']
    val_idx2 = split_info['drug_blind_split']['val_idx']
    test_idx2 = split_info['drug_blind_split']['test_idx']
    test_drugs = split_info['drug_blind_split']['test_drugs']
    print(f"  Train: {len(train_idx2):,}개")
    print(f"  Val:   {len(val_idx2):,}개")
    print(f"  Test:  {len(test_idx2):,}개 (약물 {len(test_drugs)}개 격리)")

    print("\n📍 Cell Blind Split:")
    train_idx3 = split_info['cell_blind_split']['train_idx']
    val_idx3 = split_info['cell_blind_split']['val_idx']
    test_idx3 = split_info['cell_blind_split']['test_idx']
    test_cells = split_info['cell_blind_split']['test_cells']
    print(f"  Train: {len(train_idx3):,}개")
    print(f"  Val:   {len(val_idx3):,}개")
    print(f"  Test:  {len(test_idx3):,}개 (세포주 {len(test_cells)}개 격리)")

    # 5️⃣ 특성명 확인
    print("\n[5] 특성명 분포")
    cn_features = [f for f in metadata['feature_names'] if f.startswith('CN_')]
    expr_features = [f for f in metadata['feature_names'] if f.startswith('EXPR_')]
    meth_features = [f for f in metadata['feature_names'] if f.startswith('METH_')]

    print(f"  Copy Number: {len(cn_features)}개")
    print(f"  Expression: {len(expr_features)}개")
    print(f"  Methylation: {len(meth_features)}개")

    # 6️⃣ 통계
    print("\n[6] 데이터 통계")
    print(f"\n특성 통계:")
    print(f"  평균: {X.mean():.6f}")
    print(f"  표준편차: {X.std():.6f}")
    print(f"  범위: [{X.min():.2f}, {X.max():.2f}]")

    print(f"\nIC50 통계:")
    print(f"  평균: {y.mean():.6f}")
    print(f"  표준편차: {y.std():.6f}")
    print(f"  범위: [{y.min():.2f}, {y.max():.2f}]")


if __name__ == '__main__':
    # 데이터 로드
    X, y, metadata, cell_ids, drug_ids = load_triplet_data()

    # 정보 출력
    show_data_info(X, y, metadata)

    # 분할 테스트
    print("\n[7] 분할 테스트")
    train_idx, val_idx, test_idx = get_split_indices(metadata, 'drug_blind')

    print(f"\nDrug Blind 분할 사용 예제:")
    print(f"  X_train shape: {X[train_idx].shape}")
    print(f"  y_train shape: {y[train_idx].shape}")
    print(f"  X_test shape: {X[test_idx].shape}")
    print(f"  y_test shape: {y[test_idx].shape}")

    print("\n" + "="*80)
    print("✨ 데이터 로드 완료! 모델 학습 준비됨 ✨")
    print("="*80)
