#!/usr/bin/env python3
"""
HYDRA-Onco 전처리 파이프라인
암 세포주 + IC50 약물 반응 + 다중 오믹스 데이터 통합
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYDRA-Onco 전처리 파이프라인 시작")
print("=" * 80)

# ============================================================================
# Step 1: 데이터 로드
# ============================================================================
print("\n[Step 1] 데이터 로드 중...")
data_dir = 'data'

annotations = pd.read_csv(f'{data_dir}/Cell_lines_annotations_20181226.txt', sep='\t')
ic50 = pd.read_csv(f'{data_dir}/GDSC_IC50.csv')
cn = pd.read_csv(f'{data_dir}/genomic_copynumber_561celllines_710genes_demap_features.csv', index_col=0)
expr = pd.read_csv(f'{data_dir}/genomic_expression_561celllines_697genes_demap_features.csv', index_col=0)
meth = pd.read_csv(f'{data_dir}/genomic_methylation_561celllines_808genes_demap_features.csv', index_col=0)

print(f"✓ Annotations loaded: {annotations.shape}")
print(f"✓ IC50 loaded: {ic50.shape}")
print(f"✓ Copy Number loaded: {cn.shape}")
print(f"✓ Expression loaded: {expr.shape}")
print(f"✓ Methylation loaded: {meth.shape}")

# ============================================================================
# Step 2: 데이터 정렬 (Alignment)
# ============================================================================
print("\n[Step 2] 데이터 정렬 (교집합 선택)...")

# 오믹스 데이터의 세포주 ID
omics_cells = set(cn.index)
print(f"오믹스 데이터의 세포주: {len(omics_cells)}개")

# IC50 데이터의 세포주 (열 이름)
ic50_cells = set(ic50.columns[1:])  # 첫 번째 열은 약물명
print(f"IC50 데이터의 세포주: {len(ic50_cells)}개")

# 교집합
common_cells = sorted(omics_cells & ic50_cells)
print(f"공통 세포주: {len(common_cells)}개 ✓")

# 데이터 필터링
cn_filtered = cn.loc[common_cells].copy()
expr_filtered = expr.loc[common_cells].copy()
meth_filtered = meth.loc[common_cells].copy()
ic50_filtered = ic50[['Unnamed: 0'] + common_cells].copy()

print(f"✓ 정렬 후 데이터 크기:")
print(f"  - CN: {cn_filtered.shape}")
print(f"  - Expr: {expr_filtered.shape}")
print(f"  - Meth: {meth_filtered.shape}")
print(f"  - IC50: {ic50_filtered.shape}")

# ============================================================================
# Step 3: 결측치 분석 및 필터링
# ============================================================================
print("\n[Step 3] 결측치 필터링...")

# IC50 약물 필터링 (결측률 > 50% 제거)
ic50_matrix = ic50.iloc[:, 1:].values
drug_missing_rate = np.isnan(ic50_matrix).sum(axis=1) / ic50_matrix.shape[1]
valid_drugs_idx = drug_missing_rate <= 0.5
valid_drug_names = ic50.iloc[valid_drugs_idx, 0].tolist()
ic50_filtered = ic50.iloc[valid_drugs_idx].copy()
ic50_filtered = ic50_filtered[['Unnamed: 0'] + common_cells]

print(f"약물 필터링: {(~valid_drugs_idx).sum()}개 제거 → {valid_drugs_idx.sum()}개 유지")

# 오믹스 특성 필터링 (상관계수 0인 특성 제거)
print("오믹스 특성 필터링 중... (상관계수 0 특성 제거)")

def filter_zero_variance_features(df):
    """표준편차 0인 특성 제거"""
    std_devs = df.std()
    return df.loc[:, std_devs > 0]

def handle_missing_values(df, method='drop_columns'):
    """결측치 처리 (열 단위 제거 또는 행 단위 제거)"""
    if method == 'drop_columns':
        # 결측치가 있는 열 제거
        return df.dropna(axis=1)
    elif method == 'drop_rows':
        # 결측치가 있는 행 제거
        return df.dropna(axis=0)
    elif method == 'mean_impute':
        # 평균값으로 결측치 대체
        return df.fillna(df.mean())
    return df

cn_filtered = filter_zero_variance_features(cn_filtered)
cn_filtered = handle_missing_values(cn_filtered, method='drop_columns')

expr_filtered = filter_zero_variance_features(expr_filtered)
expr_filtered = handle_missing_values(expr_filtered, method='drop_columns')

meth_filtered = filter_zero_variance_features(meth_filtered)
meth_filtered = handle_missing_values(meth_filtered, method='drop_columns')

print(f"✓ 필터링 후:")
print(f"  - CN: {cn_filtered.shape} (결측치 있는 열 제거)")
print(f"  - Expr: {expr_filtered.shape}")
print(f"  - Meth: {meth_filtered.shape}")

# ============================================================================
# Step 4: 정규화 (Normalization)
# ============================================================================
print("\n[Step 4] 데이터 정규화 (Z-score)...")

# 각 오믹스 데이터 정규화
scaler_cn = StandardScaler()
cn_normalized = pd.DataFrame(
    scaler_cn.fit_transform(cn_filtered),
    index=cn_filtered.index,
    columns=cn_filtered.columns
)

scaler_expr = StandardScaler()
expr_normalized = pd.DataFrame(
    scaler_expr.fit_transform(expr_filtered),
    index=expr_filtered.index,
    columns=expr_filtered.columns
)

scaler_meth = StandardScaler()
meth_normalized = pd.DataFrame(
    scaler_meth.fit_transform(meth_filtered),
    index=meth_filtered.index,
    columns=meth_filtered.columns
)

# IC50 정규화
ic50_values = ic50_filtered.iloc[:, 1:].values
scaler_ic50 = StandardScaler()
ic50_normalized = scaler_ic50.fit_transform(ic50_values)

print(f"✓ 정규화 완료")
print(f"  모든 특성: 평균 ~0, 표준편차 ~1")

# ============================================================================
# Step 5: 특성 병합 (Feature Merging)
# ============================================================================
print("\n[Step 5] 오믹스 특성 병합...")

# 전체 특성 병합
omics_merged = pd.concat(
    [cn_normalized, expr_normalized, meth_normalized],
    axis=1
)

print(f"✓ 병합 후 특성 개수: {omics_merged.shape[1]}")
print(f"  - Copy Number: {cn_normalized.shape[1]}")
print(f"  - Expression: {expr_normalized.shape[1]}")
print(f"  - Methylation: {meth_normalized.shape[1]}")
print(f"  - Total: {omics_merged.shape[1]}")

# ============================================================================
# Step 6: 데이터 분할 (Train/Val/Test Split)
# ============================================================================
print("\n[Step 6] 데이터 분할 (Train/Val/Test)...")

# 세포주 기반 분할
cell_indices = np.arange(len(common_cells))
train_idx, temp_idx = train_test_split(cell_indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_cells = [common_cells[i] for i in train_idx]
val_cells = [common_cells[i] for i in val_idx]
test_cells = [common_cells[i] for i in test_idx]

print(f"✓ 분할 완료:")
print(f"  - Train: {len(train_cells)}개 세포주 ({len(train_cells)/len(common_cells)*100:.1f}%)")
print(f"  - Validation: {len(val_cells)}개 세포주 ({len(val_cells)/len(common_cells)*100:.1f}%)")
print(f"  - Test: {len(test_cells)}개 세포주 ({len(test_cells)/len(common_cells)*100:.1f}%)")

# ============================================================================
# Step 7: 최종 데이터셋 생성
# ============================================================================
print("\n[Step 7] 최종 데이터셋 생성...")

# 세포주별 선택
X_train = omics_merged.loc[train_cells]
X_val = omics_merged.loc[val_cells]
X_test = omics_merged.loc[test_cells]

# IC50 데이터 정규화
ic50_filtered_set_index = ic50_filtered.set_index('Unnamed: 0')
scaler_ic50_val = StandardScaler()
ic50_normalized_values = scaler_ic50_val.fit_transform(ic50_filtered_set_index.values)
ic50_valid = pd.DataFrame(
    ic50_normalized_values,
    index=ic50_filtered_set_index.index,
    columns=ic50_filtered_set_index.columns
)

# 공통 세포주만 선택
ic50_valid = ic50_valid[common_cells]

# 세포주별 IC50 데이터 선택 (약물 × 세포주 형식 유지)
y_train = ic50_valid[train_cells].T  # 세포주 × 약물
y_val = ic50_valid[val_cells].T
y_test = ic50_valid[test_cells].T

print(f"✓ 최종 데이터셋:")
print(f"  Train:")
print(f"    X: {X_train.shape} (세포주 × 특성)")
print(f"    y: {y_train.shape} (세포주 × 약물)")
print(f"  Validation:")
print(f"    X: {X_val.shape}")
print(f"    y: {y_val.shape}")
print(f"  Test:")
print(f"    X: {X_test.shape}")
print(f"    y: {y_test.shape}")

# ============================================================================
# Step 8: 저장
# ============================================================================
print("\n[Step 8] 전처리된 데이터 저장...")

output_dir = 'preprocessed_data'
os.makedirs(output_dir, exist_ok=True)

# Train
X_train.to_csv(f'{output_dir}/X_train.csv')
y_train.to_csv(f'{output_dir}/y_train.csv')

# Validation
X_val.to_csv(f'{output_dir}/X_val.csv')
y_val.to_csv(f'{output_dir}/y_val.csv')

# Test
X_test.to_csv(f'{output_dir}/X_test.csv')
y_test.to_csv(f'{output_dir}/y_test.csv')

# Metadata
metadata = {
    'train_cells': train_cells,
    'val_cells': val_cells,
    'test_cells': test_cells,
    'feature_names': list(omics_merged.columns),
    'drug_names': list(ic50_valid.index)
}

import json
with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ 저장 완료: {output_dir}/")
print(f"  - X_train.csv, y_train.csv")
print(f"  - X_val.csv, y_val.csv")
print(f"  - X_test.csv, y_test.csv")
print(f"  - metadata.json")

# ============================================================================
# 최종 요약
# ============================================================================
print("\n" + "=" * 80)
print("전처리 완료 요약")
print("=" * 80)
print(f"""
입력 데이터:
  - Cell Annotations: 1,461개 세포주
  - IC50: 266개 약물 × 970개 세포주
  - Omics (3개): 561개 세포주 × 2,215개 특성

전처리 과정:
  1. ✓ 데이터 정렬: 공통 세포주 {len(common_cells)}개 선택
  2. ✓ 결측치 필터링: 약물 결측률 >50% 제거
  3. ✓ 정규화: Z-score (평균 0, 표준편차 1)
  4. ✓ 특성 병합: 3개 오믹스 병합 (2,215개 특성)
  5. ✓ 데이터 분할: Train/Val/Test = 70/15/15

출력 데이터:
  학습셋: {X_train.shape[0]} 세포주 × {X_train.shape[1]} 특성 → {y_train.shape[1]} 약물
  검증셋: {X_val.shape[0]} 세포주 × {X_val.shape[1]} 특성 → {y_val.shape[1]} 약물
  테스트셋: {X_test.shape[0]} 세포주 × {X_test.shape[1]} 특성 → {y_test.shape[1]} 약물

다음 단계: 머신러닝 모델 개발 (XGBoost, Neural Network 등)
""")
print("=" * 80)
