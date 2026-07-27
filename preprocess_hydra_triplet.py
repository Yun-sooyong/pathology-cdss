#!/usr/bin/env python3
"""
HYDRA-Onco 삼중항(Triplet) 전처리 파이프라인
(세포주, 약물) 단위의 삼중항 데이터 생성
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYDRA-Onco 삼중항 전처리 파이프라인 시작")
print("=" * 80)

# ============================================================================
# Step 1: 데이터 로드
# ============================================================================
print("\n[Step 1] 데이터 로드 중...")
data_dir = 'data'

ic50 = pd.read_csv(f'{data_dir}/GDSC_IC50.csv')
expr = pd.read_csv(f'{data_dir}/genomic_expression_561celllines_697genes_demap_features.csv', index_col=0)
meth = pd.read_csv(f'{data_dir}/genomic_methylation_561celllines_808genes_demap_features.csv', index_col=0)
cn = pd.read_csv(f'{data_dir}/genomic_copynumber_561celllines_710genes_demap_features.csv', index_col=0)

print(f"✓ IC50 loaded: {ic50.shape}")
print(f"✓ Expression loaded: {expr.shape}")
print(f"✓ Methylation loaded: {meth.shape}")
print(f"✓ Copy Number loaded: {cn.shape}")

# ============================================================================
# Step 2: ID 매칭 및 교집합
# ============================================================================
print("\n[Step 2] ID 매칭 및 교집합 찾기...")

# 오믹스 데이터의 세포주 ID
omics_cells = set(expr.index) & set(meth.index) & set(cn.index)
print(f"오믹스 데이터의 공통 세포주: {len(omics_cells)}개")

# IC50 데이터의 세포주 (열 이름)
ic50_cells = set(ic50.columns[1:])
print(f"IC50 데이터의 세포주: {len(ic50_cells)}개")

# 약물
ic50_drugs = ic50.iloc[:, 0].tolist()
print(f"IC50 데이터의 약물: {len(ic50_drugs)}개")

# 교집합
common_cells = sorted(omics_cells & ic50_cells)
print(f"공통 세포주: {len(common_cells)}개 ✓")

# 데이터 필터링
expr_filtered = expr.loc[common_cells].copy()
meth_filtered = meth.loc[common_cells].copy()
cn_filtered = cn.loc[common_cells].copy()
ic50_filtered = ic50[['Unnamed: 0'] + common_cells].copy()

print(f"✓ 정렬 후 데이터 크기:")
print(f"  - Expr: {expr_filtered.shape}")
print(f"  - Meth: {meth_filtered.shape}")
print(f"  - CN: {cn_filtered.shape}")
print(f"  - IC50: {ic50_filtered.shape}")

# ============================================================================
# Step 3: 결측치 필터링
# ============================================================================
print("\n[Step 3] 결측치 필터링...")

# 약물 필터링 (결측률 > 50% 제거)
ic50_matrix = ic50_filtered.iloc[:, 1:].values
drug_missing_rate = np.isnan(ic50_matrix).sum(axis=1) / ic50_matrix.shape[1]
valid_drugs_idx = drug_missing_rate <= 0.5
valid_drug_names = ic50_filtered.iloc[valid_drugs_idx, 0].tolist()
ic50_filtered = ic50_filtered.iloc[valid_drugs_idx].copy()
ic50_filtered = ic50_filtered[['Unnamed: 0'] + common_cells]

print(f"약물 필터링: {(~valid_drugs_idx).sum()}개 제거 → {valid_drugs_idx.sum()}개 유지")

# 오믹스 특성 필터링 (표준편차 0인 특성 제거)
def filter_zero_variance_features(df):
    """표준편차 0인 특성 제거"""
    std_devs = df.std()
    return df.loc[:, std_devs > 0]

def handle_missing_values(df, method='drop_columns'):
    """결측치 처리"""
    if method == 'drop_columns':
        return df.dropna(axis=1)
    elif method == 'mean_impute':
        return df.fillna(df.mean())
    return df

expr_filtered = filter_zero_variance_features(expr_filtered)
expr_filtered = handle_missing_values(expr_filtered, method='drop_columns')

meth_filtered = filter_zero_variance_features(meth_filtered)
meth_filtered = handle_missing_values(meth_filtered, method='mean_impute')

cn_filtered = filter_zero_variance_features(cn_filtered)
cn_filtered = handle_missing_values(cn_filtered, method='drop_columns')

print(f"✓ 필터링 후:")
print(f"  - Expr: {expr_filtered.shape}")
print(f"  - Meth: {meth_filtered.shape}")
print(f"  - CN: {cn_filtered.shape}")

# ============================================================================
# Step 4: 삼중항(Triplet) 구성 (분할 전)
# ============================================================================
print("\n[Step 4] 삼중항(Triplet) 구성...")

triplets = []
missing_count = 0

# 모든 (세포주, 약물) 조합 순회
for drug_idx, drug_name in enumerate(valid_drug_names):
    for cell_idx, cell_name in enumerate(common_cells):
        ic50_value = ic50_filtered.iloc[drug_idx, cell_idx + 1]

        # IC50이 NA가 아닌 경우만 포함
        if not pd.isna(ic50_value):
            triplets.append({
                'cell_id': cell_name,
                'drug_id': drug_name,
                'drug_idx': drug_idx,
                'cell_idx': cell_idx,
                'ic50': ic50_value
            })
        else:
            missing_count += 1

triplet_df = pd.DataFrame(triplets)
print(f"총 삼중항 생성: {len(triplet_df)}개")
print(f"제외된 결측 조합: {missing_count}개")

# ============================================================================
# Step 5: 특성 병합 및 정규화 (분할 전)
# ============================================================================
print("\n[Step 5] 특성 병합 및 정규화...")

# 특성 병합 (열 이름 중복 방지를 위해 prefix 추가)
cn_filtered.columns = ['CN_' + col for col in cn_filtered.columns]
expr_filtered.columns = ['EXPR_' + col for col in expr_filtered.columns]
meth_filtered.columns = ['METH_' + col for col in meth_filtered.columns]

omics_merged = pd.concat(
    [cn_filtered, expr_filtered, meth_filtered],
    axis=1
)
print(f"✓ 병합 후 특성 개수: {omics_merged.shape[1]}")
print(f"  - Copy Number: {cn_filtered.shape[1]}")
print(f"  - Expression: {expr_filtered.shape[1]}")
print(f"  - Methylation: {meth_filtered.shape[1]}")

# 특성명 저장
feature_names = list(omics_merged.columns)

# 정규화 (아직 전체 데이터에서)
print("정규화 계산 중... (최종 분할 후 적용)")
scaler_omics = StandardScaler()
omics_scaled = pd.DataFrame(
    scaler_omics.fit_transform(omics_merged),
    index=omics_merged.index,
    columns=omics_merged.columns
)

# IC50 정규화
scaler_ic50 = StandardScaler()
ic50_values = ic50_filtered.iloc[:, 1:].values
ic50_scaled = scaler_ic50.fit_transform(ic50_values)

print("✓ 정규화 완료")

# ============================================================================
# Step 6: 삼중항 데이터 매트릭스 생성
# ============================================================================
print("\n[Step 6] 삼중항 데이터 매트릭스 생성...")

triplet_omics = np.zeros((len(triplet_df), omics_scaled.shape[1]))
triplet_ic50 = np.zeros(len(triplet_df))

for idx, row in triplet_df.iterrows():
    cell_name = row['cell_id']
    drug_idx = row['drug_idx']

    # 세포주 특성
    cell_pos = list(omics_scaled.index).index(cell_name)
    triplet_omics[idx] = omics_scaled.iloc[cell_pos].values

    # IC50 값
    triplet_ic50[idx] = ic50_scaled[drug_idx, list(common_cells).index(cell_name)]

print(f"✓ 삼중항 매트릭스:")
print(f"  - 특성: {triplet_omics.shape}")
print(f"  - IC50: {triplet_ic50.shape}")

# ============================================================================
# Step 7: 데이터 분할 (3가지 방식)
# ============================================================================
print("\n[Step 7] 데이터 분할...")

# 분할 방식 1: 무작위 분할 (Random Split)
print("\n[분할 방식 1] 무작위 분할 (70/10/20)")
indices = np.arange(len(triplet_df))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, random_state=42)

random_split = {
    'train_idx': train_idx,
    'val_idx': val_idx,
    'test_idx': test_idx,
}
print(f"  Train: {len(train_idx)} ({len(train_idx)/len(triplet_df)*100:.1f}%)")
print(f"  Val: {len(val_idx)} ({len(val_idx)/len(triplet_df)*100:.1f}%)")
print(f"  Test: {len(test_idx)} ({len(test_idx)/len(triplet_df)*100:.1f}%)")

# 분할 방식 2: 약물 blind (일부 약물을 통째로 test로)
print("\n[분할 방식 2] 약물 blind (10% 약물 격리)")
unique_drugs = triplet_df['drug_id'].unique()
n_test_drugs = max(1, int(len(unique_drugs) * 0.1))
test_drugs = np.random.RandomState(42).choice(unique_drugs, n_test_drugs, replace=False)
test_drugs_set = set(test_drugs)

drug_blind_test_idx = triplet_df[triplet_df['drug_id'].isin(test_drugs_set)].index.values
drug_blind_train_idx = triplet_df[~triplet_df['drug_id'].isin(test_drugs_set)].index.values

# train/val 분할
train_idx2, val_idx2 = train_test_split(
    drug_blind_train_idx, test_size=0.15, random_state=42
)

drug_blind_split = {
    'train_idx': train_idx2,
    'val_idx': val_idx2,
    'test_idx': drug_blind_test_idx,
    'test_drugs': test_drugs.tolist(),
}
print(f"  테스트 약물: {n_test_drugs}개 ({n_test_drugs/len(unique_drugs)*100:.1f}%)")
print(f"  Train: {len(train_idx2)} ({len(train_idx2)/len(triplet_df)*100:.1f}%)")
print(f"  Val: {len(val_idx2)} ({len(val_idx2)/len(triplet_df)*100:.1f}%)")
print(f"  Test: {len(drug_blind_test_idx)} ({len(drug_blind_test_idx)/len(triplet_df)*100:.1f}%)")

# 분할 방식 3: 세포주 blind (일부 세포주를 통째로 test로)
print("\n[분할 방식 3] 세포주 blind (10% 세포주 격리)")
unique_cells = triplet_df['cell_id'].unique()
n_test_cells = max(1, int(len(unique_cells) * 0.1))
test_cells = np.random.RandomState(42).choice(unique_cells, n_test_cells, replace=False)
test_cells_set = set(test_cells)

cell_blind_test_idx = triplet_df[triplet_df['cell_id'].isin(test_cells_set)].index.values
cell_blind_train_idx = triplet_df[~triplet_df['cell_id'].isin(test_cells_set)].index.values

train_idx3, val_idx3 = train_test_split(
    cell_blind_train_idx, test_size=0.15, random_state=42
)

cell_blind_split = {
    'train_idx': train_idx3,
    'val_idx': val_idx3,
    'test_idx': cell_blind_test_idx,
    'test_cells': test_cells.tolist(),
}
print(f"  테스트 세포주: {n_test_cells}개 ({n_test_cells/len(unique_cells)*100:.1f}%)")
print(f"  Train: {len(train_idx3)} ({len(train_idx3)/len(triplet_df)*100:.1f}%)")
print(f"  Val: {len(val_idx3)} ({len(val_idx3)/len(triplet_df)*100:.1f}%)")
print(f"  Test: {len(cell_blind_test_idx)} ({len(cell_blind_test_idx)/len(triplet_df)*100:.1f}%)")

# ============================================================================
# Step 8: 저장
# ============================================================================
print("\n[Step 8] 전처리된 데이터 저장...")

output_dir = 'preprocessed_triplet'
os.makedirs(output_dir, exist_ok=True)

# 기본 데이터 저장
np.savez(
    f'{output_dir}/triplet_data.npz',
    omics=triplet_omics,
    ic50=triplet_ic50,
    triplet_df_cells=triplet_df['cell_id'].values,
    triplet_df_drugs=triplet_df['drug_id'].values,
)

# 분할 정보 저장
split_info = {
    'random_split': {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in random_split.items()},
    'drug_blind_split': {k: v.tolist() if isinstance(v, np.ndarray) else v
                         for k, v in drug_blind_split.items()},
    'cell_blind_split': {k: v.tolist() if isinstance(v, np.ndarray) else v
                         for k, v in cell_blind_split.items()},
}

# 메타데이터 저장
metadata = {
    'n_samples': len(triplet_df),
    'n_features': triplet_omics.shape[1],
    'n_drugs': len(valid_drug_names),
    'n_cells': len(common_cells),
    'feature_names': feature_names,
    'drug_names': valid_drug_names,
    'cell_names': common_cells,
    'feature_sources': {
        'copynumber': cn_filtered.shape[1],
        'expression': expr_filtered.shape[1],
        'methylation': meth_filtered.shape[1],
    },
    'split_info': split_info,
}

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ 저장 완료: {output_dir}/")
print(f"  - triplet_data.npz (특성 + IC50 + 메타데이터)")
print(f"  - metadata.json (분할 정보 + 특성명 + 약물명)")

# ============================================================================
# 최종 요약
# ============================================================================
print("\n" + "=" * 80)
print("전처리 완료 요약")
print("=" * 80)
print(f"""
입력 데이터:
  - 세포주: {len(common_cells)}개
  - 약물: {len(valid_drug_names)}개 (필터링: 266 → {len(valid_drug_names)})
  - 특성: {triplet_omics.shape[1]}개
    ├─ Copy Number: {cn_filtered.shape[1]}개
    ├─ Expression: {expr_filtered.shape[1]}개
    └─ Methylation: {meth_filtered.shape[1]}개

삼중항 데이터셋:
  - 총 샘플: {len(triplet_df):,}개 (결측 제외)
  - 특성 행렬: {triplet_omics.shape}
  - IC50 벡터: {triplet_ic50.shape}

데이터 분할:
  1️⃣ 무작위 분할:
     - Train: {len(train_idx):,}개 ({len(train_idx)/len(triplet_df)*100:.1f}%)
     - Val: {len(val_idx):,}개 ({len(val_idx)/len(triplet_df)*100:.1f}%)
     - Test: {len(test_idx):,}개 ({len(test_idx)/len(triplet_df)*100:.1f}%)

  2️⃣ 약물 blind:
     - Train: {len(train_idx2):,}개 ({len(train_idx2)/len(triplet_df)*100:.1f}%)
     - Val: {len(val_idx2):,}개 ({len(val_idx2)/len(triplet_df)*100:.1f}%)
     - Test: {len(drug_blind_test_idx):,}개 (약물 {n_test_drugs}개 격리)

  3️⃣ 세포주 blind:
     - Train: {len(train_idx3):,}개 ({len(train_idx3)/len(triplet_df)*100:.1f}%)
     - Val: {len(val_idx3):,}개 ({len(val_idx3)/len(triplet_df)*100:.1f}%)
     - Test: {len(cell_blind_test_idx):,}개 (세포주 {n_test_cells}개 격리)

정규화: Z-score (평균 ~0, 표준편차 ~1)

출력 파일:
  - triplet_data.npz: 삼중항 데이터 (omics, ic50, cell_ids, drug_ids)
  - metadata.json: 분할 정보, 특성명, 약물명, 세포주명

다음 단계: 모델 개발 (각 분할 방식별로 평가)
""")
print("=" * 80)
