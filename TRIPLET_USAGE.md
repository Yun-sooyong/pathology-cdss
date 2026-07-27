# HYDRA-Onco 삼중항 데이터 사용 가이드

## 📦 데이터 구조

```
삼중항 = (오믹스 벡터, IC50 값)

각 샘플:
  - 오믹스: 2,213차원 벡터 (CN_708 + EXPR_697 + METH_808)
  - IC50: 스칼라 값 (정규화됨)
  
특징:
  - 총 110,216개 삼중항
  - IC50 결측값 제외됨
  - Z-score 정규화 완료
  - NaN 값 없음
```

---

## 🚀 빠른 시작

### 1️⃣ 기본 로드

```python
import numpy as np
import json

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)

# 특성과 타겟
X = data['omics']        # (110216, 2213)
y = data['ic50']         # (110216,)

# 메타데이터
with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"샘플 수: {X.shape[0]:,}개")
print(f"특성 수: {X.shape[1]}")
print(f"특성명: {metadata['feature_names'][:5]}")
print(f"약물 수: {metadata['n_drugs']}")
```

### 2️⃣ 데이터 분할 로드

```python
import numpy as np
import json

data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X = data['omics']
y = data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

# 분할 정보
split_info = metadata['split_info']

# 무작위 분할
train_idx = split_info['random_split']['train_idx']
val_idx = split_info['random_split']['val_idx']
test_idx = split_info['random_split']['test_idx']

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
```

---

## 🎯 3가지 분할 전략

### 전략 1️⃣: 무작위 분할 (Random Split)

```python
split = metadata['split_info']['random_split']

X_train = X[split['train_idx']]
X_val = X[split['val_idx']]
X_test = X[split['test_idx']]

y_train = y[split['train_idx']]
y_val = y[split['val_idx']]
y_test = y[split['test_idx']]

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Val:   {X_val.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")
```

**사용**: 기본 모델 성능 평가

---

### 전략 2️⃣: 약물 Blind (Drug Blind Split)

```python
split = metadata['split_info']['drug_blind_split']

X_train = X[split['train_idx']]
X_val = X[split['val_idx']]
X_test = X[split['test_idx']]

y_train = y[split['train_idx']]
y_val = y[split['val_idx']]
y_test = y[split['test_idx']]

# 테스트 약물 확인
test_drugs = split['test_drugs']
print(f"테스트 약물 ({len(test_drugs)}개): {test_drugs[:5]}...")

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Val:   {X_val.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples (새로운 약물에 대한 일반화 평가)")
```

**사용**: 학습 중에 본 적 없는 약물에 대한 모델 성능 평가 → **실제 임상 시나리오**

---

### 전략 3️⃣: 세포주 Blind (Cell Blind Split)

```python
split = metadata['split_info']['cell_blind_split']

X_train = X[split['train_idx']]
X_val = X[split['val_idx']]
X_test = X[split['test_idx']]

y_train = y[split['train_idx']]
y_val = y[split['val_idx']]
y_test = y[split['test_idx']]

# 테스트 세포주 확인
test_cells = split['test_cells']
print(f"테스트 세포주 ({len(test_cells)}개): {test_cells[:5]}...")

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Val:   {X_val.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples (새로운 세포주에 대한 일반화 평가)")
```

**사용**: 학습 중에 본 적 없는 세포주에 대한 모델 성능 평가 → **새로운 환자/종양 시나리오**

---

## 🤖 모델 개발 예제

### 예제 1: XGBoost 모델

```python
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X, y = data['omics'], data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

# 분할 선택 (약물 blind 사용)
split = metadata['split_info']['drug_blind_split']
X_train, y_train = X[split['train_idx']], y[split['train_idx']]
X_val, y_val = X[split['val_idx']], y[split['val_idx']]
X_test, y_test = X[split['test_idx']], y[split['test_idx']]

# 모델 학습
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 평가
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")
```

### 예제 2: Neural Network (PyTorch)

```python
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X, y = data['omics'], data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

# 분할 선택
split = metadata['split_info']['drug_blind_split']
X_train = torch.FloatTensor(X[split['train_idx']])
y_train = torch.FloatTensor(y[split['train_idx']])

X_test = torch.FloatTensor(X[split['test_idx']])
y_test = torch.FloatTensor(y[split['test_idx']])

# 모델 정의
class DenseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = DenseNet(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

# 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    r2 = r2_score(y_test.numpy(), y_pred)
    print(f"Test R²: {r2:.4f}")
```

### 예제 3: 앙상블 모델 (3가지 분할로 학습)

```python
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import r2_score

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X, y = data['omics'], data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

results = {}

# 3가지 분할 전략에서 모델 학습 및 평가
for split_name in ['random_split', 'drug_blind_split', 'cell_blind_split']:
    split = metadata['split_info'][split_name]
    
    X_train = X[split['train_idx']]
    y_train = y[split['train_idx']]
    X_test = X[split['test_idx']]
    y_test = y[split['test_idx']]
    
    # 모델 학습
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 평가
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    results[split_name] = {
        'model': model,
        'r2': r2,
        'n_test': len(y_test)
    }
    
    print(f"{split_name}: R² = {r2:.4f} (테스트 샘플 {len(y_test):,}개)")

# 앙상블 예측
test_splits = [
    ('random', metadata['split_info']['random_split']['test_idx']),
    ('drug', metadata['split_info']['drug_blind_split']['test_idx']),
    ('cell', metadata['split_info']['cell_blind_split']['test_idx']),
]

for name, test_idx in test_splits:
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    # 3개 모델의 예측 평균
    predictions = [
        results['random_split']['model'].predict(X_test),
        results['drug_blind_split']['model'].predict(X_test),
        results['cell_blind_split']['model'].predict(X_test),
    ]
    
    y_pred_ensemble = np.mean(predictions, axis=0)
    r2 = r2_score(y_test, y_pred_ensemble)
    
    print(f"앙상블 ({name}_split): R² = {r2:.4f}")
```

---

## 📊 특성 중요도 분석

```python
import numpy as np
import json
import xgboost as xgb
import matplotlib.pyplot as plt

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X, y = data['omics'], data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

# 분할
split = metadata['split_info']['random_split']
X_train = X[split['train_idx']]
y_train = y[split['train_idx']]

# 모델 학습
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 특성 중요도
importances = model.feature_importances_
top_idx = np.argsort(importances)[-20:]  # 상위 20개
top_features = [metadata['feature_names'][i] for i in top_idx]
top_importances = importances[top_idx]

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances)
plt.xlabel('Feature Importance')
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("특성 중요도 상위 10:")
for i in range(min(10, len(top_features))):
    print(f"  {i+1}. {top_features[i]}: {top_importances[i]:.6f}")
```

---

## 🔍 데이터 탐색

```python
import numpy as np
import json
import pandas as pd

# 데이터 로드
data = np.load('preprocessed_triplet/triplet_data.npz', allow_pickle=True)
X, y = data['omics'], data['ic50']

with open('preprocessed_triplet/metadata.json', 'r') as f:
    metadata = json.load(f)

# 통계
print("=" * 50)
print("데이터셋 통계")
print("=" * 50)

print(f"\n샘플 수: {X.shape[0]:,}개")
print(f"특성 수: {X.shape[1]}")
print(f"약물 수: {len(metadata['drug_names'])}")
print(f"세포주 수: {len(metadata['cell_names'])}")

print(f"\n특성 통계:")
print(f"  평균: {X.mean():.6f}")
print(f"  표준편차: {X.std():.6f}")
print(f"  최소값: {X.min():.2f}")
print(f"  최대값: {X.max():.2f}")

print(f"\nIC50 통계:")
print(f"  평균: {y.mean():.6f}")
print(f"  표준편차: {y.std():.6f}")
print(f"  최소값: {y.min():.2f}")
print(f"  최대값: {y.max():.2f}")

# 특성별 통계
feature_stats = pd.DataFrame({
    'mean': X.mean(axis=0),
    'std': X.std(axis=0),
    'min': X.min(axis=0),
    'max': X.max(axis=0),
}, index=metadata['feature_names'])

print(f"\n특성별 통계 (상위 5):")
print(feature_stats.head())
```

---

## ⚠️ 주의사항

### ✅ 해야 할 것
- 무조건 train/val/test 분할 유지
- 3가지 분할 전략 모두 평가 (일반화 성능 검증)
- 약물 blind와 세포주 blind는 각각 다른 성능 나옴 (정상)

### ❌ 하지 말아야 할 것
- X와 y의 순서 섞기 (인덱스 일치 필수)
- 다시 정규화하기 (이미 정규화됨)
- train 데이터만 사용해서 test 평가

---

## 📁 파일 위치

```
preprocessed_triplet/
├── triplet_data.npz          # 특성 + IC50 + 메타정보
└── metadata.json              # 분할 정보 + 특성명 + 약물명
```

---

## 🆘 문제 해결

### Q: 메모리 부족으로 데이터 로드 안 됨
```python
# 배치 처리로 해결
batch_size = 10000
n_batches = (X.shape[0] + batch_size - 1) // batch_size

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, X.shape[0])
    X_batch = X[start:end]
    y_batch = y[start:end]
    # 모델 학습
```

### Q: 특정 약물/세포주만 필터링하고 싶음
```python
cell_ids = data['triplet_df_cells']
drug_ids = data['triplet_df_drugs']

# 특정 약물만
drug_mask = drug_ids == 'GDSC:1001'
X_filtered = X[drug_mask]
y_filtered = y[drug_mask]

# 특정 세포주만
cell_mask = cell_ids == 'ACH-000001'
X_filtered = X[cell_mask]
y_filtered = y[cell_mask]
```

---

## 📚 다음 단계

1. **기본 모델 학습** → XGBoost로 baseline 구축
2. **3가지 분할 평가** → 각 전략별 성능 비교
3. **특성 분석** → 상위 특성 해석
4. **하이퍼파라미터 튜닝** → 최적 파라미터 찾기
5. **앙상블 모델** → 여러 모델 조합
6. **생물학적 해석** → 상위 특성의 생물학적 의미 분석

---

**작성일**: 2026-07-13  
**데이터셋**: HYDRA-Onco 삼중항  
**버전**: 1.0
