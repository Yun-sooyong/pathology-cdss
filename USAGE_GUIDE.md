# HYDRA-Onco 전처리 데이터 사용 가이드

## 🚀 빠른 시작

### 전처리된 데이터 로드

```python
import pandas as pd
import numpy as np

# 학습 데이터 로드
X_train = pd.read_csv('preprocessed_data/X_train.csv', index_col=0)
y_train = pd.read_csv('preprocessed_data/y_train.csv', index_col=0)

X_val = pd.read_csv('preprocessed_data/X_val.csv', index_col=0)
y_val = pd.read_csv('preprocessed_data/y_val.csv', index_col=0)

X_test = pd.read_csv('preprocessed_data/X_test.csv', index_col=0)
y_test = pd.read_csv('preprocessed_data/y_test.csv', index_col=0)

# 메타데이터 로드
import json
with open('preprocessed_data/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
print(f"Features: {len(metadata['feature_names'])}")
print(f"Drugs: {len(metadata['drug_names'])}")
```

---

## 📊 데이터 구조 이해하기

### X (특성) 데이터
```python
# Shape: (샘플수, 특성수)
X_train.shape  # (392, 2213)

# 행: 세포주
X_train.index  # ['ACH-000001', 'ACH-000002', ...]

# 열: 유전자 특성 (정규화됨, 평균=0, 표준편차=1)
X_train.columns  # ['AKT3', 'ABI1', 'SH2B3', ...]

# 데이터 확인
X_train.iloc[0, :5]  # 첫 번째 샘플의 첫 5개 특성
```

### y (타겟) 데이터
```python
# Shape: (샘플수, 약물수)
y_train.shape  # (392, 224)

# 행: 세포주 (X_train과 동일)
y_train.index  # ['ACH-000001', 'ACH-000002', ...]

# 열: 약물 (IC50 값, 정규화됨)
y_train.columns  # ['GDSC:1001', 'GDSC:1004', ...]

# 데이터 확인
y_train.iloc[0, :5]  # 첫 번째 샘플의 첫 5개 약물 IC50 값
y_train.isna().sum()  # 각 약물별 결측치 수
```

---

## ⚙️ 샘플 사용 사례

### 1️⃣ 기본 데이터 탐색

```python
# 기본 통계
print("X_train 통계:")
print(X_train.describe())

print("\ny_train 통계 (IC50 값):")
print(y_train.describe())

# 결측치 확인
print(f"\nX 데이터 결측치: {X_train.isna().sum().sum()}")
print(f"y 데이터 결측치: {y_train.isna().sum().sum()}")
```

### 2️⃣ 특정 약물에 대한 예측 모델 (XGBoost)

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# 첫 번째 약물만 사용
drug_idx = 0
drug_name = metadata['drug_names'][drug_idx]

# 결측값이 아닌 샘플만 선택
valid_idx = ~y_train.iloc[:, drug_idx].isna()
X_train_drug = X_train[valid_idx]
y_train_drug = y_train.iloc[valid_idx, drug_idx]

# 모델 학습
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_drug, y_train_drug)

# 검증 데이터에서 평가
valid_idx_val = ~y_val.iloc[:, drug_idx].isna()
X_val_drug = X_val[valid_idx_val]
y_val_drug = y_val.iloc[valid_idx_val, drug_idx]

y_pred = model.predict(X_val_drug)
r2 = r2_score(y_val_drug, y_pred)
mse = mean_squared_error(y_val_drug, y_pred)

print(f"약물: {drug_name}")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
```

### 3️⃣ 다중 약물 예측 (Multi-task Learning)

```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# 모든 약물에 대해 동시에 학습
# 결측값 처리: 0으로 마스킹 후 결측값이 있는 위치만 제외

y_train_filled = y_train.fillna(y_train.mean())  # 평균으로 대체
y_val_filled = y_val.fillna(y_val.mean())

# 다중 출력 모델
base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train_filled)

y_pred = model.predict(X_val)
print(f"Prediction shape: {y_pred.shape}")  # (84, 224)
```

### 4️⃣ 특성 중요도 분석

```python
import matplotlib.pyplot as plt

# 약물별 특성 중요도 계산
drug_idx = 0
valid_idx = ~y_train.iloc[:, drug_idx].isna()

model = xgb.XGBRegressor()
model.fit(X_train[valid_idx], y_train.iloc[valid_idx, drug_idx])

# 상위 20개 특성
importances = model.feature_importances_
top_indices = np.argsort(importances)[-20:]
top_features = X_train.columns[top_indices]
top_importances = importances[top_indices]

plt.barh(top_features, top_importances)
plt.xlabel('Feature Importance')
plt.title(f'Top 20 Features for {metadata["drug_names"][drug_idx]}')
plt.tight_layout()
plt.show()
```

### 5️⃣ 특성 간 상관관계 분석

```python
# 상관관계 행렬 (계산 비용이 높으니 샘플링 권장)
sample_features = np.random.choice(X_train.columns, 50, replace=False)
corr_matrix = X_train[sample_features].corr()

# 상위 상관관계 쌍 찾기
import pandas as pd
corr_pairs = []
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        corr_pairs.append({
            'Feature1': corr_matrix.index[i],
            'Feature2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df = corr_df.reindex(corr_df['Correlation'].abs().argsort(ascending=False))
print(corr_df.head(10))
```

---

## 🔍 데이터 문제 해결

### 문제: NaN 값을 포함한 y 데이터

**원인**: 모든 세포주가 모든 약물로 테스트되지 않음

**해결책**:

```python
# 방법 1: 결측값이 아닌 샘플만 필터링
drug_idx = 0
valid_mask = ~y_train.iloc[:, drug_idx].isna()
X_filtered = X_train[valid_mask]
y_filtered = y_train.iloc[valid_mask, drug_idx]

# 방법 2: 약물별 평균으로 대체
y_train_imputed = y_train.fillna(y_train.mean())

# 방법 3: 0으로 대체 (마스킹 처리와 함께)
y_train_masked = y_train.fillna(0)
sample_weights = (~y_train.isna()).astype(int)
# 모델 학습 시 sample_weight 사용
```

### 문제: 불균형한 IC50 분포

**원인**: 일부 약물은 모든 세포주에서 테스트, 일부는 일부만 테스트

**해결책**:

```python
# 테스트 수가 많은 약물 필터링
test_counts = (~y_train.isna()).sum()
frequent_drugs = test_counts[test_counts > 200].index
y_train_filtered = y_train[frequent_drugs]
```

---

## 📈 성능 평가 메트릭

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """모델 성능 평가"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# 사용
evaluate_model(y_val_drug, y_pred)
```

---

## 🎯 모범 사례 (Best Practices)

### ✅ 해야 할 것

- ✅ 항상 train/val/test 분할 유지
- ✅ X 데이터는 이미 정규화됨 (추가 정규화 불필요)
- ✅ y 데이터도 정규화됨 (출력값도 정규화 스케일)
- ✅ 결측값을 명확하게 처리
- ✅ 교차 검증 시 약물별로 분층 (stratify)

### ❌ 하지 말아야 할 것

- ❌ 전체 데이터에 대해 다시 정규화
- ❌ train 통계로 val/test 정규화
- ❌ 결측값을 무시하고 학습
- ❌ test 데이터를 train/val과 혼합

---

## 💾 추가 리소스

- `preprocess_hydra_onco.py`: 전처리 스크립트 (재현 가능)
- `PREPROCESSING_SUMMARY.md`: 상세 전처리 보고서
- `metadata.json`: 특성명, 약물명, 세포주 분할 정보

---

## 🆘 문의

데이터 관련 문제가 있으시면:
1. `PREPROCESSING_SUMMARY.md`의 "데이터 품질" 섹션 참고
2. `metadata.json`에서 특성명/약물명 확인
3. 원본 데이터는 `data/` 디렉토리 참고

