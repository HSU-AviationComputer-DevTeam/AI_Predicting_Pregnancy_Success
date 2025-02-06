import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 방지

# 데이터 로드
train = pd.read_csv('C:/code/aimers/data/train.csv')
test = pd.read_csv('C:/code/aimers/data/test.csv')

# ID 컬럼 제거
if 'ID' in train.columns:
    train.drop(columns=['ID'], inplace=True)
if 'ID' in test.columns:
    test.drop(columns=['ID'], inplace=True)

# 타겟과 피처 분리
target = '임신 성공 여부'
X = train.drop(columns=[target])
y = train[target]

# 숫자형 문자열 컬럼("회" 단위 포함) 변환
numeric_str_columns = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
]

for col in numeric_str_columns:
    if col in X.columns:
        X[col] = X[col].astype(str).str.replace('회', '', regex=False)
        X[col] = pd.to_numeric(X[col], errors='coerce')
    if col in test.columns:
        test[col] = test[col].astype(str).str.replace('회', '', regex=False)
        test[col] = pd.to_numeric(test[col], errors='coerce')

# 범주형 컬럼 목록 (숫자형 문자열 컬럼은 제외)
categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형",
    "배란 자극 여부", "배란 유도 유형", "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인",
    "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "배아 생성 주요 이유", "난자 출처", "정자 출처", "난자 기증자 나이",
    "정자 기증자 나이", "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
    "대리모 여부", "PGD 시술 여부", "PGS 시술 여부"
]
# 숫자형 문자열 컬럼 제외
categorical_columns = [col for col in categorical_columns if col not in numeric_str_columns]

# 범주형 데이터 Label Encoding
for col in categorical_columns:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
for col in categorical_columns:
    if col in test.columns:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))

# 가중치를 부여할 중요한 컬럼
important_features = ['이식된 배아 수', '동결 배아 사용 여부', '배아 이식 경과일', '신선 배아 사용 여부', '난자 출처,', '저장된 배아 수', '시술 당시 나이']

# 비선형 가중치 적용 (로그 변환 + 가중치)
for feature in important_features:
    if feature in X.columns:
        X[feature] = np.log1p(X[feature]) * 2.5  # 로그 변환 후 2.5배 가중치 적용
    if feature in test.columns:
        test[feature] = np.log1p(test[feature]) * 2.5


# 누락된 값 처리: SimpleImputer를 사용해 중앙값으로 대체
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test)

# 스케일링 제거 (LightGBM은 스케일링에 덜 민감)
X_scaled = X_imputed
test_scaled = test_imputed

# LightGBM 하이퍼파라미터 튜닝
lgb_model = lgb.LGBMClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.6, 0.8],
    'num_leaves': [31, 50, 70]
}

# StratifiedKFold를 사용한 교차 검증
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_scaled, y)

print("최적 하이퍼파라미터:", grid_search.best_params_)
print("최고 AUC 점수:", grid_search.best_score_)

# 최적의 모델(best_model) 선택 후 전체 학습 데이터로 재학습
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y)

# 피처 중요도 시각화
plt.figure(figsize=(12,8))
# X 데이터의 컬럼 이름을 사용해 중요도 정렬
importance = best_model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(np.array(X.columns)[sorted_idx], importance[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('LightGBM Feature Importance')
plt.show()

# 테스트 데이터 예측
pred_proba = best_model.predict_proba(test_scaled)[:, 1]

# 제출 파일 생성
sample_submission = pd.read_csv('C:/code/aimers/data/sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('C:/code/aimers/data/lightgbm_1.csv', index=False)
