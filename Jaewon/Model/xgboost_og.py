import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# 로드
train = pd.read_csv('C:/code/aimers/data/train.csv')
test = pd.read_csv('C:/code/aimers/data/test.csv')

# ID 컬럼 제거
if 'ID' in train.columns:
    train.drop(columns=['ID'], inplace=True)
if 'ID' in test.columns:
    test.drop(columns=['ID'], inplace=True)

# 분리
target = '임신 성공 여부'
X = train.drop(columns=[target])
y = train[target]

X_t = test.drop(columns=[target])
y_t= test[target]

# 숫자형 문자열 컬럼("회") 변환
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
        
# 범주형 컬럼
categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부"
]

# 범주형 데이터 인코딩
for col in categorical_columns:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
for col in categorical_columns:
    if col in test.columns:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))

# 누락 값 처리 SimpleImputer를 사용 > 중앙값으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_t)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

#  XGBoost 하이퍼파라미터
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_scaled, y)

print("최적 하이퍼파라미터:", grid_search.best_params_)
print("최고 AUC 점수:", grid_search.best_score_)

# 최적의 모델(best_model) 선택 후 전체 학습 데이터로 재학습
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y)

# 테스트 데이터 예측
pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# 제출파일 csv
sample_submission = pd.read_csv('C:/code/aimers/data/sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('C:/code/aimers/data/xgboost_og_submit.csv', index=False)

y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
test_auc_score = roc_auc_score(y_t, y_test_pred_prob)
print("최적 모델의 테스트 데이터 AUC 점수:", test_auc_score)

#540fit
#최적 하이퍼파라미터: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
#최고 AUC 점수: 0.7393712923659637
#제출 후 점수   0.73906
