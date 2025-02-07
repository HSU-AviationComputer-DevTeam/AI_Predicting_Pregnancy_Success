!pip install pandas --upgrade
!pip install numpy --upgrade

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- 1. 데이터 로드 ---
# train.csv와 test.csv 파일에서 'ID' 컬럼은 제거합니다.
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# --- 2. "시술 당시 나이" 전처리 및 missing indicator 생성 ---
# missing indicator: "알 수 없음"이면 1, 아니면 0
train['시술 당시 나이_missing'] = train['시술 당시 나이'].apply(lambda x: 1 if x == '알 수 없음' else 0)
test['시술 당시 나이_missing'] = test['시술 당시 나이'].apply(lambda x: 1 if x == '알 수 없음' else 0)

# 나이 매핑: 높은 값이 나이가 많음을 의미 → 임신 성공 확률은 낮아짐.
age_mapping = {
    '만18-34세': 0,
    '만35-37세': 1,
    '만38-39세': 2,
    '만40-42세': 3,
    '만43-44세': 4,
    '만45-50세': 5,
    '알 수 없음': np.nan
}
train['시술 당시 나이'] = train['시술 당시 나이'].map(age_mapping)
test['시술 당시 나이'] = test['시술 당시 나이'].map(age_mapping)

# --- 3. 타겟 및 Feature 분리 ---
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# --- 4. 컬럼 목록 구성 ---
# "시술 당시 나이"와 "시술 당시 나이_missing"은 수치형으로 처리됩니다.
categorical_columns = [
    "시술 시기 코드", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형",
    "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인", "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수",
    "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수", "난자 출처", "정자 출처",
    "난자 기증자 나이", "정자 기증자 나이", "동결 배아 사용 여부", "신선 배아 사용 여부",
    "기증 배아 사용 여부", "대리모 여부", "PGD 시술 여부", "PGS 시술 여부"
]

numeric_columns = [
    "시술 당시 나이", "시술 당시 나이_missing",
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수",
    "해동된 배아 수", "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수",
    "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일",
    "배아 이식 경과일", "배아 해동 경과일"
]

# --- 5. 결측치 처리 및 인코딩 ---
# 수치형 변수: 결측치는 0으로 채웁니다.
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# 범주형 변수: pd.get_dummies를 사용 (dummy_na=True)
X_encoded = pd.get_dummies(X, columns=categorical_columns, dummy_na=True)
X_test_encoded = pd.get_dummies(test, columns=categorical_columns, dummy_na=True)

# 학습 데이터와 테스트 데이터의 컬럼을 align (누락된 dummy 컬럼은 0으로 채움)
X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# 중복 컬럼 제거 (만약 발생했다면)
X_encoded = X_encoded.loc[:, ~X_encoded.columns.duplicated()]
X_test_encoded = X_test_encoded.loc[:, ~X_test_encoded.columns.duplicated()]

# **중요**: 모든 feature 이름을 안전하게 (f0, f1, f2, ...) 재설정하여 특수문자 문제를 완전히 회피합니다.
X_encoded.columns = ["f" + str(i) for i in range(X_encoded.shape[1])]
X_test_encoded.columns = ["f" + str(i) for i in range(X_test_encoded.shape[1])]

# --- 6. 학습/검증 데이터 분할 (교차 검증용) ---
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# --- 7. Stacking 앙상블 모델 구성 ---
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# GPU 사용: XGBoost와 LGBM에 GPU 옵션 설정 (ExtraTreesClassifier는 CPU 사용)
estimators = [
    ('xgb', XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                use_label_encoder=False,
                tree_method='gpu_hist',   # GPU 사용
                predictor='gpu_predictor',
                gpu_id=0,
                n_jobs=1)),
    ('lgbm', LGBMClassifier(
                random_state=42,
                # GPU 파라미터를 제거하여 CPU 모드로 동작 (OpenCL 문제가 발생하므로)
                n_jobs=1)),
    ('etc', ExtraTreesClassifier(
                random_state=42,
                n_jobs=1))
]
stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=1  # n_jobs=1로 설정하여 병렬 처리 관련 피클링 문제 회피
)

# --- 8. GridSearchCV를 통한 최종 메타 모델 튜닝 (LogisticRegression의 C값 튜닝) ---
param_grid = {
    'final_estimator__C': [0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(stack_clf, param_grid, scoring='roc_auc', cv=5, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)
print("Best CV ROC AUC Score: ", grid_search.best_score_)

# --- 9. 검증 세트 평가 ---
val_pred_proba = grid_search.predict_proba(X_val)[:, 1]
val_roc_auc = roc_auc_score(y_val, val_pred_proba)
print("Validation ROC AUC Score: {:.4f}".format(val_roc_auc))

# --- 10. 최종 모델 재학습 (전체 학습 데이터 이용) ---
best_params = grid_search.best_params_
final_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, C=best_params['final_estimator__C']),
    cv=5,
    n_jobs=1
)
final_model.fit(X_encoded, y)

# --- 11. 테스트 데이터 예측 및 제출 파일 생성 ---
pred_proba = final_model.predict_proba(X_test_encoded)[:, 1]
submission = pd.DataFrame({
    'ID': ["TEST_" + str(i).zfill(5) for i in range(len(test))]
})
submission['probability'] = pred_proba
submission.to_csv('./improved_submit.csv', index=False)
print("Submission 파일 'improved_submit.csv'가 생성되었습니다.")

# --- 12. Test 데이터에는 실제 타겟(임신 성공 여부)이 없으므로 ROC AUC 점수는 계산하지 않습니다.
print("Test 데이터에는 실제 '임신 성공 여부'가 없으므로, ROC AUC 점수를 계산할 수 없습니다.")

