!pip install pandas --upgrade
!pip install numpy --upgrade
!pip uninstall -y scikit-learn
!pip install scikit-learn==1.3.1
!pip install catboost --upgrade

import sys
sys.setrecursionlimit(15000)

import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # 추가

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from lightgbm.basic import LightGBMError  # LightGBMError를 임포트
#########################################
# 0. 횟수 컬럼 문자열을 숫자로 변환하는 함수 (정규표현식 사용)
#########################################
def convert_count_str(val):
    if pd.isna(val):
        return 0.0
    val = str(val).strip()
    if "회 이상" in val:
        return 6.0
    m = re.search(r'(\d+)회?', val)
    if m:
        return float(m.group(1))
    return 0.0

#########################################
# 0.5. 난자/정자 기증자 나이 변환 함수 및 매핑
#########################################
donor_age_mapping = {
    '만20세 이하': 0,
    '만21-25세': 1,
    '만26-30세': 2,
    '만31-35세': 3,
    '만36-40세': 4,
    '만41-45세': 5,
    '알 수 없음': 0
}
def convert_donor_age(val):
    if pd.isna(val):
        return np.nan
    return donor_age_mapping.get(str(val).strip(), np.nan)

# 데이터프레임의 모든 문자열 컬럼을 체크하는 함수
def check_string_columns(df):
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print("\n문자열 타입 컬럼 발견:")
        for col in string_cols:
            print(f"\nColumn {col}:")
            print(df[col].unique())
    return string_cols


#########################################
# --- 1. 데이터 로드 ---
#########################################
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

#########################################
# --- 2. "시술 당시 나이" 전처리 및 missing indicator 생성 ---
#########################################
train['시술 당시 나이_missing'] = train['시술 당시 나이'].apply(lambda x: 1.0 if str(x).strip() == '알 수 없음' else 0.0)
test['시술 당시 나이_missing'] = test['시술 당시 나이'].apply(lambda x: 1.0 if str(x).strip() == '알 수 없음' else 0.0)

age_mapping = {
    '만18-34세': 0,
    '만35-37세': 1,
    '만38-39세': 2,
    '만40-42세': 3,
    '만43-44세': 4,
    '만45-50세': 5,
    '알 수 없음': np.nan
}
train['시술 당시 나이'] = train['시술 당시 나이'].apply(lambda x: float(age_mapping.get(str(x).strip(), 0)))
test['시술 당시 나이'] = test['시술 당시 나이'].apply(lambda x: float(age_mapping.get(str(x).strip(), 0)))

#########################################
# --- 2.5. 횟수 관련 컬럼 변환 (문자열 → 숫자) ---
#########################################
count_columns = ["총 임신 횟수", "총 출산 횟수", "총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", "클리닉 내 총 시술 횟수"]
for col in count_columns:
    train[col] = train[col].astype(str).str.strip().apply(convert_count_str)
    test[col] = test[col].astype(str).str.strip().apply(convert_count_str)
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

#########################################
# --- 2.6. 난자/정자 기증자 나이 변환 ---
#########################################
train['난자 기증자 나이'] = train['난자 기증자 나이'].astype(str).apply(convert_donor_age)
test['난자 기증자 나이'] = test['난자 기증자 나이'].astype(str).apply(convert_donor_age)
train['정자 기증자 나이'] = train['정자 기증자 나이'].astype(str).apply(convert_donor_age)
test['정자 기증자 나이'] = test['정자 기증자 나이'].astype(str).apply(convert_donor_age)

#########################################
# --- 3. 타겟 및 Feature 분리 ---
#########################################
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

#########################################
# --- 추가 Feature Engineering ---
#########################################
for df in [X, test]:
    df["임신_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)
    df["출산_비율"] = df["총 출산 횟수"] / (df["총 시술 횟수"] + 1)
    df["총_시술_합계"] = df["IVF 시술 횟수"] + df["DI 시술 횟수"]
    df["임신_비율2"] = df["총 임신 횟수"] / (df["총_시술_합계"] + 1)
    df["adjusted_시술_나이"] = np.exp(-df["시술 당시 나이"])
    df["adjusted_난자_기증자_나이"] = np.exp(-df["난자 기증자 나이"])
    df["adjusted_정자_기증자_나이"] = np.exp(-df["정자 기증자 나이"])

#########################################
# --- 4. 컬럼 목록 구성 ---
#########################################
categorical_columns = [
    "시술 시기 코드", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형",
    "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인", "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "배아 생성 주요 이유", "클리닉 내 총 시술 횟수",
    "난자 출처", "정자 출처",
    "PGD 시술 여부", "PGS 시술 여부"
]

numeric_columns = [
    "시술 당시 나이", "시술 당시 나이_missing",
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수",
    "해동된 배아 수", "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수",
    "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일",
    "배아 이식 경과일", "배아 해동 경과일",
    "임신_비율", "출산_비율", "총_시술_합계", "임신_비율2",
    "총 임신 횟수", "총 출산 횟수", "총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "난자 기증자 나이", "정자 기증자 나이",
    "adjusted_시술_나이", "adjusted_난자_기증자_나이", "adjusted_정자_기증자_나이"
]

#########################################
# --- 5. 결측치 처리 및 인코딩 ---
#########################################
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

X = X.fillna(0)  # numeric_columns 만이 아닌 전체 X에 적용
test = test.fillna(0)  # numeric_columns 만이 아닌 전체 test에 적용

X_encoded = pd.get_dummies(X, columns=categorical_columns, dummy_na=True)
X_test_encoded = pd.get_dummies(test, columns=categorical_columns, dummy_na=True)

# 학습/테스트 데이터의 컬럼 맞추기
X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# 중복 컬럼 제거
X_encoded = X_encoded.loc[:, ~X_encoded.columns.duplicated()]
X_test_encoded = X_test_encoded.loc[:, ~X_test_encoded.columns.duplicated()]

# 모든 feature 이름을 f0, f1, f2, ...로 재설정
X_encoded.columns = [f"f{i}" for i in range(X_encoded.shape[1])]
X_test_encoded.columns = [f"f{i}" for i in range(X_test_encoded.shape[1])]

# 최종 변환 전 확인
print("\n최종 변환 전 문자열 컬럼 체크:")
string_cols_final = check_string_columns(X_encoded)

# 문자열 컬럼이 있다면 직접 변환
if len(string_cols_final) > 0:
    for col in string_cols_final:
        print(f"\nConverting {col} to numeric...")
        X_encoded[col] = pd.to_numeric(X_encoded[col].apply(convert_count_str))
        X_test_encoded[col] = pd.to_numeric(X_test_encoded[col].apply(convert_count_str))


# 최종 변환: 모든 열을 float 타입으로 강제 변환
X_encoded = X_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)

#########################################
# --- 6. 학습/검증 데이터 분할 ---
#########################################
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

#########################################
# --- 7. Stacking 앙상블 모델 구성 (GPU 최대 활용 + 정규화 포함) ---
#########################################
from sklearn.pipeline import Pipeline

# 최종 메타 모델에 정규화를 적용하는 파이프라인 구성
final_estimator_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# GPU 사용: XGBoost, LightGBM, CatBoost 모두 GPU 모드로 설정
try:
    lgbm_model = LGBMClassifier(
        random_state=42,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_jobs=1
    )
    # 더미 데이터를 이용하여 GPU 모드가 동작하는지 확인합니다.
    dummy_X = np.random.rand(10, 5)
    dummy_y = np.random.randint(0, 2, 10)
    lgbm_model.fit(dummy_X, dummy_y)
    print("LightGBM GPU mode enabled.")
except LightGBMError as e:
    print("LightGBM GPU mode not available, using CPU mode. Error:", e)
    lgbm_model = LGBMClassifier(random_state=42, n_jobs=1)


cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    task_type='GPU',
    devices='0:1',
    verbose=0,
    random_seed=42
)

estimators = [
    ('xgb', XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                use_label_encoder=False,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                gpu_id=0,
                n_jobs=1)),
    ('lgbm', lgbm_model),
    ('cat', cat_model),
    ('etc', ExtraTreesClassifier(random_state=42, n_jobs=1))
]

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator_pipeline,
    cv=5,
    n_jobs=1
)

#########################################
# --- 8. GridSearchCV를 통한 최종 메타 모델 튜닝 (정규화된 LR의 C 튜닝) ---
#########################################
param_grid = {
    'final_estimator__lr__C': [0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(stack_clf, param_grid, scoring='roc_auc', cv=5, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)
print("Best CV ROC AUC Score: ", grid_search.best_score_)



#########################################
# --- 9. 검증 세트 평가 ---
#########################################
val_pred_proba = grid_search.predict_proba(X_val)[:, 1]
val_roc_auc = roc_auc_score(y_val, val_pred_proba)
print("Validation ROC AUC Score: {:.4f}".format(val_roc_auc))

#########################################
# --- 10. 최종 모델 재학습 (전체 학습 데이터 이용) ---
#########################################
best_params = grid_search.best_params_
final_model = StackingClassifier(
    estimators=estimators,
    final_estimator=Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=best_params['final_estimator__lr__C']))
    ]),
    cv=5,
    n_jobs=1
)
final_model.fit(X_encoded, y)

#########################################
# --- 11. 테스트 데이터 예측 및 제출 파일 생성 ---
#########################################
pred_proba = final_model.predict_proba(X_test_encoded)[:, 1]
submission = pd.DataFrame({
    'ID': [f"TEST_{i:05d}" for i in range(len(test))]
})
submission['probability'] = pred_proba
submission.to_csv('./improved_submit.csv', index=False)
print("Submission 파일 'improved_submit.csv'가 생성되었습니다.")

#########################################
# --- 12. Test 데이터에는 타겟이 없으므로 ROC AUC 점수는 계산하지 않습니다. ---
#########################################
print("Test 데이터에는 실제 '임신 성공 여부'가 없으므로, ROC AUC 점수를 계산할 수 없습니다.")
