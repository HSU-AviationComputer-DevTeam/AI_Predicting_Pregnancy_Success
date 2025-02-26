import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
import joblib
import random
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정

# 데이터 불러오기 및 전처리
def preprocess_data(train_path, test_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(test_path.replace('test.csv', 'sample_submission.csv'))

    # ID 컬럼 제거
    if 'ID' in train.columns:
        train.drop(columns=['ID'], inplace=True)
    if 'ID' in test.columns:
        test.drop(columns=['ID'], inplace=True)

    y = train['임신 성공 여부']
    X = train.drop(columns=['임신 성공 여부'])
    X_test = test.copy()

    # 결측치 처리
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
            X_test[col] = X_test[col].fillna('Unknown')
        else:
            X[col] = X[col].fillna(X[col].mean())
            X_test[col] = X_test[col].fillna(X[col].mean())

    # 범주형/수치형 변수 구분
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

    # 범주형 변수 인코딩
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_features] = ordinal_encoder.fit_transform(X[categorical_features])
    X_test[categorical_features] = ordinal_encoder.transform(X_test[categorical_features])

    # 수치형 변수 스케일링
    scaler = MinMaxScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # 이상치 처리
    for col in numerical_features:
        q1 = X[col].quantile(0.01)
        q3 = X[col].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        X[col] = np.clip(X[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    # 파워 변환으로 분포 개선
    power = PowerTransformer(method='yeo-johnson')
    X[numerical_features] = power.fit_transform(X[numerical_features])
    X_test[numerical_features] = power.transform(X_test[numerical_features])

    # 상호작용 특성 생성
    interact_features = []
    for i in range(min(10, len(numerical_features))):
        for j in range(i+1, min(11, len(numerical_features))):
            feat1, feat2 = numerical_features[i], numerical_features[j]
            feat_name = f'{feat1}_{feat2}_interact'
            interact_features.append(feat_name)
            X[feat_name] = X[feat1] * X[feat2]
            X_test[feat_name] = X_test[feat1] * X_test[feat2]

    # 변수 타입 변환
    X[numerical_features + interact_features] = X[numerical_features + interact_features].astype(float)
    X_test[numerical_features + interact_features] = X_test[numerical_features + interact_features].astype(float)
    X[categorical_features] = X[categorical_features].astype(int)
    X_test[categorical_features] = X_test[categorical_features].astype(int)

    return X, y, X_test, sample_submission, categorical_features

# 기본 CatBoost 파라미터 (고정 값)
def get_cat_base_params():
    return {
        "iterations": 1500,
        "border_count": 128,
        "random_strength": 0.5,
        "bagging_temperature": 1,
        "od_type": "Iter",
        "od_wait": 50,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,  # tuning 중에는 verbose 없이 진행
        "random_seed": 42,
        "class_weights": [1, 3]
    }

# 기본 LightGBM 파라미터 (고정 값)
def get_lgb_base_params():
    return {
        "n_estimators": 1500,
        "min_data_in_leaf": 20,
        "max_bin": 255,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_weight": 0.001,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": 3.0
    }

# CatBoost 하이퍼파라미터 튜닝 범위
cat_param_grid = {
    "learning_rate": [0.01, 0.02, 0.05],
    "depth": [6, 8],
    "l2_leaf_reg": [3, 7],
    "subsample": [0.7, 0.85]
}

# LightGBM 하이퍼파라미터 튜닝 범위
lgb_param_grid = {
    "learning_rate": [0.01, 0.02, 0.05],
    "num_leaves": [32, 64],
    "max_depth": [8, 12],
    "reg_alpha": [3, 7],
    "reg_lambda": [7, 12]
}

# CatBoost Random Search 튜닝 함수
def tune_catboost(param_grid, base_params, X_tr, y_tr, X_val, y_val, cat_features, n_iter=10):
    best_score = -np.inf
    best_params = None

    for i in range(n_iter):
        params = base_params.copy()
        # 무작위로 파라미터 선택
        params["learning_rate"] = random.choice(param_grid["learning_rate"])
        params["depth"] = random.choice(param_grid["depth"])
        params["l2_leaf_reg"] = random.choice(param_grid["l2_leaf_reg"])
        params["subsample"] = random.choice(param_grid["subsample"])

        # Pool 생성
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=0
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        # best score 업데이트
        if auc > best_score:
            best_score = auc
            best_params = params.copy()

    return best_params, best_score

# LightGBM Random Search 튜닝 함수
def tune_lgbm(param_grid, base_params, X_tr, y_tr, X_val, y_val, n_iter=10):
    best_score = -np.inf
    best_params = None

    for i in range(n_iter):
        params = base_params.copy()
        params["learning_rate"] = random.choice(param_grid["learning_rate"])
        params["num_leaves"] = random.choice(param_grid["num_leaves"])
        params["max_depth"] = random.choice(param_grid["max_depth"])
        params["reg_alpha"] = random.choice(param_grid["reg_alpha"])
        params["reg_lambda"] = random.choice(param_grid["reg_lambda"])

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        if auc > best_score:
            best_score = auc
            best_params = params.copy()

    return best_params, best_score

# 가중치 최적화 함수 (예외 처리 추가)
def optimize_weights(predictions, y_true):
    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        weighted_pred = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
        return -roc_auc_score(y_true, weighted_pred)

    n_models = len(predictions)
    initial_aucs = [roc_auc_score(y_true, pred) for pred in predictions]
    initial_weights = np.array(initial_aucs) / sum(initial_aucs)
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    try:
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except Exception as e:
        print(f"최적화 실패, 기본 가중치 사용: {e}")
        return initial_weights

# 배치 예측 함수
def batch_predict(model, X, batch_size=10000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch_pred = model.predict_proba(X.iloc[i:i+batch_size])[:, 1]
        predictions.append(batch_pred)
    return np.concatenate(predictions)

# 모델 훈련 및 예측 함수
def train_and_predict():
    # 데이터 로드 및 전처리
    X, y, X_test, sample_submission, categorical_features = preprocess_data(
        'C:/code/aimers/data/train.csv',
        'C:/code/aimers/data/test.csv'
    )

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 기본 파라미터 설정
    cat_base_params = get_cat_base_params()
    lgb_base_params = get_lgb_base_params()

    # 예측 결과 저장
    oof_preds_cat = np.zeros(len(X))
    oof_preds_lgb = np.zeros(len(X))
    test_preds_cat = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))

    best_models = {'cat': None, 'lgb': None}
    best_score = 0

    # K-Fold 훈련
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n==== Fold {fold_idx}/{n_splits} ====")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        try:
            # RandomUnderSampler 사용
            rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_tr_res, y_tr_res = rus.fit_resample(X_tr, y_tr)

            # -- CatBoost 튜닝 및 학습 --
            print("Tuning CatBoost...")
            best_cat_params, cat_tune_auc = tune_catboost(
                cat_param_grid, cat_base_params, X_tr_res, y_tr_res, X_val, y_val, categorical_features, n_iter=10
            )
            print(f"Best CatBoost Params: {best_cat_params} | AUC: {cat_tune_auc:.6f}")

            print("Training CatBoost with tuned parameters...")
            cat_model = CatBoostClassifier(**best_cat_params)
            train_pool = Pool(X_tr_res, y_tr_res, cat_features=categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=categorical_features)
            cat_model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=100,
                verbose=200
            )

            # -- LightGBM 튜닝 및 학습 --
            print("Tuning LightGBM...")
            best_lgb_params, lgb_tune_auc = tune_lgbm(
                lgb_param_grid, lgb_base_params, X_tr_res, y_tr_res, X_val, y_val, n_iter=10
            )
            print(f"Best LightGBM Params: {best_lgb_params} | AUC: {lgb_tune_auc:.6f}")

            print("Training LightGBM with tuned parameters...")
            lgb_model = LGBMClassifier(**best_lgb_params)
            lgb_model.fit(
                X_tr_res, y_tr_res,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
            )

            # 검증 세트 예측
            cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
            lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]

            oof_preds_cat[val_idx] = cat_val_pred
            oof_preds_lgb[val_idx] = lgb_val_pred

            test_preds_cat += batch_predict(cat_model, X_test) / n_splits
            test_preds_lgb += batch_predict(lgb_model, X_test) / n_splits

            cat_auc = roc_auc_score(y_val, cat_val_pred)
            lgb_auc = roc_auc_score(y_val, lgb_val_pred)
            print(f"CatBoost Fold {fold_idx} AUC: {cat_auc:.6f}")
            print(f"LightGBM Fold {fold_idx} AUC: {lgb_auc:.6f}")

            weights = optimize_weights([cat_val_pred, lgb_val_pred], y_val)
            ensemble_auc = roc_auc_score(y_val, weights[0] * cat_val_pred + weights[1] * lgb_val_pred)
            print(f"Ensemble Fold {fold_idx} AUC: {ensemble_auc:.6f} (weights: {weights})")

            if ensemble_auc > best_score:
                best_score = ensemble_auc
                best_models['cat'] = cat_model
                best_models['lgb'] = lgb_model
                print(f"New best model found! Score: {best_score:.6f}")
                best_models['cat'].save_model(f'best_cat_model.cbm')
                joblib.dump(best_models['lgb'], f'best_lgb_model.bin')

            print(f"훈련 데이터 크기: {len(X_tr)} -> 리샘플링 후: {len(X_tr_res)}")
            print(f"훈련 데이터 클래스 분포: {np.bincount(y_tr)}")
            print(f"리샘플링 후 클래스 분포: {np.bincount(y_tr_res)}")

        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            continue

        gc.collect()

    cat_oof_auc = roc_auc_score(y, oof_preds_cat)
    lgb_oof_auc = roc_auc_score(y, oof_preds_lgb)
    print(f"\nCatBoost OOF AUC: {cat_oof_auc:.6f}")
    print(f"LightGBM OOF AUC: {lgb_oof_auc:.6f}")

    final_weights = optimize_weights([oof_preds_cat, oof_preds_lgb], y)
    oof_ensemble = final_weights[0] * oof_preds_cat + final_weights[1] * oof_preds_lgb
    ensemble_oof_auc = roc_auc_score(y, oof_ensemble)
    print(f"Final Ensemble OOF AUC: {ensemble_oof_auc:.6f}")
    print(f"Final weights: CatBoost={final_weights[0]:.4f}, LightGBM={final_weights[1]:.4f}")

    final_prediction = final_weights[0] * test_preds_cat + final_weights[1] * test_preds_lgb

    if best_models['lgb'] is not None:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_models['lgb'].feature_importances_
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('LightGBM Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved.")

    sample_submission['probability'] = final_prediction
    submission_path = "C:/code/aimers/new/submit/param_test01_submit.csv"
    sample_submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved: {submission_path}")
    print(f"Final Ensemble OOF AUC(private score): {ensemble_oof_auc:.6f}")

if __name__ == "__main__":
    train_and_predict()
