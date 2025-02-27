# ====================== 환경 및 라이브러리 버전 ======================
# Python 3.9.7
# pandas 1.3.5
# numpy 1.21.6
# matplotlib 3.5.1
# seaborn 0.11.2
# scikit-learn 1.0.2
# imbalanced-learn 0.8.1
# catboost 1.0.6
# scipy 1.7.3
# ===================================================================

# 데이터 불균형 비율 약 3:1(190,123 : 66,228)
# 모델 파라미터 최적화
# 최적 모델 저장
# CatBoost 단일 모델 사용

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# 클래스별 결측치 처리 함수 추가
def fill_missing_by_class(X, y, X_test, numerical_features):
    """각 클래스별 평균으로 결측치 대체"""
    print("클래스별 결측치 처리 중...")
    
    # 훈련 데이터와 타겟 결합
    train_with_target = X.copy()
    train_with_target['target'] = y
    
    for col in numerical_features:
        if X[col].isnull().sum() > 0:  # 결측치가 있는 경우만 처리
            # 클래스별 평균 계산
            class_means = train_with_target.groupby('target')[col].mean().to_dict()
            
            # 훈련 데이터 결측치 대체
            for cls, mean_val in class_means.items():
                mask = (X[col].isna()) & (train_with_target['target'] == cls)
                X.loc[mask, col] = mean_val
            
            # 테스트 데이터는 전체 평균으로 대체
            overall_mean = X[col].mean()
            X_test.loc[X_test[col].isna(), col] = overall_mean
            
            print(f"  {col} 컬럼: 클래스별 결측치 처리 완료")
    
    return X, X_test

# 향상된 특성 공학 적용
def enhanced_feature_engineering(X, X_test, numerical_features):
    """고급 특성 공학 적용"""
    print("고급 도메인 특화 특성 엔지니어링 적용 중...")
    
    # 주의: 이전_시술_성공률, 이전_임신_출산_전환율 같은 특성이 타겟 누수 여부 확인
    potentially_leaky_features = ['이전_시술_성공률', '이전_임신_출산_전환율', '임신', '성공']
    for col in X.columns:
        for leaky in potentially_leaky_features:
            if leaky in col and col in X.columns:
                print(f"경고: 잠재적 타겟 누수 특성 검출: {col}")
                # 타겟과의 상관관계 확인
                if 'target' in X.columns:
                    correlation = X[[col, 'target']].corr().iloc[0,1]
                    print(f"  타겟과의 상관관계: {correlation:.4f}")
                # 필요시 이러한 특성 제거
    
    # 1. 나이-시술 관련 교차 특성 
    if all(col in X.columns for col in ['시술_나이_수치', '총 시술 횟수_수치']):
        X['나이_시술횟수_비율'] = X['시술_나이_수치'] / (X['총 시술 횟수_수치'] + 1)
        X_test['나이_시술횟수_비율'] = X_test['시술_나이_수치'] / (X_test['총 시술 횟수_수치'] + 1)
        numerical_features.append('나이_시술횟수_비율')
        
        # 2. 나이 제곱 특성 (비선형성 포착)
        X['시술_나이_제곱'] = X['시술_나이_수치'] ** 2
        X_test['시술_나이_제곱'] = X_test['시술_나이_수치'] ** 2
        numerical_features.append('시술_나이_제곱')
        
        # 3. 경험-나이 교차 특성 
        X['나이_경험_상호작용'] = X['시술_나이_수치'] * X['총 시술 횟수_수치']
        X_test['나이_경험_상호작용'] = X_test['시술_나이_수치'] * X_test['총 시술 횟수_수치']
        numerical_features.append('나이_경험_상호작용')
    
    # 4. 각종 성공률 지표 간 관계
    if all(col in X.columns for col in ['이전_시술_성공률', '이전_임신_출산_전환율']):
        X['성공지표_복합점수'] = X['이전_시술_성공률'] * 0.7 + X['이전_임신_출산_전환율'] * 0.3
        X_test['성공지표_복합점수'] = X_test['이전_시술_성공률'] * 0.7 + X_test['이전_임신_출산_전환율'] * 0.3
        numerical_features.append('성공지표_복합점수')
    
    # 5. 상호작용 특성 생성
    print("상호작용 특성 생성 중...")
    key_features = ['시술_나이_수치', '배아_이식_수_수치', '이전_시술_성공률', '이전_시술_횟수', '고위험_연령군'] 
    key_features = [f for f in key_features if f in X.columns]
    
    if len(key_features) >= 2:
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                interaction_name = f'{feat1}_{feat2}_interaction'
                X[interaction_name] = X[feat1] * X[feat2]
                X_test[interaction_name] = X_test[feat1] * X_test[feat2]
                numerical_features.append(interaction_name)
    
    # 6. 의학적 지식 기반 복합 지표 생성
    if all(col in X.columns for col in ['시술_나이_수치', '고위험_연령군']):
        # 난임 위험도 지수 (나이에 따른 가중치 증가)
        age_risk = X['시술_나이_수치'].copy()
        age_risk_test = X_test['시술_나이_수치'].copy()
        
        # 35세 초과시 위험 지수 가속 증가
        age_risk.loc[age_risk > 35] = 35 + (age_risk.loc[age_risk > 35] - 35) * 1.5
        age_risk_test.loc[age_risk_test > 35] = 35 + (age_risk_test.loc[age_risk_test > 35] - 35) * 1.5
        
        # 40세 초과시 위험 지수 더 급격히 증가
        age_risk.loc[age_risk > 40] = 40 + (age_risk.loc[age_risk > 40] - 40) * 2.0
        age_risk_test.loc[age_risk_test > 40] = 40 + (age_risk_test.loc[age_risk_test > 40] - 40) * 2.0
        
        X['난임_위험도_지수'] = age_risk / 30  # 정규화
        X_test['난임_위험도_지수'] = age_risk_test / 30
        numerical_features.append('난임_위험도_지수')
    
    return X, X_test, numerical_features

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
    
    # 타겟 변수 분리
    y = train['임신 성공 여부']
    X = train.drop(columns=['임신 성공 여부'])
    X_test = test.copy()

    # 범주형/수치형 변수 구분
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"범주형 변수: {len(categorical_features)}개, 수치형 변수: {len(numerical_features)}개")
    
    # 1. 의심스러운 특성 검사 - 높은 상관관계 확인
    print("\n=== 데이터 누수 의심 특성 확인 ===")
    suspect_features = []
    
    # 수치형 특성에 대해 타겟과의 상관관계 확인
    for col in numerical_features:
        if not pd.isna(X[col]).all():  # 모든 값이 NA가 아닌 경우만
            correlation = np.corrcoef(X[col].fillna(0).values, y.values)[0, 1]
            if abs(correlation) > 0.7:  # 상관관계 임계값
                print(f"경고: 높은 상관관계 특성 - {col}: {correlation:.4f}")
                suspect_features.append((col, correlation))
    
    # 특성 중요도 상위 10개 확인
    suspect_features.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n상관관계 기준 상위 의심 특성:")
    for feature, corr in suspect_features[:10]:
        print(f"  {feature}: {corr:.4f}")
    
    # 2. 명백한 누수 특성 제거
    leaky_keywords = ['성공', '임신', 'target', 'score', 'outcome', 'result', '예측', 'prediction']
    removed_features = []
    
    for col in X.columns:
        # 특성명에 의심 키워드가 포함된 경우
        if any(keyword in col.lower() for keyword in leaky_keywords):
            print(f"누수 의심 특성 제거: {col}")
            removed_features.append(col)
        
        # 난자 해동 경과일 특성은 특별히 제거 (상관관계 0.98이 나온 특성)
        if col == '난자 해동 경과일' or ('해동' in col and '경과' in col):
            print(f"높은 상관관계 특성 제거: {col}")
            removed_features.append(col)
    
    # 누수 의심 특성 제거
    X.drop(columns=removed_features, errors='ignore', inplace=True)
    X_test.drop(columns=removed_features, errors='ignore', inplace=True)
    
    # 범주형 변수 확인 및 변환 (누수 특성 제거 후)
    categorical_features = [col for col in categorical_features if col in X.columns]
    numerical_features = [col for col in numerical_features if col in X.columns]
    
    # 범주형 변수를 문자열로 변환
    for col in categorical_features:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # 수치형 변수 중 범주형에 가까운 것들 처리
    for col in numerical_features[:]:  # 복사본으로 반복
        if col in X.columns and X[col].nunique() < 10:
            X[col] = X[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            categorical_features.append(col)
            numerical_features.remove(col)
            print(f"  {col} 변수를 범주형으로 변환 (고유값: {X[col].nunique()}개)")
    
    # 나머지 전처리 계속 진행
    X, X_test = fill_missing_by_class(X, y, X_test, numerical_features)
    X, X_test, numerical_features = enhanced_feature_engineering(X, X_test, numerical_features)
    
    # CatBoost 범주형 변수 인덱스 구성
    categorical_features = list(set(categorical_features))  # 중복 제거
    cat_indices = [i for i, col in enumerate(X.columns) if col in categorical_features]
    
    print(f"\n최종 특성 수: {X.shape[1]}")
    print(f"최종 범주형 변수 수: {len(cat_indices)}")
    
    return X, y, X_test, sample_submission, cat_indices, categorical_features, test_path

# 배치 예측 함수 (모든 예측에 공통으로 사용)
def batch_predict(model, data, cat_features, batch_size=10000):
    """대용량 데이터 예측을 위한 배치 처리 함수"""
    n_samples = len(data)
    n_batches = int(np.ceil(n_samples / batch_size))
    preds = np.zeros(n_samples)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = data.iloc[start_idx:end_idx]
        
        # Pool 객체 생성
        batch_pool = Pool(data=batch_data, cat_features=cat_features)
        preds[start_idx:end_idx] = model.predict_proba(batch_pool)[:, 1]
    
    return preds

# 제출 파일 저장 개선
def save_submission(sample_submission, prediction, model_type, auc_score):
    """제출 파일 저장 함수"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    sample_submission['probability'] = prediction
    submission_path = f"/Users/sindong-u/coding/python/LgAImers/submit/{model_type}_{auc_score:.6f}_{timestamp}.csv"
    sample_submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved: {submission_path}")
    return submission_path

# 진행 로깅 함수
def log_cv_progress(fold_idx, n_splits, start_time):
    """교차 검증 진행률 표시"""
    elapsed = time.time() - start_time
    eta = elapsed / (fold_idx + 1) * (n_splits - fold_idx - 1)
    print(f"\n[진행률] {fold_idx+1}/{n_splits} 완료 ({elapsed:.1f}초 소요, 약 {eta:.1f}초 남음)")

# 특성 중요도 기반 특성 선택 추가
def select_features_by_importance(model, X, threshold=0.01):
    """특성 중요도 기반으로 중요한 특성만 선택"""
    importance = model.get_feature_importance()
    feature_names = X.columns.tolist()
    
    # 특성 중요도 정규화
    importance = importance / np.sum(importance)
    
    # 중요도가 threshold 이상인 특성 선택
    selected_features = [feature_names[i] for i in range(len(feature_names)) 
                         if importance[i] >= threshold]
    
    print(f"특성 선택: {len(feature_names)}개 -> {len(selected_features)}개 특성")
    return selected_features

# 폴드별 최적 모델 저장 강화
def save_fold_model_info(model, fold_idx, auc_score):
    """폴드별 모델 정보 저장"""
    best_iteration = model.get_best_iteration()
    model.save_model(f'best_cat_model_fold{fold_idx}.cbm')
    
    with open(f'fold{fold_idx}_info.txt', 'w') as f:
        f.write(f"Fold {fold_idx} AUC: {auc_score:.6f}\n")
        f.write(f"Best iteration: {best_iteration}\n")
        
        # 상위 20개 중요 특성 저장
        importance = model.get_feature_importance()
        features = model.feature_names_
        top_features = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)[:20]
        
        f.write("Top 20 features:\n")
        for feature, imp in top_features:
            f.write(f"{feature}: {imp:.6f}\n")

# 앙상블 전략 개선 - improved_ensemble_prediction 함수 추가
def improved_ensemble_prediction(fold_models, X_test, cat_indices, fold_aucs):
    """최고 성능 폴드에 가중치를 더 부여하는 앙상블 전략"""
    print("개선된 가중치 앙상블 적용 중...")
    
    # 최고 성능 폴드 식별
    best_fold_idx = np.argmax(fold_aucs)
    
    # 특별 가중치 계산: 기본 가중치의 1.5배
    special_weights = fold_aucs.copy()
    special_weights[best_fold_idx] *= 1.5  # 최고 성능 모델 가중치 50% 증가
    special_weights = special_weights / sum(special_weights)  # 정규화
    
    # 폴드별 가중치 및 성능 출력
    for i, weight in enumerate(special_weights):
        print(f"Fold {i} 가중치: {weight:.4f}, AUC: {fold_aucs[i]:.6f}")
    
    # 가중치 적용 앙상블 예측
    ensemble_preds = np.zeros(len(X_test))
    for i, model in enumerate(fold_models):
        if i < len(special_weights):
            fold_preds = batch_predict(model, X_test, cat_indices)
            ensemble_preds += fold_preds * special_weights[i]
    
    return ensemble_preds

# 특성 선택 개선 함수
def select_optimal_features(X, y, columns, threshold=0.8):
    """누적 중요도 기준으로 중요 특성 선택"""
    print("특성 선택 최적화 수행 중...")
    
    # 빠른 성능 평가를 위한 간단한 모델
    quick_model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, 
        loss_function='Logloss', eval_metric='AUC',
        random_seed=42, verbose=100
    )
    
    # 범주형 특성 처리
    cat_features = [i for i, col in enumerate(columns) 
                   if X[col].dtype == 'object' or X[col].dtype == 'category']
    
    # 모델 학습
    quick_model.fit(X, y, cat_features=cat_features, verbose=False)
    
    # 특성 중요도 계산 및 정렬
    feature_importance = pd.DataFrame({
        'feature': columns,
        'importance': quick_model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    # 누적 중요도 계산
    total_importance = feature_importance['importance'].sum()
    feature_importance['norm_importance'] = feature_importance['importance'] / total_importance
    feature_importance['cum_importance'] = feature_importance['norm_importance'].cumsum()
    
    # threshold 이하 누적 중요도를 가진 특성만 선택
    important_features = feature_importance[feature_importance['cum_importance'] <= threshold]['feature'].tolist()
    
    # 최소 10개 특성 보장
    if len(important_features) < 10:
        important_features = feature_importance.head(10)['feature'].tolist()
    
    print(f"특성 선택: {len(columns)}개 -> {len(important_features)}개 특성")
    
    # 선택된 중요 특성 목록 저장
    pd.DataFrame({'selected_features': important_features}).to_csv('selected_features.csv', index=False)
    
    return important_features

# 최상위 폴드 분석 함수
def analyze_best_fold(X, y, fold_indices, best_fold_idx, cat_params):
    """최고 성능 폴드 분석 및 시각화"""
    # 폴드 데이터 분리
    train_idx, val_idx = list(fold_indices)[best_fold_idx]
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 폴드별 특성 중요도 분석
    model = CatBoostClassifier(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    
    # 특성 중요도 추출 및 시각화
    importance = model.get_feature_importance()
    feature_names = X.columns
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance)[-20:]  # 상위 20개 특성
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'Top 20 Feature Importance in Best Fold {best_fold_idx}')
    plt.savefig(f'best_fold_{best_fold_idx}_importance.png')
    plt.close()
    
    return model

def train_and_predict(train_path='/Users/sindong-u/coding/python/LgAImers/data/train.csv', test_path='/Users/sindong-u/coding/python/LgAImers/data/test.csv'):
    # 데이터 전처리
    X, y, X_test, sample_submission, cat_indices, categorical_features, test_path = preprocess_data(train_path, test_path)
    
    # 중요: 전처리 이후 데이터 체크
    print("전처리 후 데이터 체크:")
    print(f"X 형태: {X.shape}, y 형태: {y.shape}")
    print(f"X_test 형태: {X_test.shape}")
    print(f"범주형 변수 수: {len(categorical_features)}")
    
    # 데이터 누수 체크 중...
    print("데이터 누수 체크 중...")
    # 먼저 적은 표본으로 테스트
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    y_sample = y.iloc[X_sample.index]  # 추가: y_sample 정의

    try:
        # 누수 체크를 위한 간단한 모델 - 문자열 변환된 범주형 특성 사용
        check_params = {
            'iterations': 50,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False,
            'random_seed': 42
        }
        
        # Pool 객체 직접 생성
        check_pool = Pool(
            data=X_sample,
            label=y_sample,
            cat_features=cat_indices
        )
        
        # 모델 학습
        check_model = CatBoostClassifier(**check_params)
        check_model.fit(check_pool, verbose=False)
        
        # 특성 중요도 확인
        importance = check_model.get_feature_importance()
        top_features = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(10)
        print("가장 중요한 10개 특성:")
        print(top_features)
        
    except Exception as e:
        print(f"누수 체크 과정에서 오류 발생: {e}")
        print("범주형 변수 처리 문제로 인한 오류일 수 있습니다. 계속 진행합니다.")
    
    # 수정: StratifiedKFold 사용 - 누수 방지
    kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)  # 10에서 7로 변경
    fold_indices = list(kf.split(X, y))
    n_folds = len(fold_indices)
    
    # CatBoost 파라미터 최적화
    cat_params = {
        'iterations': 2000,              
        'learning_rate': 0.02,           
        'depth': 8,                      
        'l2_leaf_reg': 5,                
        'random_strength': 1.0,          
        'bagging_temperature': 1.0,     
        'one_hot_max_size': 25,         
        'early_stopping_rounds': 200,    
        'class_weights': [1, 3.0],      
        'bootstrap_type': 'Bayesian',
        'grow_policy': 'Lossguide',
        'min_data_in_leaf': 20,          
        'eval_metric': 'AUC',           
        'loss_function': 'Logloss',
        'verbose': 100,
        'use_best_model': True,
        'random_seed': 42
    }
    
    # 데이터 누수 방지를 위해 특성 확인
    for col in X.columns:
        if '_target_' in col.lower() or '임신' in col or 'success' in col.lower():
            print(f"주의! 잠재적 누수 특성 발견: {col}")
    
    # 예측 저장용 변수
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # 최고 성능 기록용
    best_score = 0
    best_model = None
    
    # 모델 훈련 시작
    start_time = time.time()
    
    # 각 폴드 모델을 저장 (앙상블용)
    fold_models = []
    
    # 폴드별 성능 저장 리스트 추가
    fold_aucs = []
    
    print("\n데이터 전처리 완료. 모델 훈련 시작...")
    
    # 각 폴드별 훈련 수행
    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n==================== Fold {fold_idx} / {n_folds} ====================")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 훈련/검증 데이터 풀 생성
        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            cat_features=cat_indices
        )
        
        eval_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_indices
        )
        
        # 로그 출력
        print(f"훈련 데이터 크기: {len(X_tr)} (원본 데이터 사용)")
        print(f"훈련 데이터 클래스 분포: {np.bincount(y_tr)}")
        
        # CatBoost 모델 학습
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            train_pool, 
            eval_set=eval_pool,
            use_best_model=True
        )
        
        # 테스트 세트 예측 (배치 처리)
        test_preds += batch_predict(cat_model, X_test, cat_indices) / n_folds
        
        # 검증 데이터 예측
        val_preds = batch_predict(cat_model, X_val, cat_indices)
        oof_preds[val_idx] = val_preds
        
        # 개별 모델 성능 확인
        cat_auc = roc_auc_score(y_val, val_preds)
        print(f"CatBoost Fold {fold_idx} AUC: {cat_auc:.6f}")
        
        # 폴드 AUC 저장
        fold_aucs.append(cat_auc)
        
        # 모델 저장
        fold_models.append(cat_model)
        
        # 최고 성능 모델 업데이트
        if cat_auc > best_score:
            best_score = cat_auc
            best_model = cat_model
            print(f"New best model found! Score: {best_score:.6f}")
            
            # 모델 저장
            cat_model.save_model(f'best_cat_model_fold{fold_idx}.cbm')
        
        # 진행 상황 로깅
        elapsed = time.time() - start_time
        remaining = elapsed / (fold_idx + 1) * (n_folds - fold_idx - 1)
        print(f"[진행률] {fold_idx+1}/{n_folds} 완료 ({elapsed:.1f}초 소요, 약 {remaining:.1f}초 남음)")
        
        # 학습 곡선 시각화
        plot_learning_curve(cat_model, fold_idx)
    
    # 전체 OOF 성능 평가
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nCatBoost OOF AUC: {oof_auc:.6f}")
    
    # 개별 폴드 성능 확인
    for i, fold_auc in enumerate(fold_aucs):
        print(f"Fold {i} AUC: {fold_auc:.6f}")
    
    # 최고 성능 폴드 모델 저장 및 예측
    best_fold_idx = np.argmax(fold_aucs)
    best_fold_model = fold_models[best_fold_idx]
    print(f"\n최고 성능 폴드: Fold {best_fold_idx}, AUC: {fold_aucs[best_fold_idx]:.6f}")
    
    # 제출 파일 생성 부분 수정

    # 1. 최고 폴드 기반 제출 파일
    best_preds = batch_predict(best_fold_model, X_test, cat_indices)
    sample_submission['probability'] = best_preds  # '임신 성공 여부' 대신 'probability'로 변경
    submission_path = 'single_best_submission.csv'
    sample_submission.to_csv(submission_path, index=False)
    print(f"최고 성능 단일 폴드 제출 파일 생성 완료: {submission_path}")
    
    # 2. 앙상블 기반 제출 파일
    sample_submission['probability'] = test_preds  # '임신 성공 여부' 대신 'probability'로 변경
    ensemble_path = 'ensemble_submission.csv'
    sample_submission.to_csv(ensemble_path, index=False)
    print(f"앙상블 제출 파일 생성 완료: {ensemble_path}")
    
    # 3. 상위 3개 폴드 앙상블 제출 파일
    top3_indices = np.argsort(fold_aucs)[-3:]
    top3_preds = np.zeros(len(X_test))
    
    # 가중치 계산 (AUC 기반)
    top3_aucs = [fold_aucs[i] for i in top3_indices]
    weights = np.array(top3_aucs) / sum(top3_aucs)
    
    for i, idx in enumerate(top3_indices):
        fold_pred = batch_predict(fold_models[idx], X_test, cat_indices)
        top3_preds += fold_pred * weights[i]
        print(f"Top 폴드 {idx} 가중치: {weights[i]:.4f}")
    
    sample_submission['probability'] = top3_preds  # '임신 성공 여부' 대신 'probability'로 변경
    top3_path = 'top3_ensemble_submission.csv'
    sample_submission.to_csv(top3_path, index=False)
    print(f"상위 3 폴드 앙상블 제출 파일 생성 완료: {top3_path}")
    
    # 4. 상위 5개 폴드 앙상블 제출 파일 (7개 폴드 중 상위 5개 선택)
    top5_indices = np.argsort(fold_aucs)[-5:]  # 성능 기준 상위 5개 폴드 인덱스
    top5_preds = np.zeros(len(X_test))

    # 가중치 계산 (AUC 기반)
    top5_aucs = [fold_aucs[i] for i in top5_indices]
    weights = np.array(top5_aucs) / sum(top5_aucs)

    for i, idx in enumerate(top5_indices):
        fold_pred = batch_predict(fold_models[idx], X_test, cat_indices)
        top5_preds += fold_pred * weights[i]
        print(f"Top 폴드 {idx} 가중치: {weights[i]:.4f}")

    sample_submission['probability'] = top5_preds
    top5_path = 'top5_ensemble_submission.csv'
    sample_submission.to_csv(top5_path, index=False)
    print(f"상위 5개 폴드 앙상블 제출 파일 생성 완료: {top5_path}")
    
    # 제출 안내 메시지
    print("\n====== 제출 안내 ======")
    print("1. 생성된 CSV 파일을 대회 사이트에 제출하세요.")
    print("2. 다음 3개 파일 중 하나를 선택하여 제출하세요:")
    print(f"   - {submission_path}: 단일 최고 성능 모델")
    print(f"   - {ensemble_path}: 모든 폴드 평균 앙상블")
    print(f"   - {top3_path}: 상위 3개 폴드 가중 앙상블")
    print(f"   - {top5_path}: 상위 5개 폴드 가중 앙상블")
    print("3. 대회 서버에서 제출한 예측과 실제 정답을 비교하여 최종 AUC 점수를 계산합니다.")
    
    return best_model, oof_auc

# 모델 학습 과정에서 학습 커브 모니터링
def plot_learning_curve(model, fold_idx):
    """학습 커브 시각화 함수"""
    evals_result = model.get_evals_result()
    train_logloss = evals_result['learn']['Logloss']
    test_logloss = evals_result['validation']['Logloss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_logloss, label='train')
    plt.plot(test_logloss, label='validation')
    plt.xlabel('Iterations')
    plt.ylabel('Logloss')
    plt.legend()
    plt.title(f'Learning Curve for Fold {fold_idx}')
    plt.savefig(f'learning_curve_fold{fold_idx}.png')
    plt.close()

# 가중치 최적화 함수 추가
def optimize_weights(predictions, y_true):
    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 정규화
        weighted_pred = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
        return -roc_auc_score(y_true, weighted_pred)  # 최대화를 위해 음수 사용
    
    n_models = len(predictions)
    # 폴드별 성능 기반 초기 가중치
    initial_aucs = [roc_auc_score(y_true, pred) for pred in predictions]
    initial_weights = np.array(initial_aucs) / sum(initial_aucs)
    
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    from scipy.optimize import minimize
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 최종 제출에 사용할 최적 전략:
# 1. 가장 성능이 좋은 단일 폴드 모델 사용 (Fold 2)
# 2. 성능 기반 앙상블 (높은 성능 모델에 더 높은 가중치)
# 3. 최고 성능 폴드의 OOF 예측과 다른 폴드의 예측 간 상관관계 분석하여 
#    상관관계가 낮은 모델들만 앙상블 (다양성 확보)

# 프로그램 실행
if __name__ == "__main__":
    best_model, auc_score = train_and_predict()
    print(f"\n훈련 완료! 최종 AUC: {auc_score:.6f}")