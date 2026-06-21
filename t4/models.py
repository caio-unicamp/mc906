import pandas as pd
import numpy as np
import pickle
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
N_FOLDS = 5
PROCESSED_DIR = 'processed'
MODELS_DIR = 'models'
SUBMISSIONS_DIR = 'submissions'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def load_data():
    train = pickle.load(open(f'{PROCESSED_DIR}/train_processed.pkl', 'rb'))
    test = pickle.load(open(f'{PROCESSED_DIR}/test_processed.pkl', 'rb'))
    sample_sub = pickle.load(open(f'{PROCESSED_DIR}/sample_submission.pkl', 'rb'))

    X = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = train['TARGET']
    test_ids = test['SK_ID_CURR']
    X_test = test.drop(columns=['SK_ID_CURR'])
    feature_names = X.columns.tolist()

    print(f"Train: {X.shape}, Test: {X_test.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Target rate: {y.mean():.4f}")

    return X, y, X_test, test_ids, feature_names, sample_sub


def evaluate_model(name, model, X, y, X_test, test_ids, sample_sub,
                   scale=False, feature_select=False):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if feature_select:
        selector = VarianceThreshold(threshold=0.01)
        X_s = selector.fit_transform(X)
        X_test_s = selector.transform(X_test)
    else:
        X_s = X.values if hasattr(X, 'values') else X
        X_test_s = X_test.values if hasattr(X_test, 'values') else X_test

    if scale:
        scaler = RobustScaler()
        X_s = scaler.fit_transform(X_s)
        X_test_s = scaler.transform(X_test_s)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    scores = []
    fold_times = []

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_s, y)):
        t0 = time.time()
        X_tr, X_val = X_s[train_idx], X_s[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if hasattr(model, 'fit') and not hasattr(model, 'n_features_in_'):
            m = model
        else:
            import copy
            m = copy.deepcopy(model)

        if isinstance(m, lgb.LGBMClassifier):
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
            )
        elif isinstance(m, xgb.XGBClassifier):
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif isinstance(m, CatBoostClassifier):
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=100,
            )
        else:
            m.fit(X_tr, y_tr)

        val_pred = m.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        test_pred = m.predict_proba(X_test_s)[:, 1]
        test_preds += test_pred / N_FOLDS

        fold_time = time.time() - t0
        fold_times.append(fold_time)
        print(f"  Fold {fold+1}: AUC = {score:.6f}, Time = {fold_time:.1f}s")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    oof_score = roc_auc_score(y, oof_preds)

    print(f"\n  Mean CV AUC : {mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  OOF AUC     : {oof_score:.6f}")

    submission = sample_sub.copy()
    submission['TARGET'] = test_preds
    submission.to_csv(f'{SUBMISSIONS_DIR}/{name.replace(" ", "_").lower()}.csv', index=False)

    with open(f'{MODELS_DIR}/{name.replace(" ", "_").lower()}_preds.pkl', 'wb') as f:
        pickle.dump({'oof': oof_preds, 'test': test_preds, 'scores': scores, 'oof_auc': oof_score}, f)

    return {
        'name': name,
        'cv_mean': mean_score,
        'cv_std': std_score,
        'oof_auc': oof_score,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'scores': scores,
    }


def run_all_models():
    X, y, X_test, test_ids, feature_names, sample_sub = load_data()

    results = {}

    # Model 1: Logistic Regression (Baseline)
    print("\n" + "="*80)
    print("MODEL 1: Logistic Regression (Baseline)")
    print("="*80)
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver='saga',
        penalty='l2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results['Logistic Regression'] = evaluate_model(
        'Logistic Regression', lr, X, y, X_test, test_ids, sample_sub,
        scale=True, feature_select=True
    )

    # Model 2: XGBoost
    print("\n" + "="*80)
    print("MODEL 2: XGBoost")
    print("="*80)
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(1 - y.mean()) / y.mean(),
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    results['XGBoost'] = evaluate_model(
        'XGBoost', xgb_model, X, y, X_test, test_ids, sample_sub
    )

    gc.collect()

    # Model 3: LightGBM
    print("\n" + "="*80)
    print("MODEL 3: LightGBM")
    print("="*80)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=128,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=(1 - y.mean()) / y.mean(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    results['LightGBM'] = evaluate_model(
        'LightGBM', lgb_model, X, y, X_test, test_ids, sample_sub
    )

    gc.collect()

    # Model 4: CatBoost
    print("\n" + "="*80)
    print("MODEL 4: CatBoost")
    print("="*80)
    cb_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        scale_pos_weight=(1 - y.mean()) / y.mean(),
        eval_metric='AUC',
        task_type='CPU',
        thread_count=-1,
        verbose=False,
    )

    X_cb = X.copy()
    X_test_cb = X_test.copy()
    results['CatBoost'] = evaluate_model(
        'CatBoost', cb_model, X_cb, y, X_test_cb, test_ids, sample_sub
    )

    gc.collect()

    # Model 5: Ensemble (average of top models)
    print("\n" + "="*80)
    print("MODEL 5: Ensemble (Weighted Average)")
    print("="*80)
    ensemble_test = np.zeros(len(test_ids))
    ensemble_oof = np.zeros(len(X))

    weights = {
        'LightGBM': 0.35,
        'XGBoost': 0.30,
        'CatBoost': 0.25,
        'Logistic Regression': 0.10,
    }

    for name, w in weights.items():
        if name in results:
            ensemble_test += w * results[name]['test_preds']
            ensemble_oof += w * results[name]['oof_preds']

    ensemble_oof_auc = roc_auc_score(y, ensemble_oof)
    submission = sample_sub.copy()
    submission['TARGET'] = ensemble_test
    submission.to_csv(f'{SUBMISSIONS_DIR}/ensemble.csv', index=False)

    results['Ensemble'] = {
        'name': 'Ensemble',
        'cv_mean': None,
        'cv_std': None,
        'oof_auc': ensemble_oof_auc,
        'oof_preds': ensemble_oof,
        'test_preds': ensemble_test,
        'scores': [],
    }
    print(f"  Ensemble OOF AUC: {ensemble_oof_auc:.6f}")

    return results


def display_comparison(results):
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON (ROC AUC)")
    print("="*80)
    print(f"{'Model':<25} {'CV Mean':<12} {'CV Std':<12} {'OOF AUC':<12}")
    print("-"*61)

    comparison_data = []
    for name, res in results.items():
        cv_mean = f"{res['cv_mean']:.6f}" if res['cv_mean'] is not None else 'N/A'
        cv_std = f"{res['cv_std']:.6f}" if res['cv_std'] is not None else 'N/A'
        oof_auc = f"{res['oof_auc']:.6f}"
        print(f"{name:<25} {cv_mean:<12} {cv_std:<12} {oof_auc:<12}")
        comparison_data.append({
            'Model': name,
            'CV_Mean_AUC': res['cv_mean'],
            'CV_Std_AUC': res['cv_std'],
            'OOF_AUC': res['oof_auc'],
        })

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv('model_comparison.csv', index=False)
    print(f"\nComparison saved to model_comparison.csv")
    return df_comp


if __name__ == '__main__':
    results = run_all_models()
    display_comparison(results)
    print("\nDone! Submissions saved in 'submissions/' directory.")
