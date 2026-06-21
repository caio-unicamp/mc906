import pandas as pd
import numpy as np
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

RANDOM_STATE = 42
PROCESSED_DIR = 'processed'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data_and_preds():
    train = pickle.load(open(f'{PROCESSED_DIR}/train_processed.pkl', 'rb'))
    train_preds = pickle.load(open('models/lightgbm_preds.pkl', 'rb'))

    X = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = train['TARGET']
    oof_preds = train_preds['oof']

    return X, y, oof_preds


def shap_analysis():
    X, y, oof_preds = load_data_and_preds()

    print("Loading LightGBM for SHAP analysis...")
    import lightgbm as lgb

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

    sample_size = min(10000, len(X))
    X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)
    y_sample = y.loc[X_sample.index]

    print("Training LightGBM on sample for SHAP...")
    lgb_model.fit(X_sample, y_sample)

    X_explain = X.sample(n=2000, random_state=RANDOM_STATE)

    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_explain)

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Generating SHAP bar plot (feature importance)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(f'{RESULTS_DIR}/shap_feature_importance.csv', index=False)

    top_features = feature_importance.head(5)['feature'].tolist()
    print(f"Top 5 SHAP features: {top_features}")

    for feat in top_features[:3]:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat, shap_values, X_explain, show=False)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/shap_dependence_{feat.replace("/", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()

    del explainer, shap_values, X_explain, X_sample
    gc.collect()
    print("SHAP analysis complete.")


def error_analysis():
    X, y, oof_preds = load_data_and_preds()

    results = pd.DataFrame({
        'TARGET': y.values,
        'PREDICTION': oof_preds,
    })

    results['PRED_BINARY'] = (results['PREDICTION'] >= 0.5).astype(int)
    results['ERROR_TYPE'] = 'Correct'
    results.loc[(results['TARGET'] == 0) & (results['PRED_BINARY'] == 1), 'ERROR_TYPE'] = 'False Positive'
    results.loc[(results['TARGET'] == 1) & (results['PRED_BINARY'] == 0), 'ERROR_TYPE'] = 'False Negative'

    print("\nError Analysis:")
    print(f"False Positives: {(results['ERROR_TYPE'] == 'False Positive').sum()}")
    print(f"False Negatives: {(results['ERROR_TYPE'] == 'False Negative').sum()}")

    plt.figure(figsize=(10, 6))
    for label, color, name in [(0, '#1f77b4', 'Pays (TARGET=0)'),
                                 (1, '#ff7f0e', 'Defaults (TARGET=1)')]:
        mask = results['TARGET'] == label
        plt.hist(results.loc[mask, 'PREDICTION'], bins=50, alpha=0.6,
                 color=color, label=name, density=True)

    plt.xlabel('Predicted Probability of Default')
    plt.ylabel('Density')
    plt.title('Prediction Distribution by True Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/pred_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    thresholds = np.arange(0.1, 1.0, 0.05)
    fps, fns, f1s = [], [], []
    for t in thresholds:
        pred = (results['PREDICTION'] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        fps.append(fp)
        fns.append(fn)
        if (2*tp + fp + fn) > 0:
            f1s.append(2*tp/(2*tp + fp + fn))
        else:
            f1s.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, fps, 'b-', label='False Positives')
    ax1.plot(thresholds, fns, 'r-', label='False Negatives')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Count')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1s, 'g--', label='F1 Score')
    ax2.set_ylabel('F1 Score', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    plt.title('Error Analysis by Threshold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/error_by_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Error analysis complete.")


def feature_importance_analysis():
    X, y, oof_preds = load_data_and_preds()

    print("Computing correlation with target...")
    correlations = []
    for col in X.columns:
        corr = X[col].corr(y.astype(float))
        correlations.append({'feature': col, 'correlation': corr})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    corr_df.to_csv(f'{RESULTS_DIR}/correlations.csv', index=False)

    top_corr = corr_df.head(15)
    plt.figure(figsize=(12, 8))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_corr['correlation']]
    plt.barh(range(len(top_corr)), top_corr['correlation'].values, color=colors)
    plt.yticks(range(len(top_corr)), top_corr['feature'].values)
    plt.xlabel('Correlation with TARGET')
    plt.title('Top 15 Feature Correlations with Default')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/top_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Feature importance analysis complete.")


if __name__ == '__main__':
    print("Running interpretability analysis...")
    shap_analysis()
    error_analysis()
    feature_importance_analysis()
    print("\nAll analyses complete. Results saved in 'results/' directory.")
