#!/usr/bin/env python3
"""Home Credit Default Risk - Kaggle competition solution.
   Covers: Preprocessing, Logistic Regression, XGBoost, LightGBM, CatBoost,
   Deep Learning NN, Ensemble, SHAP interpretability, Error Analysis."""

import pandas as pd
import numpy as np
import pickle
import os
import gc
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

RANDOM_STATE = 42
N_FOLDS = 5

# ===========================================================================
# CELL 1: Data Loading (switch between Kaggle and Local)
# ===========================================================================

# --- KAGGLE PATH (uncomment for Kaggle submission) ---
# DATA_DIR = '/kaggle/input/home-credit-default-risk'

# --- LOCAL PATH (comment out for Kaggle) ---
import kagglehub
kagglehub.competition_download('home-credit-default-risk')
import shutil
cache_dir = os.path.expanduser('~/.cache/kagglehub/competitions/home-credit-default-risk')
# Unpack the archive if needed
archive_path = os.path.join(cache_dir, 'home-credit-default-risk.archive')
if os.path.exists(archive_path):
    import zipfile
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(cache_dir)
DATA_DIR = cache_dir
os.makedirs('data_local', exist_ok=True)
for f in os.listdir(DATA_DIR):
    if f.endswith('.csv'):
        shutil.copy2(os.path.join(DATA_DIR, f), os.path.join('data_local', f))
DATA_DIR = 'data_local'

# ===========================================================================
# CELL 2: Load main tables
# ===========================================================================

def read_csv_safe(path, **kw):
    try: return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', **kw)

train = read_csv_safe(f'{DATA_DIR}/application_train.csv')
test  = read_csv_safe(f'{DATA_DIR}/application_test.csv')
sub   = read_csv_safe(f'{DATA_DIR}/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target rate: {train['TARGET'].mean():.4f}")
print(f"Target dist: {train['TARGET'].value_counts().to_dict()}")

# ===========================================================================
# CELL 3: EDA Quick Overview
# ===========================================================================

# Missing value overview
missing = train.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).head(20)
print("Top 20 missing features (%):")
print(missing_pct)

# Target distribution plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(['Repays (0)', 'Defaults (1)'], train['TARGET'].value_counts().values,
            color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Target Distribution')
axes[0].set_ylabel('Count')

# Age distribution
if 'DAYS_BIRTH' in train.columns:
    train['AGE_YEARS'] = -train['DAYS_BIRTH'] / 365.25
    axes[1].hist(train[train['TARGET']==0]['AGE_YEARS'], bins=50, alpha=0.6,
                 label='Repays', color='#2ecc71')
    axes[1].hist(train[train['TARGET']==1]['AGE_YEARS'], bins=50, alpha=0.6,
                 label='Defaults', color='#e74c3c')
    axes[1].set_title('Age Distribution by Target')
    axes[1].legend()

# AMT_CREDIT distribution
if 'AMT_CREDIT' in train.columns:
    axes[2].hist(np.log1p(train[train['TARGET']==0]['AMT_CREDIT']), bins=50, alpha=0.6,
                 label='Repays', color='#2ecc71')
    axes[2].hist(np.log1p(train[train['TARGET']==1]['AMT_CREDIT']), bins=50, alpha=0.6,
                 label='Defaults', color='#e74c3c')
    axes[2].set_title('Log Credit Amount by Target')
    axes[2].legend()

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=100)
plt.close()
del train['AGE_YEARS']
print("EDA overview saved.")

# ===========================================================================
# CELL 4: Feature Engineering from Supplementary Tables
# ===========================================================================

def agg_supplement():
    """Aggregate supplementary tables into client-level features."""

    bureau = read_csv_safe(f'{DATA_DIR}/bureau.csv')
    bb     = read_csv_safe(f'{DATA_DIR}/bureau_balance.csv')
    prev   = read_csv_safe(f'{DATA_DIR}/previous_application.csv')
    pos    = read_csv_safe(f'{DATA_DIR}/POS_CASH_balance.csv')
    cc     = read_csv_safe(f'{DATA_DIR}/credit_card_balance.csv')
    ins    = read_csv_safe(f'{DATA_DIR}/installments_payments.csv')

    # --- BUREAU ---
    b_aggs = {
        'AMT_CREDIT_SUM': ['mean','sum','min','max'],
        'AMT_CREDIT_SUM_DEBT': ['mean','sum','max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean','max'],
        'CNT_CREDIT_PROLONG': ['sum'], 'AMT_ANNUITY': ['mean','max'],
        'DAYS_CREDIT': ['mean','min','max'],
        'CREDIT_DAY_OVERDUE': ['mean','max'],
        'DAYS_CREDIT_ENDDATE': ['mean','min','max'],
        'DAYS_ENDDATE_FACT': ['mean','min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean','max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean','sum'],
        'DAYS_CREDIT_UPDATE': ['mean','max'],
    }
    b_agg = bureau.groupby('SK_ID_CURR').agg(b_aggs)
    b_agg.columns = ['BUR_' + '_'.join(c) for c in b_agg.columns]
    b_agg['BUR_COUNT'] = bureau.groupby('SK_ID_CURR').size()

    for col, prefix in [('CREDIT_ACTIVE','BUR_ACT'), ('CREDIT_TYPE','BUR_TYP')]:
        d = pd.get_dummies(bureau[col], prefix=prefix)
        d['SK_ID_CURR'] = bureau['SK_ID_CURR']
        b_agg = b_agg.join(d.groupby('SK_ID_CURR').mean(), how='left')

    # bureau balance
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['size','min','max']})
    bb_agg.columns = ['BB_'+'_'.join(c) for c in bb_agg.columns]
    d = pd.get_dummies(bb['STATUS'], prefix='BB_ST')
    d['SK_ID_BUREAU'] = bb['SK_ID_BUREAU']
    bb_agg = bb_agg.join(d.groupby('SK_ID_BUREAU').mean(), how='left')
    b_agg = b_agg.merge(bb_agg, left_index=True, right_index=True, how='left')
    b_agg.fillna(0, inplace=True)
    del bureau, bb, bb_agg, d; gc.collect()

    # --- PREVIOUS APPLICATION ---
    p_aggs = {
        'AMT_ANNUITY': ['mean','min','max'],
        'AMT_APPLICATION': ['mean','min','max'],
        'AMT_CREDIT': ['mean','min','max'],
        'AMT_DOWN_PAYMENT': ['mean','min','max'],
        'AMT_GOODS_PRICE': ['mean','min','max'],
        'HOUR_APPR_PROCESS_START': ['mean','min','max'],
        'RATE_DOWN_PAYMENT': ['mean','max'],
        'RATE_INTEREST_PRIMARY': ['mean','max'],
        'RATE_INTEREST_PRIVILEGED': ['mean','max'],
        'DAYS_DECISION': ['mean','min','max'],
        'CNT_PAYMENT': ['mean','sum'],
        'DAYS_FIRST_DRAWING': ['mean','min'],
        'DAYS_FIRST_DUE': ['mean','min'],
        'DAYS_LAST_DUE_1ST_VERSION': ['mean','min','max'],
        'DAYS_LAST_DUE': ['mean','min'],
        'DAYS_TERMINATION': ['mean','min'],
        'NFLAG_INSURED_ON_APPROVAL': ['mean'],
        'SELLERPLACE_AREA': ['mean','max'],
        'NFLAG_LAST_APPL_IN_DAY': ['mean'],
    }
    p_agg = prev.groupby('SK_ID_CURR').agg(p_aggs)
    p_agg.columns = ['PRV_' + '_'.join(c) for c in p_agg.columns]
    p_agg['PRV_COUNT'] = prev.groupby('SK_ID_CURR').size()
    p_agg['PRV_APPROVED_RATIO'] = (
        prev[prev['NAME_CONTRACT_STATUS']=='Approved'].groupby('SK_ID_CURR').size()
        / p_agg['PRV_COUNT']
    )
    p_agg['PRV_REFUSED_RATIO'] = (
        prev[prev['NAME_CONTRACT_STATUS']=='Refused'].groupby('SK_ID_CURR').size()
        / p_agg['PRV_COUNT']
    )
    p_agg.fillna(0, inplace=True)
    del prev; gc.collect()

    # --- POS CASH ---
    pos_aggs = {
        'MONTHS_BALANCE': ['size','min','max'],
        'CNT_INSTALMENT': ['mean','min','max'],
        'CNT_INSTALMENT_FUTURE': ['mean','min','max','sum'],
        'SK_DPD': ['mean','max','sum'],
        'SK_DPD_DEF': ['mean','max','sum'],
    }
    pos_agg = pos.groupby('SK_ID_CURR').agg(pos_aggs)
    pos_agg.columns = ['POS_' + '_'.join(c) for c in pos_agg.columns]
    pos_agg['POS_CMPL_RATIO'] = (
        pos[pos['NAME_CONTRACT_STATUS']=='Completed'].groupby('SK_ID_CURR').size()
        / pos_agg['POS_MONTHS_BALANCE_size']
    )
    pos_agg.fillna(0, inplace=True)
    del pos; gc.collect()

    # --- CREDIT CARD ---
    cc_aggs = {
        'MONTHS_BALANCE': ['size','min'],
        'AMT_BALANCE': ['mean','max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean','max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean','max','sum'],
        'AMT_DRAWINGS_CURRENT': ['mean','max','sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['mean','max','sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean','max','sum'],
        'AMT_INST_MIN_REGULARITY': ['mean','max'],
        'AMT_PAYMENT_CURRENT': ['mean','max','sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean','max','sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean','max'],
        'AMT_RECIVABLE': ['mean','max'],
        'AMT_TOTAL_RECEIVABLE': ['mean','max'],
        'CNT_DRAWINGS_ATM_CURRENT': ['mean','max','sum'],
        'CNT_DRAWINGS_CURRENT': ['mean','max','sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['mean','max','sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean','max','sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean','max','sum'],
        'SK_DPD': ['mean','max','sum'],
        'SK_DPD_DEF': ['mean','max','sum'],
    }
    cc_agg = cc.groupby('SK_ID_CURR').agg(cc_aggs)
    cc_agg.columns = ['CC_' + '_'.join(c) for c in cc_agg.columns]
    cc_agg['CC_UTIL'] = cc_agg['CC_AMT_BALANCE_mean'] / (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_mean'] + 1)
    cc_agg['CC_PAY_RATIO'] = cc_agg['CC_AMT_PAYMENT_CURRENT_sum'] / (cc_agg['CC_AMT_BALANCE_mean'] + 1)
    cc_agg.fillna(0, inplace=True)
    cc_agg.replace([np.inf, -np.inf], 0, inplace=True)
    del cc; gc.collect()

    # --- INSTALLMENTS ---
    ins_aggs = {
        'NUM_INSTALMENT_VERSION': ['mean','sum'],
        'NUM_INSTALMENT_NUMBER': ['mean','min','max'],
        'DAYS_INSTALMENT': ['mean','min','max'],
        'DAYS_ENTRY_PAYMENT': ['mean','min','max'],
        'AMT_INSTALMENT': ['mean','min','max','sum'],
        'AMT_PAYMENT': ['mean','min','max','sum'],
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(ins_aggs)
    ins_agg.columns = ['INS_' + '_'.join(c) for c in ins_agg.columns]
    ins_agg['INS_PAY_DIFF_MEAN'] = ins.groupby('SK_ID_CURR').apply(
        lambda x: (x['AMT_INSTALMENT'] - x['AMT_PAYMENT']).mean(), include_groups=False
    )
    ins_agg['INS_LATE_MEAN'] = ins.groupby('SK_ID_CURR').apply(
        lambda x: (x['DAYS_ENTRY_PAYMENT'] - x['DAYS_INSTALMENT']).mean(), include_groups=False
    )
    ins_agg['INS_PAY_RATIO'] = (
        ins.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
        / (ins.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum() + 1)
    )
    ins_agg.fillna(0, inplace=True)
    ins_agg.replace([np.inf, -np.inf], 0, inplace=True)
    del ins; gc.collect()

    return b_agg, p_agg, pos_agg, cc_agg, ins_agg


if not os.path.exists('bureau_agg.pkl'):
    b_agg, p_agg, pos_agg, cc_agg, ins_agg = agg_supplement()
    pickle.dump(b_agg, open('bureau_agg.pkl','wb'))
    pickle.dump(p_agg, open('prev_agg.pkl','wb'))
    pickle.dump(pos_agg, open('pos_agg.pkl','wb'))
    pickle.dump(cc_agg, open('cc_agg.pkl','wb'))
    pickle.dump(ins_agg, open('ins_agg.pkl','wb'))
else:
    b_agg = pickle.load(open('bureau_agg.pkl','rb'))
    p_agg = pickle.load(open('prev_agg.pkl','rb'))
    pos_agg = pickle.load(open('pos_agg.pkl','rb'))
    cc_agg = pickle.load(open('cc_agg.pkl','rb'))
    ins_agg = pickle.load(open('ins_agg.pkl','rb'))

print(f"Bureau agg: {b_agg.shape}")
print(f"Prev app agg: {p_agg.shape}")
print(f"POS cash agg: {pos_agg.shape}")
print(f"Credit card agg: {cc_agg.shape}")
print(f"Installments agg: {ins_agg.shape}")

# ===========================================================================
# CELL 5: Merge & Preprocess Main Table
# ===========================================================================

for name, df in [('bureau', b_agg), ('prev', p_agg), ('pos', pos_agg),
                  ('cc', cc_agg), ('ins', ins_agg)]:
    train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')
    test  = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')

print(f"After merge: train {train.shape}, test {test.shape}")

y = train['TARGET'].copy()
train.drop(columns=['TARGET'], inplace=True)
train_ids = train['SK_ID_CURR'].copy()
test_ids  = test['SK_ID_CURR'].copy()

# Encode categoricals
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([
        train[col].astype(str).fillna('MISSING'),
        test[col].astype(str).fillna('MISSING')
    ])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str).fillna('MISSING'))
    test[col]  = le.transform(test[col].astype(str).fillna('MISSING'))

# Impute missing
print("Imputing missing values...")
imputer = SimpleImputer(strategy='median')
feat_names = train.drop(columns=['SK_ID_CURR']).columns.tolist()
X_all = imputer.fit_transform(train.drop(columns=['SK_ID_CURR']))
test_all = imputer.transform(test.drop(columns=['SK_ID_CURR']))

X = pd.DataFrame(X_all, columns=feat_names)
X_test = pd.DataFrame(test_all, columns=feat_names)

# Drop near-zero variance features
print("Dropping low-variance features...")
selector = VarianceThreshold(threshold=0.001)
X_sel = selector.fit_transform(X)
X_test_sel = selector.transform(X_test)
kept_features = np.array(feat_names)[selector.get_support()].tolist()
X = pd.DataFrame(X_sel, columns=kept_features)
X_test = pd.DataFrame(X_test_sel, columns=kept_features)

print(f"Final feature count: {X.shape[1]}")
print(f"Train: {X.shape}, Test: {X_test.shape}")

# ===========================================================================
# CELL 6: Scale features for LR and DL
# ===========================================================================

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_lr = X_scaled
X_test_lr = X_test_scaled

# ===========================================================================
# CELL 7: Cross-validation setup
# ===========================================================================

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def run_cv(name, model_class, model_params, X_data, use_scale=False, feat_names=None,
           cat_features=False, is_lgb=False, is_xgb=False, is_catboost=False):
    oof = np.zeros(len(X_data))
    test_preds = np.zeros(len(X_test_lr) if use_scale else len(X_test))
    scores = []

    if use_scale:
        X_t = X_test_lr
    else:
        X_t = X_test.values

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_data, y)):
        t0 = time.time()

        if use_scale:
            X_tr, X_va = X_data[tr_idx], X_data[va_idx]
        else:
            X_tr, X_va = X_data.values[tr_idx], X_data.values[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if cat_features and is_catboost:
            m = model_class(**model_params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False,
                  early_stopping_rounds=100)
        elif is_lgb:
            m = model_class(**model_params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc')
        elif is_xgb:
            m = model_class(**model_params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            m = model_class(**model_params)
            m.fit(X_tr, y_tr)

        va_pred = m.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        auc = roc_auc_score(y_va, va_pred)
        scores.append(auc)

        t_pred = m.predict_proba(X_t)[:, 1]
        test_preds += t_pred / N_FOLDS

        elapsed = time.time() - t0
        print(f"  Fold {fold+1}: AUC={auc:.6f}  [{elapsed:.1f}s]")

    oof_auc = roc_auc_score(y, oof)
    print(f"  CV Mean: {np.mean(scores):.6f} (+/-{np.std(scores):.6f})")
    print(f"  OOF AUC: {oof_auc:.6f}\n")
    return {'oof': oof, 'test': test_preds, 'oof_auc': oof_auc, 'scores': scores}

# ===========================================================================
# CELL 8: Model 1 - Logistic Regression (Baseline)
# ===========================================================================

print("="*60)
print("MODEL 1: Logistic Regression (Baseline)")
print("="*60)

lr_results = run_cv(
    'Logistic Regression',
    LogisticRegression,
    {'C': 0.1, 'max_iter': 2000, 'solver': 'saga', 'penalty': 'l2',
     'random_state': RANDOM_STATE, 'n_jobs': -1},
    X_lr, use_scale=True
)
submission_lr = sub.copy()
submission_lr['TARGET'] = lr_results['test']
submission_lr.to_csv('submission_lr.csv', index=False)

# ===========================================================================
# CELL 9: Model 2 - XGBoost
# ===========================================================================

print("="*60)
print("MODEL 2: XGBoost")
print("="*60)

xgb_params = {
    'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 6,
    'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'scale_pos_weight': (1-y.mean()) / y.mean(),
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'random_state': RANDOM_STATE, 'n_jobs': -1,
    'early_stopping_rounds': 100,
}

xgb_results = run_cv('XGBoost', xgb.XGBClassifier, xgb_params, X,
                     use_scale=False, is_xgb=True)
submission_xgb = sub.copy()
submission_xgb['TARGET'] = xgb_results['test']
submission_xgb.to_csv('submission_xgb.csv', index=False)
gc.collect()

# ===========================================================================
# CELL 10: Model 3 - LightGBM
# ===========================================================================

print("="*60)
print("MODEL 3: LightGBM")
print("="*60)

lgb_params = {
    'n_estimators': 3000, 'learning_rate': 0.05, 'num_leaves': 128,
    'max_depth': 7, 'min_child_samples': 20, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': (1-y.mean()) / y.mean(),
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
}

lgb_results = run_cv('LightGBM', lgb.LGBMClassifier, lgb_params, X,
                     use_scale=False, is_lgb=True)
submission_lgb = sub.copy()
submission_lgb['TARGET'] = lgb_results['test']
submission_lgb.to_csv('submission_lgb.csv', index=False)
gc.collect()

# ===========================================================================
# CELL 11: Model 4 - CatBoost
# ===========================================================================

print("="*60)
print("MODEL 4: CatBoost")
print("="*60)

cb_params = {
    'iterations': 2000, 'learning_rate': 0.05, 'depth': 6,
    'l2_leaf_reg': 3, 'random_seed': RANDOM_STATE,
    'scale_pos_weight': (1-y.mean()) / y.mean(),
    'eval_metric': 'AUC', 'task_type': 'CPU', 'thread_count': -1,
    'verbose': False,
}

cb_results = run_cv('CatBoost', CatBoostClassifier, cb_params, X,
                    use_scale=False, is_catboost=True)
submission_cb = sub.copy()
submission_cb['TARGET'] = cb_results['test']
submission_cb.to_csv('submission_cb.csv', index=False)
gc.collect()

# ===========================================================================
# CELL 12: Model 5 - Deep Learning NN
# ===========================================================================

print("="*60)
print("MODEL 5: Deep Learning Neural Network (PyTorch)")
print("="*60)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {DEVICE}")


class TabNN(nn.Module):
    def __init__(self, input_dim, hidden=[512,256,128,64], dropout=0.3):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_nn_fold(X_tr, y_tr, X_va, y_va, input_dim):
    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.values))
    counts = np.bincount(y_tr.astype(int))
    w = np.where(y_tr.values == 1, 1.0/counts[1], 1.0/counts[0])
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(X_tr), replacement=True)
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler)

    model = TabNN(input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(DEVICE))
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    best_auc, best_state, patience = 0, None, 0

    for ep in range(50):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = torch.sigmoid(model(torch.FloatTensor(X_va).to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(y_va, va_pred)
        sched.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break

    model.load_state_dict(best_state)
    return model


dl_oof = np.zeros(len(X_scaled))
dl_test = np.zeros(len(X_test_scaled))
dl_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_scaled, y)):
    t0 = time.time()
    X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = train_nn_fold(X_tr, y_tr, X_va, y_va, X_scaled.shape[1])

    model.eval()
    with torch.no_grad():
        va_pred = torch.sigmoid(model(torch.FloatTensor(X_va).to(DEVICE))).cpu().numpy()
        t_pred  = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE))).cpu().numpy()

    dl_oof[va_idx] = va_pred
    dl_test += t_pred / N_FOLDS
    auc = roc_auc_score(y_va, va_pred)
    dl_scores.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.6f}  [{time.time()-t0:.1f}s]")

dl_oof_auc = roc_auc_score(y, dl_oof)
print(f"  CV Mean: {np.mean(dl_scores):.6f} (+/-{np.std(dl_scores):.6f})")
print(f"  OOF AUC: {dl_oof_auc:.6f}\n")

dl_results = {'oof': dl_oof, 'test': dl_test, 'oof_auc': dl_oof_auc, 'scores': dl_scores}
submission_dl = sub.copy()
submission_dl['TARGET'] = dl_test
submission_dl.to_csv('submission_dl.csv', index=False)
gc.collect()

# ===========================================================================
# CELL 13: Model 6 - Ensemble (Weighted Average)
# ===========================================================================

print("="*60)
print("MODEL 6: Ensemble (Weighted Average)")
print("="*60)

weights = {
    'LightGBM': 0.35,
    'XGBoost': 0.30,
    'CatBoost': 0.25,
    'Deep Learning': 0.10,
}

ensemble_oof = (
    weights['LightGBM'] * lgb_results['oof'] +
    weights['XGBoost'] * xgb_results['oof'] +
    weights['CatBoost'] * cb_results['oof'] +
    weights['Deep Learning'] * dl_results['oof']
)

ensemble_test = (
    weights['LightGBM'] * lgb_results['test'] +
    weights['XGBoost'] * xgb_results['test'] +
    weights['CatBoost'] * cb_results['test'] +
    weights['Deep Learning'] * dl_results['test']
)

ensemble_oof_auc = roc_auc_score(y, ensemble_oof)
print(f"  Ensemble OOF AUC: {ensemble_oof_auc:.6f}")

submission_ens = sub.copy()
submission_ens['TARGET'] = ensemble_test
submission_ens.to_csv('submission_ensemble.csv', index=False)

# ===========================================================================
# CELL 14: Model Comparison
# ===========================================================================

print("\n" + "="*70)
print("FINAL MODEL COMPARISON (ROC AUC)")
print("="*70)

models = {
    'Logistic Regression': lr_results['oof_auc'],
    'XGBoost': xgb_results['oof_auc'],
    'LightGBM': lgb_results['oof_auc'],
    'CatBoost': cb_results['oof_auc'],
    'Deep Learning NN': dl_results['oof_auc'],
    'Ensemble': ensemble_oof_auc,
}

comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'OOF AUC': list(models.values()),
}).sort_values('OOF AUC', ascending=False)

print(comparison_df.to_string(index=False))
comparison_df.to_csv('model_comparison.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#3498db']*5 + ['#e74c3c']
bars = ax.barh(list(models.keys()), list(models.values()), color=colors)
ax.set_xlabel('ROC AUC')
ax.set_title('Model Comparison - Home Credit Default Risk')
ax.set_xlim(min(models.values()) - 0.01, max(models.values()) + 0.005)
for bar, val in zip(bars, models.values()):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.6f}', va='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
plt.close()

# ===========================================================================
# CELL 15: Interpretability - SHAP Analysis
# ===========================================================================

print("\n" + "="*60)
print("SHAP Interpretability Analysis")
print("="*60)

# Train a single LightGBM for SHAP (use sample for speed)
sample_size = min(5000, len(X))
X_shap_train = X.sample(n=sample_size, random_state=RANDOM_STATE)
y_shap_train = y.loc[X_shap_train.index]

lgb_shap = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=64, max_depth=6,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
lgb_shap.fit(X_shap_train, y_shap_train)

X_explain = X.sample(n=500, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(lgb_shap)
shap_values = explainer.shap_values(X_explain)

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_explain, show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar plot (importance)
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_explain, plot_type='bar', show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance CSV
shap_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)
shap_imp.to_csv('shap_feature_importance.csv', index=False)

top5 = shap_imp.head(5)['feature'].tolist()
print(f"Top 5 SHAP features: {top5}")

# Dependence plots for top features
for feat in top5[:3]:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat, shap_values, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feat[:30].replace("/","_")}.png', dpi=100)
    plt.close()

del explainer, shap_values, X_explain, X_shap_train
gc.collect()
print("SHAP analysis complete.")

# ===========================================================================
# CELL 16: Error Analysis
# ===========================================================================

print("\n" + "="*60)
print("Error Analysis")
print("="*60)

results_df = pd.DataFrame({
    'TARGET': y.values,
    'PREDICTION': lgb_results['oof'],
})
results_df['PRED_BIN'] = (results_df['PREDICTION'] >= 0.5).astype(int)

tn = ((results_df['TARGET']==0) & (results_df['PRED_BIN']==0)).sum()
fp = ((results_df['TARGET']==0) & (results_df['PRED_BIN']==1)).sum()
fn = ((results_df['TARGET']==1) & (results_df['PRED_BIN']==0)).sum()
tp = ((results_df['TARGET']==1) & (results_df['PRED_BIN']==1)).sum()

print(f"Confusion Matrix:")
print(f"  TN={tn:8d}  FP={fp:8d}")
print(f"  FN={fn:8d}  TP={tp:8d}")
print(f"  Precision: {tp/(tp+fp):.4f}" if (tp+fp)>0 else "")
print(f"  Recall:    {tp/(tp+fn):.4f}" if (tp+fn)>0 else "")

# Prediction distribution by class
fig, ax = plt.subplots(figsize=(10, 6))
for label, color, name in [(0, '#2ecc71', 'Pays'), (1, '#e74c3c', 'Defaults')]:
    mask = results_df['TARGET'] == label
    ax.hist(results_df.loc[mask, 'PREDICTION'], bins=50, alpha=0.6,
            color=color, label=name, density=True)
ax.set_xlabel('Predicted Probability of Default')
ax.set_ylabel('Density')
ax.set_title('Prediction Distribution by True Class')
ax.legend()
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=100)
plt.close()

# Threshold analysis
thresholds = np.arange(0.1, 1.0, 0.05)
tpr_list, fpr_list, f1_list = [], [], []
for t in thresholds:
    pr = (results_df['PREDICTION'] >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(results_df['TARGET'], pr).ravel()
    tpr_list.append(tp_ / (tp_ + fn_) if (tp_+fn_) > 0 else 0)
    fpr_list.append(fp_ / (fp_ + tn_) if (fp_+tn_) > 0 else 0)
    f1_list.append(2*tp_/(2*tp_+fp_+fn_) if (2*tp_+fp_+fn_)>0 else 0)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds, tpr_list, 'g-', label='Recall (TPR)')
ax1.plot(thresholds, fpr_list, 'r-', label='FPR')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Rate')
ax1.legend(loc='center left')

ax2 = ax1.twinx()
ax2.plot(thresholds, f1_list, 'b--', label='F1 Score')
ax2.set_ylabel('F1 Score', color='b')
ax2.legend(loc='center right')
plt.title('Metrics by Decision Threshold')
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=100)
plt.close()

# Correlation analysis
correlations = []
for col in X.columns[:100]:
    c = np.corrcoef(X[col].fillna(0), y.astype(float))[0,1]
    correlations.append({'feature': col, 'correlation': c})
corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top_corr = corr_df.head(15)
colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_corr['correlation']]
ax.barh(range(len(top_corr)), top_corr['correlation'].values, color=colors_bar)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr['feature'].values)
ax.set_xlabel('Correlation with TARGET')
ax.set_title('Top 15 Feature Correlations with Default')
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=100)
plt.close()

print("Error analysis complete.")

# ===========================================================================
# CELL 17: Generate Final Submission
# ===========================================================================

print("\n" + "="*60)
print("GENERATING FINAL SUBMISSION")
print("="*60)

final_submission = sub.copy()
final_submission['TARGET'] = ensemble_test
final_submission.to_csv('submission.csv', index=False)

print(f"Final submission shape: {final_submission.shape}")
print(f"Submission saved to: submission.csv")
print(f"Ensemble OOF AUC: {ensemble_oof_auc:.6f}")
print(f"\nBest single model (LightGBM) OOF AUC: {lgb_results['oof_auc']:.6f}")
print(f"\nAll submissions saved:")
for name, auc in models.items():
    print(f"  {name}: {auc:.6f}")

print("\n" + "="*60)
print("DONE! All models trained and submissions generated.")
print("="*60)
