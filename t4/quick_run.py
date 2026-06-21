import pandas as pd, numpy as np, pickle, os, gc, time, warnings, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

RANDOM_STATE = 42; N_FOLDS = 5
os.makedirs('submissions', exist_ok=True); os.makedirs('results', exist_ok=True)

train = pickle.load(open('processed/train_processed.pkl', 'rb'))
test = pickle.load(open('processed/test_processed.pkl', 'rb'))
sub = pickle.load(open('processed/sample_submission.pkl', 'rb'))

X_all = train.drop(columns=['SK_ID_CURR', 'TARGET'])
y = train['TARGET']
X_test_all = test.drop(columns=['SK_ID_CURR'])

selector = VarianceThreshold(threshold=0.001)
X_sel = selector.fit_transform(X_all)
X_test_sel = selector.transform(X_test_all)
kept = np.array(X_all.columns)[selector.get_support()]
X = pd.DataFrame(X_sel, columns=kept)
X_test = pd.DataFrame(X_test_sel, columns=kept)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

N = len(X); Nt = len(X_test)
lr_oof = np.zeros(N); lr_test = np.zeros(Nt); lr_scores = []
for fold, (tr, va) in enumerate(cv.split(X_scaled, y)):
    m = LogisticRegression(C=0.1, max_iter=2000, solver='liblinear', random_state=RANDOM_STATE)
    m.fit(X_scaled[tr], y.iloc[tr])
    lr_oof[va] = m.predict_proba(X_scaled[va])[:, 1]
    lr_test += m.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
    lr_scores.append(roc_auc_score(y.iloc[va], lr_oof[va]))
lr_auc = roc_auc_score(y, lr_oof)
lr_cv_mean = np.mean(lr_scores); lr_cv_std = np.std(lr_scores)

xgb_oof = np.zeros(N); xgb_test = np.zeros(Nt); xgb_scores = []
for fold, (tr, va) in enumerate(cv.split(X, y)):
    m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(1-y.mean())/y.mean(), objective='binary:logistic',
        eval_metric='auc', random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X.values[tr], y.iloc[tr], eval_set=[(X.values[va], y.iloc[va])], verbose=False)
    xgb_oof[va] = m.predict_proba(X.values[va])[:, 1]
    xgb_test += m.predict_proba(X_test.values)[:, 1] / N_FOLDS
    xgb_scores.append(roc_auc_score(y.iloc[va], xgb_oof[va]))
xgb_auc = roc_auc_score(y, xgb_oof)
xgb_cv_mean = np.mean(xgb_scores); xgb_cv_std = np.std(xgb_scores); gc.collect()

lgb_oof = np.zeros(N); lgb_test = np.zeros(Nt); lgb_scores = []
for fold, (tr, va) in enumerate(cv.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=128, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=(1-y.mean())/y.mean(),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    m.fit(X.values[tr], y.iloc[tr], eval_set=[(X.values[va], y.iloc[va])], eval_metric='auc')
    lgb_oof[va] = m.predict_proba(X.values[va])[:, 1]
    lgb_test += m.predict_proba(X_test.values)[:, 1] / N_FOLDS
    lgb_scores.append(roc_auc_score(y.iloc[va], lgb_oof[va]))
lgb_auc = roc_auc_score(y, lgb_oof)
lgb_cv_mean = np.mean(lgb_scores); lgb_cv_std = np.std(lgb_scores); gc.collect()

cb_oof = np.zeros(N); cb_test = np.zeros(Nt); cb_scores = []
for fold, (tr, va) in enumerate(cv.split(X, y)):
    m = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        scale_pos_weight=(1-y.mean())/y.mean(), random_seed=RANDOM_STATE, eval_metric='AUC',
        task_type='CPU', thread_count=-1, verbose=False)
    m.fit(X.values[tr], y.iloc[tr], eval_set=[(X.values[va], y.iloc[va])], verbose=False, early_stopping_rounds=30)
    cb_oof[va] = m.predict_proba(X.values[va])[:, 1]
    cb_test += m.predict_proba(X_test.values)[:, 1] / N_FOLDS
    cb_scores.append(roc_auc_score(y.iloc[va], cb_oof[va]))
cb_auc = roc_auc_score(y, cb_oof)
cb_cv_mean = np.mean(cb_scores); cb_cv_std = np.std(cb_scores); gc.collect()

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TabNN(nn.Module):
    def __init__(self, idim, h=[256,128,64], d=0.3):
        super().__init__(); ls=[]; di=idim
        for hi in h: ls+=[nn.Linear(di,hi),nn.BatchNorm1d(hi),nn.ReLU(),nn.Dropout(d)]; di=hi
        ls.append(nn.Linear(di,1)); self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x).squeeze(1)

dl_oof = np.zeros(N); dl_test = np.zeros(Nt); dl_scores = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_scaled, y)):
    X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]; y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    pw = (len(y_tr)-y_tr.sum())/y_tr.sum()
    cts = np.bincount(y_tr.astype(int)); sw = np.where(y_tr.values==1, 1.0/max(cts[1],1), 1.0/max(cts[0],1))
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.values)),
        batch_size=1024, sampler=WeightedRandomSampler(torch.DoubleTensor(sw), len(X_tr), replacement=True))
    model = TabNN(X_scaled.shape[1]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    ba, pa, bs = 0, 0, None
    for ep in range(50):
        model.train()
        for bx, by in loader: bx,by=bx.to(DEVICE),by.to(DEVICE); opt.zero_grad(); crit(model(bx),by).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vp = torch.sigmoid(model(torch.FloatTensor(X_va).to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(y_va, vp); sched.step(auc)
        if auc > ba: ba, pa = auc, 0; bs = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else: pa += 1
        if pa >= 8: break
    model.load_state_dict(bs); model.eval()
    with torch.no_grad():
        dl_oof[va_idx] = torch.sigmoid(model(torch.FloatTensor(X_va).to(DEVICE))).cpu().numpy()
        dl_test += torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE))).cpu().numpy() / N_FOLDS
    dl_scores.append(roc_auc_score(y_va, dl_oof[va_idx])); del model; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
dl_auc = roc_auc_score(y, dl_oof); dl_cv_mean = np.mean(dl_scores); dl_cv_std = np.std(dl_scores); gc.collect()

ens_oof = 0.35*lgb_oof + 0.30*xgb_oof + 0.25*cb_oof + 0.10*dl_oof
ens_test = 0.35*lgb_test + 0.30*xgb_test + 0.25*cb_test + 0.10*dl_test
ens_auc = roc_auc_score(y, ens_oof)

comp_df = pd.DataFrame({
    'Model': ['Logistic Regression','XGBoost','LightGBM','CatBoost','Deep Learning NN','Ensemble'],
    'OOF_AUC': [lr_auc, xgb_auc, lgb_auc, cb_auc, dl_auc, ens_auc],
    'CV_Mean': [lr_cv_mean, xgb_cv_mean, lgb_cv_mean, cb_cv_mean, dl_cv_mean, np.nan],
    'CV_Std': [lr_cv_std, xgb_cv_std, lgb_cv_std, cb_cv_std, dl_cv_std, np.nan],
}).sort_values('OOF_AUC', ascending=False).reset_index(drop=True)
comp_df.to_csv('results/model_comparison.csv', index=False)
display(comp_df)

for name, t in [('logistic_regression',lr_test),('xgboost',xgb_test),('lightgbm',lgb_test),
    ('catboost',cb_test),('deep_learning_nn',dl_test),('ensemble',ens_test)]:
    s=sub.copy(); s['TARGET']=t; s.to_csv(f'submissions/submission_{name}.csv', index=False)
sub_final=sub.copy(); sub_final['TARGET']=ens_test; sub_final.to_csv('submission.csv', index=False)

fig, ax = plt.subplots(figsize=(10,5))
models_list = comp_df['Model'].tolist(); aucs = comp_df['OOF_AUC'].tolist()
colors = ['#3498db']*4 + ['#9b59b6'] + ['#e74c3c']
bars = ax.barh(models_list, aucs, color=colors)
ax.set_xlabel('ROC AUC'); ax.set_title('Model Comparison')
ax.set_xlim(min(aucs)-0.01, max(aucs)+0.005)
for bar, v in zip(bars, aucs): ax.text(v+0.0005, bar.get_y()+bar.get_height()/2, f'{v:.6f}', va='center')
plt.tight_layout(); plt.savefig('results/model_comparison.png', dpi=150); plt.close()

n_sample = min(2000, len(X))
X_shap_tr = X.sample(n=n_sample, random_state=RANDOM_STATE)
y_shap_tr = y.loc[X_shap_tr.index]
lgb_shap = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=64, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
lgb_shap.fit(X_shap_tr, y_shap_tr)
X_explain = X.sample(n=200, random_state=RANDOM_STATE)
expl = shap.TreeExplainer(lgb_shap)
sv = expl.shap_values(X_explain)

plt.figure(figsize=(12,8))
shap.summary_plot(sv, X_explain, show=False, max_display=15)
plt.tight_layout(); plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(12,7))
shap.summary_plot(sv, X_explain, plot_type='bar', show=False, max_display=15)
plt.tight_layout(); plt.savefig('results/shap_importance.png', dpi=150, bbox_inches='tight'); plt.close()

shap_imp = pd.DataFrame({'feature': X.columns, 'importance': np.abs(sv).mean(axis=0)}).sort_values('importance', ascending=False)
shap_imp.to_csv('results/shap_feature_importance.csv', index=False)
display(shap_imp.head(10))

correlations = []
for col in X.columns[:200]:
    c = np.corrcoef(X[col].fillna(0), y.astype(float))[0,1]
    correlations.append({'feature': col, 'correlation': c})
corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
corr_df.to_csv('results/feature_correlations.csv', index=False)

pred_bin = (lgb_oof >= 0.5).astype(int)
tn = ((y==0) & (pred_bin==0)).sum(); fp = ((y==0) & (pred_bin==1)).sum()
fn = ((y==1) & (pred_bin==0)).sum(); tp = ((y==1) & (pred_bin==1)).sum()

fig, ax = plt.subplots(figsize=(10,6))
for label, color, name in [(0, '#2ecc71', 'Pays'), (1, '#e74c3c', 'Defaults')]:
    mask = y.values == label
    ax.hist(lgb_oof[mask], bins=50, alpha=0.6, color=color, label=name, density=True)
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density'); ax.legend()
plt.tight_layout(); plt.savefig('results/error_distribution.png', dpi=150); plt.close()

json.dump({
    'models': {'lr':float(lr_auc),'xgb':float(xgb_auc),'lgb':float(lgb_auc),'cb':float(cb_auc),'dl':float(dl_auc),'ens':float(ens_auc)},
    'lr_cv':(float(lr_cv_mean),float(lr_cv_std)),'xgb_cv':(float(xgb_cv_mean),float(xgb_cv_std)),
    'lgb_cv':(float(lgb_cv_mean),float(lgb_cv_std)),'cb_cv':(float(cb_cv_mean),float(cb_cv_std)),
    'dl_cv':(float(dl_cv_mean),float(dl_cv_std)),
    'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
    'target_rate':float(y.mean()), 'n_features':int(X.shape[1]), 'n_samples':int(len(X)),
    'top_shap':shap_imp.head(10).to_dict('records'),
    'top_corr':corr_df.head(10).to_dict('records'),
}, open('results/report_data.json', 'w'), indent=2)
