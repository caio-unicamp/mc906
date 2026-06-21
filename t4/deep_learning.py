import pandas as pd
import numpy as np
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DIR = 'processed'
MODELS_DIR = 'models'
SUBMISSIONS_DIR = 'submissions'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            outputs = torch.sigmoid(model(batch_x))
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    return np.array(all_preds), np.array(all_targets)


def train_nn(X_train, y_train, X_val, y_val, input_dim, pos_weight=None):
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train.values)
    )

    counts = np.bincount(y_train.astype(int))
    if pos_weight is None:
        pos_weight = counts[0] / counts[1]

    class_counts = counts.astype(float)
    weights_per_sample = np.where(y_train.values == 1,
                                  1.0 / class_counts[1],
                                  1.0 / class_counts[0])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights_per_sample),
        num_samples=len(X_train),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, sampler=sampler)

    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val.values)
    )
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    model = TabularNN(input_dim).to(DEVICE)

    pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_preds, val_targets = evaluate(model, val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: Loss = {train_loss:.4f}, Val AUC = {val_auc:.6f}")

        if patience_counter >= 10:
            break

    model.load_state_dict(best_state)
    return model


def run_nn_model():
    train = pickle.load(open(f'{PROCESSED_DIR}/train_processed.pkl', 'rb'))
    test = pickle.load(open(f'{PROCESSED_DIR}/test_processed.pkl', 'rb'))
    sample_sub = pickle.load(open(f'{PROCESSED_DIR}/sample_submission.pkl', 'rb'))

    X = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = train['TARGET']
    test_ids = test['SK_ID_CURR']
    X_test = test.drop(columns=['SK_ID_CURR'])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    print(f"\n{'='*60}")
    print(f"Model: Deep Learning NN (PyTorch)")
    print(f"Input dim: {X_scaled.shape[1]}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = train_nn(X_tr, y_tr, X_val, y_val, X_scaled.shape[1])

        val_preds, val_targets = evaluate(
            model,
            DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val.values)),
                      batch_size=4096)
        )
        val_preds = 1 / (1 + np.exp(-val_preds))
        oof_preds[val_idx] = val_preds
        auc = roc_auc_score(val_targets, val_preds)
        scores.append(auc)

        with torch.no_grad():
            test_out = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE)))
            test_preds += test_out.cpu().numpy() / 5

        print(f"  Fold {fold+1}: AUC = {auc:.6f}")

        del model
        gc.collect()
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    oof_auc = roc_auc_score(y, oof_preds)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"\n  Mean CV AUC : {mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  OOF AUC     : {oof_auc:.6f}")

    submission = sample_sub.copy()
    submission['TARGET'] = test_preds
    submission.to_csv(f'{SUBMISSIONS_DIR}/deep_learning_nn.csv', index=False)

    with open(f'{MODELS_DIR}/deep_learning_nn_preds.pkl', 'wb') as f:
        pickle.dump({'oof': oof_preds, 'test': test_preds, 'scores': scores, 'oof_auc': oof_auc}, f)

    result = {
        'name': 'Deep Learning NN',
        'cv_mean': mean_score,
        'cv_std': std_score,
        'oof_auc': oof_auc,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'scores': scores,
    }

    return result


if __name__ == '__main__':
    result = run_nn_model()
    print("\nDeep Learning NN training complete.")
    print(f"OOF AUC: {result['oof_auc']:.6f}")
