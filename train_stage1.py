"""
Stage 1 Training - Gesture Entry Detection (Binary Classification)
Grid search over window sizes, step sizes, and model architectures.
"""

import os
import glob
import copy
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    STAGE1_DATA_DIR, MODEL_DIR, REPORT_DIR, IMU_CHANNELS,
    FEATURE_COLS, FEATURE_IDX, SAMPLE_RATE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, EARLY_STOP, SEED
)

# Grid search parameters
WINDOW_SECS = [1.0, 1.5, 2.0]
STEP_SECS = [0.5, 0.75, 1.0]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_TYPES = ["mlp", "lstm", "gru", "tcn", "cnn1d", "cnn_lstm"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Session:
    sid: str
    data: np.ndarray
    labels: np.ndarray
    times: np.ndarray


# =============================================================================
# Data Loading
# =============================================================================

def load_sessions(data_dir: str) -> List[Session]:
    """Load all sessions from data directory."""
    dirs = sorted(glob.glob(os.path.join(data_dir, "session_*")))
    if not dirs:
        raise FileNotFoundError(f"No sessions in {data_dir}")

    sessions = []
    for d in dirs:
        # Support both new and old file naming
        csv_path = os.path.join(d, "data.csv")
        pkl_path = os.path.join(d, "data.pkl")
        csv_path_old = os.path.join(d, "continuous_detection.csv")
        pkl_path_old = os.path.join(d, "continuous_detection.pkl")

        # Prefer CSV for compatibility
        if os.path.exists(csv_path):
            actual_csv = csv_path
        elif os.path.exists(csv_path_old):
            actual_csv = csv_path_old
        else:
            actual_csv = None

        if os.path.exists(pkl_path):
            actual_pkl = pkl_path
        elif os.path.exists(pkl_path_old):
            actual_pkl = pkl_path_old
        else:
            actual_pkl = None

        if actual_csv:
            df = pd.read_csv(actual_csv)
            data = df[list(IMU_CHANNELS.keys())].values.astype(np.float32)
            labels = df['label'].values.astype(np.int32)
            times = df['relative_time'].values.astype(np.float32)
            sid = os.path.basename(d).replace("session_", "")
        elif actual_pkl:
            with open(actual_pkl, 'rb') as f:
                raw = pickle.load(f)
            samples = raw['data']
            data = np.array([s['data'] for s in samples], dtype=np.float32)
            labels = np.array([s.get('label', 0) for s in samples], dtype=np.int32)
            times = np.array([s.get('relative_time', s.get('t', 0)) for s in samples], dtype=np.float32)
            sid = raw.get('session_id', os.path.basename(d))
        else:
            continue

        sessions.append(Session(sid=sid, data=data, labels=labels, times=times))
        pos = int(labels.sum())
        print(f"  {sid}: {len(labels)} samples, {pos} positive")

    if not sessions:
        raise RuntimeError(f"No valid sessions in {data_dir}")
    return sessions


def resample_to_rate(session: Session, target_fs: float) -> Session:
    """Resample session to target sampling rate."""
    times = session.times
    if len(times) < 2:
        return session

    dt = np.median(np.diff(times[times > 0]))
    if dt <= 0:
        return session

    new_times = np.arange(times[0], times[-1], 1.0 / target_fs, dtype=np.float32)
    if len(new_times) < 2:
        return session

    new_data = np.zeros((len(new_times), session.data.shape[1]), dtype=np.float32)
    for ch in range(session.data.shape[1]):
        new_data[:, ch] = np.interp(new_times, times, session.data[:, ch])

    label_f = np.interp(new_times, times, session.labels.astype(np.float32))
    new_labels = (label_f >= 0.5).astype(np.int32)

    return Session(sid=session.sid, data=new_data, labels=new_labels, times=new_times)


# =============================================================================
# Window Creation
# =============================================================================

def create_windows(session: Session, win_sec: float, step_sec: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows with center-label approach."""
    dt = np.median(np.diff(session.times))
    if dt <= 0:
        dt = 1.0 / SAMPLE_RATE

    win_len = int(round(win_sec / dt))
    step_len = max(1, int(round(step_sec / dt)))

    X_list, y_list, t_list = [], [], []
    N = len(session.data)

    for start in range(0, N - win_len + 1, step_len):
        end = start + win_len
        window = session.data[start:end]
        if window.shape[0] != win_len:
            continue

        center = start + win_len // 2
        label = session.labels[center]
        center_t = session.times[center]

        X_list.append(window)
        y_list.append(int(label))
        t_list.append(float(center_t))

    if not X_list:
        return np.empty((0, win_len, 55)), np.empty((0,), dtype=np.int32), np.empty((0,))

    return np.stack(X_list), np.array(y_list, dtype=np.int32), np.array(t_list)


def create_all_windows(sessions: List[Session], win_sec: float, step_sec: float
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create windows from all sessions."""
    all_X, all_y, all_t, all_sid = [], [], [], []

    for sess in sessions:
        X, y, t = create_windows(sess, win_sec, step_sec)
        if X.size == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_t.append(t)
        all_sid.extend([sess.sid] * len(y))

    if not all_X:
        raise RuntimeError("No windows created")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    sid = np.array(all_sid)

    # Extract only feature columns
    X = X[:, :, FEATURE_IDX]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"  Windows: {X.shape[0]} (pos={pos}, neg={neg})")

    return X, y, sid, t


# =============================================================================
# Train/Val/Test Split
# =============================================================================

def split_by_time(X, y, sid, t) -> Tuple:
    """Time-based split per session."""
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        times = t[idx]
        order = np.argsort(times)
        idx_sorted = idx[order]

        m = len(idx_sorted)
        if m < 5:
            train_mask[idx_sorted] = True
            continue

        n_train = int(round(m * TRAIN_RATIO))
        n_val = int(round(m * VAL_RATIO))

        train_mask[idx_sorted[:n_train]] = True
        val_mask[idx_sorted[n_train:n_train + n_val]] = True
        test_mask[idx_sorted[n_train + n_val:]] = True

    return (X[train_mask], y[train_mask],
            X[val_mask], y[val_mask],
            X[test_mask], y[test_mask])


# =============================================================================
# Normalization
# =============================================================================

def normalize(X_train, X_val, X_test, eps=1e-6, clip=1e4):
    """Channel-wise z-score normalization."""
    def safe(x):
        x = x.astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(x, -clip, clip)

    X_train = safe(X_train)
    X_val = safe(X_val)
    X_test = safe(X_test)

    N, T, D = X_train.shape
    flat = X_train.reshape(N * T, D)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < eps] = 1.0

    def zscore(x):
        x = (x - mean) / std
        return np.nan_to_num(x, nan=0.0).astype(np.float32)

    stats = {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}
    return zscore(X_train), zscore(X_val), zscore(X_test), stats


# =============================================================================
# Models
# =============================================================================

class MLP(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * D, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class LSTM(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.rnn = nn.LSTM(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1]).squeeze(1)


class GRU(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.rnn = nn.GRU(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1]).squeeze(1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, drop=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=d * (k - 1))
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)[..., :x.size(-1)]
        out = self.drop(self.bn(out))
        res = self.down(x) if self.down else x
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.tcn = nn.Sequential(
            TCNBlock(D, 64, d=1), TCNBlock(64, 64, d=2), TCNBlock(64, 64, d=4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.pool(self.tcn(x)).squeeze(-1)
        return self.fc(out).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.pool(self.cnn(x)).squeeze(-1)
        return self.fc(out).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


def build_model(name: str, shape):
    models = {
        'mlp': MLP, 'lstm': LSTM, 'gru': GRU,
        'tcn': TCN, 'cnn1d': CNN1D, 'cnn_lstm': CNNLSTM
    }
    return models[name](shape)


# =============================================================================
# Training
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping."""
    X_tr = torch.from_numpy(X_train).to(DEVICE)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    X_va = torch.from_numpy(X_val).to(DEVICE)
    y_va = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)

    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / pos], device=DEVICE) if pos > 0 and neg > 0 else None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, X_test, y_test, thresholds):
    """Evaluate model at multiple thresholds."""
    X_t = torch.from_numpy(X_test).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()

    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = np.nan

    results = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        acc = accuracy_score(y_test, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, labels=[1], average='binary', zero_division=0
        )
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'threshold': thr, 'accuracy': acc,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc
        })

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(df, save_dir):
    """Generate performance visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Best model per type
    best_per_model = df.loc[df.groupby('model')['f1'].idxmax()]

    # Bar chart: F1 by model
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(best_per_model)))
    bars = plt.bar(best_per_model['model'], best_per_model['f1'], color=colors)
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Stage 1: Best F1 Score by Model')
    plt.ylim(0, 1)
    for bar, val in zip(bars, best_per_model['f1']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage1_f1_by_model.png'), dpi=150)
    plt.close()

    # Heatmap: F1 by window/step for best model
    best_model = df.loc[df['f1'].idxmax(), 'model']
    subset = df[df['model'] == best_model]
    pivot = subset.pivot_table(values='f1', index='window_sec', columns='step_sec', aggfunc='max')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title(f'Stage 1: F1 Heatmap ({best_model})')
    plt.xlabel('Step (sec)')
    plt.ylabel('Window (sec)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage1_f1_heatmap.png'), dpi=150)
    plt.close()

    # Line chart: Threshold vs metrics
    best_row = df.loc[df['f1'].idxmax()]
    subset = df[(df['model'] == best_row['model']) &
                (df['window_sec'] == best_row['window_sec']) &
                (df['step_sec'] == best_row['step_sec'])]
    plt.figure(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(subset['threshold'], subset[metric], 'o-', label=metric)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Stage 1: Metrics vs Threshold ({best_row["model"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage1_threshold_curve.png'), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument('--data_dir', default=STAGE1_DATA_DIR, help='Data directory')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING - Gesture Entry Detection")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data_dir}")
    print("=" * 60 + "\n")

    # Load data
    print("[1] Loading sessions...")
    sessions = load_sessions(args.data_dir)
    sessions = [resample_to_rate(s, SAMPLE_RATE) for s in sessions]

    all_results = []
    best_f1 = -1
    best_info = None
    best_state = None
    best_stats = None

    # Grid search
    print("\n[2] Grid search...")
    for win_sec in WINDOW_SECS:
        for step_sec in STEP_SECS:
            print(f"\n--- Window={win_sec}s, Step={step_sec}s ---")

            X, y, sid, t = create_all_windows(sessions, win_sec, step_sec)
            if len(np.unique(y)) < 2:
                print("  Skipped: single class")
                continue

            X_tr, y_tr, X_va, y_va, X_te, y_te = split_by_time(X, y, sid, t)
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print("  Skipped: single class in split")
                continue

            X_tr, X_va, X_te, stats = normalize(X_tr, X_va, X_te)
            shape = X_tr.shape[1:]

            for model_type in MODEL_TYPES:
                print(f"  Training {model_type}...", end=" ", flush=True)
                model = build_model(model_type, shape).to(DEVICE)
                model = train_model(model, X_tr, y_tr, X_va, y_va)
                results = evaluate_model(model, X_te, y_te, THRESHOLDS)

                for r in results:
                    r['model'] = model_type
                    r['window_sec'] = win_sec
                    r['step_sec'] = step_sec
                all_results.extend(results)

                local_best = max(results, key=lambda x: x['f1'])
                print(f"F1={local_best['f1']:.3f}")

                if local_best['f1'] > best_f1:
                    best_f1 = local_best['f1']
                    best_info = {
                        'model': model_type,
                        'window_sec': win_sec,
                        'step_sec': step_sec,
                        'threshold': local_best['threshold'],
                        **{k: local_best[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
                    }
                    best_state = copy.deepcopy(model.state_dict())
                    best_stats = stats
                    best_shape = shape

    # Save results
    print("\n[3] Saving results...")
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(REPORT_DIR, 'stage1_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results: {csv_path}")

    # Generate plots
    plot_results(df, REPORT_DIR)
    print(f"  Plots: {REPORT_DIR}/stage1_*.png")

    # Save best model
    if best_state and best_info:
        ckpt_path = os.path.join(
            MODEL_DIR,
            f"stage1_{best_info['model']}_w{best_info['window_sec']}_s{best_info['step_sec']}.pt"
        )
        torch.save({
            'model_type': best_info['model'],
            'input_shape': best_shape,
            'window_sec': best_info['window_sec'],
            'step_sec': best_info['step_sec'],
            'threshold': best_info['threshold'],
            'target_fs': SAMPLE_RATE,
            'detection_channels': FEATURE_COLS,
            'norm_mean': best_stats['mean'],
            'norm_std': best_stats['std'],
            'metrics': {k: best_info[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'state_dict': best_state
        }, ckpt_path)
        print(f"  Model: {ckpt_path}")

        print("\n" + "=" * 60)
        print("BEST MODEL")
        print("=" * 60)
        print(f"  Type      : {best_info['model']}")
        print(f"  Window    : {best_info['window_sec']}s")
        print(f"  Step      : {best_info['step_sec']}s")
        print(f"  Threshold : {best_info['threshold']}")
        print(f"  Accuracy  : {best_info['accuracy']:.4f}")
        print(f"  Precision : {best_info['precision']:.4f}")
        print(f"  Recall    : {best_info['recall']:.4f}")
        print(f"  F1        : {best_info['f1']:.4f}")
        print(f"  AUC       : {best_info['auc']:.4f}")

    # Top 10
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)
    top10 = df.nlargest(10, 'f1')[['model', 'window_sec', 'step_sec', 'threshold', 'f1', 'auc']]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
