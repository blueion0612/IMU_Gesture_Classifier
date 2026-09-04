"""
Stage 2 Training - Gesture Classification (15-class)
Grid search over model architectures.
"""

import os
import glob
import copy
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    STAGE2_DATA_DIR, MODEL_DIR, REPORT_DIR, IMU_CHANNELS,
    FEATURE_COLS, GESTURE_NAMES, NUM_GESTURES, SAMPLE_RATE, SEQ_LEN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, EARLY_STOP, SEED
)

MODEL_TYPES = ["mlp", "lstm", "gru", "tcn", "cnn1d", "cnn_lstm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================

def center_crop_pad(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Center crop or pad sequence to target length."""
    L, D = seq.shape
    if L == target_len:
        return seq.astype(np.float32)
    if L > target_len:
        start = (L - target_len) // 2
        return seq[start:start + target_len].astype(np.float32)
    out = np.zeros((target_len, D), dtype=np.float32)
    start = (target_len - L) // 2
    out[start:start + L] = seq.astype(np.float32)
    return out


def extract_gesture_window(df: pd.DataFrame, seq_len: int, target_fs: float) -> np.ndarray:
    """Extract gesture-centered window from fragment."""
    if 'relative_time' in df.columns:
        times = df['relative_time'].values.astype(np.float64)
    else:
        t_raw = df['timestamp'].values.astype(np.float64)
        times = t_raw - t_raw[0]

    imu = df[FEATURE_COLS].values.astype(np.float32)
    L, D = imu.shape
    if L == 0:
        raise ValueError("Empty fragment")

    # Find gesture center
    if 'is_gesture' in df.columns:
        mask = df['is_gesture'].values > 0.5
        if mask.any():
            t_g = times[mask]
            center_t = 0.5 * (t_g.min() + t_g.max())
        else:
            center_t = 0.5 * (times[0] + times[-1])
    else:
        center_t = 0.5 * (times[0] + times[-1])

    # Define window
    win_sec = seq_len / target_fs
    half = win_sec / 2.0
    t_start = center_t - half
    t_end = center_t + half

    # Clamp to fragment bounds
    if t_start < times[0]:
        shift = times[0] - t_start
        t_start += shift
        t_end += shift
    if t_end > times[-1]:
        shift = t_end - times[-1]
        t_start -= shift
        t_end -= shift

    if t_start < times[0] or t_end > times[-1]:
        return center_crop_pad(imu, seq_len)

    # Resample
    new_times = np.linspace(t_start, t_end, seq_len, dtype=np.float64)
    new_seq = np.zeros((seq_len, D), dtype=np.float32)
    for d in range(D):
        new_seq[:, d] = np.interp(new_times, times, imu[:, d])
    return new_seq


def load_fragments(data_dir: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all gesture fragments."""
    pattern = os.path.join(data_dir, "session_*", "gesture_*.csv")
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if not p.endswith("_backup.csv")]

    if not paths:
        raise FileNotFoundError(f"No fragments in {data_dir}")

    print(f"  Found {len(paths)} fragments")

    X_list, y_list, sid_list = [], [], []

    for path in paths:
        df = pd.read_csv(path)

        # Check required columns
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing or 'label' not in df.columns:
            continue

        try:
            seq = extract_gesture_window(df, seq_len, SAMPLE_RATE)
            label = int(df['label'].iloc[0])
            sid = os.path.basename(os.path.dirname(path))
            X_list.append(seq)
            y_list.append(label)
            sid_list.append(sid)
        except Exception as e:
            print(f"  Skip {path}: {e}")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    sid = np.array(sid_list)

    print(f"  Loaded: {X.shape}")
    print("  Distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        print(f"    {label:2d} ({GESTURE_NAMES.get(label, '?'):15s}): {count}")

    return X, y, sid


# =============================================================================
# Split & Normalize
# =============================================================================

def stratified_split(X, y, seed=SEED):
    """Stratified train/val/test split."""
    idx = np.arange(len(y))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=(1 - TRAIN_RATIO), stratify=y, random_state=seed
    )

    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=(1 - val_frac), stratify=y_temp, random_state=seed + 1
    )

    print(f"\n  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return (X[train_idx], y_train, X[val_idx], y_val, X[test_idx], y_test)


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
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * D, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class LSTM(nn.Module):
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.rnn = nn.LSTM(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1])


class GRU(nn.Module):
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.rnn = nn.GRU(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


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
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.tcn = nn.Sequential(
            TCNBlock(D, 64, d=1), TCNBlock(64, 64, d=2), TCNBlock(64, 64, d=4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.pool(self.tcn(x)).squeeze(-1)
        return self.fc(out)


class CNN1D(nn.Module):
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.pool(self.cnn(x)).squeeze(-1)
        return self.fc(out)


class CNNLSTM(nn.Module):
    def __init__(self, shape, num_classes):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


def build_model(name: str, shape, num_classes):
    models = {
        'mlp': MLP, 'lstm': LSTM, 'gru': GRU,
        'tcn': TCN, 'cnn1d': CNN1D, 'cnn_lstm': CNNLSTM
    }
    return models[name](shape, num_classes)


# =============================================================================
# Training
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping."""
    X_tr = torch.from_numpy(X_train).to(DEVICE)
    y_tr = torch.from_numpy(y_train).to(DEVICE)
    X_va = torch.from_numpy(X_val).to(DEVICE)
    y_va = torch.from_numpy(y_val).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

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


def evaluate_model(model, X, y):
    """Evaluate model and return metrics."""
    model.eval()
    X_t = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    pred = logits.argmax(axis=1)

    acc = accuracy_score(y, pred)
    f1_macro = f1_score(y, pred, average='macro')

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'y_true': y,
        'y_pred': pred
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot and save confusion matrix."""
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')

    names = [GESTURE_NAMES.get(int(i), f'{i}') for i in labels]
    plt.xticks(np.arange(len(labels)) + 0.5, names, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)) + 0.5, names, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_results(df, save_dir):
    """Generate performance visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Bar chart: Accuracy & F1 by model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    bars = axes[0].bar(df['model'], df['test_acc'], color=colors)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Stage 2: Test Accuracy by Model')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, df['test_acc']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=9)

    bars = axes[1].bar(df['model'], df['test_f1'], color=colors)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Stage 2: Test Macro F1 by Model')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars, df['test_f1']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage2_metrics_by_model.png'), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Training")
    parser.add_argument('--data_dir', default=STAGE2_DATA_DIR, help='Data directory')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN, help='Sequence length')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print("STAGE 2 TRAINING - Gesture Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data_dir}")
    print(f"Seq len: {args.seq_len}")
    print("=" * 60 + "\n")

    # Load data
    print("[1] Loading fragments...")
    X, y, sid = load_fragments(args.data_dir, args.seq_len)
    num_classes = len(np.unique(y))

    # Split
    print("\n[2] Splitting data...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = stratified_split(X, y, args.seed)

    # Normalize
    print("\n[3] Normalizing...")
    X_tr, X_va, X_te, stats = normalize(X_tr, X_va, X_te)
    shape = X_tr.shape[1:]

    # Train models
    print("\n[4] Training models...")
    all_results = []
    best_acc = -1
    best_info = None
    best_state = None

    for model_type in MODEL_TYPES:
        print(f"\n  {model_type}...", end=" ", flush=True)
        model = build_model(model_type, shape, num_classes).to(DEVICE)
        model = train_model(model, X_tr, y_tr, X_va, y_va)

        val_metrics = evaluate_model(model, X_va, y_va)
        test_metrics = evaluate_model(model, X_te, y_te)

        print(f"val_acc={val_metrics['accuracy']:.3f}, test_acc={test_metrics['accuracy']:.3f}")

        result = {
            'model': model_type,
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_macro'],
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_macro']
        }
        all_results.append(result)

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_info = {
                'model': model_type,
                **result,
                'y_true': test_metrics['y_true'],
                'y_pred': test_metrics['y_pred']
            }
            best_state = copy.deepcopy(model.state_dict())

    # Save results
    print("\n[5] Saving results...")
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(REPORT_DIR, 'stage2_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results: {csv_path}")

    # Generate plots
    plot_results(df, REPORT_DIR)
    print(f"  Plots: {REPORT_DIR}/stage2_*.png")

    # Confusion matrix for best model
    if best_info:
        cm_path = os.path.join(REPORT_DIR, f"stage2_confusion_{best_info['model']}.png")
        plot_confusion_matrix(
            best_info['y_true'], best_info['y_pred'],
            f"Stage 2: {best_info['model']} Confusion Matrix",
            cm_path
        )
        print(f"  Confusion: {cm_path}")

    # Save best model
    if best_state and best_info:
        ckpt_path = os.path.join(MODEL_DIR, f"stage2_{best_info['model']}_seq{args.seq_len}.pt")
        torch.save({
            'model_type': best_info['model'],
            'input_shape': shape,
            'seq_len': args.seq_len,
            'num_classes': num_classes,
            'class_id_to_name': GESTURE_NAMES,
            'target_fs': SAMPLE_RATE,
            'detection_channels': FEATURE_COLS,
            'norm_mean': stats['mean'],
            'norm_std': stats['std'],
            'metrics': {
                'val_acc': best_info['val_acc'],
                'val_f1': best_info['val_f1'],
                'test_acc': best_info['test_acc'],
                'test_f1': best_info['test_f1']
            },
            'state_dict': best_state
        }, ckpt_path)
        print(f"  Model: {ckpt_path}")

        print("\n" + "=" * 60)
        print("BEST MODEL")
        print("=" * 60)
        print(f"  Type        : {best_info['model']}")
        print(f"  Val Acc     : {best_info['val_acc']:.4f}")
        print(f"  Val F1      : {best_info['val_f1']:.4f}")
        print(f"  Test Acc    : {best_info['test_acc']:.4f}")
        print(f"  Test F1     : {best_info['test_f1']:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("ALL MODELS")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
