"""
Real-time Two-Stage Gesture Recognition
Stage 1: Entry detection -> Stage 2: Classification
"""

import argparse
import socket
import struct
import time
from collections import deque
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMU_CHANNELS, MSG_SIZE, UDP_PORT, FEATURE_COLS, GESTURE_NAMES

FEATURE_IDX = [IMU_CHANNELS[c] for c in FEATURE_COLS]


# =============================================================================
# UDP Receiver
# =============================================================================

class UDPReceiver:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = None
        self.running = False

    def start(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.ip, self.port))
            self.sock.settimeout(0.1)
            self.running = True
            print(f"[OK] Listening on {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[ERR] Bind failed: {e}")
            return False

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("[OK] Receiver stopped")

    def recv(self, timeout=0.1) -> Optional[Tuple[float, np.ndarray]]:
        if not self.running:
            return None
        self.sock.settimeout(timeout)
        try:
            data, _ = self.sock.recvfrom(MSG_SIZE)
            if len(data) != MSG_SIZE:
                return None
            values = struct.unpack('!' + 'f' * 55, data)
            return time.time(), np.array(values, dtype=np.float32)
        except socket.timeout:
            return None
        except Exception as e:
            print(f"[ERR] Receive: {e}")
            return None


# =============================================================================
# Ring Buffer
# =============================================================================

class RingBuffer:
    def __init__(self, maxlen: int):
        self.times = deque(maxlen=maxlen)
        self.frames = deque(maxlen=maxlen)

    def add(self, ts: float, data: np.ndarray):
        self.times.append(ts)
        self.frames.append(data.astype(np.float32))

    def __len__(self):
        return len(self.frames)

    def get_range(self, t_start: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
        times, frames = [], []
        for ts, fr in zip(self.times, self.frames):
            if t_start <= ts <= t_end:
                times.append(ts)
                frames.append(fr)
        if not times:
            return np.zeros((0,)), np.zeros((0, 55))
        return np.array(times), np.stack(frames)

    def clear(self):
        self.times.clear()
        self.frames.clear()


# =============================================================================
# Models (Stage 1 - Binary)
# =============================================================================

class S1_MLP(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * D, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class S1_LSTM(nn.Module):
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


class S1_GRU(nn.Module):
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


class S1_TCNBlock(nn.Module):
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


class S1_TCN(nn.Module):
    def __init__(self, shape):
        super().__init__()
        T, D = shape
        self.tcn = nn.Sequential(
            S1_TCNBlock(D, 64, d=1), S1_TCNBlock(64, 64, d=2), S1_TCNBlock(64, 64, d=4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.pool(self.tcn(x)).squeeze(-1)).squeeze(1)


class S1_CNN1D(nn.Module):
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
        return self.fc(self.pool(self.cnn(x)).squeeze(-1)).squeeze(1)


class S1_CNNLSTM(nn.Module):
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
        return self.fc(torch.cat([h[0], h[1]], dim=1)).squeeze(1)


# =============================================================================
# Models (Stage 2 - Multi-class)
# =============================================================================

class S2_MLP(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * D, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_cls)
        )

    def forward(self, x):
        return self.net(x)


class S2_LSTM(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.rnn = nn.LSTM(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, n_cls)
        )

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1])


class S2_GRU(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.rnn = nn.GRU(D, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, n_cls)
        )

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


class S2_TCNBlock(nn.Module):
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


class S2_TCN(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.tcn = nn.Sequential(
            S2_TCNBlock(D, 64, d=1), S2_TCNBlock(64, 64, d=2), S2_TCNBlock(64, 64, d=4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.pool(self.tcn(x)).squeeze(-1))


class S2_CNN1D(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.pool(self.cnn(x)).squeeze(-1))


class S2_CNNLSTM(nn.Module):
    def __init__(self, shape, n_cls):
        super().__init__()
        T, D = shape
        self.cnn = nn.Sequential(
            nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, n_cls)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat([h[0], h[1]], dim=1))


def build_s1(name, shape):
    models = {'mlp': S1_MLP, 'lstm': S1_LSTM, 'gru': S1_GRU,
              'tcn': S1_TCN, 'cnn1d': S1_CNN1D, 'cnn_lstm': S1_CNNLSTM}
    return models[name](shape)


def build_s2(name, shape, n_cls):
    models = {'mlp': S2_MLP, 'lstm': S2_LSTM, 'gru': S2_GRU,
              'tcn': S2_TCN, 'cnn1d': S2_CNN1D, 'cnn_lstm': S2_CNNLSTM}
    return models[name](shape, n_cls)


# =============================================================================
# Detector Wrappers
# =============================================================================

class Stage1Detector:
    def __init__(self, ckpt_path: str, buffer: RingBuffer, device):
        self.buffer = buffer
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)
        self.model_type = ckpt['model_type']
        self.shape = tuple(ckpt['input_shape'])
        self.win_sec = float(ckpt.get('window_sec', 1.0))
        self.step_sec = float(ckpt.get('step_sec', 0.5))
        self.threshold = float(ckpt.get('threshold', 0.5))
        self.fs = float(ckpt.get('target_fs', 50.0))

        self.mean = np.array(ckpt['norm_mean'], dtype=np.float32)
        self.std = np.array(ckpt['norm_std'], dtype=np.float32)

        self.model = build_s1(self.model_type, self.shape).to(device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.last_time = None

        print(f"[S1] Loaded: {self.model_type}, win={self.win_sec}s, thr={self.threshold}")

    def _preprocess(self, X):
        X = np.clip(X.astype(np.float64), -1e4, 1e4)
        std = np.where(self.std < 1e-6, 1.0, self.std)
        return ((X - self.mean) / std).astype(np.float32)

    def detect(self, now: float) -> Tuple[bool, Optional[float]]:
        win_len = self.shape[0]
        if len(self.buffer) < win_len:
            return False, None

        if self.last_time and (now - self.last_time) < self.step_sec:
            return False, None

        t_end = now
        t_start = t_end - self.win_sec
        times, frames = self.buffer.get_range(t_start, t_end)

        if len(times) < 2:
            return False, None

        # Extract features & resample
        imu = frames[:, FEATURE_IDX]
        dt = 1.0 / self.fs
        t_grid = t_start + np.arange(win_len) * dt
        X = np.zeros((win_len, len(FEATURE_IDX)), dtype=np.float32)
        for ch in range(len(FEATURE_IDX)):
            X[:, ch] = np.interp(t_grid, times, imu[:, ch])

        X = self._preprocess(X)
        x_t = torch.from_numpy(X[None, ...]).to(self.device)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x_t)).item()

        self.last_time = now
        return prob >= self.threshold, prob


class Stage2Classifier:
    def __init__(self, ckpt_path: str, buffer: RingBuffer, device):
        self.buffer = buffer
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)
        self.model_type = ckpt['model_type']
        self.shape = tuple(ckpt['input_shape'])
        self.seq_len = int(ckpt.get('seq_len', self.shape[0]))
        self.n_cls = int(ckpt['num_classes'])
        self.names = ckpt.get('class_id_to_name', GESTURE_NAMES)
        self.fs = float(ckpt.get('target_fs', 50.0))

        self.mean = np.array(ckpt['norm_mean'], dtype=np.float32)
        self.std = np.array(ckpt['norm_std'], dtype=np.float32)

        self.model = build_s2(self.model_type, self.shape, self.n_cls).to(device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        print(f"[S2] Loaded: {self.model_type}, seq={self.seq_len}, classes={self.n_cls}")

    def _preprocess(self, X):
        X = np.clip(X.astype(np.float64), -1e4, 1e4)
        std = np.where(self.std < 1e-6, 1.0, self.std)
        return ((X - self.mean) / std).astype(np.float32)

    def _crop_pad(self, seq, target_len):
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

    def classify(self, t_start: float, t_end: float, step_sec=0.5
                 ) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        times, frames = self.buffer.get_range(t_start, t_end)
        if len(times) < 2:
            return None, None, None

        imu = frames[:, FEATURE_IDX]

        # Resample
        dt = 1.0 / self.fs
        duration = max(t_end - t_start, 0.0)
        res_len = int(round(duration * self.fs))
        if res_len < 2:
            return None, None, None

        t_grid = t_start + np.arange(res_len) * dt
        imu_res = np.zeros((res_len, len(FEATURE_IDX)), dtype=np.float32)
        for ch in range(len(FEATURE_IDX)):
            imu_res[:, ch] = np.interp(t_grid, times, imu[:, ch])

        win_len = self.seq_len
        N = imu_res.shape[0]

        if N < win_len:
            seg = self._crop_pad(imu_res, win_len)
            seg = self._preprocess(seg)
            x_t = torch.from_numpy(seg[None, ...]).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(x_t), dim=1).cpu().numpy()[0]
            best_id = int(np.argmax(probs))
            return best_id, self.names.get(best_id, f'{best_id}'), float(probs[best_id])

        # Sliding window
        step_frames = max(1, int(round(step_sec * self.fs)))
        best_prob, best_id = -1.0, None

        for start in range(0, N - win_len + 1, step_frames):
            seg = imu_res[start:start + win_len]
            seg = self._preprocess(seg)
            x_t = torch.from_numpy(seg[None, ...]).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(x_t), dim=1).cpu().numpy()[0]
            p_max = float(probs.max())
            if p_max > best_prob:
                best_prob = p_max
                best_id = int(np.argmax(probs))

        if best_id is None:
            return None, None, None
        return best_id, self.names.get(best_id, f'{best_id}'), best_prob


# =============================================================================
# Main Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-time Gesture Recognition")
    parser.add_argument('--ip', default='0.0.0.0', help='UDP bind IP')
    parser.add_argument('--port', type=int, default=UDP_PORT, help='UDP port')
    parser.add_argument('--stage1', required=True, help='Stage 1 checkpoint')
    parser.add_argument('--stage2', required=True, help='Stage 2 checkpoint')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Detection cooldown (sec)')
    parser.add_argument('--collect_sec', type=float, default=2.5, help='Stage 2 collection time')
    parser.add_argument('--step_sec', type=float, default=0.5, help='Stage 2 window step')
    parser.add_argument('--device', default='auto', help='cpu/cuda/auto')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Buffer size
    tmp1 = torch.load(args.stage1, map_location='cpu')
    tmp2 = torch.load(args.stage2, map_location='cpu')
    fs = float(tmp1.get('target_fs', 50.0))
    T1 = int(tmp1['input_shape'][0])
    T2 = int(tmp2.get('seq_len', tmp2['input_shape'][0]))
    buf_sec = max(T1, T2) / fs * 4.0
    buf_size = int(buf_sec * fs)

    buffer = RingBuffer(buf_size)
    s1 = Stage1Detector(args.stage1, buffer, device)
    s2 = Stage2Classifier(args.stage2, buffer, device)

    receiver = UDPReceiver(args.ip, args.port)
    if not receiver.start():
        return

    print("\n" + "=" * 50)
    print("REAL-TIME INFERENCE")
    print("=" * 50)
    print("  Stage 1: Entry detection")
    print(f"  Stage 2: {args.collect_sec}s collection, step={args.step_sec}s")
    print("  Ctrl+C to stop")
    print("=" * 50 + "\n")

    last_s1_time = -1e9
    s2_pending = False
    s2_start = None

    try:
        while True:
            pkt = receiver.recv(0.1)

            # Check S2 timeout
            now = time.time()
            if s2_pending and now >= s2_start + args.collect_sec:
                gid, name, conf = s2.classify(s2_start, s2_start + args.collect_sec, args.step_sec)
                if gid is not None:
                    print(f"[GESTURE] {gid}: {name} (conf={conf:.3f})")
                else:
                    print("[GESTURE] No valid prediction")
                s2_pending = False
                buffer.clear()
                s1.last_time = None
                last_s1_time = time.time()
                continue

            if pkt is None:
                continue

            ts, values = pkt
            buffer.add(ts, values)

            # S2 timeout check (packet-triggered)
            if s2_pending and ts >= s2_start + args.collect_sec:
                gid, name, conf = s2.classify(s2_start, s2_start + args.collect_sec, args.step_sec)
                if gid is not None:
                    print(f"[GESTURE] {gid}: {name} (conf={conf:.3f})")
                else:
                    print("[GESTURE] No valid prediction")
                s2_pending = False
                buffer.clear()
                s1.last_time = None
                last_s1_time = time.time()
                continue

            # S1 detection
            if not s2_pending:
                detected, prob = s1.detect(ts)
                if not detected:
                    continue
                if ts - last_s1_time < args.cooldown:
                    continue

                buffer.clear()
                s1.last_time = None
                last_s1_time = ts
                s2_pending = True
                s2_start = ts

                print(f"[ENTRY] Detected (prob={prob:.3f})")
                print(f"        Perform gesture in {args.collect_sec}s...")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
