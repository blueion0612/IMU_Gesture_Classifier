"""
IMU Data Collector
- Mode 1: Continuous recording for gesture entry detection (Stage 1)
- Mode 2: Fragment recording for gesture classification (Stage 2)
"""

import os
import csv
import json
import time
import socket
import struct
import pickle
import argparse
import threading
import queue
from datetime import datetime
from collections import deque, defaultdict

from .config import (
    UDP_PORT, MSG_SIZE, IMU_CHANNELS, GESTURE_NAMES,
    STAGE1_DATA_DIR, STAGE2_DATA_DIR, SAMPLE_RATE
)


class UDPReceiver:
    """Receives IMU data via UDP socket."""

    def __init__(self, ip, port=UDP_PORT):
        self.ip = ip
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        self.queue = queue.Queue()
        self.buffer = deque(maxlen=150)  # 3 sec @ 50Hz

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.ip, self.port))
            self.sock.settimeout(0.1)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            print(f"[OK] Listening on {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[ERR] Failed to bind {self.ip}:{self.port} - {e}")
            return False

    def _loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(MSG_SIZE)
                if len(data) == MSG_SIZE:
                    values = struct.unpack('!' + 'f' * 55, data)
                    ts = time.time()
                    self.queue.put((ts, values))
                    self.buffer.append((ts, values))
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[ERR] Receive error: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.sock:
            self.sock.close()
        print("[OK] Receiver stopped")

    def get(self, timeout=0.02):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def snapshot(self):
        return list(self.buffer)

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class ContinuousRecorder:
    """Records continuous IMU data for Stage 1 training."""

    def __init__(self, output_dir=STAGE1_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.samples = []
        self.markers = []
        self.recording = False

    def record(self, receiver, duration=60):
        print("\n" + "=" * 50)
        print("CONTINUOUS RECORDING (Stage 1)")
        print("=" * 50)
        print(f"Duration: {duration} sec")
        print("\nControls:")
        print("  Enter  : Mark gesture entry")
        print("  q+Enter: Stop recording")
        print("-" * 50 + "\n")

        self.recording = True
        t0 = time.time()

        def rec_loop():
            while self.recording:
                pkt = receiver.get()
                if pkt:
                    ts, vals = pkt
                    self.samples.append({
                        'ts': ts,
                        't': ts - t0,
                        'data': vals,
                        'label': 0
                    })

        thread = threading.Thread(target=rec_loop, daemon=True)
        thread.start()

        try:
            while self.recording:
                elapsed = time.time() - t0
                if elapsed >= duration:
                    print(f"\nTime limit ({duration}s) reached.")
                    break

                cmd = input().strip()
                if cmd.lower() == 'q':
                    print("\nStopping...")
                    break
                elif cmd == '':
                    print("Prepare... ", end="", flush=True)
                    for i in [3, 2, 1]:
                        print(f"{i}...", end="", flush=True)
                        time.sleep(0.33)

                    start_idx = len(self.samples)
                    start_time = time.time()
                    print(" GO!")
                    time.sleep(1.0)
                    end_idx = len(self.samples)

                    for i in range(start_idx, min(end_idx, len(self.samples))):
                        if i < len(self.samples):
                            self.samples[i]['label'] = 1

                    self.markers.append({
                        'time': start_time - t0,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
                    print(f"Marked! (samples {start_idx}-{end_idx})")

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.recording = False
            thread.join(timeout=1)

        self._save()

    def _save(self):
        if not self.samples:
            print("No data to save")
            return

        sess_dir = os.path.join(self.output_dir, f"session_{self.session_id}")
        os.makedirs(sess_dir, exist_ok=True)

        # Save pickle
        pkl_path = os.path.join(sess_dir, "data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'session_id': self.session_id,
                'data': self.samples,
                'markers': self.markers,
                'fields': list(IMU_CHANNELS.keys())
            }, f)

        # Save CSV
        csv_path = os.path.join(sess_dir, "data.csv")
        cols = ['timestamp', 'relative_time', 'label'] + list(IMU_CHANNELS.keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for s in self.samples:
                row = {'timestamp': s['ts'], 'relative_time': s['t'], 'label': s['label']}
                for name, idx in IMU_CHANNELS.items():
                    row[name] = s['data'][idx]
                writer.writerow(row)

        # Save markers
        json_path = os.path.join(sess_dir, "markers.json")
        with open(json_path, 'w') as f:
            json.dump(self.markers, f, indent=2)

        n_pos = sum(1 for s in self.samples if s['label'] == 1)
        n_neg = len(self.samples) - n_pos
        dur = self.samples[-1]['t'] if self.samples else 0

        print("\n" + "=" * 50)
        print("SAVED")
        print("=" * 50)
        print(f"  Directory : {sess_dir}")
        print(f"  Samples   : {len(self.samples)}")
        print(f"  Duration  : {dur:.1f} sec")
        print(f"  Positive  : {n_pos}")
        print(f"  Negative  : {n_neg}")
        print(f"  Markers   : {len(self.markers)}")


class FragmentRecorder:
    """Records gesture fragments for Stage 2 training."""

    def __init__(self, output_dir=STAGE2_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sess_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.sess_dir, exist_ok=True)
        self.counts = defaultdict(int)
        self.pre_sec = 0.5
        self.gesture_sec = 1.5
        self.post_sec = 1.0
        self.total_sec = self.pre_sec + self.gesture_sec + self.post_sec

    def collect(self, receiver):
        print("\n" + "=" * 50)
        print("FRAGMENT COLLECTION (Stage 2)")
        print("=" * 50)
        print(f"Format: {self.pre_sec}s + {self.gesture_sec}s + {self.post_sec}s = {self.total_sec}s")
        print("\nControls:")
        print("  0-14   : Gesture ID")
        print("  q      : Quit")
        print("-" * 50)

        rec_buf = deque(maxlen=200)
        lock = threading.Lock()
        running = threading.Event()
        running.set()

        def buf_loop():
            while running.is_set():
                pkt = receiver.get()
                if pkt:
                    with lock:
                        rec_buf.append(pkt)

        thread = threading.Thread(target=buf_loop, daemon=True)
        thread.start()

        try:
            while True:
                self._show_status()
                cmd = input("\nGesture ID (0-14) or 'q': ").strip()

                if cmd.lower() == 'q':
                    break

                try:
                    gid = int(cmd)
                    if gid not in GESTURE_NAMES:
                        print("Invalid ID. Use 0-14.")
                        continue
                except ValueError:
                    print("Invalid input.")
                    continue

                name = GESTURE_NAMES[gid]
                trial = self.counts[gid] + 1
                print(f"\nRecording: {name} (ID:{gid}, Trial:{trial})")

                time.sleep(0.5)
                print("Prepare... ", end="", flush=True)
                for i in [3, 2, 1]:
                    print(f"{i}...", end="", flush=True)
                    time.sleep(0.33)

                print(" GO!")
                time.sleep(self.gesture_sec)
                print("Hold...")
                time.sleep(self.post_sec)

                with lock:
                    buf_snap = list(rec_buf)

                t_end = time.time()
                t_start = t_end - self.total_sec

                samples = []
                for ts, vals in buf_snap:
                    if t_start <= ts <= t_end:
                        rel_t = ts - t_start
                        in_gesture = self.pre_sec <= rel_t < (self.pre_sec + self.gesture_sec)
                        samples.append({
                            'ts': ts,
                            't': rel_t,
                            'data': vals,
                            'label': gid,
                            'is_gesture': 1 if in_gesture else 0
                        })

                print(f"Captured {len(samples)} samples")
                self._save_fragment(samples, gid, name, trial)
                self.counts[gid] += 1

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            running.clear()
            thread.join(timeout=1)
            self._save_summary()

    def _show_status(self):
        print("\n" + "-" * 50)
        print("STATUS")
        print("-" * 50)
        for row in range(5):
            line = ""
            for col in range(3):
                idx = row + col * 5
                if idx < len(GESTURE_NAMES):
                    cnt = self.counts[idx]
                    mark = "*" if cnt > 0 else " "
                    line += f"{mark}{idx:2d}: {GESTURE_NAMES[idx]:15s}[{cnt:2d}]  "
            print(line)
        total = sum(self.counts.values())
        unique = len([k for k, v in self.counts.items() if v > 0])
        print("-" * 50)
        print(f"Total: {total} | Unique: {unique}/15")

    def _save_fragment(self, samples, gid, name, trial):
        if not samples:
            return

        fname = f"gesture_{gid:02d}_{name}_trial_{trial:03d}"

        # CSV
        csv_path = os.path.join(self.sess_dir, f"{fname}.csv")
        cols = ['timestamp', 'relative_time', 'label', 'is_gesture'] + list(IMU_CHANNELS.keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for s in samples:
                row = {
                    'timestamp': s['ts'],
                    'relative_time': s['t'],
                    'label': s['label'],
                    'is_gesture': s['is_gesture']
                }
                for name, idx in IMU_CHANNELS.items():
                    row[name] = s['data'][idx]
                writer.writerow(row)

        # Pickle
        pkl_path = os.path.join(self.sess_dir, f"{fname}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'gesture_id': gid,
                'gesture_name': name,
                'trial': trial,
                'samples': samples,
                'duration': self.total_sec,
                'pre_sec': self.pre_sec,
                'gesture_sec': self.gesture_sec,
                'post_sec': self.post_sec
            }, f)

    def _save_summary(self):
        summary = {
            'session_id': self.session_id,
            'total': sum(self.counts.values()),
            'distribution': dict(self.counts),
            'unique': len([k for k, v in self.counts.items() if v > 0]),
            'params': {
                'pre_sec': self.pre_sec,
                'gesture_sec': self.gesture_sec,
                'post_sec': self.post_sec,
                'total_sec': self.total_sec
            }
        }

        json_path = os.path.join(self.sess_dir, "summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 50)
        print("SESSION COMPLETE")
        print("=" * 50)
        print(f"  Directory   : {self.sess_dir}")
        print(f"  Total       : {summary['total']} fragments")
        print(f"  Unique      : {summary['unique']}/15 gestures")
        if summary['distribution']:
            print("\nDistribution:")
            for gid, cnt in sorted(summary['distribution'].items()):
                print(f"  {gid:2d}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="IMU Data Collector")
    parser.add_argument('mode', choices=['continuous', 'fragment'],
                        help='Recording mode')
    parser.add_argument('ip', help='IP address for UDP')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration for continuous mode (sec)')
    parser.add_argument('--port', type=int, default=UDP_PORT,
                        help=f'UDP port (default: {UDP_PORT})')
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("IMU DATA COLLECTOR")
    print("=" * 50)
    print(f"  Mode : {args.mode}")
    print(f"  IP   : {args.ip}")
    print(f"  Port : {args.port}")
    print("=" * 50)

    receiver = UDPReceiver(args.ip, args.port)

    try:
        if receiver.start():
            print("Starting in 2 seconds...\n")
            time.sleep(2)

            if args.mode == 'continuous':
                rec = ContinuousRecorder()
                rec.record(receiver, duration=args.duration)
            else:
                rec = FragmentRecorder()
                rec.collect(receiver)
        else:
            print("\nFailed to start receiver")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        receiver.stop()
        print("\nDone")


if __name__ == "__main__":
    main()
