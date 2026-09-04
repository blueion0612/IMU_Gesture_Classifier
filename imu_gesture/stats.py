"""
Data Statistics Generator
Generates statistics report for Stage 1 and Stage 2 training data.
"""

import os
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    STAGE1_DATA_DIR, STAGE2_DATA_DIR, REPORT_DIR,
    IMU_CHANNELS, FEATURE_COLS, GESTURE_NAMES, SAMPLE_RATE
)


def analyze_stage1(data_dir: str) -> dict:
    """Analyze Stage 1 (continuous detection) data."""
    dirs = sorted(glob.glob(os.path.join(data_dir, "session_*")))

    stats = {
        'total_sessions': 0,
        'total_samples': 0,
        'total_seconds': 0.0,
        'total_minutes': 0.0,
        'positive_samples': 0,
        'negative_samples': 0,
        'sessions': []
    }

    for d in dirs:
        csv_path = os.path.join(d, "data.csv")
        if not os.path.exists(csv_path):
            # Try old format
            csv_path = os.path.join(d, "continuous_detection.csv")
            if not os.path.exists(csv_path):
                continue

        df = pd.read_csv(csv_path)
        if 'label' not in df.columns:
            continue

        sid = os.path.basename(d).replace("session_", "")
        n_samples = len(df)

        # Duration from relative_time
        if 'relative_time' in df.columns:
            duration = df['relative_time'].max()
        else:
            duration = n_samples / SAMPLE_RATE

        n_pos = int((df['label'] == 1).sum())
        n_neg = n_samples - n_pos

        stats['total_sessions'] += 1
        stats['total_samples'] += n_samples
        stats['total_seconds'] += duration
        stats['positive_samples'] += n_pos
        stats['negative_samples'] += n_neg

        stats['sessions'].append({
            'session_id': sid,
            'samples': n_samples,
            'duration_sec': round(duration, 1),
            'positive': n_pos,
            'negative': n_neg
        })

    stats['total_minutes'] = round(stats['total_seconds'] / 60, 2)
    stats['total_seconds'] = round(stats['total_seconds'], 1)

    return stats


def analyze_stage2(data_dir: str) -> dict:
    """Analyze Stage 2 (fragment classification) data."""
    pattern = os.path.join(data_dir, "session_*", "gesture_*.csv")
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if not p.endswith("_backup.csv")]

    stats = {
        'total_sessions': 0,
        'total_fragments': 0,
        'total_samples': 0,
        'total_seconds': 0.0,
        'total_minutes': 0.0,
        'label_distribution': {},
        'sessions': {}
    }

    for path in paths:
        df = pd.read_csv(path)
        if 'label' not in df.columns:
            continue

        label = int(df['label'].iloc[0])
        sid = os.path.basename(os.path.dirname(path))
        n_samples = len(df)

        # Duration from relative_time
        if 'relative_time' in df.columns:
            duration = df['relative_time'].max()
        else:
            duration = n_samples / SAMPLE_RATE

        stats['total_fragments'] += 1
        stats['total_samples'] += n_samples
        stats['total_seconds'] += duration

        # Label distribution
        if label not in stats['label_distribution']:
            stats['label_distribution'][label] = 0
        stats['label_distribution'][label] += 1

        # Session tracking
        if sid not in stats['sessions']:
            stats['sessions'][sid] = {'fragments': 0, 'labels': {}}
            stats['total_sessions'] += 1
        stats['sessions'][sid]['fragments'] += 1
        if label not in stats['sessions'][sid]['labels']:
            stats['sessions'][sid]['labels'][label] = 0
        stats['sessions'][sid]['labels'][label] += 1

    stats['total_minutes'] = round(stats['total_seconds'] / 60, 2)
    stats['total_seconds'] = round(stats['total_seconds'], 1)

    return stats


def generate_report(s1_stats: dict, s2_stats: dict, output_dir: str):
    """Generate statistics report as CSV and plots."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # Stage 1 Report
    # =========================================================================
    s1_summary = {
        'metric': [
            'Total Sessions',
            'Total Samples',
            'Total Duration (min)',
            'Positive Samples',
            'Negative Samples',
            'Positive Ratio (%)'
        ],
        'value': [
            s1_stats['total_sessions'],
            s1_stats['total_samples'],
            s1_stats['total_minutes'],
            s1_stats['positive_samples'],
            s1_stats['negative_samples'],
            round(s1_stats['positive_samples'] / max(s1_stats['total_samples'], 1) * 100, 2)
        ]
    }
    df_s1 = pd.DataFrame(s1_summary)
    df_s1.to_csv(os.path.join(output_dir, 'data_stats_stage1.csv'), index=False)

    # Session details
    if s1_stats['sessions']:
        df_s1_sess = pd.DataFrame(s1_stats['sessions'])
        df_s1_sess.to_csv(os.path.join(output_dir, 'data_stats_stage1_sessions.csv'), index=False)

    # =========================================================================
    # Stage 2 Report
    # =========================================================================
    s2_summary = {
        'metric': [
            'Total Sessions',
            'Total Fragments',
            'Total Samples',
            'Total Duration (min)',
            'Unique Labels'
        ],
        'value': [
            s2_stats['total_sessions'],
            s2_stats['total_fragments'],
            s2_stats['total_samples'],
            s2_stats['total_minutes'],
            len(s2_stats['label_distribution'])
        ]
    }
    df_s2 = pd.DataFrame(s2_summary)
    df_s2.to_csv(os.path.join(output_dir, 'data_stats_stage2.csv'), index=False)

    # Label distribution
    label_data = []
    for label, count in sorted(s2_stats['label_distribution'].items()):
        name = GESTURE_NAMES.get(label, f'label_{label}')
        label_data.append({'label_id': label, 'gesture_name': name, 'count': count})
    df_labels = pd.DataFrame(label_data)
    df_labels.to_csv(os.path.join(output_dir, 'data_stats_stage2_labels.csv'), index=False)

    # =========================================================================
    # Plots
    # =========================================================================

    # Stage 1: Positive vs Negative
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Positive\n(Gesture)', 'Negative\n(Non-gesture)']
    values = [s1_stats['positive_samples'], s1_stats['negative_samples']]
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Samples')
    ax.set_title('Stage 1: Class Distribution')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_stats_stage1_dist.png'), dpi=150)
    plt.close()

    # Stage 2: Label distribution
    if s2_stats['label_distribution']:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels_sorted = sorted(s2_stats['label_distribution'].keys())
        names = [GESTURE_NAMES.get(l, f'{l}') for l in labels_sorted]
        counts = [s2_stats['label_distribution'][l] for l in labels_sorted]

        colors = plt.cm.Set3(np.linspace(0, 1, len(labels_sorted)))
        bars = ax.bar(range(len(names)), counts, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Fragments')
        ax.set_xlabel('Gesture')
        ax.set_title('Stage 2: Gesture Distribution')

        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_stats_stage2_dist.png'), dpi=150)
        plt.close()

    return s1_summary, s2_summary


def print_report(s1_stats: dict, s2_stats: dict):
    """Print statistics to console."""
    print("\n" + "=" * 60)
    print("DATA STATISTICS REPORT")
    print("=" * 60)

    print("\n[STAGE 1 - Gesture Entry Detection]")
    print("-" * 40)
    print(f"  Sessions      : {s1_stats['total_sessions']}")
    print(f"  Total Samples : {s1_stats['total_samples']:,}")
    print(f"  Duration      : {s1_stats['total_minutes']:.1f} min ({s1_stats['total_seconds']:.0f} sec)")
    print(f"  Positive      : {s1_stats['positive_samples']:,}")
    print(f"  Negative      : {s1_stats['negative_samples']:,}")
    if s1_stats['total_samples'] > 0:
        ratio = s1_stats['positive_samples'] / s1_stats['total_samples'] * 100
        print(f"  Positive Rate : {ratio:.1f}%")

    print("\n[STAGE 2 - Gesture Classification]")
    print("-" * 40)
    print(f"  Sessions      : {s2_stats['total_sessions']}")
    print(f"  Fragments     : {s2_stats['total_fragments']}")
    print(f"  Total Samples : {s2_stats['total_samples']:,}")
    print(f"  Duration      : {s2_stats['total_minutes']:.1f} min ({s2_stats['total_seconds']:.0f} sec)")
    print(f"  Labels        : {len(s2_stats['label_distribution'])}")

    if s2_stats['label_distribution']:
        print("\n  Distribution:")
        for label in sorted(s2_stats['label_distribution'].keys()):
            count = s2_stats['label_distribution'][label]
            name = GESTURE_NAMES.get(label, f'label_{label}')
            print(f"    {label:2d} ({name:15s}): {count:4d}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Data Statistics Generator")
    parser.add_argument('--stage1_dir', default=STAGE1_DATA_DIR, help='Stage 1 data directory')
    parser.add_argument('--stage2_dir', default=STAGE2_DATA_DIR, help='Stage 2 data directory')
    parser.add_argument('--output_dir', default=REPORT_DIR, help='Output directory')
    args = parser.parse_args()

    print("Analyzing Stage 1 data...")
    s1_stats = analyze_stage1(args.stage1_dir)

    print("Analyzing Stage 2 data...")
    s2_stats = analyze_stage2(args.stage2_dir)

    print("Generating report...")
    generate_report(s1_stats, s2_stats, args.output_dir)

    print_report(s1_stats, s2_stats)

    print(f"\nReports saved to: {args.output_dir}")
    print("  - data_stats_stage1.csv")
    print("  - data_stats_stage1_sessions.csv")
    print("  - data_stats_stage1_dist.png")
    print("  - data_stats_stage2.csv")
    print("  - data_stats_stage2_labels.csv")
    print("  - data_stats_stage2_dist.png")


if __name__ == "__main__":
    main()
