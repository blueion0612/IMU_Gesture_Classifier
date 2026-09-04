"""Tests for the channel map and the model builders.

Nothing here needs recordings or a trained checkpoint: the models are built
and run on random tensors of the documented shape. Run with `pytest`, or
directly with `python tests/test_models.py`.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_gesture import config  # noqa: E402
from imu_gesture.train_stage1 import build_model as build_stage1  # noqa: E402
from imu_gesture.train_stage2 import build_model as build_stage2  # noqa: E402

MODELS = ["mlp", "lstm", "gru", "tcn", "cnn1d", "cnn_lstm"]


def test_channel_map_is_the_upstream_55_float_layout():
    """The collector reads the upstream packet, which is 55 floats."""
    assert len(config.IMU_CHANNELS) == 55
    assert config.MSG_SIZE == 55 * 4
    assert sorted(config.IMU_CHANNELS.values()) == list(range(55))


def test_feature_columns_resolve_to_watch_accelerometer_and_gyroscope():
    """Training uses six watch channels, and their indices come from the map."""
    assert config.FEATURE_COLS == [
        "sw_lacc_x", "sw_lacc_y", "sw_lacc_z",
        "sw_gyro_x", "sw_gyro_y", "sw_gyro_z",
    ]
    assert config.NUM_FEATURES == 6
    assert config.FEATURE_IDX == [config.IMU_CHANNELS[c] for c in config.FEATURE_COLS]
    assert all(0 <= i < 55 for i in config.FEATURE_IDX)


def test_gesture_labels_are_contiguous_and_unique():
    ids = sorted(config.GESTURE_NAMES)
    assert ids == list(range(15))
    assert config.NUM_GESTURES == 15
    assert len(set(config.GESTURE_NAMES.values())) == 15


def test_data_split_covers_the_whole_set():
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-9


def test_stage1_models_emit_one_logit_per_sample():
    """Stage 1 is a binary detector over a window of six channels."""
    seq, batch = 50, 4
    x = torch.randn(batch, seq, config.NUM_FEATURES)
    for name in MODELS:
        model = build_stage1(name, (seq, config.NUM_FEATURES))
        out = model(x)
        assert out.shape[0] == batch, f"{name} lost the batch dimension"
        assert out.numel() == batch, f"{name} did not produce one logit per sample"


def test_stage2_models_emit_one_logit_per_class():
    """Stage 2 classifies into the 15 gestures."""
    seq, batch = config.SEQ_LEN, 4
    x = torch.randn(batch, seq, config.NUM_FEATURES)
    for name in MODELS:
        model = build_stage2(name, (seq, config.NUM_FEATURES), config.NUM_GESTURES)
        out = model(x)
        assert tuple(out.shape) == (batch, config.NUM_GESTURES), f"{name} shape {tuple(out.shape)}"


def test_models_are_deterministic_in_eval_mode():
    """Dropout and batch norm must be off, or inference would not repeat."""
    torch.manual_seed(0)
    x = torch.randn(2, config.SEQ_LEN, config.NUM_FEATURES)
    model = build_stage2("tcn", (config.SEQ_LEN, config.NUM_FEATURES), config.NUM_GESTURES).eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.allclose(a, b)


def test_feature_extraction_selects_the_right_columns():
    """Indexing a packet by FEATURE_IDX returns exactly the six training channels."""
    packet = np.arange(55, dtype=np.float32)
    picked = packet[config.FEATURE_IDX]
    assert picked.tolist() == [float(config.IMU_CHANNELS[c]) for c in config.FEATURE_COLS]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
