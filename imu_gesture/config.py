"""
IMU Gesture Recognition System - Configuration
"""

import os

# =============================================================================
# Network
# =============================================================================
UDP_IP = "10.204.12.3"
UDP_PORT = 65000

# =============================================================================
# Sampling
# =============================================================================
SAMPLE_RATE = 50  # Hz
GESTURE_SEC = 1.5
BUFFER_SIZE = 300  # 6 sec @ 50Hz
SEQ_LEN = 100

# =============================================================================
# Gesture Labels
# =============================================================================
GESTURE_NAMES = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "circle_cw",
    5: "circle_ccw",
    6: "double_left",
    7: "double_right",
    8: "x_shape",
    9: "double_tap",
    10: "turn_left_90",
    11: "turn_right_90",
    12: "figure_8",
    13: "square",
    14: "triangle",
}

NUM_GESTURES = len(GESTURE_NAMES)

# =============================================================================
# IMU Channel Map (55 channels total)
# =============================================================================
IMU_CHANNELS = {
    # Watch - time
    "sw_dt": 0, "sw_h": 1, "sw_m": 2, "sw_s": 3, "sw_ns": 4,
    # Watch - rotation vector
    "sw_rotvec_w": 5, "sw_rotvec_x": 6, "sw_rotvec_y": 7, "sw_rotvec_z": 8, "sw_rotvec_conf": 9,
    # Watch - gyroscope
    "sw_gyro_x": 10, "sw_gyro_y": 11, "sw_gyro_z": 12,
    # Watch - linear velocity
    "sw_lvel_x": 13, "sw_lvel_y": 14, "sw_lvel_z": 15,
    # Watch - linear acceleration
    "sw_lacc_x": 16, "sw_lacc_y": 17, "sw_lacc_z": 18,
    # Watch - pressure
    "sw_pres": 19,
    # Watch - gravity
    "sw_grav_x": 20, "sw_grav_y": 21, "sw_grav_z": 22,

    # Phone - time
    "ph_dt": 23, "ph_h": 24, "ph_m": 25, "ph_s": 26, "ph_ns": 27,
    # Phone - rotation vector
    "ph_rotvec_w": 28, "ph_rotvec_x": 29, "ph_rotvec_y": 30, "ph_rotvec_z": 31, "ph_rotvec_conf": 32,
    # Phone - gyroscope
    "ph_gyro_x": 33, "ph_gyro_y": 34, "ph_gyro_z": 35,
    # Phone - linear velocity
    "ph_lvel_x": 36, "ph_lvel_y": 37, "ph_lvel_z": 38,
    # Phone - linear acceleration
    "ph_lacc_x": 39, "ph_lacc_y": 40, "ph_lacc_z": 41,
    # Phone - pressure
    "ph_pres": 42,
    # Phone - gravity
    "ph_grav_x": 43, "ph_grav_y": 44, "ph_grav_z": 45,

    # Calibration
    "sw_forward_w": 46, "sw_forward_x": 47, "sw_forward_y": 48, "sw_forward_z": 49,
    "ph_forward_w": 50, "ph_forward_x": 51, "ph_forward_y": 52, "ph_forward_z": 53,
    "sw_init_pres": 54,
}

MSG_SIZE = 55 * 4  # 55 floats x 4 bytes

# Features used for training (watch accelerometer + gyroscope)
FEATURE_COLS = ["sw_lacc_x", "sw_lacc_y", "sw_lacc_z", "sw_gyro_x", "sw_gyro_y", "sw_gyro_z"]
FEATURE_IDX = [IMU_CHANNELS[c] for c in FEATURE_COLS]
NUM_FEATURES = len(FEATURE_COLS)

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = "./gesture_data"
STAGE1_DATA_DIR = os.path.join(DATA_DIR, "continuous", "detection")
STAGE2_DATA_DIR = os.path.join(DATA_DIR, "gestures", "fragments")
MODEL_DIR = os.path.join(DATA_DIR, "models")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

for d in [DATA_DIR, STAGE1_DATA_DIR, STAGE2_DATA_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Training Defaults
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 1000
EARLY_STOP = 15
SEED = 42
