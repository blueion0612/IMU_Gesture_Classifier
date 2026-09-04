# IMU Gesture Classifier

A two-stage deep learning system for real-time gesture recognition using smartwatch IMU sensors.

## Overview

This system recognizes 15 hand gestures from IMU (accelerometer + gyroscope) data streamed via UDP from a smartwatch at 50Hz.

**Two-Stage Pipeline:**
1. **Stage 1 - Entry Detection**: Binary classifier detects when a gesture begins
2. **Stage 2 - Classification**: Multi-class classifier identifies which gesture (15 classes)

## Supported Gestures

| ID | Gesture | ID | Gesture | ID | Gesture |
|----|---------|----|---------|----|---------|
| 0 | left | 5 | circle_ccw | 10 | turn_left_90 |
| 1 | right | 6 | double_left | 11 | turn_right_90 |
| 2 | up | 7 | double_right | 12 | figure_8 |
| 3 | down | 8 | x_shape | 13 | square |
| 4 | circle_cw | 9 | double_tap | 14 | triangle |

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## Project Structure

```
IMU_Gesture_Classifier/
├── config.py          # Configuration (network, channels, paths)
├── collector.py       # Data collection (continuous/fragment modes)
├── train_stage1.py    # Stage 1 training (entry detection)
├── train_stage2.py    # Stage 2 training (gesture classification)
├── inference.py       # Real-time inference
├── stats.py           # Data statistics generator
└── README.md
```

## Data & Pretrained Models

Download the training data and pretrained models from Google Drive:

**[Download Data & Models](https://drive.google.com/drive/folders/1mgxz30EclogKA0BMhB4Qonne6MmlWgjL?usp=sharing)**

Place the downloaded `gesture_data/` folder in the project root:

```
IMU_Gesture_Classifier/
├── gesture_data/
│   ├── continuous/detection/   # Stage 1 training data
│   ├── gestures/fragments/     # Stage 2 training data
│   ├── models/                 # Pretrained models (.pt)
│   └── reports/                # Training results (CSV/PNG)
└── ...
```

## IMU Streaming App

To stream IMU data from your smartwatch, use the companion Android app:

**[IMU Stream App (GitHub)](https://github.com/blueion0612/IMU_Stream_APP_MJU)**

The app streams IMU data via UDP at 50Hz with 55 channels (smartwatch + phone sensors).

## Usage

### 1. Data Collection

Ensure the IMU Stream App is running and streaming to your PC's IP address.

```bash
# Continuous recording for Stage 1 (gesture entry detection)
python collector.py continuous <PHONE_IP> --duration 60

# Fragment recording for Stage 2 (individual gestures)
python collector.py fragment <PHONE_IP>
```

### 2. View Data Statistics

```bash
python stats.py
```

Outputs:
- `reports/data_stats_stage1.csv` - Stage 1 summary
- `reports/data_stats_stage2.csv` - Stage 2 summary
- `reports/data_stats_stage2_labels.csv` - Per-label distribution
- `reports/data_stats_*.png` - Distribution plots

### 3. Training

```bash
# Train Stage 1 (grid search over window sizes and models)
python train_stage1.py

# Train Stage 2 (grid search over models)
python train_stage2.py
```

Outputs:
- `reports/stage1_results.csv` - All configuration results
- `reports/stage2_results.csv` - All model results
- `reports/stage1_*.png` - Performance visualizations
- `reports/stage2_*.png` - Performance visualizations
- `models/stage1_*.pt` - Best Stage 1 model
- `models/stage2_*.pt` - Best Stage 2 model

### 4. Real-time Inference

```bash
python inference.py \
    --stage1 gesture_data/models/stage1_lstm_w1.0_s0.5.pt \
    --stage2 gesture_data/models/stage2_tcn_seq100.pt
```

Options:
- `--ip`: UDP bind IP (default: 0.0.0.0)
- `--port`: UDP port (default: 65000)
- `--cooldown`: Detection cooldown in seconds (default: 2.0)
- `--collect_sec`: Stage 2 collection time (default: 2.5)

## Pipeline Architecture

```
UDP Stream (50Hz)
       │
       ▼
┌─────────────────┐
│  Stage 1        │ ─→ Sliding window detection
│  Binary Detect  │
└───────┬─────────┘
        │ Gesture entry detected
        ▼
┌─────────────────┐
│  2.5s Buffer    │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Stage 2        │ ─→ Multi-window search, best confidence
│  15-Class       │
└───────┬─────────┘
        ▼
   Gesture Output
```

## Model Architectures

| Model | Description |
|-------|-------------|
| mlp | Multi-layer perceptron |
| lstm | LSTM recurrent network |
| gru | GRU recurrent network |
| tcn | Temporal Convolutional Network |
| cnn1d | 1D Convolutional Network |
| cnn_lstm | CNN + Bidirectional LSTM |

## IMU Channels Used

Only smartwatch IMU data is used for training (6 channels):
- `sw_lacc_x`, `sw_lacc_y`, `sw_lacc_z` (linear acceleration)
- `sw_gyro_x`, `sw_gyro_y`, `sw_gyro_z` (gyroscope)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 50 Hz |
| Batch Size | 64 (S1), 32 (S2) |
| Learning Rate | 1e-3 |
| Early Stopping | 15 epochs |
| Data Split | 70/15/15 (train/val/test) |

## License

MIT License

## Author

blueion0612
