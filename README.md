<div align="center">

# IMU Gesture Classifier

Yuhyeon Lee · 2025

[![tests](https://img.shields.io/github/actions/workflow/status/blueion0612/IMU_Gesture_Classifier/tests.yml?branch=main&label=tests)](https://github.com/blueion0612/IMU_Gesture_Classifier/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/blueion0612/IMU_Gesture_Classifier)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20code-orange)](#limitations)

[**Recordings and checkpoints**](https://drive.google.com/drive/folders/1mgxz30EclogKA0BMhB4Qonne6MmlWgjL?usp=sharing) · [**Streaming app**](https://github.com/wearable-motion-capture/sensor-stream-apps) · [**Related**](#related)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/hero_pipeline-dark.png">
  <img alt="UDP stream feeds a binary entry detector, a detection opens a 2.5 second buffer, and a 15-class model reads that buffer" src="docs/figures/hero_pipeline.png">
</picture>

</div>

*The two-stage pipeline. Green is the entry detector that runs on every window,
gold the classifier that runs only after a detection. The hero is the pipeline and
not a results figure because no numbers are committed here; see Results.*

**IMU Gesture Classifier** recognizes fifteen hand gestures from a smartwatch, live,
using only the watch's accelerometer and gyroscope. Detection and classification are
split: a cheap binary model watches the stream continuously and decides when a
gesture has started, and only then does a fifteen-class model run on the 2.5 seconds
that follow.

## Results

No benchmark numbers are committed to this repository, so none are claimed here.
Training writes its own, and the protocol that produces them is fixed:

| Setting | Value |
|---|---|
| Split | 70 / 15 / 15, train / validation / test |
| Seed | 42 |
| Early stopping | 15 epochs without improvement, maximum 1000 |
| Batch size | 64 for stage 1, 32 for stage 2 |
| Learning rate | 1e-3 |
| Reported metrics | accuracy and macro F1, plus a confusion matrix |

Both training scripts sweep every model in the table below and write the full grid,
not the winner alone, to `gesture_data/reports/stage{1,2}_results.csv` with matching
plots. Read those rather than trusting a summary.

| Model | Description |
|---|---|
| `mlp` | Multi-layer perceptron over the flattened window |
| `lstm` | LSTM |
| `gru` | GRU |
| `tcn` | Temporal convolutional network |
| `cnn1d` | 1D convolutional network |
| `cnn_lstm` | Convolutional front end into a bidirectional LSTM |

## Quick start

```bash
git clone https://github.com/blueion0612/IMU_Gesture_Classifier
cd IMU_Gesture_Classifier
pip install -e .
```

Download `gesture_data/` (see [Data](#data)), then:

```bash
# grid search, stage 1: when does a gesture begin
python -m imu_gesture.train_stage1

# grid search, stage 2: which gesture was it
python -m imu_gesture.train_stage2

# live recognition
python -m imu_gesture.inference \
    --stage1 gesture_data/models/stage1_lstm_w1.0_s0.5.pt \
    --stage2 gesture_data/models/stage2_tcn_seq100.pt
```

Each module is also installed as a command: `imu-collector`, `imu-stats`,
`imu-train-stage1`, `imu-train-stage2` and `imu-inference` take the same arguments.

## Method

### Why two stages

A fifteen-class model run continuously on a sliding window has to
decide, at every step, both whether anything is happening and what it is. Splitting
those questions lets the first model stay small enough to run on every window while
the second sees a complete gesture rather than a fragment of one.

### Stage 1, entry detection

A binary model over a sliding window of the six watch
channels. The window length and stride are part of the grid search, not fixed in
advance.

### Stage 2, classification

A detection opens a 2.5 second buffer. The classifier is
then run over several windows inside that buffer and the highest-confidence result
wins, so the gesture does not have to start exactly where the detector fired.

### Channels

Six, all from the watch: linear acceleration and angular rate in three
axes each. The phone's sensors are received but not used for training.

### Cooldown

After a recognized gesture the detector is suppressed for a
configurable interval, two seconds by default, so that one movement cannot produce a
burst of detections.

## Usage

**Record training data.** The collector has two modes, one per stage:

```bash
python -m imu_gesture.collector continuous <PHONE_IP> --duration 60   # stage 1
python -m imu_gesture.collector fragment <PHONE_IP>                   # stage 2
```

**Inspect what you recorded** before training on it:

```bash
python -m imu_gesture.stats
```

**Inference options:** `--ip` and `--port` for the UDP socket, `--cooldown` for the
suppression interval, `--collect_sec` for the stage 2 buffer length.

## Repository layout

```
imu_gesture/
  config.py          channel map, gesture labels, split ratios, paths
  collector.py       UDP capture, continuous and fragment modes
  stats.py           dataset summaries and distribution plots
  train_stage1.py    entry detector, grid search over window and model
  train_stage2.py    15-class model, grid search over model
  inference.py       live recognition from the stream
tests/               channel map and model shape checks
docs/figures/        README figure, the script that draws it, figstyle.py
gesture_data/        recordings, checkpoints and reports, not in git
pyproject.toml       package definition, dependencies, console scripts
```

## Tests

Eight tests, none of which need recordings or a checkpoint. They check that the
channel map is a complete 55-entry layout with no gaps, that the six training
channels resolve to the right indices, that the gesture labels are contiguous and
unique, that the split ratios sum to one, and that every advertised model builds and
returns one logit per sample for stage 1 and fifteen per sample for stage 2.

```bash
python -m pytest -q
python tests/test_models.py  # without pytest
```

## Data

Recordings, pretrained checkpoints and the training reports are on
[Google Drive](https://drive.google.com/drive/folders/1mgxz30EclogKA0BMhB4Qonne6MmlWgjL?usp=sharing).
Unpack `gesture_data/` into the repository root:

```
gesture_data/
  continuous/detection/    stage 1 recordings
  gestures/fragments/      stage 2 recordings
  models/                  checkpoints
  reports/                 result tables and plots
```

**The packet format matters.** This collector expects **55 big-endian floats** on UDP
65000, which is the layout of
[wearable-motion-capture/sensor-stream-apps](https://github.com/wearable-motion-capture/sensor-stream-apps).
The sibling repository
[IMU_Stream_APP_MJU](https://github.com/blueion0612/IMU_Stream_APP_MJU) sends a
reduced **30-float** packet and is therefore **not** a drop-in source: pointing this
collector at it captures nothing, because the length check rejects every datagram.
Using it would mean rewriting the channel map in `config.py` and re-recording, since
the two layouts order their fields differently.

## Limitations

- **No committed results.** The reports live with the data, so the repository alone
  proves nothing about accuracy.
- **Gesture set is fixed at fifteen** in `config.py`, and the checkpoints are tied to
  it. Adding a gesture means retraining both stages.
- **Single wearer.** Nothing here evaluates whether the models transfer to another
  person's movements.
- **The 2.5 second buffer bounds gesture length.** Anything slower is truncated.
- **UDP with no sequence number**, so dropped packets shorten a window silently.

## Related

- [IVO](https://github.com/blueion0612/IVO): the presentation controller that loads
  the checkpoints this repository trains, reading the 30-float packet and remapping
  the six channels.
- [IMU_Stream_APP_MJU](https://github.com/blueion0612/IMU_Stream_APP_MJU): the
  sibling streaming app. It sends the 30-float packet, so it is not a source for this
  collector; see Data.
- [PPG_Classifier](https://github.com/blueion0612/PPG_Classifier) and
  [sEMG_Gesture_Classifier](https://github.com/blueion0612/sEMG_Gesture_Classifier):
  the same question, hand state from a wrist signal, asked of photoplethysmography
  and of surface electromyography.

## Citation

```bibtex
@misc{lee2025imugesture,
  author  = {Yuhyeon Lee},
  title   = {IMU Gesture Classifier: two-stage gesture recognition from smartwatch inertial data},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/blueion0612/IMU_Gesture_Classifier},
  note    = {Unpublished}
}
```

## License

MIT. See [LICENSE](LICENSE).
