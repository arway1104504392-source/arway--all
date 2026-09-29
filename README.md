# Disability Intuition Detection Authorized Testbed
# Disability Intuition Detection Authorized Testbed
# 残疾人直觉检测授权靶场

Lesson 2: Intuition model training, real-time inference, and feedback integration
第二课时：直觉检测模型训练、实时推理与反馈集成

This testbed is fully authorized for teaching, research, and development.
本靶场已获得完整授权，可直接用于教学、研究和开发。

---

## Overview / 项目说明

The testbed collects and evaluates intuition control signals for assistive devices used by people with disabilities. The system reads EMG, IMU, and tactile array data, detects user movement intention, and outputs haptic, audio, or device control signals.

本靶场用于采集和评估残疾人辅助设备的直觉控制信号。系统读取 EMG、IMU、触觉阵列等传感器数据，检测用户运动意图，并输出触觉、音频反馈或辅助设备控制指令。

Lesson 1 covered sensor setup, data acquisition, and baseline classifiers. Lesson 2 focuses on three tasks:

第一课时已完成传感器接入、数据采集和基础分类器测试。本课时主要做三件事：

- Train time-series models for intention classification
- Add sliding-window inference for real-time output
- Map detection results to haptic and audio feedback and run the testbed protocol

- 训练时序信号模型，完成意图分类
- 接入滑动窗口推理，输出实时检测结果
- 将检测结果映射到触觉和音频反馈，跑通靶场测试流程

The default model is a 1D-CNN. It can be replaced with LSTM, GRU, or a lightweight Transformer later.

默认模型从 1D-CNN 起步，后续可以替换为 LSTM、GRU 或轻量 Transformer。

---

## Hardware / 硬件环境

| Module | Model / Notes |
|--------|---------------|
| EMG | Myo Armband / OpenBCI / 8-channel custom EMG |
| IMU | MPU6050 / BNO055 |
| Tactile sensing | Flexible piezoresistive array + TCA9548A |
| Haptic feedback | ERM motor / LRA motor |
| Audio output | 3.5mm headphone or USB speaker |
| Controller | Raspberry Pi 4B/5, Jetson Nano |
| DAQ | Arduino Due, RP2040, NI USB-6008 |

| 模块 | 型号 / 说明 |
|------|------------|
| 肌电采集 | Myo Armband / OpenBCI / 8 通道自建 EMG |
| 惯性测量 | MPU6050 / BNO055 |
| 触觉采集 | 柔性压阻阵列 + TCA9548A |
| 触觉反馈 | ERM 马达 / LRA 马达 |
| 音频输出 | 3.5mm 耳机或 USB 扬声器 |
| 主控 | Raspberry Pi 4B/5、Jetson Nano |
| 采集卡 | Arduino Due、RP2040、NI USB-6008 |

Sensor placement, electrode patches, and IMU mounting must stay consistent across sessions. Otherwise, data distributions drift noticeably between runs.

传感器佩戴位置、电极贴片和 IMU 固定方式需要保持一致，否则不同场次数据分布会明显漂移。

---

## Software Requirements / 软件依赖

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24
- SciPy >= 1.11
- scikit-learn >= 1.3
- pandas >= 2.0
- Matplotlib >= 3.7

Install / 安装：

```bash
git clone https://github.com/<your-org>/disability-intuition-detection-authorized-testbed.git
cd disability-intuition-detection-authorized-testbed

python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Repository Layout / 目录结构

```text
disability-intuition-detection-authorized-testbed/
├── README.md
├── requirements.txt
├── configs/
│   ├── sensors.yaml
│   └── model_config.yaml
├── datasets/
│   └── intuition_detection/
│       ├── raw/
│       ├── processed/
│       └── metadata/
├── scripts/
│   ├── preprocess.py
│   ├── train_intuition.py
│   ├── evaluate.py
│   ├── realtime_inference.py
│   └── run_protocol.py
├── models/
│   ├── cnn1d.py
│   ├── lstm_model.py
│   ├── transformer_model.py
│   └── fusion.py
├── utils/
│   ├── signal_processing.py
│   ├── feature_extraction.py
│   ├── feedback.py
│   └── metrics.py
├── runs/
└── docs/
    └── lesson2_guide.md
```

---

## Data Acquisition / 数据采集

Data is stored by participant and session.

数据按参与者、场次存放：

```text
datasets/intuition_detection/raw/
├── participant_001/
│   ├── session_01/
│   │   ├── emg_data.csv
│   │   ├── imu_data.csv
│   │   ├── tactile_data.csv
│   │   └── labels.csv
│   └── session_02/
└── participant_002/
```

Collection procedure / 采集流程：

1. 30s rest baseline / 静息基线 30 秒
2. Each action repeated 10 times, 5s interval / 单个动作重复 10 次，间隔 5 秒
3. Continuous action sequence simulating real use / 连续动作序列，模拟真实使用
4. Record user subjective feedback / 记录用户主观反馈

At least 10 participants per group, at least 20 repetitions per person.

建议每组不少于 10 名参与者，每人不低于 20 轮重复动作。

Label definition / 标签定义：

```python
INTENT_LABELS = {
    0: "idle",
    1: "grasp_open",
    2: "grasp_close",
    3: "reach_forward",
    4: "retract",
    5: "rotate_cw",
    6: "rotate_ccw",
    7: "push",
    8: "pull",
}
```

---

## Preprocessing / 数据预处理

Preprocessing includes band-pass filtering, notch filtering, sliding-window segmentation, and normalization.

预处理包括带通滤波、工频陷波、滑动窗口分段和归一化。

```bash
python scripts/preprocess.py \
  --raw_dir datasets/intuition_detection/raw \
  --output_dir datasets/intuition_detection/processed \
  --window_size 200 \
  --overlap 0.5
```

Default window is 200 samples. For high-rate EMG, downsample or adjust the window length first. IMU and tactile data must be aligned to EMG timestamps.

默认窗口 200 个采样点。EMG 采样率较高时，可以先降采样或调整窗口长度。IMU 和触觉数据需要与 EMG 时间戳对齐。

---

## Model Training / 模型训练

1D-CNN example / 1D-CNN 示例：

```python
import torch
import torch.nn as nn

class IntuitionCNN1D(nn.Module):
    def __init__(self, n_channels=8, n_classes=9):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_blocks(x))
```

Train / 训练：

```bash
python scripts/train_intuition.py \
  --data_dir datasets/intuition_detection/processed \
  --model 1dcnn \
  --epochs 150 \
  --batch_size 64 \
  --lr 0.001 \
  --window_size 200 \
  --output_dir runs/intuition/
```

If inter-participant variance is large, pretrain on healthy volunteer data, then fine-tune on a small sample from the target user.

如果不同参与者之间差异较大，可以先用健康志愿者数据预训练，再用目标用户的小样本做微调。

---

## Evaluation / 模型评估

```bash
python scripts/evaluate.py \
  --model runs/intuition/best_model.pth \
  --data_dir datasets/intuition_detection/processed \
  --output_dir runs/intuition/eval/
```

Key metrics / 主要指标：

| Metric | Target |
|--------|--------|
| Classification accuracy | >= 90% |
| Macro F1 | >= 0.88 |
| Online detection latency | <= 100ms |
| Feedback latency | <= 150ms |
| Resting false triggers | <= 5 / min |
| Subjective rating | >= 4/5 |

| 指标 | 目标 |
|------|------|
| 分类准确率 | >= 90% |
| Macro F1 | >= 0.88 |
| 在线检测延迟 | <= 100ms |
| 反馈延迟 | <= 150ms |
| 静息误触发 | <= 5 次/分钟 |
| 用户主观评分 | >= 4/5 |

Do not only look at overall accuracy. Check the confusion matrix as well. `idle` and similar actions are the easiest to confuse.

评估时不要只看总体准确率，还要看混淆矩阵。`idle` 和相近动作之间最容易混。

---

## Real-Time Inference / 实时推理

```bash
python scripts/realtime_inference.py \
  --model runs/intuition/best_model.pth \
  --sensor_config configs/sensors.yaml \
  --feedback_mode haptic
```

Sliding-window inference / 滑动窗口推理逻辑：

```python
import numpy as np
import torch
from collections import deque

class RealTimeInference:
    def __init__(self, model, window_size=200, stride=50, smoothing=5):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.prob_history = deque(maxlen=smoothing)

    def update(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) < self.window_size:
            return None
        if len(self.buffer) % self.stride != 0:
            return None

        window = np.array(self.buffer).T
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(window).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)

        self.prob_history.append(probs.numpy().flatten())
        avg_probs = np.mean(self.prob_history, axis=0)
        return int(np.argmax(avg_probs)), float(np.max(avg_probs))
```

---

## Feedback Mapping / 反馈映射

Keep haptic patterns under 5. More than that is hard for users to distinguish.

触觉反馈模式建议控制在 5 种以内，否则用户很难区分。

```python
PATTERNS = {
    "grasp":     [(200, 100), (200, 100), (200, 100)],
    "reach":     [(500, 200)],
    "rotate_cw": [(100, 50), (100, 50)],
    "retract":   [(300, 150), (300, 150)],
    "idle":      [],
}
```

Audio feedback supplements direction, distance, or status. Do not trigger haptic and audio at high frequency at the same time.

音频反馈用于补充方向、距离或状态信息。触觉和音频不要同时高频触发，否则会互相干扰。

---

## Testbed Protocol / 靶场测试流程

| Stage | Content | Recorded |
|-------|---------|----------|
| Preparation | Sensor calibration, rest baseline | Signal quality, baseline noise |
| Single action | Each intention repeated 10 times | Accuracy, confusion matrix |
| Continuous action | Random action sequence | Switch latency, false triggers |
| Task simulation | Grasping cup, pushing object, etc. | Completion rate, time |
| Feedback evaluation | With/without feedback comparison | NASA-TLX, subjective rating |

| 阶段 | 内容 | 记录 |
|------|------|------|
| 准备 | 传感器校准、静息基线 | 信号质量、基线噪声 |
| 单动作 | 每个意图重复 10 次 | 准确率、混淆矩阵 |
| 连续动作 | 随机动作序列 | 切换延迟、误触发 |
| 任务模拟 | 抓取杯子、推动物体等 | 完成率、耗时 |
| 反馈评估 | 对比有/无反馈 | NASA-TLX、主观评分 |

Keep environment, mounting, and task order consistent. Re-collect the rest baseline after each participant change.

测试时保持环境、佩戴方式和任务顺序一致。每次更换参与者后重新采集静息基线。

---

## FAQ / 常见问题

**High training accuracy, poor real-time performance**
**训练准确率高，实时效果差**

Check that normalization parameters match between training and real-time data. Before real-time inference, collect 10 to 20 rest samples to update mean and variance.

检查训练数据和实时数据的归一化参数是否一致。实时推理前建议采集 10 到 20 个静息样本更新均值和方差。

**Model fails after switching users**
**换人后模型不准**

Increase participant count, or fine-tune with a small sample from the target user. Participant-adversarial training is another option.

增加参与者数量，或使用少量目标用户样本做微调。也可以尝试参与者对抗训练。

**Multi-modal fusion performs worse**
**多模态融合后效果下降**

Check time alignment first. EMG, IMU, and tactile have different sampling rates and delays. Concatenating directly introduces noise. Train single-modal models first, then compare early and late fusion.

先检查时间对齐。EMG、IMU、触觉的采样率和延迟不同，直接拼接容易引入噪声。可以先分别训练单模态模型，再比较早期融合和晚期融合。

**Haptic patterns are hard to tell apart**
**触觉模式分不清**

Reduce the number of patterns, or use motors at different positions to distinguish intentions. Combine with audio for important alerts.

减少模式数量，或者用不同位置的马达区分不同意图。重要提示可以同时用音频反馈。

---

## References / 参考

1. i-GRIP: A Grasping Movement Intention Estimator for Intuitive Control of Assistive Devices. arXiv, 2023.
2. ElectroGrasp: Electrotactile Aids for Visually Impaired Individuals in Anticipatory Planning and Control of Grasp. ACM, 2023.
3. Soft Robotic Glove with Integrated Sensing for Intuitive Grasping Assistance Post Spinal Cord Injury. IEEE, 2019.
4. Vision-based interface for grasping intention detection and grip selection: towards intuitive upper-limb assistive devices. Congrès annuel de la Société de Biomécanique, 2023.

---

## License / 许可

This testbed is fully authorized. Code is released under the MIT License and may be freely used for teaching, research, and development.

本靶场已获完整授权，代码采用 MIT 许可证，可自由用于教学、研究和开发。
