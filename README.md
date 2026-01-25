# Glucose Alarm ML Classification

End-to-end pipeline for collecting, preparing, and classifying glucose alarm sounds using machine learning.

## 🎯 Project Goal

Detect glucose alarm sounds in real-world environments using audio classification with convolutional neural networks.

## 📁 Project Structure

### Recording Sessions
- **`session_recorder.ipynb`** - Record audio sessions with labeled data
- **`SESSION_RECORDER_GUIDE.md`** - Complete recording guide
- **`sessions/`** - Directory where session recordings are saved

### Dataset Preparation
- **`prepare_dataset.ipynb`** - Convert sessions into windowed training data
- **`DATASET_PREPARATION_GUIDE.md`** - Complete dataset preparation guide
- **`DATASET_STRUCTURE.md`** - Dataset structure reference
- **`dataset/`** - Directory where windowed dataset is saved
  - `train/` - Training windows
  - `val/` - Validation windows
  - `dataset_metadata.csv` - Complete metadata

### Model Training
- **`train_cnn.ipynb`** - Train CNN with session-level split (original)
- **`train_cnn_window_split.ipynb`** - Train CNN with window-level split (recommended)
- **`CNN_TRAINING_GUIDE.md`** - Complete training guide
- **`WINDOW_VS_SESSION_SPLIT.md`** - Comparison of split strategies
- **`models/`** - Directory where trained models are saved
  - `glucose_alarm_cnn.pth` - Session-level split model
  - `glucose_alarm_cnn_window_split.pth` - Window-level split model (better performance)

### Live Inference
- **`live_inference.ipynb`** - Real-time alarm detection with temporal aggregation
- **`LIVE_INFERENCE_README.md`** - Live inference guide and configuration

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# All dependencies
pip install sounddevice scipy numpy jupyter librosa soundfile pandas torch matplotlib scikit-learn
```

Or install separately:
```bash
# Recording
pip install sounddevice scipy numpy jupyter

# Dataset preparation
pip install librosa soundfile pandas

# Model training
pip install torch matplotlib scikit-learn
```

### Step 2: Record Sessions

```bash
jupyter notebook session_recorder.ipynb
```

**Important**: Grant microphone permission in System Preferences → Security & Privacy → Privacy → Microphone

Configure and record:
```python
SESSION_TYPE = "glucose_alarm"          # or "no_glucose_alarm"
BACKGROUND_NOISE = "background_noise"   # or "no_background_noise"
DURATION_SECONDS = 60
```

Record at least 4 sessions:
- 2 `glucose_alarm` sessions
- 2 `no_glucose_alarm` sessions

### Step 3: Prepare Dataset

```bash
jupyter notebook prepare_dataset.ipynb
```

This will:
1. Parse session metadata from filenames
2. Split sessions into train/val (session-level split)
3. Slice sessions into 1-second overlapping windows
4. Export windowed audio and metadata CSV

### Step 4: Train CNN Model

**Recommended: Use window-level split for better performance**

```bash
jupyter notebook train_cnn_window_split.ipynb
```

This will:
1. Load windowed audio and compute log-mel spectrograms
2. Train a CNN to classify glucose alarm vs non-alarm windows
3. Use window-level split (80/20 from all sessions)
4. Evaluate on validation set
5. Save trained model to `models/glucose_alarm_cnn_window_split.pth`

**Expected Performance**: ~85% accuracy, 86% recall

### Step 5: Run Live Inference

```bash
jupyter notebook live_inference.ipynb
```

This will:
1. Load trained model from `models/` folder
2. Capture live audio from microphone
3. Use temporal aggregation (voting across 5 windows)
4. Trigger alerts when alarm is detected

**Features**:
- Configurable model selection
- Adjustable sensitivity thresholds
- Temporal voting to reduce false alarms
- Real-time visualization

## 📊 Data Flow

```
1. Record Sessions
   └─> sessions/session_<timestamp>__<label>__<context>.wav

2. Prepare Dataset
   └─> dataset/
       ├── train/<session_id>_window_<index>.wav
       ├── val/<session_id>_window_<index>.wav
       └── dataset_metadata.csv

3. Train Model
   └─> Load windows → Compute spectrograms → Train CNN → models/*.pth

4. Live Inference
   └─> Load model → Capture audio → Temporal voting → Alert
```

## 📝 Session Recording

**Output format:**
- Format: WAV (16-bit PCM)
- Sample Rate: 16,000 Hz
- Channels: Mono
- Filename: `session_<timestamp>__<label>__<context>.wav`

**Example:** `session_20260111_213045__glucose_alarm__background_noise.wav`

See **SESSION_RECORDER_GUIDE.md** for detailed instructions.

## 🔧 Dataset Preparation

**Window parameters:**
- Window length: 1.0 second
- Hop length: 0.25 seconds
- Overlap: 75%

**Output:**
- Windowed WAV files in `dataset/train/` and `dataset/val/`
- Metadata CSV with labels and split information
- Session-level split (no data leakage)

See **DATASET_PREPARATION_GUIDE.md** for detailed instructions.

## 📚 Documentation

### Guides
- **QUICKSTART_CHECKLIST.md** - Complete step-by-step checklist
- **SESSION_RECORDER_GUIDE.md** - Recording setup and usage
- **DATASET_PREPARATION_GUIDE.md** - Dataset preparation workflow
- **CNN_TRAINING_GUIDE.md** - Model training guide
- **LIVE_INFERENCE_README.md** - Live inference setup and configuration

### Reference
- **DATASET_STRUCTURE.md** - Dataset structure reference
- **MODEL_ARCHITECTURE.md** - CNN architecture details
- **WINDOW_VS_SESSION_SPLIT.md** - Comparison of training split strategies
- **EXAMPLE_OUTPUT.md** - Example console outputs

## 🔧 Requirements

- Python 3.7+
- macOS (tested) or Linux
- Microphone access (for recording)
- Dependencies:
  - Recording: `sounddevice`, `scipy`, `numpy`, `jupyter`
  - Dataset prep: `librosa`, `soundfile`, `pandas`
  - Model training: `torch`, `matplotlib`, `scikit-learn`

## 🎯 Current Status

✅ **Complete**:
- Session recording with metadata
- Dataset preparation with session-level and window-level splits
- CNN training with evaluation (85.7% accuracy)
- Live inference with temporal aggregation
- Real-time alarm detection

🚧 **Coming Soon**:
- Model optimization (higher accuracy)
- Mobile app integration
- Deployment to edge devices (Raspberry Pi, etc.)

