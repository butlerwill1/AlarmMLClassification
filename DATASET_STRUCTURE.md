# Dataset Structure Reference

Quick reference for the windowed audio dataset structure.

---

## Directory Structure

```
Alarm ML classification/
│
├── sessions/                                    # Input: Session-level recordings
│   ├── session_<timestamp>__<label>__<context>.wav
│   └── ...
│
├── dataset/                                     # Output: Windowed dataset
│   ├── train/                                   # Training windows
│   │   ├── <session_id>_window_0000.wav
│   │   ├── <session_id>_window_0001.wav
│   │   └── ...
│   │
│   ├── val/                                     # Validation windows
│   │   ├── <session_id>_window_0000.wav
│   │   ├── <session_id>_window_0001.wav
│   │   └── ...
│   │
│   └── dataset_metadata.csv                     # Complete metadata
│
├── session_recorder.ipynb                       # Record sessions
└── prepare_dataset.ipynb                        # Create windowed dataset
```

---

## File Naming Conventions

### Session Files (Input)

**Format:**
```
session_<timestamp>__<label>__<context>.wav
```

**Components:**
- `timestamp`: `YYYYMMDD_HHMMSS` (e.g., `20260111_204751`)
- `label`: `glucose_alarm` or `no_glucose_alarm`
- `context`: Additional metadata (e.g., `background_noise`, `no_background_noise`)

**Examples:**
```
session_20260111_204751__glucose_alarm__no_background_noise.wav
session_20260111_205217__glucose_alarm__background_noise.wav
session_20260111_205814__no_glucose_alarm__alarms.wav
session_20260111_210208__no_glucose_alarm__background_noise.wav
```

### Window Files (Output)

**Format:**
```
<session_id>_window_<index>.wav
```

**Components:**
- `session_id`: Timestamp from source session (e.g., `20260111_204751`)
- `index`: 4-digit zero-padded window number (e.g., `0000`, `0042`, `0237`)

**Examples:**
```
20260111_204751_window_0000.wav
20260111_204751_window_0042.wav
20260111_205217_window_0000.wav
20260111_210208_window_0123.wav
```

---

## Metadata CSV Schema

**File:** `dataset/dataset_metadata.csv`

### Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `filename` | string | Window filename | `20260111_204751_window_0042.wav` |
| `session_id` | string | Source session timestamp | `20260111_204751` |
| `window_index` | int | Window number within session | `42` |
| `label` | int | Binary label (0 or 1) | `1` |
| `split` | string | Dataset split | `train` or `val` |
| `start_time_seconds` | float | Window start time in session | `10.5` |
| `context` | string | Session context | `background_noise` |

### Label Mapping

- **0** = `no_glucose_alarm` (negative class)
- **1** = `glucose_alarm` (positive class)

### Example Rows

```csv
filename,session_id,window_index,label,split,start_time_seconds,context
20260111_204751_window_0000.wav,20260111_204751,0,1,train,0.0,no_background_noise
20260111_204751_window_0001.wav,20260111_204751,1,1,train,0.25,no_background_noise
20260111_204751_window_0042.wav,20260111_204751,42,1,train,10.5,no_background_noise
20260111_205217_window_0000.wav,20260111_205217,0,1,val,0.0,background_noise
20260111_210208_window_0000.wav,20260111_210208,0,0,train,0.0,background_noise
```

---

## Audio Specifications

### Session Files

- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Duration**: Variable (typically 60-300 seconds)

### Window Files

- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Duration**: 1.0 second (16,000 samples)
- **Overlap**: 75% (0.75 seconds between consecutive windows)

---

## Dataset Statistics

### Window Calculation

For a session of duration `D` seconds:

```
Number of windows = floor((D - window_length) / hop_length) + 1
                  = floor((D - 1.0) / 0.25) + 1
```

**Examples:**
- 60s session → ~237 windows
- 120s session → ~477 windows
- 180s session → ~717 windows

### Typical Dataset Size

With 4 sessions (2 per class, 60s each):
- **Total windows**: ~948
- **Train windows**: ~474 (2 sessions)
- **Val windows**: ~474 (2 sessions)
- **Class balance**: 50/50 (if sessions are balanced)

---

## Data Loading Examples

### Load Metadata

```python
import pandas as pd

# Load metadata
metadata = pd.read_csv('dataset/dataset_metadata.csv')

# Filter by split
train_metadata = metadata[metadata['split'] == 'train']
val_metadata = metadata[metadata['split'] == 'val']

# Filter by label
positive_samples = metadata[metadata['label'] == 1]
negative_samples = metadata[metadata['label'] == 0]
```

### Load Audio Window

```python
import librosa
import soundfile as sf

# Using librosa
audio, sr = librosa.load('dataset/train/20260111_204751_window_0042.wav', sr=16000)

# Using soundfile
audio, sr = sf.read('dataset/train/20260111_204751_window_0042.wav')
```

### Load Batch of Windows

```python
import numpy as np
import soundfile as sf
import pandas as pd

def load_windows(metadata_df, base_dir):
    """Load all windows from metadata."""
    audio_data = []
    labels = []
    
    for idx, row in metadata_df.iterrows():
        filepath = base_dir / row['filename']
        audio, sr = sf.read(filepath)
        audio_data.append(audio)
        labels.append(row['label'])
    
    return np.array(audio_data), np.array(labels)

# Load training data
train_audio, train_labels = load_windows(train_metadata, Path('dataset/train'))
```

---

## Session-Level Split Guarantee

**Key principle**: Windows from the same session are NEVER split between train and val.

**Verification:**
```python
import pandas as pd

metadata = pd.read_csv('dataset/dataset_metadata.csv')

train_sessions = set(metadata[metadata['split'] == 'train']['session_id'])
val_sessions = set(metadata[metadata['split'] == 'val']['session_id'])

overlap = train_sessions & val_sessions
assert len(overlap) == 0, "Session leakage detected!"
```

This ensures the model is evaluated on completely unseen sessions, not just unseen windows from seen sessions.

---

## Quick Reference

| Aspect | Value |
|--------|-------|
| Window length | 1.0 second |
| Hop length | 0.25 seconds |
| Overlap | 75% |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Format | 16-bit PCM WAV |
| Label 0 | no_glucose_alarm |
| Label 1 | glucose_alarm |
| Split levels | Session-level (not window-level) |

