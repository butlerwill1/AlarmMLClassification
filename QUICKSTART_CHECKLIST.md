# Quick Start Checklist

Follow this checklist to go from zero to a prepared dataset ready for CNN training.

---

## ✅ Prerequisites

- [ ] Python 3.7+ installed
- [ ] Microphone access granted (macOS: System Preferences → Security & Privacy → Privacy → Microphone)
- [ ] Terminal or command prompt access

---

## 📦 Step 1: Install Dependencies

```bash
# Install all required packages
pip install sounddevice scipy numpy jupyter librosa soundfile pandas torch matplotlib scikit-learn
```

**Verify installation:**
```bash
python -c "import sounddevice, scipy, numpy, librosa, soundfile, pandas, torch, matplotlib, sklearn; print('✓ All packages installed')"
```

- [ ] All packages installed successfully

---

## 🎤 Step 2: Record Sessions

### Launch Notebook

```bash
cd "/Users/wbutler/Documents/Github/Alarm ML classification"
jupyter notebook session_recorder.ipynb
```

- [ ] Jupyter notebook opened in browser

### Record Session 1: Glucose Alarm + Background Noise

1. Configure:
   ```python
   SESSION_TYPE = "glucose_alarm"
   BACKGROUND_NOISE = "background_noise"
   DURATION_SECONDS = 60
   ```

2. Run all cells
3. Wait for countdown
4. Trigger glucose alarm during recording

- [ ] Session 1 recorded: `session_YYYYMMDD_HHMMSS__glucose_alarm__background_noise.wav`

### Record Session 2: Glucose Alarm + No Background Noise

1. Configure:
   ```python
   SESSION_TYPE = "glucose_alarm"
   BACKGROUND_NOISE = "no_background_noise"
   DURATION_SECONDS = 60
   ```

2. Run all cells
3. Record in quiet environment
4. Trigger glucose alarm during recording

- [ ] Session 2 recorded: `session_YYYYMMDD_HHMMSS__glucose_alarm__no_background_noise.wav`

### Record Session 3: No Glucose Alarm + Background Noise

1. Configure:
   ```python
   SESSION_TYPE = "no_glucose_alarm"
   BACKGROUND_NOISE = "background_noise"
   DURATION_SECONDS = 60
   ```

2. Run all cells
3. Record normal environment (no alarm)

- [ ] Session 3 recorded: `session_YYYYMMDD_HHMMSS__no_glucose_alarm__background_noise.wav`

### Record Session 4: No Glucose Alarm + No Background Noise

1. Configure:
   ```python
   SESSION_TYPE = "no_glucose_alarm"
   BACKGROUND_NOISE = "no_background_noise"
   DURATION_SECONDS = 60
   ```

2. Run all cells
3. Record in quiet environment (no alarm)

- [ ] Session 4 recorded: `session_YYYYMMDD_HHMMSS__no_glucose_alarm__no_background_noise.wav`

### Verify Sessions

```bash
ls -lh sessions/
```

Expected output: 4 WAV files

- [ ] 4 session files in `sessions/` directory
- [ ] 2 files with `glucose_alarm` in filename
- [ ] 2 files with `no_glucose_alarm` in filename

---

## 🔧 Step 3: Prepare Dataset

### Launch Notebook

```bash
jupyter notebook prepare_dataset.ipynb
```

- [ ] Jupyter notebook opened in browser

### Run All Cells

1. Run all cells in the notebook
2. Watch console output for progress
3. Check for any errors or warnings

- [ ] All cells executed successfully
- [ ] No errors in output

### Verify Output

**Check directories:**
```bash
ls -lh dataset/train/
ls -lh dataset/val/
ls -lh dataset/dataset_metadata.csv
```

Expected:
- `dataset/train/` contains ~474 WAV files
- `dataset/val/` contains ~474 WAV files
- `dataset/dataset_metadata.csv` exists

- [ ] `dataset/train/` directory created with window files
- [ ] `dataset/val/` directory created with window files
- [ ] `dataset/dataset_metadata.csv` file created

### Verify Metadata

```bash
head -20 dataset/dataset_metadata.csv
```

Expected columns: `filename`, `session_id`, `window_index`, `label`, `split`, `start_time_seconds`, `context`

- [ ] Metadata CSV has correct columns
- [ ] Metadata CSV has ~948 rows (header + data)

### Verify No Session Leakage

Check the notebook output for:
```
✓ No session leakage - train and val are properly separated
```

- [ ] No session leakage detected

---

## 📊 Step 4: Inspect Dataset

### Load Metadata

```python
import pandas as pd

metadata = pd.read_csv('dataset/dataset_metadata.csv')
print(f"Total windows: {len(metadata)}")
print(f"\nSplit distribution:")
print(metadata['split'].value_counts())
print(f"\nClass balance:")
print(metadata.groupby(['split', 'label']).size())
```

- [ ] Metadata loaded successfully
- [ ] Train and val splits are balanced
- [ ] Class distribution looks reasonable

### Load Sample Window

```python
import soundfile as sf

audio, sr = sf.read('dataset/train/20260111_204751_window_0000.wav')
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio) / sr:.2f}s")
```

Expected:
- Audio shape: `(16000,)`
- Sample rate: `16000`
- Duration: `1.00s`

- [ ] Sample window loaded successfully
- [ ] Window is 1.0 second long
- [ ] Sample rate is 16,000 Hz

---

## ✅ Success Criteria (Dataset Preparation)

You're ready for model training if:

- [x] 4 session files recorded
- [x] ~948 window files created (split between train/val)
- [x] Metadata CSV created with correct schema
- [x] No session leakage between train and val
- [x] Windows are 1.0 second, 16kHz, mono

---

## 🤖 Step 5: Train CNN Model

### Launch Notebook

```bash
jupyter notebook train_cnn.ipynb
```

- [ ] Jupyter notebook opened in browser

### Run All Cells

1. Run all cells in the notebook
2. Watch training progress (20 epochs)
3. Monitor validation accuracy

- [ ] All cells executed successfully
- [ ] Training completed without errors

### Verify Training Results

**Check console output:**
- Training and validation loss should decrease
- Validation accuracy should increase
- Best model saved message appears

Expected output:
```
Epoch [20/20] | Train Loss: 0.XXXX | Val Loss: 0.XXXX | Val Acc: 0.XXXX
  → Best model saved (val_loss: 0.XXXX)

Training complete!
Best model from epoch XX with val_loss: 0.XXXX
Model saved to: glucose_alarm_cnn.pth
```

- [ ] Training loss decreased over epochs
- [ ] Validation accuracy > 0.60 (60%)
- [ ] Model saved to `glucose_alarm_cnn.pth`

### Verify Evaluation Results

**Check metrics:**
```
Classification Metrics:
  Accuracy:  0.XXXX
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1 Score:  0.XXXX
```

- [ ] Confusion matrix displayed
- [ ] Precision, recall, F1 computed
- [ ] Training/validation loss plots shown

### Verify Sanity Checks

**Check sample predictions:**
- 6 spectrograms displayed
- Ground truth vs prediction shown
- Prediction probabilities displayed

- [ ] Sanity check visualizations displayed
- [ ] Predictions look reasonable
- [ ] Model is learning meaningful patterns

---

## ✅ Success Criteria (Complete Pipeline)

You've successfully completed the full pipeline if:

- [x] 4 session files recorded
- [x] ~948 window files created
- [x] Metadata CSV created
- [x] No session leakage
- [x] CNN trained for 20 epochs
- [x] Validation accuracy > 60%
- [x] Model saved to `glucose_alarm_cnn.pth`
- [x] Evaluation metrics computed
- [x] Sanity checks passed

---

## 🎯 Next Steps

Now that your model is trained:

1. **Analyze performance**
   - Review confusion matrix
   - Identify misclassified samples
   - Understand failure modes

2. **Improve model**
   - Collect more training data
   - Add data augmentation
   - Tune hyperparameters
   - Try different architectures

3. **Deploy model**
   - Implement real-time inference
   - Add temporal smoothing
   - Deploy to production

4. **Iterate**
   - Record more diverse sessions
   - Experiment with features
   - Optimize for your use case

---

## 🆘 Troubleshooting

### "No module named 'sounddevice'"
```bash
pip install sounddevice
```

### "No sessions found"
Make sure you're in the correct directory:
```bash
pwd  # Should show: /Users/wbutler/Documents/Github/Alarm ML classification
ls sessions/  # Should show WAV files
```

### "Only 1 glucose_alarm session found"
You need at least 2 sessions of each type. Record more sessions.

### "Session leakage detected"
This shouldn't happen with the default code. Check for modifications to the split logic.

### Microphone not working
- macOS: System Preferences → Security & Privacy → Privacy → Microphone → Enable for Terminal/Jupyter
- Test: `python -c "import sounddevice; print(sounddevice.query_devices())"`

---

## 📚 Additional Resources

- **SESSION_RECORDER_GUIDE.md** - Detailed recording instructions
- **DATASET_PREPARATION_GUIDE.md** - Detailed dataset preparation guide
- **DATASET_STRUCTURE.md** - Dataset structure reference
- **README.md** - Project overview

