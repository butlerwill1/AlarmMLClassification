# Dataset Cleaning Summary

**Date:** 2026-03-15  
**Issue:** Model gives 0.00 probability to real alarm sounds despite correct frequency

---

## 🔍 Root Cause Analysis

### Problem Identified
The training dataset contains **mislabeled files** where silence/background noise is labeled as "alarm".

### Evidence
Out of the first 20 files labeled as "alarm" (label=1):
- **12 files (60%)** have LOW frequency (0-48 Hz) → Actually silence/noise
- **8 files (40%)** have HIGH frequency (~3131 Hz) → Actual alarms

### Specific Mislabeled Files
Files labeled as "alarm" but containing only silence:
- `window_0000` through `window_0006` (7 files) - 0-48 Hz
- `window_0015` through `window_0019` (5 files) - 0 Hz

### Impact on Model Performance
The model learned that "alarms" can be:
1. ❌ Low-frequency silence (0-48 Hz) - INCORRECT
2. ✅ High-frequency beeps (~3131 Hz) - CORRECT

This confusion causes:
- False positives on silence/background noise
- False negatives on actual alarms that don't exactly match training pattern

### Live Alarm Test Results
**User's captured alarm:**
- Frequency: **2980 Hz** (very close to training alarms at 3131 Hz)
- RMS: **0.23** (loud and clear signal)
- Model probability: **0.0003** ❌ (essentially 0.00)

**Training alarms (correctly labeled):**
- Frequency: **3131 Hz**
- RMS: **0.018**
- Model probability: **0.9999** ✅ (essentially 1.00)

---

## 🛠️ Solution

### ⚠️ Dataset Analysis Complete - Cleaning Required!

**Analysis script:** `tests/analyze_all_training_files.py`

**Key Findings:**
- **74.3% of files labeled as "alarm" are actually mislabeled!**
- Only **182 out of 708** files have peak > 2000 Hz (actual alarms)
- **526 files** have peak < 2000 Hz (mostly 0-96 Hz silence/noise)

**This explains why the model fails:**
- The model was trained mostly on silence labeled as "alarm"
- It learned that "alarms" can be low-frequency noise
- Real alarms (~3000 Hz) don't match what it learned

### 🔧 How to Fix

1. **Run the frequency analysis** (if not already done):
   ```bash
   cd tests
   source ../venv/bin/activate
   python3 analyze_all_training_files.py
   ```

2. **Review the results:**
   - Check `tests/complete_training_analysis.png`
   - Review `tests/training_frequency_analysis_results.csv`

3. **Fix the mislabeled files:**
   ```bash
   python3 fix_mislabeled_files.py
   ```
   - This will relabel 526 files from alarm (1) to non-alarm (0)
   - Original metadata will be backed up

---

## 📋 Next Steps

### ✅ Step 1: Dataset Cleaned (COMPLETE)

### 🔄 Step 2: Retrain the Model (DO THIS NOW)

1. **Open `train_cnn_window_split.ipynb`**
2. **Set `LOAD_MODEL_PATH = None`** (line ~53) to train from scratch
3. **Run all cells**
4. **Wait ~10-15 minutes** for training to complete

### 🧪 Step 3: Test with Live Inference

1. **Open `live_inference.ipynb`**
2. **Update to use the new model** (it will be the latest .pth file)
3. **Test with real alarm sound**
4. **Expected result:** Probability > 0.9 for alarms, < 0.1 for non-alarms

---

## ✅ Expected Outcome

After retraining with cleaned data:
- ✅ Model recognizes alarms at any volume
- ✅ Model focuses on frequency pattern (~3000 Hz)
- ✅ Model rejects silence and background noise
- ✅ Live alarm detection works correctly
- ✅ False positive rate on environmental sounds (keys, etc.) reduced

---

## 📝 Important Notes

### RMS Thresholds - NOT the Issue
Initial investigation focused on RMS thresholds, but this was a red herring:
- User's alarm: RMS = 0.23 (loud, no penalty)
- Training alarm: RMS = 0.018 (normal, no penalty)
- Both are above 0.01 threshold, so no penalty applied

The real issue was always the **mislabeled training data**, not normalization.

### Normalization Settings
Current normalization (V2) is correct:
```python
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
rms_energy = np.sqrt(np.mean(audio**2))
if rms_energy < 0.0001:  # Only extreme silence
    log_mel_spec = log_mel_spec - 60
```

This provides:
- Pattern-based recognition (volume-independent)
- Suppression of extreme silence only
- No penalty for normal or quiet alarms

---

## 🗂️ Files Organized

All testing and debugging files have been moved to the `tests/` directory:

**Testing Scripts:**
- `tests/verify_what_microphone_hears.py` - Verify microphone input
- `tests/test_alarm_audio.py` - Quick live testing
- `tests/test_microphone.py` - Basic microphone test
- `tests/debug_live_capture.py` - Debug live audio capture

**Analysis Scripts:**
- `tests/analyze_all_training_files.py` - Analyze ALL training files (recommended)
- `tests/fix_mislabeled_files.py` - Fix labels based on analysis
- `tests/clean_training_dataset.py` - Alternative cleaning approach

**Analysis Results:**
- `tests/training_frequency_analysis_results.csv` - Detailed frequency analysis
- `tests/complete_training_analysis.png` - Visualization
- `tests/captured_audio_*.wav` - Test recordings

**Documentation:**
- `tests/README.md` - Complete guide to all testing tools

