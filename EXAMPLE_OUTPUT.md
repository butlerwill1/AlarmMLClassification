# Example Session Recording Output

This shows what you'll see when running the `session_recorder.ipynb` notebook.

## Example 1: Glucose Alarm Session with Background Noise

### Configuration
```python
SESSION_TYPE = "glucose_alarm"
BACKGROUND_NOISE = "background_noise"
DURATION_SECONDS = 60
```

### Console Output
```
======================================================================
AUDIO SESSION RECORDING
======================================================================
Session Type: glucose_alarm
Background Noise: background_noise
Duration: 60 seconds
Sample Rate: 16000 Hz
Channels: 1 (mono)
Format: 16-bit PCM WAV
Output: sessions/session_20260111_213045__glucose_alarm__background_noise.wav
======================================================================

🎤 Recording will start in 3 seconds...
   3...
   2...
   1...

🔴 RECORDING NOW (60 seconds)

✓ Recording complete!

======================================================================
SESSION SAVED
======================================================================
File: sessions/session_20260111_213045__glucose_alarm__background_noise.wav
Size: 1.83 MB
Samples: 960,000
======================================================================
```

## Example 2: No Alarm Session in Quiet Environment

### Configuration
```python
SESSION_TYPE = "no_glucose_alarm"
BACKGROUND_NOISE = "no_background_noise"
DURATION_SECONDS = 120
```

### Console Output
```
======================================================================
AUDIO SESSION RECORDING
======================================================================
Session Type: no_glucose_alarm
Background Noise: no_background_noise
Duration: 120 seconds
Sample Rate: 16000 Hz
Channels: 1 (mono)
Format: 16-bit PCM WAV
Output: sessions/session_20260111_214530__no_glucose_alarm__no_background_noise.wav
======================================================================

🎤 Recording will start in 3 seconds...
   3...
   2...
   1...

🔴 RECORDING NOW (120 seconds)

✓ Recording complete!

======================================================================
SESSION SAVED
======================================================================
File: sessions/session_20260111_214530__no_glucose_alarm__no_background_noise.wav
Size: 3.66 MB
Samples: 1,920,000
======================================================================
```

## Typical Recording Session Workflow

### Morning Session (Background Noise, No Alarm)
```python
SESSION_TYPE = "no_glucose_alarm"
BACKGROUND_NOISE = "background_noise"
DURATION_SECONDS = 180  # 3 minutes of ambient sound
```

### Alarm Test Session (Quiet Environment)
```python
SESSION_TYPE = "glucose_alarm"
BACKGROUND_NOISE = "no_background_noise"
DURATION_SECONDS = 90  # 1.5 minutes with alarm
```

### Evening Session (Background Noise, No Alarm)
```python
SESSION_TYPE = "no_glucose_alarm"
BACKGROUND_NOISE = "background_noise"
DURATION_SECONDS = 180
```

## File Organization After Multiple Sessions

```
sessions/
├── session_20260111_080000__no_glucose_alarm__background_noise.wav           (3.66 MB)
├── session_20260111_083000__glucose_alarm__no_background_noise.wav           (1.83 MB)
├── session_20260111_090000__glucose_alarm__background_noise.wav              (1.83 MB)
├── session_20260111_120000__no_glucose_alarm__no_background_noise.wav        (5.49 MB)
├── session_20260111_150000__glucose_alarm__background_noise.wav              (2.75 MB)
└── session_20260111_200000__no_glucose_alarm__background_noise.wav           (3.66 MB)
```

## Data Collection Tips

1. **Vary recording times** - Different times of day capture different ambient sounds
2. **Multiple alarm recordings** - Capture alarm from different distances/angles
3. **Background variety** - Record in different rooms/environments
4. **Consistent labeling** - Always set SESSION_TYPE before recording
5. **Verify files** - Check file size after each recording to ensure audio was captured

## Expected File Sizes

| Duration | File Size | Samples   |
|----------|-----------|-----------|
| 60s      | ~1.83 MB  | 960,000   |
| 90s      | ~2.75 MB  | 1,440,000 |
| 120s     | ~3.66 MB  | 1,920,000 |
| 180s     | ~5.49 MB  | 2,880,000 |
| 300s     | ~9.16 MB  | 4,800,000 |

Formula: `duration × 16000 samples/sec × 2 bytes/sample = file size`

