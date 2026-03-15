#!/usr/bin/env python3
"""
Clean the training dataset by identifying and removing mislabeled files.

This script:
1. Analyzes all training files labeled as "alarm"
2. Identifies files with low frequency (<2000 Hz) that are likely mislabeled
3. Creates a cleaned metadata CSV with corrected labels
4. Optionally moves mislabeled files to a separate directory
"""
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import shutil

print("="*70)
print("TRAINING DATASET CLEANING")
print("="*70)
print()

# Configuration
DATASET_DIR = Path("../datasets/dataset_w1.5s_h0.25s_20260314")
METADATA_FILE = DATASET_DIR / "dataset_metadata.csv"
TRAIN_DIR = DATASET_DIR / "train"
BACKUP_DIR = DATASET_DIR / "mislabeled_backup"

# Threshold for determining if a file is mislabeled
ALARM_FREQ_THRESHOLD = 2000  # Hz - alarms should be above this

print(f"Dataset directory: {DATASET_DIR}")
print(f"Metadata file: {METADATA_FILE}")
print()

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(METADATA_FILE)
print(f"Total files: {len(metadata)}")
print(f"Training files: {len(metadata[metadata['split'] == 'train'])}")
print(f"Files labeled as 'alarm': {len(metadata[metadata['label'] == 1])}")
print()

# Analyze all training files labeled as "alarm"
train_alarms = metadata[(metadata['split'] == 'train') & (metadata['label'] == 1)]

print(f"Analyzing {len(train_alarms)} training files labeled as 'alarm'...")
print("(This may take a few minutes...)")
print()

mislabeled_files = []
correct_files = []

total = len(train_alarms)
for i, (idx, row) in enumerate(train_alarms.iterrows()):
    filename = row['filename']
    filepath = TRAIN_DIR / filename

    # Show progress every 50 files
    if (i + 1) % 50 == 0:
        print(f"Progress: {i + 1}/{total} files analyzed...")

    if not filepath.exists():
        print(f"⚠️  File not found: {filename}")
        continue

    # Load and analyze
    audio, sr = librosa.load(filepath, sr=16000, mono=True)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Get peak frequency
    freq_profile = mel_db.mean(axis=1)
    peak_band = np.argmax(freq_profile)
    peak_hz = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=8000)[peak_band]

    # Check if mislabeled
    if peak_hz < ALARM_FREQ_THRESHOLD:
        mislabeled_files.append({
            'index': idx,
            'filename': filename,
            'peak_hz': peak_hz,
            'session_id': row['session_id'],
            'window_index': row['window_index']
        })
    else:
        correct_files.append({
            'filename': filename,
            'peak_hz': peak_hz
        })

print(f"Progress: {total}/{total} files analyzed... DONE!")
print()

print("="*70)
print("ANALYSIS RESULTS:")
print("="*70)
print(f"Correctly labeled alarm files: {len(correct_files)}")
print(f"Mislabeled files (low frequency): {len(mislabeled_files)}")
print(f"Percentage mislabeled: {len(mislabeled_files) / len(train_alarms) * 100:.1f}%")
print()

if mislabeled_files:
    print("Mislabeled files (first 20):")
    print("-"*70)
    for item in mislabeled_files[:20]:
        print(f"  {item['filename']:50s} Peak: {item['peak_hz']:5.0f} Hz")
    if len(mislabeled_files) > 20:
        print(f"  ... and {len(mislabeled_files) - 20} more")
    print()

# Ask user what to do
print("="*70)
print("CLEANING OPTIONS:")
print("="*70)
print("1. Change label from 1 (alarm) to 0 (non-alarm) for mislabeled files")
print("2. Move mislabeled files to backup directory and remove from dataset")
print("3. Cancel (no changes)")
print()

choice = input("Enter your choice (1/2/3): ").strip()

if choice == "1":
    print()
    print("Changing labels for mislabeled files...")
    
    # Create a copy of metadata
    metadata_cleaned = metadata.copy()
    
    # Change labels
    for item in mislabeled_files:
        metadata_cleaned.loc[item['index'], 'label'] = 0
    
    # Save cleaned metadata
    backup_metadata = METADATA_FILE.with_suffix('.csv.backup')
    shutil.copy(METADATA_FILE, backup_metadata)
    print(f"✓ Original metadata backed up to: {backup_metadata}")
    
    metadata_cleaned.to_csv(METADATA_FILE, index=False)
    print(f"✓ Cleaned metadata saved to: {METADATA_FILE}")
    print()
    print(f"Changed {len(mislabeled_files)} files from label=1 to label=0")
    print()
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Open train_cnn_window_split.ipynb")
    print("2. Set LOAD_MODEL_PATH = None (to train from scratch)")
    print("3. Run all cells to retrain with cleaned data")
    print()

elif choice == "2":
    print()
    print("Moving mislabeled files to backup directory...")
    
    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)
    
    # Move files
    moved_count = 0
    for item in mislabeled_files:
        src = TRAIN_DIR / item['filename']
        dst = BACKUP_DIR / item['filename']
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_count += 1
    
    print(f"✓ Moved {moved_count} files to: {BACKUP_DIR}")
    
    # Remove from metadata
    metadata_cleaned = metadata.copy()
    indices_to_remove = [item['index'] for item in mislabeled_files]
    metadata_cleaned = metadata_cleaned.drop(indices_to_remove)
    
    # Save cleaned metadata
    backup_metadata = METADATA_FILE.with_suffix('.csv.backup')
    shutil.copy(METADATA_FILE, backup_metadata)
    print(f"✓ Original metadata backed up to: {backup_metadata}")
    
    metadata_cleaned.to_csv(METADATA_FILE, index=False)
    print(f"✓ Cleaned metadata saved to: {METADATA_FILE}")
    print()
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Open train_cnn_window_split.ipynb")
    print("2. Set LOAD_MODEL_PATH = None (to train from scratch)")
    print("3. Run all cells to retrain with cleaned data")
    print()

else:
    print()
    print("No changes made.")
    print()

