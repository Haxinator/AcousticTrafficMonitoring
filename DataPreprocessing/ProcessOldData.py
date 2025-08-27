from rockpool.devices.xylo.syns61201 import AFESim
from rockpool.timeseries import TSContinuous
import librosa
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
UNIFIED_CSV = os.path.join(BASE_DIR, "all_audio_labels.csv")
VEHICLE_SEG_DIR = os.path.join(BASE_DIR,"vehicle_segments_oldDatapeak")
VEHICLE_SEG_CSV = os.path.join(VEHICLE_SEG_DIR,"vehicle_clips.csv")
VEHICLE_SEG_BG_DIR = os.path.join(VEHICLE_SEG_DIR, "background")
VEHICLE_SEG_BG_CLIPS_CSV = os.path.join(VEHICLE_SEG_BG_DIR, "vehicle_clips_background.csv")
VEHICLE_SEG_CV_AUG_DIR = os.path.join(BASE_DIR, "cv_aug")

loc_folders = [f"loc{i}" for i in range(1, 7)]
splits = ["train", "val"]

all_entries = []

for loc in loc_folders:
    for split in splits:
        csv_path = os.path.join(BASE_DIR, loc, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Calculate total number of cars and CVs
        df["car_total"] = df["car_left"] + df["car_right"]
        df["cv_total"] = df["cv_left"] + df["cv_right"]

        # Assign label based on vehicle type
        def classify(row):
            if row["car_total"] > 0 and row["cv_total"] == 0:
                return 0  # only car
            elif row["cv_total"] > 0 and row["car_total"] == 0:
                return 1  # only CV
            else:
                return 2  # mixed or none

        df["label"] = df.apply(classify, axis=1)

        # Generate absolute path (path field includes .flac already)
        df["abs_path"] = df["path"].apply(lambda x: os.path.join(BASE_DIR, loc, x.replace("/", os.sep)))

        # Add metadata
        df["loc"] = loc
        df["split"] = split

        all_entries.append(df[["abs_path", "label", "loc", "split"]])

# Combine all rows
all_data = pd.concat(all_entries, ignore_index=True)

# Save to unified CSV
all_data.to_csv(UNIFIED_CSV, index=False)
print(f"Saved unified label file: {UNIFIED_CSV}")

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Process a .flac file and extract 1s segments above energy threshold
def process_flac_and_save_segments(row, output_base, segment_duration=1.0, sr=16000):
    abs_path = row["abs_path"]
    label = int(row["label"])
    loc = row["loc"]
    split = row["split"]

    if label not in [0, 1]:
        return []

    try:
        y, _ = librosa.load(abs_path, sr=sr)
    except Exception as e:
        print(f"Error loading {abs_path}: {e}")
        return []

    entries = []
    segment_samples = int(sr * segment_duration)
 
    if label == 2:
        # === Background: fixed segmentation ===
        seg_len = int(sr * segment_duration)
        num_segs = len(y) // seg_len
        for i in range(num_segs):
            # Only take one background segment for balancing
            if len(entries) >= 1:
                break
            start = i * seg_len
            end = start + seg_len
            segment = y[start:end]
            if len(segment) < seg_len:
                continue
            filename = f"{loc}_{split}_{os.path.basename(abs_path).replace('.flac','')}_bg_{i+1:03d}.wav"
            out_dir = os.path.join(output_base, "background")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            sf.write(out_path, segment, sr)
            entries.append({"filepath": os.path.relpath(out_path, output_base), "label": label})
 
    elif label in [0, 1]:
        # === Car / Truck: extract one segment centered at RMS peak ===
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if len(rms) == 0:
            return entries
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        peak_idx = np.argmax(rms)
        peak_sample = int(times[peak_idx] * sr)
 
        start = max(0, peak_sample - segment_samples // 2)
        end = min(len(y), start + segment_samples)
        if end - start < segment_samples:
            return entries
 
        segment = y[start:end]
        class_dir = "car" if label == 0 else "cv"
        out_dir = os.path.join(output_base, class_dir)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{loc}_{split}_{os.path.basename(abs_path).replace('.flac','')}_peak.wav"
        out_path = os.path.join(out_dir, filename)
        sf.write(out_path, segment, sr)
        entries.append({"filepath": os.path.relpath(out_path, output_base), "label": label})
 
    return entries

# ==== Main Process ====
df = pd.read_csv(UNIFIED_CSV)
df = df[df["label"].isin([0, 1])]  # process only pure vehicle samples

# 1. Sample 1000 instances from each class
df_car = df[df["label"] == 0].sample(n=1000, random_state=42)
#or however many trucks there are.
df_truck = df[df["label"] == 1].sample(n=len(df[df["label"] == 1]), random_state=42)

df_balanced = pd.concat([df_car, df_truck], ignore_index=True)

all_entries = []

for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
    entries = process_flac_and_save_segments(row, VEHICLE_SEG_DIR)
    all_entries.extend(entries)

# Save final CSV
df_out = pd.DataFrame(all_entries)
df_out.to_csv(VEHICLE_SEG_CSV, index=False)
print(f"\n✅ Saved segmented vehicle clips to {VEHICLE_SEG_CSV}")

import pandas as pd

df = pd.read_csv(VEHICLE_SEG_CSV)

car_count = (df["label"] == 0).sum()
cv_count = (df["label"] == 1).sum()

print(f"Number of car segments (label=0): {car_count}")
print(f"Number of cv segments (label=1):  {cv_count}")
