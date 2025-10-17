from rockpool.devices.xylo.syns61201 import AFESim
from rockpool.timeseries import TSContinuous
import librosa
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

from rockpool.devices.xylo.syns65302 import AFESimExternal
from rockpool.timeseries import TSContinuous
import librosa
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
from pathlib import Path
from glob import glob
from multiprocessing import Pool, cpu_count
import random
from scipy.signal import butter, filtfilt

# === AFESim audio to spike encoding ===


def audio_to_features(input_path: str, output_dir: str, label: int):

    if not os.path.exists(input_path):
        logging.error(f"File does not exist: {input_path}")
        return '', None

    # Preprocessing steps for accurate AFE Simulation
    target_sr = 110000  # 110.0 kHz
    # highest center frequency = 16822, BW = 2804, center frequency + (0.5*BW)
    cutoff_freq = 18224  # Hz

    sample, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # The AFE expects a maximum input of 112mV RMS. librosa normalizes to a max of 1.0.
    rms_current = np.sqrt(np.mean(sample**2))

    # Calculate the scaling factor and apply it.
    target_rms = 0.112  # 112mV in Volts
    if rms_current > 0:
        scaling_factor = target_rms / rms_current
        sample_scaled = sample * scaling_factor
    else:
        sample_scaled = sample

    # Using a Butterworth filter for a flat passband.
    # The filter order and critical frequency need to be defined.
    # The cutoff frequency is normalized to the Nyquist frequency (0.5 * target_sr).
    order = 4
    nyquist_freq = 0.5 * target_sr
    normalized_cutoff = cutoff_freq / nyquist_freq

    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    sample_filtered = filtfilt(b, a, sample_scaled)

    dt_s = 0.009994
    # Threshold = (Time steps per spike) * (Max Signal Amplitude)
    # Max Signal Amplitude = 2**22
    # time_steps_per_spike = 100
    base_threshold = 2**29
    # Lowering threshold for higher frequency channel
    fixed_threshold_vec = tuple([base_threshold] * 10
                                + [base_threshold / 2, base_threshold / 4, base_threshold / 8,
                                   base_threshold / 16, base_threshold / 32, base_threshold / 64])

    low_pass_averaging_window = 42e-3 #To encouraging higher-frequency spikes. defualt: 84e-3.
    rate_scale_factor = 32 #Higher value results in fewer spikes.


    afesim_external = AFESimExternal.from_specification(spike_gen_mode="divisive_norm",
                                                        fixed_threshold_vec=None,
                                                        rate_scale_factor=rate_scale_factor,
                                                        low_pass_averaging_window=low_pass_averaging_window,
                                                        dn_EPS=32,
                                                        dt=dt_s,
                                                        )

    """ afesim_external = AFESimExternal.from_specification(spike_gen_mode="threshold",
                                                        fixed_threshold_vec=fixed_threshold_vec,
                                                        rate_scale_factor=63,
                                                        low_pass_averaging_window=84e-3,
                                                        dn_EPS=32,
                                                        dt=dt_s,
                                                        ) """

    sample = np.expand_dims(sample, axis=0)[0]

    out_external, _, _ = afesim_external((sample_filtered, target_sr))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"{base}.npy"
    out_path = os.path.join(output_dir, out_name)

    return out_path, out_external


def validate_npy_file(npy_path: str, max_channels: int = 16) -> bool:
    try:
        data = np.load(npy_path)
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return False
        if data.shape[1] > max_channels or not np.issubdtype(data.dtype, np.integer):
            return False
        if np.isnan(data).any():
            return False
        return True
    except:
        return False


def process_clip(args):
    input_path, split, data_dir = args
    out_path = None
    try:
        cls = Path(input_path).parent.name.lower()
        label_map = {'car': 0, 'cv': 1, 'background': 2, 'bg': 2, 'cv_aug': 1}
        class_map = {0: 'Car', 1: 'CommercialVehicle', 2: 'Background'}

        if cls not in label_map:
            raise ValueError(f"Unknown class '{cls}'")
        label = label_map[cls]
        class_name = class_map[label]

        output_dir = os.path.join(data_dir, split, class_name)
        os.makedirs(output_dir, exist_ok=True)

        out_path, out_external = audio_to_features(
            input_path, output_dir, label)
        if not out_path:
            raise RuntimeError("audio_to_features returned no path")

        np.save(out_path, out_external)
        # if not validate_npy_file(out_path):
        # os.remove(out_path)
        # return False
        return True

    except Exception as e:
        logging.error(f"[FAIL] {input_path}: {e}")
        if out_path and os.path.exists(out_path):
            os.remove(out_path)
        return False


def stratify_paths(paths, frac=0.15, seed=42):
    random.seed(seed)
    random.shuffle(paths)
    n = len(paths)
    n_val = int(frac * n)
    n_test = n_val
    n_train = n - n_val - n_test
    return paths[:n_train], paths[n_train:n_train+n_val], paths[n_train+n_val:]


if __name__ == '__main__':
    sample_size = 50000
    logging.basicConfig(
        filename='spike_test.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    base_dir = os.path.dirname(os.path.abspath('__file__'))
    seg_dir = os.path.join(base_dir, 'processed_segments')
    input_csv = os.path.join(seg_dir, 'processed_segments_labels.csv')
    data_dir = os.path.join(base_dir, 'newdataspikes')
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        logging.error(f"input CSV not found at: {input_csv}")
        exit(1)

    df = pd.read_csv(input_csv)
    df['filepath'] = df['filepath'].apply(
        lambda x: os.path.join('processed_segments', x.replace("\\", "/")))
    
    df_car = df[df['label'] == 0]
    df_cv = df[df['label'] == 1]
    df_bg = df[df['label'] == 2]

    df_car = df_car.sample(n=min(sample_size, len(df_car)), random_state=42)
    df_cv = df_cv.sample(n=min(sample_size, len(df_cv)), random_state=42)
    df_bg = df_bg.sample(n=min(sample_size, len(df_bg)), random_state=42)

    all_paths = df_car['filepath'].tolist(
    ) + df_cv['filepath'].tolist() + df_bg['filepath'].tolist()

    print(f"Total to process: {len(all_paths)} segments")

    # 统一标记 split 为 "Full"（用于 process_clip）
    tasks = [(p, "npy", data_dir) for p in all_paths]

    from tqdm import tqdm
    from multiprocessing import Pool

    # set threads
    max_threads = 6

    with Pool(processes=max_threads) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_clip, tasks), total=len(tasks)))

    succ = sum(results)
    tot = len(results)
    print(f"Done: {succ}/{tot} succeeded across all tasks.")
    logging.info(f"Finished encoding – success {succ}/{tot}.")

import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# === set path ===
base_dir = r'newdataspikes/npy'  # ← Replace it with the path where your three category directories are 
label_map = {
    'Car': 0,
    'CommercialVehicle': 1,
    'Background': 2,
}

target_shape = (100, 16)  # ← Set the shape of the data you wish to harmonize (e.g. from the most common samples)

X_all = []
y_all = []

# === Iterate over all category files ===
for cls_name, cls_label in label_map.items():
    cls_folder = os.path.join(base_dir, cls_name)
    npy_files = sorted(glob(os.path.join(cls_folder, '*.npy')))
    print(f"🔍 Class [{cls_name}]：{len(npy_files)} samples")

    for path in npy_files:
        try:
            data = np.load(path)

            # --- 裁剪 ---
            if data.shape[0] > target_shape[0]:
                data = data[:target_shape[0], :]

            # --- 填充 ---
            elif data.shape[0] < target_shape[0]:
                pad_len = target_shape[0] - data.shape[0]
                pad = np.zeros((pad_len, data.shape[1]))
                data = np.vstack((data, pad))

            # --- 维度对齐验证 ---
            if data.shape != target_shape:
                print(f"⚠️ Shape dismatch，skip：{path} current shape={data.shape}")
                continue

            X_all.append(data)
            y_all.append(cls_label)
        except Exception as e:
            print(f"❌ reading error：{path}，false：{e}")

X_all = np.stack(X_all)
y_all = np.array(y_all)
print(f"\n✅ Merger completed, total samples：{len(X_all)}，uniform shape：{X_all.shape[1:]}")

import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)



# === save ===
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\n🎉 complete saving：")
print(f" - training set：X_train.npy shape = {X_train.shape}")
print(f" - training set：y_train.npy shape = {y_train.shape}")
print(f" - validation set：X_val.npy shape = {X_val.shape}")
print(f" - validation set：y_val.npy shape = {y_val.shape}")
print(f" - test set：X_test.npy shape = {X_test.shape}")
print(f" - test set：y_test.npy shape = {y_test.shape}")