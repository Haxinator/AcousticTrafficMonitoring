# Use SynNet to start. We Will need to develop our own architecture later.
# Still able to make some small adjustments to the network though.
from rockpool.nn.networks import SynNet
from rockpool.nn.modules import LIFExodus
from rockpool.transform import quantize_methods as q
from rockpool import TSEvent, TSContinuous
from rockpool.devices.xylo import find_xylo_hdks
from rockpool.nn.modules import LIFTorch

# Use fastapi to connect with the frontend
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# - Import torch training utilities
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# simulation
from starlette.websockets import WebSocketDisconnect
import random
import librosa
from scipy.signal import butter, filtfilt
from rockpool.devices.xylo.syns65302 import AFESimExternal

# Thresholding and merics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import itertools
import numpy as np
import pandas as pd

# for paths
import os
import logging

# visualisation
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

xylo_board_name = 'XyloAudio3'

# - Imports dependent on your HDK
# - XyloAudio 2
if xylo_board_name == 'XyloAudio2':
    import rockpool.devices.xylo.syns61201 as xa2
    from rockpool.devices.xylo.syns61201 import xa2_devkit_utils as xa2utils
# - XyloAudio 3
elif xylo_board_name == 'XyloAudio3':
    import rockpool.devices.xylo.syns65302 as xa3
    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as xa3utils

# To run the backend, type "uvicorn Xylo:app --reload --port 3000" in the terminal
# ----------------------------------- API ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------------------------

n_labels = 3
net_in_channels = 16
dt = 1e-3
n_batches = 256
skip_window = 30
class_names = ["Car", "Commercial", "Background"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath('__file__'))
model_path = os.path.join(current_dir, "Best_Model.json")
data_path = os.path.join(current_dir, "DataPreprocessing")
os.makedirs("plots", exist_ok=True)
os.makedirs("vmem_calibration", exist_ok=True)

# # ----------------------------------- Thresholding and Analysis ------------------------------
vmem_net = SynNet(
    neuron_model=LIFTorch,
    output="vmem",
    n_channels=net_in_channels,
    n_classes=n_labels,
    size_hidden_layers=[24, 24, 24],
    time_constants_per_layer=[2, 4, 8],
    p_dropout=0.1
).to(device)

try:
    vmem_net.load(model_path)
    vmem_net.eval()
    print("Successfully loaded trained model for vmem collection.")
except FileNotFoundError:
    print(f"Trained model not found at {model_path}. Please check the path.")
    exit()

try:
    X_val = torch.from_numpy(
        np.load(os.path.join(data_path, "X_val.npy"))).float()
    y_val = torch.from_numpy(
        np.load(os.path.join(data_path, "y_val.npy"))).long()
except FileNotFoundError:
    print("Validation data not found. Please check the path.")
    exit()

val_ds = TensorDataset(X_val, y_val)
val_dl = DataLoader(val_ds, batch_size=n_batches, shuffle=False)

# # -------------------- Collect Vmems ------------------------
# vmem_file = os.path.join("vmem_calibration", "all_vmems.npy")
# labels_file = os.path.join("vmem_calibration", "all_labels.npy")
# all_vmems = []
# all_labels = []

# if os.path.exists(vmem_file) and os.path.exists(labels_file):
#     print("Found existing Vmems and labels. Loading from file...")
#     all_vmems = np.load(vmem_file)
#     all_labels = np.load(labels_file)
# else:
#     print("No saved Vmems found. Running network to collect Vmems...")
#     all_vmems = []
#     all_labels = []
#     with torch.no_grad():
#         for events, labels in val_dl:
#             events, labels = events.to(device), labels.to(device)
#             out, _, _ = vmem_net(events)
#             output_vmems = out[:, skip_window:, :].mean(dim=1)
#             all_vmems.append(output_vmems.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     all_vmems = np.concatenate(all_vmems, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     print("----- Vmem collection complete ------")
#     print("vmem shape:", all_vmems.shape, "labels shape:", all_labels.shape)

#     np.save(vmem_file, all_vmems)
#     np.save(labels_file, all_labels)

# # -------------------- Vmem Range Analysis -------------------
# lowest_vmem = np.min(all_vmems)
# highest_vmem = np.max(all_vmems)
# print(f"Lowest Vmem recorded: {lowest_vmem:.4f}")
# print(f"Highest Vmem recorded: {highest_vmem:.4f}")

# # ------------------- Automated Grid-Search Thresholding -------------------
# print("Starting automated grid search for optimal thresholds...")

# threshold_ranges = {
#     0: np.arange(-5.0, 10.0, 0.5),   # Car
#     1: np.arange(-5.0, 10.0, 0.5),   # Commercial
#     2: np.arange(-5.0, 10.0, 0.5),   # Background
# }

# best_macro_f1 = -1
# grid_search_thresholds = None

# all_threshold_combinations = list(itertools.product(
#     threshold_ranges[0],
#     threshold_ranges[1],
#     threshold_ranges[2]
# ))

# for tset in all_threshold_combinations:
#     spikes = (all_vmems >= tset).astype(int)
#     preds = np.argmax(spikes, axis=1)

#     acc = accuracy_score(all_labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         all_labels, preds, average=None, zero_division=0
#     )
#     macro_f1 = np.mean(f1)

#     if macro_f1 > best_macro_f1:
#         best_macro_f1 = macro_f1
#         grid_search_thresholds = tset
# # ------------------- Iterative Threshold Adjustment -------------------
# print("Starting iterative threshold adjustment...")

# thresholds = np.zeros(n_labels)  # [0.0, 0.0, 0.0]
# step_size = 0.1
# max_iterations = 100
# tol = 1e-4

# best_macro_f1 = -1
# interative_thresholds = thresholds.copy()

# for it in range(max_iterations):
#     improved = False
#     for idx in range(n_labels):
#         for delta in [-step_size, step_size]:
#             temp_thresholds = thresholds.copy()
#             temp_thresholds[idx] += delta

#             spikes = (all_vmems >= temp_thresholds).astype(int)
#             preds = np.argmax(spikes, axis=1)

#             _, _, f1, _ = precision_recall_fscore_support(
#                 all_labels, preds, average=None, zero_division=0
#             )
#             macro_f1 = np.mean(f1)

#             if macro_f1 > best_macro_f1 + tol:
#                 best_macro_f1 = macro_f1
#                 interative_thresholds = temp_thresholds.copy()
#                 improved = True

#     if not improved:
#         break
#     thresholds = interative_thresholds .copy()
# # ------------------- Manual Threshold Comparison -------------------
# print("Starting manual threshold comparison...")

# best_macro_f1_manual = -1
# opt_thresholds = None

# # Define the threshold sets you want to test
# manual_thresholds_to_test = [
#     [0.1, 0.2, 0.0],
#     [-1.0, 6.0, 4.0],
#     [1.0, 1.0, 1.0],
#     [3.0, 4.0, 5.0],
#     grid_search_thresholds,
#     interative_thresholds,
# ]

# for idx, tset in enumerate(manual_thresholds_to_test):
#     print(f"\n----- Evaluating Threshold Set {idx+1}: {tset} -----")

#     spikes = (all_vmems >= tset).astype(int)
#     preds = np.argmax(spikes, axis=1)

#     acc = accuracy_score(all_labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         all_labels, preds, average=None, zero_division=0
#     )
#     macro_f1 = np.mean(f1)
#     conf_matrix = confusion_matrix(all_labels, preds)

#     print(f"Macro F1: {macro_f1:.4f}")
#     print(f"Accuracy: {acc:.4f}")
#     for c_idx, cname in enumerate(class_names):
#         print(
#             f"{cname}: Precision={precision[c_idx]:.3f}, Recall={recall[c_idx]:.3f}, F1={f1[c_idx]:.3f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)

#     if macro_f1 > best_macro_f1_manual:
#         best_macro_f1_manual = macro_f1
#         opt_thresholds = tset

opt_thresholds = (np.float64(0.5), np.float64(1.0), np.float64(-5.0))

print(
    f"\nSelected optimal thresholds based on manual comparison: {opt_thresholds}")
# ----------------------------------------------------------------------
def initialize_hardware():
    net = SynNet(
        # neuron_model = LIFExodus,
        # output="vmem",                         # Use the membrane potential as the output of the network.
        # probability of dropout (good to prevent overfitting).
        p_dropout=0.2,

        # time constants and threshold are not trainable by default.
        # NOTE if not using SynNet then they will be by default.

        # Number of input channels (always 16)
        n_channels=net_in_channels,
        # Number of output classes (car, commercial, background noise).
        n_classes=n_labels,
        # Number of neurons in each hidden layer (taken from tutorial)
        size_hidden_layers=[24, 24, 24],
        # Number of time constants in each hidden layer (taken from tutorial)
        time_constants_per_layer=[2, 4, 8],
    )
    
    net.load("Best_Model.json")
    net.seq.out_neurons = LIFTorch([3, 3], threshold=opt_thresholds)
    spec = None

    # - Call the Xylo mapper on the extracted computational graph
    # - For XyloAudio 2
    if xylo_board_name == 'XyloAudio2':
        spec = xa2.mapper(net.as_graph(),  weight_dtype='float',
                        threshold_dtype='float', dash_dtype='float')
    # - For XyloAudio 3
    elif xylo_board_name == 'XyloAudio3':
        spec = xa3.mapper(net.as_graph(),  weight_dtype='float',
                        threshold_dtype='float', dash_dtype='float')

    unquantised_spec = spec.copy()
    spec.update(q.channel_quantize(**spec))
    
    # - Use rockpool.devices.xylo.config_from_specification to convert it to a hardware configuration
    # - For XyloAudio 2
    if xylo_board_name == 'XyloAudio2':
        config, is_valid, msg = xa2.config_from_specification(**spec)
    # - For XyloAudio 3
    elif xylo_board_name == 'XyloAudio3':
        config, is_valid, msg = xa3.config_from_specification(**spec)
    

    if not is_valid:
        print(f"Invalid configuration: {msg}")
        return None, None, None, None
    
    # - Use rockpool.devices.xylo.find_xylo_hdks to connect to an HDK
    xylo_hdk_nodes, modules, versions = find_xylo_hdks()
    print(f'HDK versions detected: {versions}')

    hdk = None
    modSamna = None
    modSim = None  # define globally
    modMonitor = None
    
    for version, xylo in zip(versions, xylo_hdk_nodes):
        if version == "syns61201":
            hdk = xylo
        # - For XyloAudio 3
        elif version == "syns65302":
            hdk = xylo

    # Deploy to HDK or simulation
    if hdk:
        if xylo_board_name == 'XyloAudio2':
            modSamna = xa2.XyloSamna(hdk, config, dt=dt)
            if modSamna:
                modSamna.reset_state()
        elif xylo_board_name == 'XyloAudio3':
            modSamna = xa3.XyloSamna(hdk, config, dt=dt)
    else:
        print('HDK not detected, running simulation.')
        if xylo_board_name == 'XyloAudio2':
            modSim = xa2.XyloSim.from_config(config, dt=dt)
        elif xylo_board_name == 'XyloAudio3':
            modSim = xa3.XyloSim.from_config(config, dt=dt)

    # Setup for free inference mode
    freeInferenceMode = True
    if freeInferenceMode and hdk:
        print('Free Inference Mode enabled.')
        output_mode = "Vmem"
        amplify_level = "low"
        hibernation = False
        DN = True

        if xylo_board_name == 'XyloAudio2':
            modMonitor = xa2.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode,
                                        amplify_level=amplify_level, hibernation_mode=hibernation, divisive_norm=DN)
        elif xylo_board_name == 'XyloAudio3':
            modMonitor = xa3.XyloMonitor(
                hdk, config, dt=dt, output_mode=output_mode, hibernation_mode=hibernation, dn_active=DN)

    return modSamna, modSim, hdk, modMonitor

# ------------------ Initialize Hardware -----------------------------------------
modSamna, modSim, hdk, modMonitor = initialize_hardware()
# --------------------------------------------------------------------------------
AUDIO_FOLDERS = [
    "vehicle_segments_cleaned_timon/background",
    "vehicle_segments_cleaned_timon/car",
    "vehicle_segments_cleaned_timon/cv"
]
audio_files_by_category = {
    "background": [],
    "car": [],
    "cv": []
}

# AFE simulation parameters (from your training code)
target_sr = 110000  # 110.0 kHz
cutoff_freq = 18224  # Hz
dt_s = 0.009994
low_pass_averaging_window = 42e-3
rate_scale_factor = 32

def load_audio_files():
    """Load all audio files and categorize them."""
    global audio_files_by_category
    
    for folder in AUDIO_FOLDERS:
        category = os.path.basename(folder)
        audio_path = os.path.join(data_path, folder)
        print(audio_path)
        if os.path.exists(audio_path) and os.path.isdir(audio_path):
            wav_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                        if f.lower().endswith('.wav')]
            audio_files_by_category[category].extend(wav_files)
            print(f"Loaded {len(wav_files)} audio files from {audio_path}")
        else:
            print(f"Audio folder '{audio_path}' not found.")
    
    total_files = sum(len(files) for files in audio_files_by_category.values())
    print(f"Total audio files loaded: {total_files}")

# Initial load of audio files
load_audio_files()

def process_audio_file(file_path):
    """
    Process a WAV file using the same AFESimExternal pipeline as training
    Returns the spike features compatible with the model
    """
    try:
        # Load audio with librosa
        sample, sr = librosa.load(file_path, sr=target_sr, mono=True)

        # Scale audio to target RMS (112mV RMS)
        rms_current = np.sqrt(np.mean(sample**2))
        target_rms = 0.112  # 112mV in Volts
        if rms_current > 0:
            scaling_factor = target_rms / rms_current
            sample_scaled = sample * scaling_factor
        else:
            sample_scaled = sample

        # Apply low-pass filter
        order = 4
        nyquist_freq = 0.5 * target_sr
        normalized_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        sample_filtered = filtfilt(b, a, sample_scaled)

        # Create AFESimExternal instance
        afesim_external = AFESimExternal.from_specification(
            spike_gen_mode="divisive_norm",
            fixed_threshold_vec=None,
            rate_scale_factor=rate_scale_factor,
            low_pass_averaging_window=low_pass_averaging_window,
            dn_EPS=32,
            dt=dt_s,
        )

        # Process audio through AFE
        out_external, _, _ = afesim_external((sample_filtered, target_sr))
        
        # Convert to numpy array (spike raster)
        if hasattr(out_external, 'raster'):
            spike_raster = out_external.raster(dt=dt_s, duration=len(sample_filtered)/target_sr)
        else:
            spike_raster = out_external
            
        if spike_raster.shape[1] != net_in_channels:
            if spike_raster.shape[0] == net_in_channels:
                spike_raster = spike_raster.T
            else:
                spike_raster = np.zeros((100, net_in_channels))
                
        return spike_raster, sample_filtered

    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_features_from_spikes(spike_raster, target_length=100):
    """
    Extract features from spike raster to match model input expectations
    This function needs to be adjusted based on your specific feature extraction
    """
    if spike_raster is None:
        return np.random.rand(target_length, net_in_channels) < 0.1
    
    if spike_raster.shape[0] > target_length:
        step = spike_raster.shape[0] // target_length
        features = spike_raster[::step, :]
        features = features[:target_length, :]
    else:
        features = np.zeros((target_length, net_in_channels))
        features[:spike_raster.shape[0], :] = spike_raster
    
    return features

current_last_car = None
current_power = 0
total_commercial = 0
total_normal = 0
power_history = []
cars_history = []
connection_active = False

async def simulate_audio_processing_with_list(websocket, file_list):
    """Simulate audio processing using a specific list of WAV files and then an empty input."""
    global total_commercial, total_normal, current_last_car, current_power, power_history, cars_history

    if not file_list:
        print("No audio files provided. Skipping simulation.")
        return

    print("Starting audio file simulation based on provided list.")
    
    for audio_file in file_list:
        try:

            print(f"Processing audio file: {os.path.basename(audio_file)}")
            
            spike_raster, _ = process_audio_file(audio_file)
            
            if spike_raster is None or spike_raster.size == 0:
                print(f"Failed to extract features from {audio_file}. Skipping.")
                await asyncio.sleep(1)
                continue
            
            features = extract_features_from_spikes(spike_raster, target_length=100)
            
            if hdk:
                if modMonitor:
                    output, _, r_d = modMonitor(input_data=features, record_power=True)
                else:
                    output, _, r_d = modSamna(features, record=True, record_power=True)
            else:
                output, _, r_d = modSim(features, record=True)

            prediction = np.argmax(np.sum(output, axis=0))
            
            power = 0.0
            if hdk:
                if xylo_board_name == 'XyloAudio3':
                    io_p = r_d.get('io_power', [])
                    analog_p = r_d.get('analog_power', [])
                    digital_p = r_d.get('digital_power', [])
                    if io_p:
                        power = np.mean(io_p) + np.mean(analog_p) + np.mean(digital_p)
                elif xylo_board_name == 'XyloAudio2':
                    io_p = r_d.get('io_power', [])
                    afe_core_p = r_d.get('afe_core_power', [])
                    afe_ldo_p = r_d.get('afe_ldo_power', [])
                    snn_core_p = r_d.get('snn_core_power', [])
                    if io_p:
                        power = np.mean(io_p) + np.mean(afe_core_p) + np.mean(afe_ldo_p) + np.mean(snn_core_p)

            if prediction != 2:
                current_last_car = class_names[prediction]
                if prediction == 0:
                    total_normal += 1
                elif prediction == 1:
                    total_commercial += 1

            current_power = power * 1e6
            
            power_history.append(current_power)
            cars_history.append({"normal": total_normal, "commercial": total_commercial})
            if len(power_history) > 60: power_history.pop(0)
            if len(cars_history) > 60: cars_history.pop(0)

            power_data = [{"time": f"{i}s", "power": p} for i, p in enumerate(power_history[-6:])]
            cars_data = [{"time": f"{i}s", "normal": d["normal"], "commercial": d["commercial"]} for i, d in enumerate(cars_history[-6:])]
            
            try:
                await websocket.send_json({
                    "lastCar": current_last_car,
                    "power": current_power,
                    "totalCommercial": total_commercial,
                    "totalNormal": total_normal,
                    "totalVehicles": total_commercial + total_normal,
                    "powerData": power_data,
                    "carsData": cars_data
                })
            except WebSocketDisconnect:
                print("Client disconnected during data send.")
                return

            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error processing audio file {audio_file}: {e}")
            await asyncio.sleep(1)

    # --- Simulate "empty" input after all files are processed ---
    print("All files processed. Simulating no audio input.")
    await asyncio.sleep(2)  # Give a slight pause
    try:
        await websocket.send_json({
            "lastCar": "None",
            "power": 0,
            "totalCommercial": total_commercial,
            "totalNormal": total_normal,
            "totalVehicles": total_commercial + total_normal,
            "powerData": [],
            "carsData": []
        })
        await websocket.close()
    except WebSocketDisconnect:
        print("Client disconnected while sending final data or closing.")
        return

@app.websocket("/ws/simulate")
async def websocket_simulate_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket (simulate) connected!")

    try:
        counts = {"car": 10, "background": 5, "cv": 20}

        print(f"Received request to simulate with counts: {counts}")

        selected_files = []
        for category, count in counts.items():
            if category in audio_files_by_category:
                files = audio_files_by_category[category]
                if len(files) < count:
                    print(f"Warning: Not enough files in category '{category}'. Found {len(files)}, requested {count}.")
                    selected_files.extend(files)
                else:
                    selected_files.extend(random.sample(files, count))
            else:
                print(f"Warning: Category '{category}' not found.")

        random.shuffle(selected_files)
        print(f"Total files selected for simulation: {len(selected_files)}")

        await simulate_audio_processing_with_list(websocket, selected_files)

    except WebSocketDisconnect:
        print("WebSocket (simulate) disconnected")
    except Exception as e:
        print(f"Error in WebSocket (simulate) connection: {e}")
    finally:
        print("WebSocket (simulate) connection closed")

@app.websocket("/ws")
async def websocket_hardware_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket (hardware) connected!")

    try:
        # Main processing loop for live/hardware input
        while True:
            T = 1000  # one second of recording

            if hdk:
                if modMonitor:
                    modMonitor.reset_state()
                    output, _, r_d = modMonitor(input_data=np.zeros((T, net_in_channels)), record_power=True)
                else:
                    input_spikes = np.random.rand(T, net_in_channels) < 0.1
                    output, _, r_d = modSamna(input_spikes, record=True, record_power=True)
            else:
                input_spikes = np.random.rand(T, net_in_channels) < 0.1
                output, _, r_d = modSim(input_spikes, record=True)

            prediction = np.argmax(np.sum(output, axis=0))
            power = 0
            if hdk and (modMonitor or modSamna):
                if xylo_board_name == 'XyloAudio3':
                    if 'io_power' in r_d.keys() and len(r_d['io_power']) > 0:
                        power = np.mean(r_d['io_power']) + np.mean(r_d['analog_power']) + np.mean(r_d['digital_power'])
                elif xylo_board_name == 'XyloAudio2':
                    if 'io_power' in r_d.keys() and len(r_d['io_power']) > 0:
                        power = np.mean(r_d['io_power']) + np.mean(r_d['afe_core_power']) + \
                               np.mean(r_d['afe_ldo_power']) + np.mean(r_d['snn_core_power'])

            global current_last_car, current_power, total_commercial, total_normal, power_history, cars_history

            if prediction != 2:
                current_last_car = {0: "Normal", 1: "Commercial"}.get(prediction, "Invalid")
                if prediction == 0:
                    total_normal += 1
                elif prediction == 1:
                    total_commercial += 1

            current_power = power * 1e6
            power_history.append(current_power)
            if len(power_history) > 60: power_history.pop(0)
            cars_history.append({"normal": total_normal, "commercial": total_commercial})
            if len(cars_history) > 60: cars_history.pop(0)

            power_data = [{"time": f"{i*10}s", "power": p} for i, p in enumerate(power_history[-6:])]
            cars_data = [{"time": f"{i*10}s", "normal": d["normal"], "commercial": d["commercial"]} 
                        for i, d in enumerate(cars_history[-6:])]

            await websocket.send_json({
                "lastCar": current_last_car,
                "power": current_power,
                "totalCommercial": total_commercial,
                "totalNormal": total_normal,
                "totalVehicles": total_commercial + total_normal,
                "powerData": power_data,
                "carsData": cars_data
            })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("WebSocket (hardware) disconnected")
    except Exception as e:
        print(f"Error in WebSocket (hardware) connection: {e}")
    finally:
        print("WebSocket (hardware) connection closed")

@app.get("/api/status")
async def get_status():
    return {
        "lastCar": current_last_car,
        "power": current_power,
        "totalCommercial": total_commercial,
        "totalNormal": total_normal,
        "totalVehicles": total_commercial + total_normal
    }