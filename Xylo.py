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
# --- Hardware and Network Initialization ---
hdk_initialized = False
modSamna = None
modSim = None
modMonitor = None
hdk = None

def initialize_hardware():
    global hdk_initialized, modSamna, modSim, modMonitor, hdk

    if hdk_initialized:
        print("Hardware already initialized.")
        return

    print("Initializing hardware and network...")
    
    net = SynNet(
        p_dropout=0.2,
        n_channels=net_in_channels,
        n_classes=n_labels,
        size_hidden_layers=[24, 24, 24],
        time_constants_per_layer=[2, 4, 8],
    )
    
    net.load(model_path)
    net.seq.out_neurons = LIFTorch([3, 3], threshold=opt_thresholds)
    
    spec = None
    if xylo_board_name == 'XyloAudio2':
        spec = xa2.mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
    elif xylo_board_name == 'XyloAudio3':
        spec = xa3.mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

    unquantised_spec = spec.copy()
    spec.update(q.channel_quantize(**spec))
    
    config, is_valid, msg = (xa2.config_from_specification(**spec) if xylo_board_name == 'XyloAudio2'
                             else xa3.config_from_specification(**spec))
    assert is_valid, msg
    
    xylo_hdk_nodes, modules, versions = find_xylo_hdks()
    print(f'HDK versions detected: {versions}')

    for version, xylo in zip(versions, xylo_hdk_nodes):
        if version == "syns61201" or version == "syns65302":
            hdk = xylo
            break

    if hdk:
        if xylo_board_name == 'XyloAudio2':
            modSamna = xa2.XyloSamna(hdk, config, dt=dt)
        elif xylo_board_name == 'XyloAudio3':
            modSamna = xa3.XyloSamna(hdk, config, dt=dt)
    else:
        print('HDK not detected, running simulation.')
        if xylo_board_name == 'XyloAudio2':
            modSim = xa2.XyloSim.from_config(config, dt=dt)
        elif xylo_board_name == 'XyloAudio3':
            modSim = xa3.XyloSim.from_config(config, dt=dt)

    if hdk and True: # assuming freeInferenceMode is always true for this path
        output_mode = "Spike"
        amplify_level = "low"
        hibernation = False
        DN = True
        T = 100
        if xylo_board_name == 'XyloAudio2':
            modMonitor = xa2.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode,
                                         amplify_level=amplify_level, hibernation_mode=hibernation, divisive_norm=DN)
        elif xylo_board_name == 'XyloAudio3':
            modMonitor = xa3.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode, hibernation_mode=hibernation, dn_active=DN)

    hdk_initialized = True
    print("Hardware initialization complete.")

initialize_hardware()

# ----------------------------------- WebSocket Endpoint ------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global modSamna, modSim, modMonitor, hdk
    await ws.accept()
    print("WebSocket connected!")

    total_normal = 0
    total_commercial = 0
    current_last_car = "None"

    historical_power = []
    historical_normal = []
    historical_commercial = []

    class_0_indices = np.where(y_val == 0)[0][:3]
    class_1_indices = np.where(y_val == 1)[0][:3]
    class_2_indices = np.where(y_val == 2)[0][:3]
    test_samples = []
    test_samples.extend(X_val[class_0_indices])
    test_samples.extend(X_val[class_1_indices])
    test_samples.extend(X_val[class_2_indices])

    sample_index = 0

    try:
        while True:

            input_tensor = test_samples[sample_index % len(test_samples)]
            input_data = input_tensor.numpy()
            sample_index += 1

            if modMonitor:
                modMonitor.reset_state()
                output, _, r_d = modMonitor(input_data=input_data, record_power=True)
            elif modSamna:
                output, _, r_d = modSamna(input_data, record=True, record_power=True)
            elif modSim:
                print("Running in simulation mode with test data.")
                output, _, r_d = modSim(input_data, record=True)

            print(output)

            prediction_threshold = 500
            prediction_sum = np.sum(output, axis=0)
            prediction = 2

            if prediction_sum[0] > prediction_threshold:
                prediction = 0
            elif prediction_sum[1] > prediction_threshold:
                prediction = 1

            print(f"Prediction sum:{prediction_sum}, Prediction:{prediction}")

            power = 0
            if hdk and 'io_power' in r_d:
                power = np.mean(r_d['io_power'])
                if xylo_board_name == 'XyloAudio3':
                    power += np.mean(r_d['analog_power']) + np.mean(r_d['digital_power'])
                elif xylo_board_name == 'XyloAudio2':
                    power += np.mean(r_d['afe_core_power']) + np.mean(r_d['afe_ldo_power']) + np.mean(r_d['snn_core_power'])

            if prediction != 2:
                current_last_car = {0: "Normal", 1: "Commercial"}.get(prediction, "Invalid")
                if prediction == 0:
                    total_normal += 1
                elif prediction == 1:
                    total_commercial += 1

            historical_power.append(int(power*1e6))
            historical_normal.append(total_normal)
            historical_commercial.append(total_commercial)

            await ws.send_json({
                "lastCar": current_last_car,
                "power": int(power*1e6),
                "totalNormal": total_normal,
                "totalCommercial": total_commercial,
                "totalVehicles": total_normal + total_commercial,
                "historicalPower": historical_power,
                "historicalNormal": historical_normal,
                "historicalCommercial": historical_commercial
            })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("WebSocket closing.")