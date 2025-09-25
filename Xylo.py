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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
vmem_file = os.path.join("vmem_calibration", "all_vmems.npy")
labels_file = os.path.join("vmem_calibration", "all_labels.npy")
all_vmems = []
all_labels = []

if os.path.exists(vmem_file) and os.path.exists(labels_file):
    print("Found existing Vmems and labels. Loading from file...")
    all_vmems = np.load(vmem_file)
    all_labels = np.load(labels_file)
else:
    print("No saved Vmems found. Running network to collect Vmems...")
    all_vmems = []
    all_labels = []
    with torch.no_grad():
        for events, labels in val_dl:
            events, labels = events.to(device), labels.to(device)
            out, _, _ = vmem_net(events)
            all_vmems.append(out[:, skip_window:, :].cpu().numpy())
            all_labels.append(labels.cpu().numpy().tolist())

all_vmems = np.concatenate(all_vmems, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
print("----- Vmem collection complete ------")
print("vmem shape:", all_vmems.shape, "labels shape:", all_labels.shape)

""" np.save(vmem_file, all_vmems)
np.save(labels_file, all_labels) """

opt_thresholds = (np.float64(0.5), np.float64(1.0), np.float64(-5.0))

print(
    f"\nSelected optimal thresholds based on manual comparison: {opt_thresholds}")

y_true_bin = label_binarize(all_labels, classes=np.arange(n_labels))

# --------- ROC using Vmem values ---------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
vmem_scores = all_vmems.mean(axis=1)

for i, cname in enumerate(class_names):
    scores = vmem_scores[:, i]

    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={roc_auc:.2f})")

    step = max(1, len(thresholds) // 10)
    for j in range(0, len(thresholds), step):
        plt.text(fpr[j] + 0.01, tpr[j] - 0.01, f"{thresholds[j]:.2f}",
                 color="blue", fontsize=7, alpha=0.7)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC per Label (Vmem outputs)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# --------- ROC using Spike counts  ---------
plt.subplot(1, 2, 2)

spike_counts_for_roc = np.sum((all_vmems >= 0).astype(int), axis=1)

for i, cname in enumerate(class_names):
    scores = spike_counts_for_roc[:, i]

    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{cname} (AUC={roc_auc:.2f})')

    step = max(1, len(thresholds) // 10)
    for j in range(0, len(thresholds), step):
        plt.text(fpr[j] + 0.01, tpr[j] - 0.01, f'{thresholds[j]:.0f}',
                 color='blue', fontsize=7, alpha=0.7)

plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Using Spike Counts)')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig(os.path.join("plots", "ROC_vmem_vs_spikes_fixed.png"))
plt.show()

# --------- Vehicle combined (with Background) ---------

y_true_vehicle = np.logical_or(y_true_bin[:, 0], y_true_bin[:, 1]).astype(int)

vmem_scores_vehicle = np.maximum(vmem_scores[:, 0], vmem_scores[:, 1])
vmem_scores_background = vmem_scores[:, 2] 

spike_counts_vehicle = np.maximum(spike_counts_for_roc[:, 0], spike_counts_for_roc[:, 1])

plt.figure(figsize=(8, 6)) # Increased figure size for better visibility

# --- Plot Vmem-based Vehicle ROC ---
fpr_vmem_vehicle, tpr_vmem_vehicle, thresholds_vmem_vehicle = roc_curve(y_true_vehicle, vmem_scores_vehicle)
roc_auc_vmem_vehicle = auc(fpr_vmem_vehicle, tpr_vmem_vehicle)

plt.plot(fpr_vmem_vehicle, tpr_vmem_vehicle, lw=3, color='darkorange',
         label=f'Vehicle (Combined Vmem) (AUC={roc_auc_vmem_vehicle:.2f})')

step_vmem_vehicle = max(1, len(thresholds_vmem_vehicle) // 10)
for j in range(0, len(thresholds_vmem_vehicle), step_vmem_vehicle):
    plt.text(fpr_vmem_vehicle[j] + 0.01, tpr_vmem_vehicle[j] - 0.01, f"{thresholds_vmem_vehicle[j]:.2f}",
             color="blue", fontsize=7, alpha=0.7)

# --- Plot Vmem-based Background ROC ---
fpr_vmem_bg, tpr_vmem_bg, thresholds_vmem_bg = roc_curve(y_true_bin[:, 2], vmem_scores_background)
roc_auc_vmem_bg = auc(fpr_vmem_bg, tpr_vmem_bg)

plt.plot(fpr_vmem_bg, tpr_vmem_bg, lw=2, color='gray', linestyle=':',
         label=f'Background (Individual Vmem) (AUC={roc_auc_vmem_bg:.2f})')

# Add threshold labels for Background scores (using a different color for clarity)
step_vmem_bg = max(1, len(thresholds_vmem_bg) // 10)
for j in range(0, len(thresholds_vmem_bg), step_vmem_bg):
    plt.text(fpr_vmem_bg[j] + 0.01, tpr_vmem_bg[j] - 0.01, f"{thresholds_vmem_bg[j]:.2f}",
             color="red", fontsize=7, alpha=0.7)


# --- C. Plot Spike-based Vehicle ROC ---
fpr_spikes, tpr_spikes, thresholds_spikes = roc_curve(y_true_vehicle, spike_counts_vehicle)
roc_auc_spikes = auc(fpr_spikes, tpr_spikes)

plt.plot(fpr_spikes, tpr_spikes, lw=2, color='green', linestyle='--',
         label=f'Vehicle (Combined Spikes) (AUC={roc_auc_spikes:.2f})')

step_spikes = max(1, len(thresholds_spikes) // 10)
for j in range(0, len(thresholds_spikes), step_spikes):
    plt.text(fpr_spikes[j] + 0.01, tpr_spikes[j] - 0.01, f'{thresholds_spikes[j]:.0f}',
             color='darkgreen', fontsize=7, alpha=0.7)

# --- Final Plotting details ---
plt.plot([0, 1], [0, 1], linestyle="--", color="black",
         alpha=0.5, label='Chance (AUC = 0.5)')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve: Vehicle Detection vs. Background Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig(os.path.join("plots", "ROC_Vehicle_Background_Comparison.png"))
plt.show()

scores = spike_counts_for_roc[:, 0]
fpr, tpr, thresholds = roc_curve(y_true_bin[:, 0], scores)
distances = (fpr - 0)**2 + (tpr - 1)**2
optimal_index = np.argmin(distances)
optimal_threshold_class0 = thresholds[optimal_index]
print(f"Optimal threshold for Class 0: {optimal_threshold_class0}")

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
        spec = xa2.mapper(net.as_graph(), weight_dtype='float',
                          threshold_dtype='float', dash_dtype='float')
    elif xylo_board_name == 'XyloAudio3':
        spec = xa3.mapper(net.as_graph(), weight_dtype='float',
                          threshold_dtype='float', dash_dtype='float')

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

    if hdk and True:
        output_mode = "Spike"
        amplify_level = "low"
        hibernation = False
        DN = True
        T = 100
        if xylo_board_name == 'XyloAudio2':
            modMonitor = xa2.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode,
                                         amplify_level=amplify_level, hibernation_mode=hibernation, divisive_norm=DN)
        elif xylo_board_name == 'XyloAudio3':
            modMonitor = xa3.XyloMonitor(
                hdk, config, dt=dt, output_mode=output_mode, hibernation_mode=hibernation, dn_active=DN)

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
                output, _, r_d = modMonitor(
                    input_data=input_data, record_power=True)
            elif modSamna:
                output, _, r_d = modSamna(
                    input_data, record=True, record_power=True)
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
                    power += np.mean(r_d['analog_power']) + \
                        np.mean(r_d['digital_power'])
                elif xylo_board_name == 'XyloAudio2':
                    power += np.mean(r_d['afe_core_power']) + np.mean(
                        r_d['afe_ldo_power']) + np.mean(r_d['snn_core_power'])

            if prediction != 2:
                current_last_car = {0: "Normal", 1: "Commercial"}.get(
                    prediction, "Invalid")
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
