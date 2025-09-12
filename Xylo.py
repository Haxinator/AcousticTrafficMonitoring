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

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connected!")

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

    # load the best model
    net.load("Best_Model.json")
    net.seq.out_neurons = LIFTorch([3, 3], threshold=opt_thresholds)
    spec = None

    print(f"threshold used: {opt_thresholds}")

    # - Call the Xylo mapper on the extracted computational graph
    # - For XyloAudio 2
    if xylo_board_name == 'XyloAudio2':
        spec = xa2.mapper(net.as_graph(),  weight_dtype='float',
                        threshold_dtype='float', dash_dtype='float')
    # - For XyloAudio 3
    elif xylo_board_name == 'XyloAudio3':
        spec = xa3.mapper(net.as_graph(),  weight_dtype='float',
                        threshold_dtype='float', dash_dtype='float')

    # - Quantize the specification
    # spec.update(q.global_quantize(**spec))

    # you can also try channel-wise quantization
    unquantised_spec = spec.copy()
    spec.update(q.channel_quantize(**spec))
    # print(spec)

    # - Use rockpool.devices.xylo.config_from_specification to convert it to a hardware configuration
    # - For XyloAudio 2
    if xylo_board_name == 'XyloAudio2':
        config, is_valid, msg = xa2.config_from_specification(**spec)
    # - For XyloAudio 3
    elif xylo_board_name == 'XyloAudio3':
        config, is_valid, msg = xa3.config_from_specification(**spec)
    if not is_valid:
        # stop execution
        assert False, msg

    # - Use rockpool.devices.xylo.find_xylo_hdks to connect to an HDK
    xylo_hdk_nodes, modules, versions = find_xylo_hdks()
    print(f'HDK versions detected: {versions}')

    hdk = None
    modSamna = None
    modSim = None  # define globally


    for version, xylo in zip(versions, xylo_hdk_nodes):
        if version == "syns61201":
            hdk = xylo
        # - For XyloAudio 3
        elif version == "syns65302":
            hdk = xylo

    # - Use XyloSamna to deploy to the HDK
    if hdk:
        # - For XyloAudio 2
        if xylo_board_name == 'XyloAudio2':
            modSamna = xa2.XyloSamna(hdk, config, dt=dt)
        # - For XyloAudio 3
        elif xylo_board_name == 'XyloAudio3':
            modSamna = xa3.XyloSamna(hdk, config, dt=dt)

    # Use Simulation instead.
    if hdk is None:
        print('HDK not detected, running simulation.')
        # - For XyloAudio 2
        if xylo_board_name == 'XyloAudio2':
            modSim = xa2.XyloSim.from_config(config, dt=dt)
        # - For XyloAudio 3
        elif xylo_board_name == 'XyloAudio3':
            modSim = xa3.XyloSim.from_config(config, dt=dt)

    # - Evolve the network on the Xylo HDK
    # - `reset_state` is only needed for XyloAudio 2
    if xylo_board_name == 'XyloAudio2' and modSamna:
        modSamna.reset_state()

    # ---------- Generate some Poisson input (for testing)-------------------#
    T = 100
    f = 0.9
    input_spikes = np.random.rand(T, net_in_channels) < f
    TSEvent.from_raster(input_spikes, dt, name='Poisson input events').plot()

    # ----------- Load car, vehicle and bg sound (for testing)--------------#
    # dir = os.path.dirname(os.path.abspath('__file__'))
    # base_dir = os.path.join(dir, "DataPreprocessing", "small", "samples")
    # car_sample = np.load(os.path.join(base_dir, "car.npy"), allow_pickle=True)
    # cv_sample = np.load(os.path.join(base_dir, "cv.npy"), allow_pickle=True)
    # bg_sample = np.load(os.path.join(base_dir, "bg.npy"), allow_pickle=True)

    # samples = [car_sample, cv_sample, bg_sample]

    # Find indices for each class
    class_0_indices = np.where(y_val == 0)[0][:3]  # first 3 samples of class 0
    class_1_indices = np.where(y_val == 1)[0][:3]  # first 3 samples of class 1
    class_2_indices = np.where(y_val == 2)[0][:3]  # first 3 samples of class 2

    # Now interleave them: 0,1,2,0,1,2,0,1,2
    samples = []

    for i in range(3):
        samples.append(X_val[class_0_indices[i]])

    for i in range(3):
        samples.append(X_val[class_0_indices[i]])

    for i in range(3):
        samples.append(X_val[class_2_indices[i]])

    for i in range(3):
        samples.append(X_val[class_1_indices[i]])

    for i in range(3):
        samples.append(X_val[class_1_indices[i]])

    # for i in range(3):
    #     samples.append(X_val[class_0_indices[i]])
    #     samples.append(X_val[class_2_indices[i]])
    #     samples.append(X_val[class_1_indices[i]])
        

    # Convert to a NumPy array
    samples = np.array(samples)
    print("Shape of selected samples:", samples.shape)


    # ----------------SEND DATA TO FRONTEND----------------#
    # To run the backend, type "uvicorn main:app --reload --port 3000" in the terminal

    # Global state (accessible by API)
    current_last_car = "None"
    current_power = 0

    # # Create API
    # @app.get("/api/lastcar")
    # def get_last_car():
    #     return {"lastCar": current_last_car}

    # @app.get("/api/power")
    # def get_power():
    #     return {"power": current_power}


    # -----------------------RUN--------------------------#
    # need to add microphone stuff here when actually running in free-inference (input from mic)
    freeInferenceMode = True
    modMonitor = None

    if freeInferenceMode:
        print('Free Inference Mode enabled.')
        # - Use XyloMonitor to deploy to the HDK
        output_mode = "Vmem"
        amplify_level = "low"
        hibernation = False
        DN = True
        T = 100

        # - For XyloAudio 2
        # - For XyloAudio 2 you need to wait 45s until the AFE auto-calibration is done
        if xylo_board_name == 'XyloAudio2':
            modMonitor = xa2.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode,
                                        amplify_level=amplify_level, hibernation_mode=hibernation, divisive_norm=DN)

        # - For XyloAudio 3
        # - XyloAudio 3 does not have the amplify_level parameter and does not do AFE auto-calibration
        elif xylo_board_name == 'XyloAudio3':
            modMonitor = xa3.XyloMonitor(
                hdk, config, dt=dt, output_mode=output_mode, hibernation_mode=hibernation, dn_active=DN)

    list_of_detected_cars = []

    if modMonitor:
        # loop once only
        samples = range(100)
        T = 1000 # one second of recording then output result



    
    fignum = 0
    power = -1

    for sample in samples:
        print("Loop iteration started")
        fignum = fignum + 1
        # dynamically choose whether to sim or run on hdk with or without free-inference mode.
        if hdk:
            if modMonitor:
                # - Perform inference on the Xylo board
                # - The following line will evolve XyloMonitor for T time steps.
                # - Keep in mind that this mode is using the microphone as input, hence the output might change according to the ambience noise
                modMonitor.reset_state()
                output, _, r_d = modMonitor(input_data=np.zeros(
                    (T, net_in_channels)), record_power=True)
            else:
                output, _, r_d = modSamna(sample, record=True, record_power=True)
        else:
            output, _, r_d = modSim(sample, record=True)

        print(output)

        prediction = np.argmax(np.sum(output, axis=0))

        print(prediction)
        list_of_detected_cars.append(prediction)

        # out_old, _, _ = net(torch.from_numpy(sample))

        # out_old = out_old[:, 30:, :].mean(dim=1)  # using skip and mean value vmem

        # floatingmodel_prediction = torch.argmax(out_old, 1)
        # print(floatingmodel_prediction)

        if modMonitor or modSamna:
            # Measure power in Watts
            if xylo_board_name == 'XyloAudio3':
                if 'io_power' in r_d.keys() and len(r_d['io_power'])>0:
                    power = np.mean(
                        r_d['io_power']) + np.mean(r_d['analog_power']) + np.mean(r_d['digital_power'])
                else:
                    power = 0.
                print(f"Total Power Consumption: {power * 1e6:.0f} µW")
            if xylo_board_name == 'XyloAudio2':
                power = np.mean(r_d['io_power']) + np.mean(r_d['afe_core_power']) + \
                    np.mean(r_d['afe_ldo_power']) + np.mean('snn_core_power')
                print(f"Total Power Consumption: {power * 1e6:.0f} µW")

        # Send updates to frontend
        if prediction != 2: # only update last car if it is non-background
            current_last_car = {0: "Normal", 1: "Commercial"}.get(prediction, "Invalid")
        await ws.send_json({"lastCar": current_last_car, "power": power})
        await asyncio.sleep(1)  # allow async event loop to handle WS

        #---------------------PLOT OUTPUT-------------------#
        # # - Plot some internal state variables
        # plt.figure()
        # plt.imshow(r_d['Spikes'].T, aspect='auto', origin='lower')
        # plt.title('Hidden spikes')
        # plt.ylabel('Channel')
        # plt.savefig(os.path.join('plots', f'Spikes{fignum}.png'))

        # plt.figure()
        # if hdk:
        #     TSContinuous(r_d['times'], r_d['Isyn'],
        #                 name='Hidden synaptic currents').plot(stagger=127)
        # else:
        #     TSContinuous.from_clocked(
        #         r_d['Isyn'], dt, name='Hidden synaptic currents').plot(stagger=127)
        # plt.savefig(os.path.join('plots', f'SynaticCurrents{fignum}.png'))

        # plt.figure()
        # if hdk:
        #     TSContinuous(r_d['times'], r_d['Vmem'],
        #                 name='Hidden membrane potentials').plot(stagger=127)
        # else:
        #     TSContinuous.from_clocked(
        #         r_d['Vmem'], dt, name='Hidden membrane potentials').plot(stagger=127)
        # plt.savefig(os.path.join('plots', f'MembranePotential{fignum}.png'))
        # print('Figures Saved')

        # # ---- PLOT QUANTISATION---------------------------#
        # fig = plt.figure(figsize=(16, 10))
        # ax = fig.add_subplot(321)
        # ax.set_title("w_inp float")
        # ax.hist(np.ravel(unquantised_spec["weights_in"]
        #         [unquantised_spec["weights_in"] != 0]), bins=2**8)

        # ax = fig.add_subplot(322)
        # ax.set_title("w_inp quant")
        # ax.hist(np.ravel(spec["weights_in"][spec["weights_in"] != 0]), bins=2**8)

        # ax = fig.add_subplot(323)
        # ax.set_title("w_rec float")
        # ax.hist(np.ravel(unquantised_spec["weights_rec"]
        #         [unquantised_spec["weights_rec"] != 0]), bins=2**8)

        # ax = fig.add_subplot(324)
        # ax.set_title("w_rec quant")
        # ax.hist(
        #     np.ravel(spec["weights_rec"][spec["weights_rec"] != 0]), bins=2**8
        # )

        # ax = fig.add_subplot(325)
        # ax.set_title("w_out float")
        # ax.hist(np.ravel(unquantised_spec["weights_out"]
        #         [unquantised_spec["weights_out"] != 0]), bins=2**8)

        # ax = fig.add_subplot(326)
        # ax.set_title("w_out quant")
        # ax.hist(
        #     np.ravel(spec["weights_out"][spec["weights_out"] != 0]), bins=2**8
        # )

        # plt.savefig(os.path.join('plots', f'QuantisationComparison.png'))
    # free memory just in case
    if hdk is None:
        del modSim
    else:
        if modMonitor:
            del modMonitor

        del modSamna

    await ws.close()

