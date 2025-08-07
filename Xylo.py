#Use SynNet to start. We Will need to develop our own architecture later.
#Still able to make some small adjustments to the network though.
from rockpool.nn.networks import SynNet
from rockpool.nn.modules import LIFExodus
from rockpool.transform import quantize_methods as q
from rockpool import TSEvent, TSContinuous
from rockpool.devices.xylo import find_xylo_hdks

# - Import torch training utilities
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

#for paths
import os
import logging

#visualisation
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

n_labels = 3
net_in_channels = 16
dt = 1e-3

net = SynNet(
    # neuron_model = LIFExodus,
   # output="vmem",                         # Use the membrane potential as the output of the network.
    p_dropout=0.1,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.
    #NOTE if not using SynNet then they will be by default.

    n_channels = net_in_channels,                        # Number of input channels (always 16)
    n_classes = n_labels,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = [24, 24, 24],      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = [2, 4, 8],   # Number of time constants in each hidden layer (taken from tutorial)
)

spec = None

# - Call the Xylo mapper on the extracted computational graph
# - For XyloAudio 2
if xylo_board_name == 'XyloAudio2':
    spec = xa2.mapper(net.as_graph(),  weight_dtype='float', threshold_dtype='float', dash_dtype='float')
# - For XyloAudio 3
elif xylo_board_name == 'XyloAudio3':
    spec = xa3.mapper(net.as_graph(),  weight_dtype='float', threshold_dtype='float', dash_dtype='float')

# - Quantize the specification
# spec.update(q.global_quantize(**spec))

# you can also try channel-wise quantization
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
    #stop execution
    assert False, msg 

# - Use rockpool.devices.xylo.find_xylo_hdks to connect to an HDK
xylo_hdk_nodes, modules, versions = find_xylo_hdks()
print(f'HDK versions detected: {versions}')

hdk = None
modSamna = None

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
        modSamna = xa2.XyloSamna(hdk, config, dt = dt)
    # - For XyloAudio 3
    elif xylo_board_name == 'XyloAudio3':
        modSamna = xa3.XyloSamna(hdk, config, dt = dt)

#Use Simulation instead.
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
if xylo_board_name == 'XyloAudio2':
    modSamna.reset_state()

# ---------- Generate some Poisson input (for testing)-------------------#
T = 100
f = 0.9
input_spikes = np.random.rand(T, net_in_channels) < f
TSEvent.from_raster(input_spikes, dt, name = 'Poisson input events').plot()

# ----------- Load car, vehicle and bg sound (for testing)--------------#
dir = os.path.dirname(os.path.abspath('__file__'))
base_dir = os.path.join(dir, "DataPreprocessing", "small", "samples")
car_sample = np.load(os.path.join(base_dir,"car.npy"), allow_pickle=True)
cv_sample = np.load(os.path.join(base_dir,"cv.npy"), allow_pickle=True)
bg_sample = np.load(os.path.join(base_dir,"bg.npy"), allow_pickle=True)

samples = [car_sample, cv_sample, bg_sample]

#-----------------------RUN--------------------------#
#need to add microphone stuff here when actually running in free-inference (input from mic)
freeInferenceMode = False
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
        modMonitor = xa2.XyloMonitor(hdk, config, dt=dt, output_mode=output_mode, amplify_level=amplify_level, hibernation_mode=hibernation, divisive_norm=DN)

    # - For XyloAudio 3
    # - XyloAudio 3 does not have the amplify_level parameter and does not do AFE auto-calibration
    elif xylo_board_name == 'XyloAudio3':
        modMonitor = xa3.XyloMonitor(hdk, config, dt = dt, output_mode=output_mode, hibernation_mode=hibernation, dn_active=DN)

list_of_detected_cars = []

if modMonitor:
    #loop once only
    samples = []

fignum = 0

for sample in samples:
    fignum = fignum + 1
    #dynamically choose whether to sim or run on hdk with or without free-inference mode.
    if hdk:
        if modMonitor:
            # - Perform inference on the Xylo board
            # - The following line will evolve XyloMonitor for T time steps.
            # - Keep in mind that this mode is using the microphone as input, hence the output might change according to the ambience noise
            output, _, r_d = modMonitor(input_data=np.zeros((T, net_in_channels)))
        else:
            output, _, r_d = modSamna(sample, record = True)
    else:
        output, _, r_d = modSim(sample, record = True)

    print(output)

    prediction = np.argmax(np.sum(output, axis = 0))

    print(prediction)
    list_of_detected_cars.append(prediction)

    #----------------SEND DATA TO FRONTEND----------------#

    #ERIC TO DO

    #---------------------PLOT OUTPUT-------------------#
    # - Plot some internal state variables
    plt.figure()
    plt.imshow(r_d['Spikes'].T, aspect = 'auto', origin = 'lower')
    plt.title('Hidden spikes')
    plt.ylabel('Channel')
    plt.savefig(os.path.join('plots', f'Spikes{fignum}.png'))

    plt.figure()
    if hdk:
        TSContinuous(r_d['times'], r_d['Isyn'], name = 'Hidden synaptic currents').plot(stagger = 127)
    else:
        TSContinuous.from_clocked(r_d['Isyn'], dt, name = 'Hidden synaptic currents').plot(stagger = 127)
    plt.savefig(os.path.join('plots', f'SynaticCurrents{fignum}.png'))

    plt.figure()
    if hdk:
        TSContinuous(r_d['times'], r_d['Vmem'], name = 'Hidden membrane potentials').plot(stagger = 127)
    else:
        TSContinuous.from_clocked(r_d['Vmem'], dt, name = 'Hidden membrane potentials').plot(stagger = 127)
    plt.savefig(os.path.join('plots', f'MembranePotential{fignum}.png'))
    print('Figures Saved')

# free memory just in case
if hdk is None:
    del modSim
else:
    if modMonitor:
        del modMonitor
    
    del modSamna
        