#NOTES
'''
In 100 epochs:
 - Using vmem as output increases accuracy by about 8 percent.
 - Using MSE increases accuracy by about 2 percent.
'''

#Use SynNet to start. We Will need to develop our own architecture later.
#Still able to make some small adjustments to the network though.
from rockpool.nn.networks import SynNet
# - Import torch training utilities
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
#data transformations and caching.
import tonic
import tonic.transforms as T
from tonic import datasets

#gives error
from tqdm import trange

#visualisation
import matplotlib.pyplot as plt

#array handling
import numpy as np

# A manual seed ensures repeatability
torch.manual_seed(1234) 

#model tracking
import wandb
wandb.login() #create an account to retrieve the api key it requests

# Define constants for the model training process. Note some variables have been commented out as they are not currently in use
n_epochs = 5        #need about 37000 epochs to get good accuracy (?)
# loss_method = 'mse'
loss_method = 'cross_entropy'
optimiser_lr = 1e-3
train_batch_size = 128
train_num_workers = 0
test_batch_size = 256
test_num_workers = 0
# n_batches = 1
# n_time = 100

#ONLY FOR DUMMY DATASET----------------------------------------------
# Define constants for the SHD dummy data characteristics
shd_timestep = 1e-6
shd_channels = 700
shd_classes = 20

# Define constants for the dataset transformations and SynNet architecture
# n_labels = 3         # consider subsetting for 3 classes
n_labels = 20
net_channels = 16
net_dt = 10e-3
sample_T = 250
net_p_dropout = 0.1
# net_output_type = "vmem"
net_output_type = "spike"
net_size_hidden_layers = [24, 24, 24]      # Number of neurons in each hidden layer (taken from tutorial)
net_time_constants_per_layer = [2, 4, 8]   # Number of time constants in each hidden layer (taken from tutorial)


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="cits5551-traffic-monitoring",
    # Set the wandb project where this run will be logged.
    project="shd-dummy-model",
    # Track hyperparameters and run metadata.
    config={
        "architecture": "SynNet",
        "dataset": "SHD",
        "shd_timestep": shd_timestep,
        "shd_channels": shd_channels,
        "shd_classes": shd_classes,
        "epochs": n_epochs,
        "loss_method": loss_method,
        "optimiser_lr": optimiser_lr,
        "train_batch_size": train_batch_size,
        "train_num_workers": train_num_workers,
        "test_batch_size": test_batch_size,
        "test_num_workers": test_num_workers,
        "n_labels": n_labels,
        "net_channels": net_channels,
        "net_dt": net_dt,
        "sample_T": sample_T,
        "net_p_dropout": net_p_dropout,
        "net_output_type": net_output_type,

        "size_hidden_layers": net_size_hidden_layers,
        "time_constants_per_layer": net_time_constants_per_layer
    },
)

# Code for visualisation (from tutorial)
#events, label = train_data[1]
#times = events['t'] * shd_timestep
#channels = events['x']
#plt.plot(times, channels, '|')
#plt.show()

class ToRaster():
    def __init__(self, encoding_dim, sample_T = 100):
        self.encoding_dim = encoding_dim
        self.sample_T = sample_T

    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        return tensor[:self.sample_T,:]
    
transform = T.Compose([
    #downsamples files to 16 channels and reduces timstep
    T.Downsample(time_factor=shd_timestep/net_dt, spatial_factor= net_channels/shd_channels),
    #rasterise
    # T.ToFrame(sensor_size=(net_channels, 1, 1), time_window=1),
    # #convert to tensor
    # torch.Tensor,
    # #trim in time
    # lambda m: torch.squeeze(m)[:sample_T:, :],

    ToRaster(net_channels, sample_T = sample_T),
    torch.Tensor,
])

#loads SHD dataset and puts it into ./data folder
#applies transformation pipeline described above
train_data = tonic.datasets.SHD('./data', transform=transform) #Download SHD dataset into data folder (in SHD subfolder)

raster, label = train_data[1]

#show a downsampled audio file.
#plt.imshow(raster.T, aspect='auto')
#plt.show()

#class which subsets dataset labels (we only want to use 3 digits of the dataset)
class SubsetClasses(torch.utils.data.Dataset):
    def __init__(self, dataset, matchinglabels):
        indicies = []

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in matchinglabels:
                indicies.append(idx)

        self._subset_ds = torch.utils.data.Subset(dataset, indicies)
        self._len = len(indicies)

    def __getitem__(self, index):
        return self._subset_ds[index]
    
    def __len__(self):
        return self._len
    
dataloader_kwargs = dict(
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=train_num_workers,
)

train_dl = torch.utils.data.DataLoader(
    # train_data,
    tonic.DiskCachedDataset(
        dataset=SubsetClasses(train_data, range(n_labels)),
        # dataset=train_data,
        cache_path=f"cache/{train_data.__class__.__name__}/train/{net_channels}/{net_dt}",
        reset_cache = True,
    ),
    **dataloader_kwargs
)
############# END TEST DATA------------------------------------------

#in reality need to use dataloader to open file.

# - Build a simple SynNet with three hidden layers
# Need to experiment with number of layers, neurons and time constants.
net = SynNet(
    #Dylan recommended this since it will make the optimiser work better.
    # output="vmem",                         # Use membrane potential as the output of the network.
    p_dropout=net_p_dropout,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.
    #NOTE if not using SynNet then they will be by default.

    n_channels = net_channels,                        # Number of input channels (always 16)
    n_classes = n_labels,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = net_size_hidden_layers,      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = net_time_constants_per_layer,   # Number of time constants in each hidden layer (taken from tutorial)
)

wandb.watch(net, log="all")

# Change network output type if needed (not sure if there's a more elegant way to do this)
# if net_output_type == "spike":
#     net.output = "spike"

#print(net)
#show trainable parameters (time constants should be empty dict otherwise they will be trained)
#print(net.parameters)

#pass parameters to optimise and the learning rate (lr) respectively to adam.
optimiser = Adam(net.parameters().astorch(), lr=optimiser_lr)

#Dylan recommends using MSE for loss
#results in slightly higher accuracy compared to other functions
if loss_method == 'mse':
    loss_function = MSELoss()
elif loss_method == 'cross_entropy':
    loss_function = CrossEntropyLoss()

#very basic barebones training loop.
#no constraints used
#no regularisations used
#no validation accuracy used
#no acceleration used
#GPU much faster.

#Training loop
#trange gives cool progress bar
for epoch in trange(n_epochs):

    #for calculating accuracy
    correct = 0
    total = 0

    k = 0

    #batching done by torch/tonic dataloader
    for events, labels in train_dl:
        k += 1
        print("batch: " + str(k) + "output shape: ")

        optimiser.zero_grad()

        output, _, _ = net(events)
        # output, _, _ = net(torch.Tensor(events).float())

        # DEBUGGING
        # print(output)
        # print(output.size())
        # print(labels)

        #number of spikes on output channels gives prediction
        #target channel should have most number of spikes

        #TODO MSE LOSS LOGIC (vmem)
        # loss = 0
        # vmem_val = output[:,-1,:]
        # print(vmem_val)
        # loss += loss_function(output, targets_spikes)

        #TODO MSE LOSS LOGIC
        # sum = torch.cumsum(output, dim=1)
        # predicted = torch.argmax(sum[:,-1,:], 1)
        # print(predicted)
        # predicted = predicted.float()
        # predicted.requires_grad_()
        # loss = loss_function(predicted, labels.float())
        
        # ---- don't uncomment:
        # print(sum[:,-1,:])
        # print(predicted)
        # loss = loss_function(sum[:,-1,:], labels)
        # print(predicted)
        # print(labels.float())
        #pred = torch.sum(output, dim=1)

        # CROSS ENTROPY LOSS LOGIC
        sum = torch.cumsum(output, dim=1)
        loss = loss_function(sum[:,-1,:], labels)
        print(torch.bincount(labels))

        loss.backward()
        optimiser.step()

        # For each sample in the batch, determine the class with the most spikes at the last time step
        predicted = torch.argmax(sum[:,-1,:], 1)
        print(predicted)

        #to get total number of datafiles and calculate number of correct guesses
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate and log accuracy on the train set for this epoch
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{n_epochs}, loss {loss.item():.2f}, accuracy {accuracy:.2f}')
    wandb.log({"train_loss": accuracy})

#with our trainned net make a predicition on a file
'''
events, label = train_data[4]

out, _, rd = net(events, record = True)

time, channels = torch.where(out[0])

#plot predition.
plt.plot(time * net_dt, channels, '|')
plt.xlim([0,1])
plt.ylim([-1,3])
plt.plot(0.01, label, '>', ms=18) #show highlight correct label on plot
plt.show()

train_dl = torch.utils.data.DataLoader(
    tonic.DiskCachedDataset(
        dataset=SubsetClasses(train_data, range(3)),
        cache_path=f"cache/{train_data.__class__.__name__}/train/{net_channels}/{net_dt}",
        reset_cache = False,
    ),
    **dataloader_kwargs
)
'''

# Test our model on unseen data and report accuracy
test_data = datasets.SHD('./data', train=False, transform=transform)
test_data = SubsetClasses(test_data, range(n_labels))
test_dl = DataLoader(test_data, num_workers=test_num_workers, batch_size=test_batch_size,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0

    for events, labels in test_dl:
        output, _, _ = net(torch.Tensor(events).float())

        sum = torch.cumsum(output, dim=1)

        predicted = torch.argmax(sum[:,-1,:], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy_test = (correct/total)*100

print(f"Test Accuracy: {accuracy_test:.3f}%")

wandb.log({"test_loss": accuracy_test})

# print(net.parameters)
# print(net.parameters().astorch())

wandb.finish()