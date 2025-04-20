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

#need about 37000 epochs to get good accuracy
n_epochs = 500
n_batches = 1
n_time = 100
n_labels = 3

#ONLY FOR TEST----------------------------------------------
shd_timestep = 1e-6
shd_channels = 700
shd_classes = 20
net_channels = 16
net_dt = 10e-3
sample_T = 250
#events, label = train_data[1]
#times = events['t'] * shd_timestep
#channels = events['x']
#plt.plot(times, channels, '|')
#plt.show()

transform = T.Compose([
    #downsamples files to 16 channels and reduces timstep
    T.Downsample(time_factor=shd_timestep/net_dt, spatial_factor= net_channels/shd_channels),
    #rasterise
    T.ToFrame(sensor_size=(net_channels, 1, 1), time_window=1),
    #convert to tensor
    torch.Tensor,
    #trim in time
    lambda m: torch.squeeze(m)[:sample_T:, :],
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
    batch_size = 128,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=0,
)

train_dl = torch.utils.data.DataLoader(
    tonic.DiskCachedDataset(
        dataset=SubsetClasses(train_data, range(3)),
        cache_path=f"cache/{train_data.__class__.__name__}/train/{net_channels}/{net_dt}",
        reset_cache = False,
    ),
    **dataloader_kwargs
)
############# END TEST DATA------------------------------------------



#in reality need to use dataloader to open file.

# - Build a simple SynNet with three hidden layers
# Need to experiment with number of layers, neurons and time constants.
net = SynNet(
    #Dylan recommended this since it will make the optimiser work better.
    output="vmem",                         # Use the membrane potential as the output of the network.
    p_dropout=0.1,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.
    #NOTE if not using SynNet then they will be by default.

    n_channels = net_channels,                        # Number of input channels (always 16)
    n_classes = n_labels,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = [24, 24, 24],      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = [2, 4, 8],   # Number of time constants in each hidden layer (taken from tutorial)
)

#print(net)
#show trainable parameters (time constants should be empty dict otherwise they will be trained)
#print(net.parameters)

#pass parameters to optimise and the learning rate (lr) respectively to adam.
optimiser = Adam(net.parameters().astorch(), lr=1e-3)

#Dylan recommends using MSE for loss
#results in slightly higher accuracy compared to other functions
loss_function = MSELoss()

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

    #batching done by torch/tonic dataloader
    for events, labels in train_dl:
        optimiser.zero_grad()

        output, _, _ = net(events)

        #number of spikes on output channels gives prediction
        #target channel should have most number of spikes

        #TODO figure this out so I can measure accuracy
        sum = torch.cumsum(output, dim=1)
        loss = loss_function(sum[:,-1,-1].to(torch.float32), labels.to(torch.float32))

        #pred = torch.sum(output, dim=1)

        #loss = loss_function(pred, labels)

        loss.backward()
        optimiser.step()

        #Calculate the number of correct answers
        predicted = torch.argmax(sum[:,-1,:], 1)

        #to get number of datafiles and number of correct guesses
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch}/{n_epochs}, loss {loss.item():.2}, accuracy {accuracy:.2}')

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
'''

train_dl = torch.utils.data.DataLoader(
    tonic.DiskCachedDataset(
        dataset=SubsetClasses(train_data, range(3)),
        cache_path=f"cache/{train_data.__class__.__name__}/train/{net_channels}/{net_dt}",
        reset_cache = False,
    ),
    **dataloader_kwargs
)

# TEST LOOP
test_data = datasets.SHD('./data', train=False, transform=transform)
test_dl = DataLoader(SubsetClasses(test_data, range(3)), num_workers=8, batch_size=256,
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