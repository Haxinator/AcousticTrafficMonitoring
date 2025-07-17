#NOTES
'''
In 100 epochs:
 - Using vmem as output increases accuracy by about 8 percent.
 - Using MSE increases accuracy by about 2 percent.
'''

#Use SynNet to start. We Will need to develop our own architecture later.
#Still able to make some small adjustments to the network though.
from rockpool.nn.networks import SynNet
from rockpool.nn.modules import LIFExodus
# - Import torch training utilities
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#data transformations and caching.
import tonic
import tonic.transforms as T
#from tonic import datasets

#for paths
import os
import logging

#gives error
from tqdm import trange

#visualisation
import matplotlib.pyplot as plt
import numpy as np

#need about 37000 epochs to get good accuracy
n_epochs = 100
n_batches = 1
n_time = 100
n_labels = 3
net_channels = 16

# - Use a GPU if available for faster training
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)


base_dir = os.path.dirname(os.path.abspath('__file__'))
base_dir = os.path.join(base_dir, "DataPreprocessing/npy")
train_path = os.path.join(base_dir, "Train")
test_path = os.path.join(base_dir, "Test")
val_path = os.path.join(base_dir, "Val")

if not os.path.exists(train_path):
        logging.error(f"folder not found at: {train_path}")
        exit(1)
if not os.path.exists(test_path):
        logging.error(f"folder not found at: {test_path}")
        exit(1)
if not os.path.exists(val_path):
        logging.error(f"folder not found at: {val_path}")
        exit(1)

#for loading npy files
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

training_data = datasets.DatasetFolder(
    root=train_path,
    loader=npy_loader,
    extensions=['.npy'],
   # transform=ToTensor()
)

test_data = datasets.DatasetFolder(
    root=test_path,
    loader=npy_loader,
    extensions=['.npy'],
  #  transform=ToTensor()
)

val_data = datasets.DatasetFolder(
    root=val_path,
    loader=npy_loader,
    extensions=['.npy'],
  #  transform=ToTensor()
)

dataloader_kwargs = dict(
    batch_size=128,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=8,
)

disk_train_dataset = tonic.DiskCachedDataset(
    dataset=training_data,
    # transform = torch.Tensor,lambda x: torch.tensor(x).to_sparse(),
    cache_path=f"cache/{training_data.__class__.__name__}/train/{net_channels}/",
    # target_transform=lambda x: torch.tensor(x),
    reset_cache = True,
  )

disk_val_dataset = tonic.DiskCachedDataset(
    dataset=val_data ,
    # transform = torch.Tensor,lambda x: torch.tensor(x).to_sparse(),
    cache_path=f"cache/{val_data.__class__.__name__}/val/{net_channels}/",
    # target_transform=lambda x: torch.tensor(x),
    reset_cache = True,
  )


#probably don't need to cache test data lol
disk_test_dataset = tonic.DiskCachedDataset(
    dataset=test_data,
    # transform = torch.Tensor,lambda x: torch.tensor(x).to_sparse(),
    cache_path=f"cache/{test_data.__class__.__name__}/test/{net_channels}/",
    # target_transform=lambda x: torch.tensor(x),
    reset_cache = True,
  )

train_dl = DataLoader(disk_train_dataset, **dataloader_kwargs)
val_dl = DataLoader(disk_val_dataset, **dataloader_kwargs)
test_dl = DataLoader(disk_test_dataset, **dataloader_kwargs)

#in reality need to use dataloader to open file.

# A manual seed ensures repeatability
torch.manual_seed(1234) 

# - Build a simple SynNet with three hidden layers
# Need to experiment with number of layers, neurons and time constants.
net = SynNet(
    #Dylan recommended this since it will make the optimiser work better.
    neuron_model = LIFExodus,
    #output="vmem",                         # Use the membrane potential as the output of the network.
    p_dropout=0.1,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.
    #NOTE if not using SynNet then they will be by default.

    n_channels = net_channels,                        # Number of input channels (always 16)
    n_classes = n_labels,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = [24, 24, 24],      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = [2, 4, 8],   # Number of time constants in each hidden layer (taken from tutorial)
).to(device)

#compile model to make it FAST
#net.compile()

#print(net)
#show trainable parameters (time constants should be empty dict otherwise they will be trained)
#print(net.parameters)

#pass parameters to optimise and the learning rate (lr) respectively to adam.
optimiser = Adam(net.parameters().astorch(), lr=1e-3)

#Dylan recommends using MSE for loss
#results in slightly higher accuracy compared to other functions
#loss_function = MSELoss()
loss_function = CrossEntropyLoss()

#no constraints used
#no regularisations used

best_val_acc = -1
best_bot = {}
train_acc_list = []
train_loss_list = []
val_acc_list = []


#in function so we can compile training to make it sonic speed.
#@torch.compile
def train(net, train_dl, val_dl, test_dl):
    global train_acc_list, train_loss_list, val_acc_list, best_bot, best_val_acc, optimiser, loss_function, device

    #Training loop
    #trange gives cool progress bar
    for epoch in trange(n_epochs):
        net.train()

        #for calculating accuracy
        correct = 0
        total = 0

        #batching done by torch/tonic dataloader
        for events, labels in train_dl:
            events, labels = events.to(device), labels.to(device)
            optimiser.zero_grad()

            #output, _, _ = net(events)
            output, _, _ = net(torch.Tensor(events).float())

            #number of spikes on output channels gives prediction
            #target channel should have most number of spikes

            #TODO figure this out so I can measure accuracy
            sum = torch.cumsum(output, dim=1)
            loss = loss_function(sum[:,-1,:], labels)

            #pred = torch.sum(output, dim=1)

            #loss = loss_function(pred, labels)

            loss.backward()
            optimiser.step()

            #Calculate the number of correct answers
            predicted = torch.argmax(sum[:,-1,:], 1)

            #to get number of datafiles and number of correct guesses
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # VAL LOOP
        accuracy_val = -1

        with torch.no_grad():
            net.eval()
            val_correct = 0
            val_total = 0

            for events, labels in val_dl:
                events, labels = events.to(device), labels.to(device)
                output, _, _ = net(torch.Tensor(events).float())

                sum = torch.cumsum(output, dim=1)

                predicted = torch.argmax(sum[:,-1,:], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            accuracy_val = (val_correct/val_total)*100

        if(accuracy_val > best_val_acc):
            best_val_acc = accuracy_val

            # save model
            net.save("Synnet.json")
            print("New best model saved!")

        accuracy = 100 * correct / total
        train_acc_list.append(accuracy)
        val_acc_list.append(accuracy_val)
        train_loss_list.append(loss.item())
        print(f'Epoch {epoch}/{n_epochs}, Loss {loss.item():.2f}, Training Accuracy {accuracy:.2f}, Val Accuracy {accuracy_val:.2f}')


    # TEST LOOP (after training)
    with torch.no_grad():
        correct = 0
        total = 0

        for events, labels in test_dl:
            events, labels = events.to(device), labels.to(device)
            output, _, _ = net(torch.Tensor(events).float())

            sum = torch.cumsum(output, dim=1)

            predicted = torch.argmax(sum[:,-1,:], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy_test = (correct/total)*100

    print(f"Test Accuracy: {accuracy_test:.3f}%")


#lol train
train(net, train_dl, val_dl, test_dl)

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