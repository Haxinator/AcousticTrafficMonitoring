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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

#data transformations and caching.
import tonic
import tonic.transforms as T

#for paths
import os
import logging

#progress bar
from tqdm import trange

#saving and loading json for stats
import json

#visualisation
import matplotlib.pyplot as plt
import numpy as np

#need about 37000 epochs to get good accuracy
n_epochs = 2000
n_batches = 8
n_time = 100
n_labels = 3
net_channels = 16

# - Use a GPU if available for faster training
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print(f"⚙️ Using device: {device}")

# ----------------------------- LOAD DATA --------------------------#

dataloader_kwargs = dict(
    batch_size=n_batches,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=8,
)

dataloader_val_kwargs = dict(
    batch_size=n_batches,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=8,
)

dir = os.path.dirname(os.path.abspath('__file__'))
base_dir = os.path.join(dir, "DataPreprocessing", "new_npy")
'''
train_path = os.path.join(base_dir, "Train")
test_path = os.path.join(base_dir, "Test")
val_path = os.path.join(base_dir, "Val")




#if files can be found
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


#caching data
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


train_dl = DataLoader(training_data, **dataloader_kwargs)
val_dl = DataLoader(val_data, **dataloader_kwargs)
test_dl = DataLoader(test_data, **dataloader_kwargs)


'''
mingDataPath = os.path.join(dir, "DataPreprocessing", "delete_npy")

#CUDA results in x5 speed increase.
#ming data results in x6 speed increase.
#data caching significantly slower (6x slower) than loading ming data.
X_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_train.npy"))).float()
y_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_train.npy"))).long()
X_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_val.npy"))).float()
y_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_val.npy"))).long()
X_test = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_test.npy"))).float()
y_test = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_test.npy"))).long()

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

'''
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)


train_dl = DataLoader(train_ds, **dataloader_kwargs)
val_dl = DataLoader(val_ds, **dataloader_val_kwargs)
test_dl = DataLoader(test_ds, **dataloader_val_kwargs)
'''

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=n_batches, shuffle=True, drop_last=True, pin_memory=True, num_workers=6)
val_dl = DataLoader(val_ds, batch_size=n_batches, shuffle=False, drop_last=False, pin_memory=True, num_workers=6)
test_dl = DataLoader(test_ds, batch_size=n_batches, shuffle=False, drop_last=False, pin_memory=True, num_workers=6)

# A manual seed ensures repeatability
torch.manual_seed(1234) 

# - Build a simple SynNet with three hidden layers
# Need to experiment with number of layers, neurons and time constants.
net = SynNet(
    neuron_model = LIFExodus,
    output="vmem",                       # Use the membrane potential as the output of the network.
    p_dropout=0.1,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.
    #NOTE if not using SynNet then they will be by default.

    n_channels = net_channels,                        # Number of input channels (always 16)
    n_classes = n_labels,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = [24, 24, 24],      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = [3, 6, 9],   # Number of time constants in each hidden layer (taken from tutorial)
).to(device)

#compile model to make it FAST (doesn't work =( )
#net.compile()

#print(net)
#show trainable parameters (time constants should be empty dict otherwise they will be trained)
#print(net.parameters)

#pass parameters to optimise and the learning rate (lr) respectively to adam.
#must be done after model is moved to GPU (otherwise learning won't occur)
optimiser = Adam(net.parameters().astorch(), lr=1e-3)

#Dylan recommends using MSE for loss
#results in slightly higher accuracy compared to other functions
loss_function = MSELoss()
#loss_function = CrossEntropyLoss()

# ==== 4. MSE defining ====
def float_target_mse_loss(outputs, labels, num_classes, pos_val=5.0, neg_val=-1.0):
    device = outputs.device
    B = labels.shape[0]
    target = torch.full((B, num_classes), neg_val, device=device)
    target[torch.arange(B, device=device), labels] = pos_val
    return F.mse_loss(outputs, target)


#no constraints used
#no regularisations used

#where the model and statistics will be saved
best_val_acc = -1
correct = 0
total = 0
total_loss = 0
total_epochs = -1
best_bot = {}
train_acc_list = []
train_loss_list = []
val_acc_list = []
skip_window = 30

#initialise stat list to empty (for now)
stats = {"train_acc_list": train_acc_list, "train_loss_list": train_loss_list, "val_acc_list": val_acc_list, "correct": correct, "total": total, "total_loss": total_loss, "best_val_acc": best_val_acc, "total_epochs": total_epochs, "test_acc": 0}
best_stat = {"train_acc_list": train_acc_list, "train_loss_list": train_loss_list, "val_acc_list": val_acc_list, "correct": correct, "total": total, "total_loss": total_loss, "best_val_acc": best_val_acc, "total_epochs": total_epochs, "test_acc": 0}
#in function so we can compile training to make it sonic speed.
#@torch.compile (doesn't work =( )
def train(net, train_dl, val_dl, test_dl):
    global train_acc_list, train_loss_list, val_acc_list, best_bot, best_val_acc, optimiser, skip_window, device

    #Training loop
    #trange gives cool progress bar
    for _ in trange(n_epochs):
        stats["total_epochs"] += 1
        net.train()
        loss = 0.0

        #batching done by torch/tonic dataloader
        for events, labels in train_dl:
            events, labels = events.to(device), labels.to(device)
            #prevent exploding gradients by reseting gradients every loop
            optimiser.zero_grad()
            
            out, _, _ = net(events)
            output = out[:, skip_window:, :].mean(dim=1) 
            loss = float_target_mse_loss(output, labels, num_classes=n_labels)

            
            loss.backward()
            #step must be done after calling backward.
            optimiser.step()

            predicted = torch.argmax(output, 1).to(device)

            #to get number of datafiles and number of correct guesses
            stats["total"] += labels.size(0)
            stats["correct"] += (predicted == labels).sum().item()
            stats["total_loss"] += loss.item() * events.size(0)

        # VAL LOOP
        accuracy_val = -1

        with torch.no_grad():
            net.eval()
            val_correct = 0
            val_total = 0

            for events, labels in val_dl:
                events, labels = events.to(device), labels.to(device)
                out, _, _ = net(events)

                output = out[:, skip_window:, :].mean(dim=1)  # using skip and mean value vmem

                predicted = torch.argmax(output, 1).to(device)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            accuracy_val = (val_correct/val_total)*100

        accuracy = 100 * stats["correct"] / stats["total"]
        avg_loss = stats["total_loss"] / stats["total"]

        stats["train_acc_list"].append(accuracy)
        stats["val_acc_list"].append(accuracy_val)
        stats["train_loss_list"].append(avg_loss)

        if(accuracy_val > stats["best_val_acc"]):
            stats["best_val_acc"] = accuracy_val

            # save model
            net.save(model_file_name)
            print("💾 New best model saved!")

            #save stats
            with open(stat_file_name, "w", encoding="utf-8") as stat_file:
                json.dump(stats, stat_file, indent=4) #indent makes json pretty

        print(f'Epoch {stats["total_epochs"]}: Loss {avg_loss:.4f}, Training Accuracy {accuracy:.2f}%, Val Accuracy {accuracy_val:.2f}%')


    #load best model found
    net.load(model_file_name)
    net.to(device)

    # TEST LOOP (after training)
    with torch.no_grad():
        correct = 0
        total = 0

        for events, labels in test_dl:
            events, labels = events.to(device), labels.to(device)
            output, _, _ = net(events)

            out = output[:, skip_window:, :].mean(dim=1)  # using skip and mean value vmem

            predicted = torch.argmax(out, 1).to(device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy_test = (correct/total)*100
        stats['test_acc'] = accuracy_test

        if (stats['test_acc'] > best_stat["test_acc"]):
            with open(bestStat_file_name, "w", encoding="utf-8") as stat_file:
                json.dump(stats, stat_file, indent=4) #indent makes json pretty
                # save model
                net.save(bestModel_file_name)
                print("💾 Record breaking model saved!")

             

    print(f"Test Accuracy: {accuracy_test:.3f}%")


# ---------------- Program Start---------------------#
#%%
stat_file_name = "New_Stats.json"
model_file_name = "New_Model.json"
bestStat_file_name = "Best_Stats.json"
bestModel_file_name = "Best_Model.json"
#%%
#load model if save exists.
if os.path.exists(os.path.join(dir, model_file_name)):
    net.load(model_file_name)
    net.to(device)
#load statistics if save exists
if os.path.exists(os.path.join(dir, stat_file_name)):
    with open(stat_file_name, "r", encoding="utf-8") as stat_file:
        stats = json.load(stat_file)
#load best stats if save exists
if os.path.exists(os.path.join(dir, bestStat_file_name)):
    with open(bestStat_file_name, "r", encoding="utf-8") as stat_file:
        best_stat = json.load(stat_file)

trainModel = True

if trainModel:
    #lol train
    train(net, train_dl, val_dl, test_dl)
else:
    #%%
    import matplotlib.pyplot as plt
    import json
    import os
    dir = os.path.dirname(os.path.abspath('__file__'))
    stats = {}

    if os.path.exists(os.path.join(dir, stat_file_name)):
        with open(stat_file_name, "r", encoding="utf-8") as stat_file:
            stats = json.load(stat_file)

    #show stats
    epochs = list(range(stats["total_epochs"]+1))

    # Accuracy Curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats["train_acc_list"], label='Train Accuracy')
    plt.plot(epochs, stats["val_acc_list"], label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, stats["train_loss_list"], label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("plots","lastestDataset"))
    plt.show()

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
# %%
