#Use SynNet to start. We Will need to develop our own architecture later.
#Still able to make some small adjustments to the network though.
from rockpool.nn.networks import SynNet
# - Import torch training utilities
import torch
from torch.optim import Adam
from torch.nn import MSELoss

#gives error
from tqdm import trange

n_epochs = 100
n_batches = 1
n_time = 100

# - Build a simple SynNet with three hidden layers
# Need to experiment with number of layers, neurons and time constants.
net = SynNet(
    #Dylan recommended this since it will make the optimiser work better.
    output="vmem",                         # Use the membrane potential as the output of the network.
    p_dropout=0.1,                         # probability of dropout (good to prevent overfitting).

    #time constants and threshold are not trainable by default.

    n_channels = 16,                        # Number of input channels (always 16)
    n_classes = 3,                          # Number of output classes (car, commercial, background noise).
    size_hidden_layers = [24, 24, 24],      # Number of neurons in each hidden layer (taken from tutorial)
    time_constants_per_layer = [2, 4, 8],   # Number of time constants in each hidden layer (taken from tutorial)
)

print(net)

#pass parameters to optimise and the learning rate (lr) respectively to adam.
optimiser = Adam(net.parameters().astorch(), lr=1e-3)

#Dylan recommends using MSE for loss
loss_function = MSELoss()

#pointless task
input_sp = (torch.rand(1, 100, 16) < 0.01) * 1.0
target_sp = torch.ones(1, 100, 3)

#Training loop
for epoch in trange(n_epochs):
    optimiser.zero_grad()

    output, _, _ = net(input_sp)

    loss = loss_function(output, target_sp)

    loss.backward()
    optimiser.step()