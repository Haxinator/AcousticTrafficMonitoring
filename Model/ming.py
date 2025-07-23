import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rockpool.nn.modules import LIFExodus
from rockpool.nn.networks import SynNet
from torch.optim import Adam
from tqdm import tqdm
import os

# ==== 1. 配置硬件 ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 2. 加载数据 ====
dir = os.path.dirname(os.path.abspath('__file__'))
mingDataPath = os.path.join(dir, "../npy")

#CUDA results in x5 speed increase.
#ming data results in x6 speed increase.
#data caching significantly slower (6x slower) than loading ming data.

#load ming data
X_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_train.npy"))).float()
y_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_train.npy"))).long()
X_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_val.npy"))).float()
y_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_val.npy"))).long()

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)


train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True, pin_memory=True, num_workers=6)
val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False, pin_memory=True, num_workers=6)

# ==== 3. 构建模型 ====
n_channels = 16
n_classes = 3

net = SynNet(
    neuron_model = LIFExodus,
    output="vmem",
    p_dropout=0,
    n_channels=n_channels,
    n_classes=n_classes,
    size_hidden_layers=[24, 24, 24],
    time_constants_per_layer=[2, 4, 8],
).to(device)

optimizer = Adam(net.parameters().astorch(), lr=1e-3)

# ==== 4. 定义损失函数 ====
def one_hot_mse_loss(outputs, labels, num_classes):
    target_onehot = F.one_hot(labels, num_classes=num_classes).float().to(outputs.device)
    return F.mse_loss(outputs, target_onehot)

# ==== 5. 开始训练 ====
n_epochs = 100
best_val_acc = 0.0
best_state = None
train_acc_list = []
train_loss_list = []
val_acc_list = []
skip_window = 30
for epoch in range(n_epochs):
    net.train()
    correct = 0
    total = 0
    total_loss = 0

    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        out, _, _ = net(xb)
        sum_out = torch.cumsum(out[:,skip_window:,:], dim=1)[:, -1, :]  # time axis = 1
        loss = one_hot_mse_loss(sum_out, yb, num_classes=n_classes)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = torch.argmax(sum_out, dim=1)
        correct += (pred == yb.to(pred.device)).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    acc = 100 * correct / total
    train_acc_list.append(acc)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")

    # ==== 验证 ====
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            out, _, _ = net(xb)
            sum_out = torch.cumsum(out[:,skip_window:,:], dim=1)[:, -1, :]
            pred = torch.argmax(sum_out, dim=1)
            correct += (pred == yb.to(pred.device)).sum().sum().item()
            total += yb.size(0)

    val_acc = 100 * correct / total
    val_acc_list.append(val_acc)

    print(f"  [Val] Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = net.state_dict()
        net.save("ming.json")
        print(f"  ✅ New best model saved at epoch {epoch} with val acc {val_acc:.2f}%")

print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")