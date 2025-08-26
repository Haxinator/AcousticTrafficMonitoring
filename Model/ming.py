import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rockpool.nn.modules import LIFExodus
from rockpool.nn.networks import SynNet
from torch.optim import Adam
from tqdm import tqdm
import os

# ==== 1. Importing Dependencies and Hardware Configuration ====
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rockpool.nn.networks import SynNet
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dir = os.path.dirname(os.path.abspath('__file__'))
mingDataPath = os.path.join(dir, "..", "DataPreprocessing", "delete_npy")

# ==== 2. loading data ====
#load ming data
X_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_train.npy"))).float()
y_train = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_train.npy"))).long()
X_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_val.npy"))).float()
y_val = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_val.npy"))).long()
X_test = torch.from_numpy(np.load(os.path.join(mingDataPath, "X_test.npy"))).float()
y_test = torch.from_numpy(np.load(os.path.join(mingDataPath, "y_test.npy"))).long()

print(f"Using device: {X_train.shape}")
print(f"Using device: {y_train.shape}")
print(f"Using device: {X_val.shape}")
print(f"Using device: {y_val.shape}")

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)


train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, pin_memory=True, num_workers=6)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=False, pin_memory=True, num_workers=6)

# ==== 3. model ====
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

#checkpoint_path = "/content/drive/My Drive/SD-project/best_snn_model_with_newMSE.pth"
#net.load_state_dict(torch.load(checkpoint_path, map_location=device))
optimizer = Adam(net.parameters().astorch(), lr=1e-3)
import matplotlib.pyplot as plt
import torch.nn.functional as F

# get a sample
xb, yb = next(iter(val_dl))
xb, yb = xb.to(device), yb.to(device)

net.eval()
with torch.no_grad():
    out, _, _ = net(xb)  # [B, T, C]

# get the output
vmem = out[0].cpu().numpy()  # shape: [T, C]
cumsum_vmem = vmem.cumsum(axis=0)  # shape: [T, C]

# plt fig
plt.figure(figsize=(8, 5))
for i in range(cumsum_vmem.shape[1]):
    plt.plot(cumsum_vmem[:, i], label=f'Class {i}')

plt.axvline(x=20, color='red', linestyle='--', label='Skip Window')
plt.title('Cumulative Membrane Potential over Time')
plt.xlabel('Time step')
plt.ylabel('Cumulative vmem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== 4. MSE defining ====
def float_target_mse_loss(outputs, labels, num_classes, pos_val=5.0, neg_val=-1.0):
    device = outputs.device
    B = labels.shape[0]
    target = torch.full((B, num_classes), neg_val, device=device)
    target[torch.arange(B, device=device), labels] = pos_val
    return F.mse_loss(outputs, target)

# ==== 5. training ====
n_epochs = 1000
best_val_acc = 0.0
best_state = None
train_acc_list = []
train_loss_list = []
val_acc_list = []
skip_window = 20
for epoch in range(n_epochs):
    net.train()
    correct = 0
    total = 0
    total_loss = 0

    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        out, _, _ = net(xb)                   # shape: [B, T, C]
        output = out[:, skip_window:, :].mean(dim=1)               # using sikp and mean value [B, C]
        loss = float_target_mse_loss(output, yb, num_classes=n_classes)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = torch.argmax(output, dim=1)
        correct += (pred == yb.to(pred.device)).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    acc = 100 * correct / total
    train_acc_list.append(acc)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")

    # ==== 验证 ====
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            out, _, _ = net(xb)
            output = out[:, skip_window:, :].mean(dim=1)                    # using sikp and mean value vmem
            pred = torch.argmax(output, dim=1)
            correct += (pred == yb.to(pred.device)).sum().item()
            total += yb.size(0)

    val_acc = 100 * correct / total
    val_acc_list.append(val_acc)
    print(f"  [Val] Accuracy: {val_acc:.2f}%")

    # ==== 保存最佳模型 ====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = net.state_dict()
        torch.save(best_state, "best_snn_model_with_newMSE_final.pth")
        print(f"  ✅ New best model saved at epoch {epoch} with val acc {val_acc:.2f}%")

print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
