# ==== 1. Importing Dependencies and Hardware Configuration ====
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rockpool.nn.networks import SynNet
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from rockpool.nn.networks import SynNet
from rockpool.nn.modules import LIFExodus
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dir = os.path.dirname(os.path.abspath('__file__'))
mingDataPath = os.path.join(dir, "..", "DataPreprocessing", "CustomDataset")

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

# ========= 0. Imports =========
import os, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
 
# ========= 1. Reproducibility =========
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
 
# ========= 2. Model definition =========
n_channels = 16
n_classes = 3
p_dropout = 0     # Suggested: 0.2–0.3; set to 0.0 if you want to disable dropout
time_cs = [2, 4, 8] # Time constants per layer
hidden = [24, 24, 24]
 
net = SynNet(
    neuron_model=LIFExodus,
    output="vmem",
    p_dropout=p_dropout,
    n_channels=n_channels,
    n_classes=n_classes,
    size_hidden_layers=hidden,
    time_constants_per_layer=time_cs,
).to(device)
 
# ========= 3. Model profile =========
total_params = sum(p.numel() for p in net.parameters().astorch())
trainable_params = sum(p.numel() for p in net.parameters().astorch() if p.requires_grad)
model_size_mb = total_params * 4 / 1024 / 1024  # fp32 precision
print(f"[Profile] Total params: {total_params:,} | Trainable: {trainable_params:,} | ~{model_size_mb:.2f} MB")
 
# ========= 4. Optimizer & LR scheduler =========
base_lr = 1e-3
weight_decay = 1e-4
optimizer = AdamW(net.parameters().astorch(), lr=base_lr, betas=(0.9, 0.98), weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)  # Smooth decay, no restart
max_grad_norm = 1.0
 
# If you prefer validation-based scheduling, replace with:
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5,
#                               threshold=1e-3, cooldown=2, min_lr=1e-6)
 
# ========= 5. Loss =========
def float_target_mse_loss(outputs, labels, num_classes, pos_val=8.0, neg_val=-2.0):
    device_ = outputs.device
    B = labels.shape[0]
    target = torch.full((B, num_classes), neg_val, device=device_)
    target[torch.arange(B, device=device_), labels] = pos_val
    return F.mse_loss(outputs, target)
 
# ========= 6. EMA (Exponential Moving Average) =========
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)
 
ema = EMA(net, decay=0.999)
 
# ========= 7. Eval helper =========
def evaluate(model, data_loader, use_ema=True, skip_window=20):
    model.eval()
    # Backup current weights and temporarily switch to EMA weights for evaluation
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    if use_ema:
        ema.apply_to(model)
 
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            out, _, _ = model(xb)              # [B, T, C]
            output = out[:, skip_window:, :].mean(1)  # Temporal mean
            pred = output.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
 
    # Restore original weights
    model.load_state_dict(backup, strict=False)
    return 100.0 * correct / total
 
# ========= 8. Train loop =========
n_epochs = 1000
skip_window = 20
ckpt_path = "val_best_model.pth"
ckpt_path1 = "last_step_model.pth"
best_val_acc = 0.0
patience = 100
bad_epochs = 0
 
train_acc_list, train_loss_list, val_acc_list, lr_hist = [], [], [], []
 
for epoch in range(n_epochs):
    net.train()
    correct, total, total_loss = 0, 0, 0.0
 
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
 
        out, _, _ = net(xb)                     # [B, T, C]
        output = out[:, skip_window:, :].mean(1)
        loss = float_target_mse_loss(output, yb, num_classes=n_classes)
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters().astorch(), max_grad_norm)
        optimizer.step()
        ema.update(net)  # Update EMA at every step
 
        total_loss += loss.item() * xb.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
 
    avg_loss = total_loss / total
    train_acc = 100.0 * correct / total
    train_loss_list.append(avg_loss)
    train_acc_list.append(train_acc)
 
    # Validation (using EMA weights)
    val_acc = evaluate(net, val_dl, use_ema=True, skip_window=skip_window)
    val_acc_list.append(val_acc)
 
    # Step the scheduler
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    # If using ReduceLROnPlateau, replace with: scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    lr_hist.append(current_lr)
 
    print(f"Epoch {epoch:03d} | LR={current_lr:.3e} | TrainLoss={avg_loss:.4f} | "
          f"TrainAcc={train_acc:.2f}% | [Val]Acc={val_acc:.2f}%")
 
    # Best checkpoint saving & Early Stopping
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        torch.save({
            "model": net.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": best_val_acc,
            "config": {
                "n_channels": n_channels, "n_classes": n_classes,
                "hidden": hidden, "time_cs": time_cs, "p_dropout": p_dropout
            }
        }, ckpt_path)
        net.save(ckpt_path)
        print(f"  ✅ New best saved @ epoch {epoch} (val {val_acc:.2f}%)")
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"⏹ Early stop at epoch {epoch}. Best Val Acc = {best_val_acc:.2f}%")
            torch.save({
            "model": net.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": best_val_acc,
            "config": {
                "n_channels": n_channels, "n_classes": n_classes,
                "hidden": hidden, "time_cs": time_cs, "p_dropout": p_dropout
            }
        }, ckpt_path1)
            
        net.save(ckpt_path1)
        break
 
print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
 
