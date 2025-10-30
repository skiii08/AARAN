# -*- coding: utf-8 -*-
"""
train_mlp_baseline_usernorm.py
---------------------------------------
- 学習: y_user_norm（ユーザー平均0・標準偏差1）
- 評価: ユーザー単位で逆正規化し、10段階スケールでMAE/RMSE算出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# Config
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 256

# ==================================================
# Load Data
# ==================================================
def load_graph(path):
    data = torch.load(path, weights_only=False)
    X_user = data["user_features"]
    X_movie = data["movie_features"]
    X_review = data["review_signals"]

    # user_normターゲットで学習
    y = data.get("ratings_user_norm", None)
    if y is None:
        raise KeyError(f"{path.name} に ratings_user_norm が存在しません")

    user_idx = data["user_indices"]
    movie_idx = data["movie_indices"]
    user_names = np.array(data["user_names"])[user_idx.cpu().numpy()]

    # 特徴連結
    X = torch.cat([X_user[user_idx], X_movie[movie_idx], X_review], dim=1)

    # 逆正規化用に、各ユーザーの mean/std をロード
    user_stats = pd.read_csv(DATA_DIR / "user_stats.csv") if (DATA_DIR / "user_stats.csv").exists() else None
    user_mean_map, user_std_map = {}, {}
    if user_stats is not None:
        for _, row in user_stats.iterrows():
            user_mean_map[row["user_name"]] = row["mean"]
            user_std_map[row["user_name"]] = row["std"]

    return X, y, user_names, user_mean_map, user_std_map


train_X, train_y, train_users, mean_map, std_map = load_graph(DATA_DIR / "hetero_graph_train.pt")
val_X, val_y, val_users, _, _ = load_graph(DATA_DIR / "hetero_graph_val.pt")
test_X, test_y, test_users, _, _ = load_graph(DATA_DIR / "hetero_graph_test.pt")

print(f"Device: {DEVICE}")
print(f"Train samples: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")

# ==================================================
# Dataset
# ==================================================
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=False)

# ==================================================
# Model
# ==================================================
class MLPBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x).squeeze(1)

model = MLPBaseline(train_X.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ==================================================
# Utility: user-wise denormalization
# ==================================================
def denormalize(preds, users, mean_map, std_map):
    preds_np = preds.cpu().numpy()
    out = []
    for val, u in zip(preds_np, users):
        if u in mean_map:
            out.append(val * std_map[u] + mean_map[u])
        else:
            out.append(val * 2.0 + 7.5)  # fallback (global mean/std)
    return np.array(out)

# ==================================================
# Evaluation (with de-normalization)
# ==================================================
def evaluate(model, loader, user_names, mean_map, std_map):
    model.eval()
    preds, targets, names = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            start = i * loader.batch_size
            end = start + len(x)
            batch_users = user_names[start:end]
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.append(out.cpu())
            targets.append(y.cpu())
            names.extend(batch_users)
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # user単位で逆正規化（10段階スケールへ）
    preds_10 = denormalize(preds, names, mean_map, std_map)
    targs_10 = denormalize(targets, names, mean_map, std_map)

    mae = np.mean(np.abs(preds_10 - targs_10))
    rmse = np.sqrt(np.mean((preds_10 - targs_10) ** 2))
    rho = np.corrcoef(preds_10, targs_10)[0, 1]
    return mae, rmse, rho


# ==================================================
# Training Loop
# ==================================================
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    train_loss = total_loss / len(train_loader.dataset)

    val_mae, val_rmse, val_rho = evaluate(model, val_loader, val_users, mean_map, std_map)
    flag = "✓" if val_mae < best_val else ""
    if flag:
        best_val = val_mae
        torch.save(model.state_dict(), DATA_DIR / "mlp_usernorm_best.pt")

    print(f"Ep {epoch:03d} | Loss {train_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f} {flag}")

print("=== TEST ===")
model.load_state_dict(torch.load(DATA_DIR / "mlp_usernorm_best.pt"))
test_mae, test_rmse, test_rho = evaluate(model, test_loader, test_users, mean_map, std_map)
print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")
