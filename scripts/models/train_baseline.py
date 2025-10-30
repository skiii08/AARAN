#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Baseline (MSE only, per-user normalized target)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from aaran_model_baseline import AARANLinearBaseline
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_PCA = True
LR = 2e-4
EPOCHS = 100
PATIENCE = 20

def load_graph(split):
    suffix = "_pca" if USE_PCA else ""
    return torch.load(DATA_DIR / f"hetero_graph_{split}{suffix}.pt", weights_only=False)

# ======================================================
# 🔹 Per-user normalization utilities
# ======================================================
def compute_user_stats(g):
    """ユーザーごとの平均と標準偏差を計算"""
    user_ids = g['user_indices'].cpu().numpy()
    ratings = g['ratings'].cpu().numpy()
    user_mean = defaultdict(lambda: 0.0)
    user_std  = defaultdict(lambda: 1.0)

    for uid in np.unique(user_ids):
        r = ratings[user_ids == uid]
        user_mean[uid] = r.mean()
        std = r.std()
        user_std[uid] = std if std > 1e-6 else 1.0
    return user_mean, user_std

def normalize_per_user(g, user_mean, user_std):
    """ユーザー単位でratingsを正規化"""
    user_ids = g['user_indices']
    ratings_norm = torch.zeros_like(g['ratings'])
    for i in range(len(ratings_norm)):
        uid = int(user_ids[i])
        ratings_norm[i] = (g['ratings'][i] - user_mean[uid]) / user_std[uid]
    g['ratings_user_norm'] = ratings_norm
    return g

def denormalize_per_user(y_pred, user_ids, user_mean, user_std):
    """ユーザー単位で逆正規化"""
    y_denorm = torch.zeros_like(y_pred)
    for i in range(len(y_pred)):
        uid = int(user_ids[i])
        y_denorm[i] = y_pred[i] * user_std[uid] + user_mean[uid]
    return y_denorm

# ======================================================
# 🔹 Evaluation
# ======================================================
def evaluate(model, g, device, user_mean, user_std):
    model.eval()
    g = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in g.items()}
    with torch.no_grad():
        out = model(g)
        pred_norm = out['rating']
        user_ids = g['user_indices']
        pred = denormalize_per_user(pred_norm, user_ids, user_mean, user_std).cpu().numpy()
        tgt  = denormalize_per_user(g['ratings_user_norm'], user_ids, user_mean, user_std).cpu().numpy()
        mae  = np.mean(np.abs(pred - tgt))
        rmse = np.sqrt(np.mean((pred - tgt)**2))
        spear = spearmanr(pred, tgt).correlation
    return mae, rmse, spear

# ======================================================
# 🔹 Main
# ======================================================
if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_g, val_g, test_g = [load_graph(s) for s in ("train","val","test")]
    u_dim = train_g['user_features'].shape[1]
    m_dim = train_g['movie_features'].shape[1]
    r_dim = train_g['review_signals'].shape[1]
    print(f"User:{u_dim}, Movie:{m_dim}, Review:{r_dim}")

    model = AARANLinearBaseline(u_dim, m_dim, r_dim, dropout=0.3).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()

    # ==================================================
    # 🔹 User-level normalization setup
    # ==================================================
    user_mean, user_std = compute_user_stats(train_g)
    train_g = normalize_per_user(train_g, user_mean, user_std)
    val_g = normalize_per_user(val_g, user_mean, user_std)
    test_g = normalize_per_user(test_g, user_mean, user_std)

    best_val = 1e9
    patience = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        g = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in train_g.items()}
        opt.zero_grad()
        out = model(g)
        loss = crit(out['rating'], g['ratings_user_norm'])
        loss.backward()
        opt.step()

        val_mae, val_rmse, val_spear = evaluate(model, val_g, device, user_mean, user_std)
        log = f"Ep {ep:03d} | Loss {loss.item():.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_spear:.3f}"
        if val_mae < best_val:
            best_val = val_mae
            patience = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_baseline_usernorm.pt")
            print(log + " ✓")
        else:
            patience += 1
            print(log)
            if patience >= PATIENCE:
                print(f"\nEarly stopping @ {ep}")
                break

    print("\n=== TEST ===")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_baseline_usernorm.pt", weights_only=False))
    test_mae, test_rmse, test_spear = evaluate(model, test_g, device, user_mean, user_std)
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_spear:.3f}")
