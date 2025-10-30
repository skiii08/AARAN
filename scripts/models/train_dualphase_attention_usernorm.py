#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dualphase_attention_usernorm.py

AARAN 向け DualPhase + Cross-Interaction Attention（ユーザー×映画）モデル。
- 目的値: user-normalized rating（zスコア）
- 学習は user_norm（z）で行い、評価はユーザー別の μ/σ を学習データから回帰で推定して 0-10 スケールへデノーマライズして報告
- 入力:
    - user_features: (N_user, 686D)
    - movie_features: (N_movie, 1202D)
    - review_signals: (N_edge, 22D)  # 18 aspect + 4 person-attention
    - ratings_raw:   (N_edge,)
    - ratings_user_norm: (N_edge,)   # 学習ターゲット
    - user_indices / movie_indices: (N_edge,)
"""

import os
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

# =============================
# Config
# =============================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
BATCH_SIZE = 1024
LR = 3e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 400
PATIENCE = 40
HIDDEN_DIM = 256
RS_DIM = 64         # review_signals embedding dim
DROPOUT = 0.15

# =============================
# Utils
# =============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(spearmanr(a, b, nan_policy="omit").correlation)
    except Exception:
        return float("nan")

# =============================
# Data Loading
# =============================
def _load_split(path: Path):
    d = torch.load(path, weights_only=False)
    # 取り出し
    user_feats = d["user_features"].float()
    movie_feats = d["movie_features"].float()
    rs = d["review_signals"].float()
    u_idx = d["user_indices"].long()
    m_idx = d["movie_indices"].long()
    y_raw = d.get("ratings_raw", None)
    if isinstance(y_raw, torch.Tensor):
        y_raw = y_raw.float()
    y_norm = d.get("ratings_user_norm", None)
    if isinstance(y_norm, torch.Tensor):
        y_norm = y_norm.float()

    # バリデーション
    assert rs.shape[0] == u_idx.shape[0] == m_idx.shape[0], "edge 次元不一致"
    assert y_raw is not None and y_norm is not None, "ratings_raw / ratings_user_norm が必要です"

    # ユーザー名→idx マップ（μ,σ 推定で便利用）
    # 保存側は Python dict のはず。torch.saveでそのまま入っている想定。
    user_name_to_idx = d.get("user_name_to_idx", None)
    user_names = d.get("user_names", None)
    return {
        "user_features": user_feats,
        "movie_features": movie_feats,
        "review_signals": rs,
        "user_indices": u_idx,
        "movie_indices": m_idx,
        "y_raw": y_raw,
        "y_norm": y_norm,
        "user_name_to_idx": user_name_to_idx,
        "user_names": user_names,
    }

def load_all_splits(data_dir: Path):
    train = _load_split(data_dir / "hetero_graph_train.pt")
    val   = _load_split(data_dir / "hetero_graph_val.pt")
    test  = _load_split(data_dir / "hetero_graph_test.pt")
    return train, val, test

# =============================
# μ/σ の推定（ユーザー毎の x ≈ μ + σ * z を最小二乗で回帰）
# =============================
def fit_user_denorm_maps(train: dict, min_count: int = 2) -> Tuple[Dict[int, float], Dict[int, float]]:
    u_idx = train["user_indices"].cpu().numpy()
    x = train["y_raw"].cpu().numpy()
    z = train["y_norm"].cpu().numpy()

    num_users = train["user_features"].shape[0]
    mu_map: Dict[int, float] = {}
    sigma_map: Dict[int, float] = {}

    # 事前に全ユーザーの index → サンプル集合 を作る
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, u in enumerate(u_idx):
        buckets[int(u)].append(i)

    global_mean = float(np.mean(x))
    global_std = float(np.std(x) + 1e-6)

    for u in range(num_users):
        idxs = buckets.get(u, [])
        if len(idxs) < min_count:
            mu_map[u] = global_mean
            sigma_map[u] = max(global_std, 1e-3)
            continue

        xu = x[idxs]  # raw
        zu = z[idxs]  # norm target

        # 最小二乗: xu ≈ a + b * zu
        # [1, zu] のデザイン行列
        A = np.vstack([np.ones_like(zu), zu]).T  # shape (n, 2)
        try:
            coef, _, _, _ = np.linalg.lstsq(A, xu, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            a, b = float(np.mean(xu)), float(np.std(xu) + 1e-6)

        # 安全措置
        if not np.isfinite(a): a = global_mean
        if (not np.isfinite(b)) or abs(b) < 1e-6:
            b = global_std

        mu_map[u] = a
        sigma_map[u] = max(abs(b), 1e-6)

    return mu_map, sigma_map

def denorm_by_user(u_idx: torch.Tensor, z_pred: torch.Tensor,
                   mu_map: Dict[int, float], sigma_map: Dict[int, float]) -> torch.Tensor:
    # x_pred = mu + sigma * z_pred
    out = torch.empty_like(z_pred)
    up = u_idx.detach().cpu().numpy()
    zp = z_pred.detach().cpu().numpy()
    xp = np.empty_like(zp)
    for i in range(len(zp)):
        u = int(up[i])
        mu = mu_map.get(u, 7.5)
        sg = sigma_map.get(u, 2.0)
        xp[i] = mu + sg * zp[i]
    return torch.tensor(xp, dtype=torch.float32, device=z_pred.device)

# =============================
# Dataset
# =============================
class EdgeDataset(Dataset):
    def __init__(self, split: dict):
        self.user_features = split["user_features"]
        self.movie_features = split["movie_features"]
        self.rs = split["review_signals"]
        self.u_idx = split["user_indices"]
        self.m_idx = split["movie_indices"]
        self.y = split["y_norm"]  # 学習は z を直接
        assert self.rs.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return (self.u_idx[i], self.m_idx[i], self.rs[i], self.y[i])

# =============================
# Model
# =============================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.1, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            act(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)

class CrossInteraction(nn.Module):
    """
    user_emb と movie_emb の相互作用を注意ゲートで融合。
    g = sigmoid(W [u, m, u*m, |u-m|])  （feature-wise gate）
    out = u + m + g * (u * m)
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, u, m):
        z = torch.cat([u, m, u * m, torch.abs(u - m)], dim=-1)
        g = self.gate(z)
        return u + m + g * (u * m)

class DualPhaseAttentionModel(nn.Module):
    """
    Phase1: [u_enc, m_enc, rs_enc] -> base z
    Phase2: CrossInteraction(u_enc, m_enc) + [u_enc, m_enc, rs_enc, base z] -> delta
    out_z = base + delta
    """
    def __init__(self, user_dim, movie_dim, rs_dim=22, hidden=256, rs_emb=64, dropout=0.15):
        super().__init__()
        self.user_enc = MLP(user_dim, hidden, hidden=hidden, dropout=dropout)
        self.movie_enc = MLP(movie_dim, hidden, hidden=hidden, dropout=dropout)
        self.rs_enc = nn.Sequential(
            nn.Linear(rs_dim, rs_emb),
            nn.GELU(),
            nn.LayerNorm(rs_emb),
        )
        self.cross = CrossInteraction(hidden)

        # Phase1 head
        p1_in = hidden + hidden + rs_emb
        self.p1 = nn.Sequential(
            nn.Linear(p1_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # Phase2 head
        p2_in = hidden + hidden + rs_emb + 1  # cross(u,m)=hidden を別に足しても良いが、ここは base を追加
        self.p2 = nn.Sequential(
            nn.Linear(p2_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, user_feats, movie_feats, rs_feats, u_idx, m_idx):
        # 取り出し & encode
        u = self.user_enc(user_feats[u_idx])    # (B, H)
        m = self.movie_enc(movie_feats[m_idx])  # (B, H)
        r = self.rs_enc(rs_feats)               # (B, rs_emb)

        # Phase1
        p1_in = torch.cat([u, m, r], dim=-1)
        base = self.p1(p1_in).squeeze(-1)       # (B,)

        # Cross-interaction
        cm = self.cross(u, m)                   # (B, H)

        # Phase2
        p2_in = torch.cat([u, m, r, base.unsqueeze(-1)], dim=-1)
        delta = self.p2(p2_in).squeeze(-1)

        out = base + delta
        return out

# =============================
# Train / Eval
# =============================
@torch.no_grad()
def evaluate(model, loader, device, user_feats, movie_feats, mu_map, sigma_map):
    model.eval()
    zs, xs, uids = [], [], []
    for u_idx, m_idx, rs, y in loader:
        u_idx = u_idx.to(device)
        m_idx = m_idx.to(device)
        rs = rs.to(device)
        y = y.to(device)

        z_pred = model(user_feats, movie_feats, rs, u_idx, m_idx)
        zs.append(z_pred.detach().cpu())
        xs.append(y.detach().cpu())   # 注意: これは z の正解（学習ターゲット）
        uids.append(u_idx.detach().cpu())

    z_pred = torch.cat(zs, dim=0)         # predicted z
    z_true = torch.cat(xs, dim=0)         # true z
    u_all  = torch.cat(uids, dim=0)

    # z スケールでの MAE/RMSE
    with torch.no_grad():
        mae_z = torch.mean(torch.abs(z_pred - z_true)).item()
        rmse_z = torch.sqrt(torch.mean((z_pred - z_true) ** 2)).item()

    # 0-10 スケールへデノーマライズして評価
    x_pred = denorm_by_user(u_all, z_pred, mu_map, sigma_map)   # (B,)
    # z_true -> raw への変換には本来 y_raw が必要だが、相関だけ見たいなら z_true と順序は同一。
    # MAE/RMSE は raw の正解が必要。検証では loader が y_norm しか持っていないので、
    # raw 正解は別途入れる or 簡易的に z 上のスコアを報告。今回は相関は z で、MAE/RMSE は x_pred と y_raw を別APIで。
    # → 評価時は別関数 evaluate_with_raw を使う（下で定義）。

    return mae_z, rmse_z, float("nan"), float("nan"), float("nan")  # placeholder

@torch.no_grad()
def evaluate_with_raw(model, split: dict, batch_size, device, mu_map, sigma_map):
    ds = EdgeDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model.eval()
    z_preds, uids = [], []
    for u_idx, m_idx, rs, y in loader:
        u_idx = u_idx.to(device)
        m_idx = m_idx.to(device)
        rs = rs.to(device)
        z_pred = model(split["user_features"].to(device),
                       split["movie_features"].to(device),
                       rs, u_idx, m_idx)
        z_preds.append(z_pred.detach().cpu())
        uids.append(u_idx.detach().cpu())

    z_pred = torch.cat(z_preds, dim=0)       # (N_edge,)
    u_all  = torch.cat(uids, dim=0)

    # denorm
    x_pred = denorm_by_user(u_all, z_pred, mu_map, sigma_map).cpu().numpy()
    x_true = split["y_raw"].cpu().numpy()

    mae = float(np.mean(np.abs(x_pred - x_true)))
    rmse = float(np.sqrt(np.mean((x_pred - x_true) ** 2)))
    rho = spearman_corr(x_pred, x_true)

    return mae, rmse, rho

def train():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device.type}")

    train_split, val_split, test_split = load_all_splits(DATA_DIR)

    # user/movie features を to(device) しても良いが、バッチで gather するため常駐させる。
    user_feats = train_split["user_features"].to(device)
    movie_feats = train_split["movie_features"].to(device)

    # μ/σ を学習 split から推定（x ≈ a + b z）
    mu_map, sigma_map = fit_user_denorm_maps(train_split)

    # Dataset / Loader
    train_ds = EdgeDataset(train_split)
    val_ds = EdgeDataset(val_split)
    test_ds = EdgeDataset(test_split)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    user_dim = user_feats.shape[1]
    movie_dim = movie_feats.shape[1]
    rs_dim = train_split["review_signals"].shape[1]

    model = DualPhaseAttentionModel(user_dim, movie_dim, rs_dim=rs_dim,
                                    hidden=HIDDEN_DIM, rs_emb=RS_DIM, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience = PATIENCE
    no_improve = 0

    print(f"User:{user_feats.shape[0]}, Movie:{movie_feats.shape[0]}, Review:{rs_dim}")
    print(f"Train samples: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        cnt = 0

        for u_idx, m_idx, rs, y in train_loader:
            u_idx = u_idx.to(device)
            m_idx = m_idx.to(device)
            rs = rs.to(device)
            y = y.to(device)  # z target

            z_pred = model(user_feats, movie_feats, rs, u_idx, m_idx)
            loss = criterion(z_pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            total_loss += loss.item() * y.size(0)
            cnt += y.size(0)

        sched.step()
        train_loss = total_loss / max(cnt, 1)

        # 検証: ここでは raw MAE/RMSE/ρ を出す
        val_mae, val_rmse, val_rho = evaluate_with_raw(
            model, val_split, BATCH_SIZE, device, mu_map, sigma_map
        )

        improved = val_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "✓"
        else:
            no_improve += 1
            flag = " "

        print(f"Ep {ep:03d} | Loss {train_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f} {flag}")

        if no_improve >= patience:
            print(f"\nEarly stopping @ {ep}")
            break

    # テスト評価（ベストモデル）
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae, test_rmse, test_rho = evaluate_with_raw(
        model, test_split, BATCH_SIZE, device, mu_map, sigma_map
    )
    print("\n=== TEST ===")
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")

if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(SEED)
    train()
