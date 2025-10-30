#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dualphase_usernorm.py

AARAN-DualPhaseモデル（User / Movie / Review分岐 MLP）を使用した学習スクリプト。
- ターゲット: ratings_user_norm
- 評価時にユーザーごとに逆正規化して10段階スケールに復元
- 出力: MAE / RMSE / Spearman
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LR = 1e-4
EPOCHS = 600
PATIENCE = 20
DROPOUT = 0.3


# =========================================================
# Model Definition
# =========================================================
class AARANDualPhase(nn.Module):
    """
    AARANに近い三分岐構造:
    User / Movie / Review の埋め込みをそれぞれMLPでエンコードして融合
    """
    def __init__(self, user_dim, movie_dim, review_dim, dropout=0.3):
        super().__init__()

        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )
        self.movie_encoder = nn.Sequential(
            nn.Linear(movie_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )
        self.review_encoder = nn.Sequential(
            nn.Linear(review_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 * 2 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # ← これを追加
        )

    def forward(self, g):
        u = g['user_features'][g['user_indices']]
        m = g['movie_features'][g['movie_indices']]
        r = g['review_signals']

        u = self.user_encoder(u)
        m = self.movie_encoder(m)
        r = self.review_encoder(r)

        x = torch.cat([u, m, r], dim=1)
        y_hat = self.fusion(x).squeeze(-1)
        return {'rating': y_hat}


# =========================================================
# Utility functions
# =========================================================
def load_graph(path: Path):
    data = torch.load(path, weights_only=False)
    user_idx = data['user_indices']
    movie_idx = data['movie_indices']
    user_names = data['user_names']

    y = data['ratings_user_norm']  # ← user_normを使用
    y_raw = data['ratings_raw']
    user_features = data['user_features']
    movie_features = data['movie_features']
    review_signals = data['review_signals']

    # user_nameごとのmean, stdを辞書として保持
    mean_map = data.get('user_mean_map', {})
    std_map = data.get('user_std_map', {})

    return {
        'user_features': user_features,
        'movie_features': movie_features,
        'review_signals': review_signals,
        'user_indices': user_idx,
        'movie_indices': movie_idx,
        'ratings_user_norm': y,
        'ratings_raw': y_raw,
        'user_names': user_names,
        'user_mean_map': mean_map,
        'user_std_map': std_map,
    }


def denormalize_per_user(pred, user_names, mean_map, std_map):
    preds_denorm = []
    for i, name in enumerate(user_names):
        mean = mean_map.get(name, 7.5)
        std = std_map.get(name, 2.0)
        preds_denorm.append(pred[i].item() * std + mean)
    return np.array(preds_denorm)


def evaluate(model, g, device):
    model.eval()
    with torch.no_grad():
        g = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in g.items()}
        out = model(g)

        pred = out['rating'].cpu().numpy()
        tgt_norm = g['ratings_user_norm'].cpu().numpy()
        user_idx = g['user_indices'].cpu().numpy()
        user_names = [g['user_names'][i] for i in user_idx]

        # denorm to 10-scale
        pred_denorm = denormalize_per_user(pred, user_names, g['user_mean_map'], g['user_std_map'])
        tgt_denorm = denormalize_per_user(tgt_norm, user_names, g['user_mean_map'], g['user_std_map'])

        mae = np.mean(np.abs(pred_denorm - tgt_denorm))
        rmse = np.sqrt(np.mean((pred_denorm - tgt_denorm) ** 2))
        spear = spearmanr(pred_denorm, tgt_denorm).correlation
    return mae, rmse, spear


# =========================================================
# Training
# =========================================================
if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data
    train_g = load_graph(DATA_DIR / "hetero_graph_train.pt")
    val_g = load_graph(DATA_DIR / "hetero_graph_val.pt")
    test_g = load_graph(DATA_DIR / "hetero_graph_test.pt")

    user_dim = train_g['user_features'].shape[1]
    movie_dim = train_g['movie_features'].shape[1]
    review_dim = train_g['review_signals'].shape[1]
    print(f"User:{user_dim}, Movie:{movie_dim}, Review:{review_dim}")
    print(f"Train samples: {len(train_g['ratings_user_norm'])}, Val: {len(val_g['ratings_user_norm'])}, Test: {len(test_g['ratings_user_norm'])}")

    model = AARANDualPhase(user_dim, movie_dim, review_dim, dropout=DROPOUT).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()

    best_val = float('inf')
    patience = 0

    # =====================================================
    # Training Loop
    # =====================================================
    for ep in range(1, EPOCHS + 1):
        model.train()
        g = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in train_g.items()}
        opt.zero_grad()
        out = model(g)
        loss = crit(out['rating'], g['ratings_user_norm'])
        loss.backward()
        opt.step()

        val_mae, val_rmse, val_spear = evaluate(model, val_g, device)
        log = f"Ep {ep:03d} | Loss {loss.item():.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_spear:.3f}"

        if val_mae < best_val:
            best_val = val_mae
            patience = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "dualphase_usernorm_best.pt")
            print(log + " ✓")
        else:
            patience += 1
            print(log)
            if patience >= PATIENCE:
                print(f"\nEarly stopping @ {ep}")
                break

    # =====================================================
    # Test Evaluation
    # =====================================================
    print("\n=== TEST ===")
    model.load_state_dict(torch.load(OUTPUT_DIR / "dualphase_usernorm_best.pt", weights_only=False))
    test_mae, test_rmse, test_spear = evaluate(model, test_g, device)
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_spear:.3f}")
