# -*- coding: utf-8 -*-
"""
train_aaran_v3_full_refined.py
AARAN v3 (Full Fusion) - 統一版トレーナー
- v2(GAT) と同じ訓練・評価設計
- 目的変数: user-normalized rating（学習）
- 評価: userごとに denorm → MAE/RMSE/ρ を 0-10 の raw スケールで算出
- モデル: AARANv3FullModel (Encoder + GAT + Aspect + Fusion)
- 損失: FocalRegressionLoss + λ * MSE
- チェックポイント: outputs/aaran/checkpoints/aaran_v3_full_YYYYMMDD_HHMM.pt
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import random
import os

import numpy as np
import torch
import torch.nn as nn

# ===== Imports =====
from scripts.models.aaran_v3_full import AARANv3FullModel
from scripts.models.loss_functions import FocalRegressionLoss
from scripts.models.utils import build_user_stats
from scripts.models.eval import evaluate_raw


# -------------------------
# 乱数固定
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# グラフ読み込み（v1/v2と同形式）
# -------------------------
def load_graph(path: Path) -> dict:
    data = torch.load(path, weights_only=False)
    blobs = {
        "user_features": data["user_features"],
        "movie_features": data["movie_features"],
        "user_indices": data["user_indices"],
        "movie_indices": data["movie_indices"],
        "review_signals": data["review_signals"],
        "ratings_user_norm": data["ratings_user_norm"],
        "ratings_raw": data["ratings_raw"],
    }
    return blobs


# -------------------------
# バッチ生成
# -------------------------
def make_batches(blobs: dict, batch_size: int):
    N = blobs["user_indices"].size(0)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        yield {
            "user_indices": blobs["user_indices"][s:e],
            "movie_indices": blobs["movie_indices"][s:e],
            "review_signals": blobs["review_signals"][s:e],
            "ratings_user_norm": blobs["ratings_user_norm"][s:e],
        }


# -------------------------
# 学習（1epoch）
# -------------------------
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_blobs: dict,
    device: torch.device,
    focal_loss: nn.Module,
    mse_loss: nn.Module,
    lambda_reg: float,
    batch_size: int,
) -> float:
    model.train()
    total = 0.0
    num_samples = train_blobs["user_indices"].size(0)

    user_feat = train_blobs["user_features"].to(device)
    movie_feat = train_blobs["movie_features"].to(device)

    for batch in make_batches(train_blobs, batch_size):
        u_idx = batch["user_indices"].to(device)
        m_idx = batch["movie_indices"].to(device)
        r_x = batch["review_signals"].to(device)
        y_norm = batch["ratings_user_norm"].to(device)

        u_x = user_feat[u_idx]
        m_x = movie_feat[m_idx]

        # ランダム aspect 信号（将来: review_signals から抽出に変更予定）
        aspect_signals = torch.randn(u_x.size(0), 18, device=device)

        y_hat = model(u_x, m_x, r_x, aspect_signals)
        loss = focal_loss(y_hat, y_norm) + lambda_reg * mse_loss(y_hat, y_norm)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total += float(loss.item()) * u_x.size(0)

    return total / num_samples


# -------------------------
# メイン
# -------------------------
def main():
    set_seed(42)

    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / "processed"
    OUT_DIR = ROOT / "outputs" / "aaran" / "checkpoints"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path = OUT_DIR / f"aaran_v3_full_{timestamp}.pt"

    # データ読み込み
    train_blobs = load_graph(DATA_DIR / "hetero_graph_train.pt")
    val_blobs = load_graph(DATA_DIR / "hetero_graph_val.pt")
    test_blobs = load_graph(DATA_DIR / "hetero_graph_test.pt")

    mu_map, std_map = build_user_stats(
        train_blobs["user_indices"],
        train_blobs["ratings_raw"],
    )

    # モデル構築
    model = AARANv3FullModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    focal_loss = FocalRegressionLoss(alpha=0.25, gamma=2.0)
    mse_loss = nn.MSELoss()

    # ハイパーパラメータ
    EPOCHS = 200
    BATCH = 1024
    LAMBDA = 0.1
    PATIENCE = 30

    best = {"mae": 1e9, "rmse": 1e9, "rho": -1.0}
    bad = 0

    print(f"Device: {DEVICE}")
    print("Start training AARAN v3 (Full Fusion)…")

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model, optimizer, train_blobs, DEVICE,
            focal_loss, mse_loss, LAMBDA, BATCH
        )

        val_mae, val_rmse, val_rho = evaluate_raw(model, val_blobs, mu_map, std_map, DEVICE)

        scheduler.step(val_mae)
        print(f"Ep {ep:03d} | Loss {tr_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f}")

        if val_mae < best["mae"] - 1e-4:
            best = {"mae": val_mae, "rmse": val_rmse, "rho": val_rho}
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"✅ Saved best checkpoint → {ckpt_path}")
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            print(f"Early stopping @ {ep}")
            break

    test_mae, test_rmse, test_rho = evaluate_raw(model, test_blobs, mu_map, std_map, DEVICE)
    print("=== TEST ===")
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")


if __name__ == "__main__":
    main()
