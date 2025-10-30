# -*- coding: utf-8 -*-
"""
train_aaran_v3_full_ranked.py
AARAN v3 (Full Fusion) + Pairwise Ranking
- 学習: user-normalized rating を回帰（Focal + λ*MSE）
- 追加: 同一ユーザー内ペアに対する pairwise ranking loss（ヒンジ）を併用
- 評価: user別 μ,σ で逆正規化 → raw(1-10) で MAE/RMSE/ρ
- 既存の evaluate_raw_v3_full を使用（aspect_signals を自動抽出）
- チェックポイント: outputs/aaran/checkpoints/aaran_v3_full_ranked_YYYYMMDD_HHMM.pt
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# ===== Project Imports =====
from scripts.models.aaran_v3_full import AARANv3FullModel
from scripts.models.loss_functions import FocalRegressionLoss
from scripts.models.utils import build_user_stats
from scripts.models.eval import evaluate_raw # v3用（aspect_signals対応）

# -------------------------
# 乱数固定（再現性）
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# グラフ読み込み（v1/v2 と同形式）
# -------------------------
def load_graph(path: Path) -> dict:
    data = torch.load(path, weights_only=False)
    blobs = {
        "user_features": data["user_features"],       # (U, 686)
        "movie_features": data["movie_features"],     # (M, 1202)
        "user_indices": data["user_indices"],         # (E,)
        "movie_indices": data["movie_indices"],       # (E,)
        "review_signals": data["review_signals"],     # (E, 22)  # [:, :18] が aspect signals
        "ratings_user_norm": data["ratings_user_norm"],  # (E,)
        "ratings_raw": data["ratings_raw"],           # (E,)   # μ,σ 推定用
    }
    return blobs

# -------------------------
# バッチ生成（slice方式：v1/v2と統一）
# -------------------------
def make_batches(blobs: dict, batch_size: int):
    N = blobs["user_indices"].size(0)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        sl = slice(s, e)
        yield {
            "user_indices": blobs["user_indices"][sl],
            "movie_indices": blobs["movie_indices"][sl],
            "review_signals": blobs["review_signals"][sl],       # (B, 22)
            "ratings_user_norm": blobs["ratings_user_norm"][sl], # (B,)
        }

# -------------------------
# Pairwise Ranking Loss（同一ユーザー内）
# -------------------------
class PairwiseRankLoss(nn.Module):
    """
    簡易ヒンジ損失: max(0, margin - (ŷ_i - ŷ_j)) for pairs with y_i > y_j
    - 同一ユーザー内で、上位(positives)と下位(negatives)からランダムに K 組を作る
    - 入力は "normalized" 予測/真値（user-wise）
    """
    def __init__(self, margin: float = 0.3, num_neg: int = 2):
        super().__init__()
        self.margin = margin
        self.num_neg = num_neg

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        """
        y_hat: (B,)
        y_true:(B,)
        user_idx:(B,)
        """
        device = y_hat.device
        loss_terms = []
        # ユーザー毎にインデックスを収集
        u_cpu = user_idx.detach().cpu().numpy()
        by_user: Dict[int, np.ndarray] = {}
        for i, u in enumerate(u_cpu):
            by_user.setdefault(int(u), []).append(i)

        for u, idx_list in by_user.items():
            idx = torch.tensor(idx_list, dtype=torch.long, device=device)
            if idx.numel() < 2:
                continue
            y_u = y_true[idx]
            yhat_u = y_hat[idx]

            # 正例（高評価）と負例（低評価）を分ける簡易閾値
            # 連続値でも扱えるよう、中央値で二分
            med = torch.median(y_u)
            pos_mask = y_u > med
            neg_mask = y_u <= med
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                # すべて同評価近傍ならスキップ
                continue

            pos_idx = idx[pos_mask]
            neg_idx = idx[neg_mask]

            # K個までサンプル
            Kp = min(pos_idx.numel(), self.num_neg)
            Kn = min(neg_idx.numel(), self.num_neg)
            if Kp == 0 or Kn == 0:
                continue
            # ランダムサンプリング
            p_sel = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:Kp]]
            n_sel = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:Kn]]

            # 全組合せ (Kp x Kn) を作っても小規模なのでOK
            P = yhat_u[(p_sel - idx[0]) + idx[0]]  # yhat[p_sel]
            N = yhat_u[(n_sel - idx[0]) + idx[0]]  # yhat[n_sel]
            # Broadcasting
            diff = P.view(-1, 1) - N.view(1, -1)   # (Kp, Kn)
            hinge = torch.clamp(self.margin - diff, min=0.0)
            loss_terms.append(hinge.mean())

        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(loss_terms).mean()

# -------------------------
# 1epoch 学習
# -------------------------
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_blobs: dict,
    device: torch.device,
    focal_loss: nn.Module,
    mse_loss: nn.Module,
    rank_loss_fn: PairwiseRankLoss,
    lambda_reg: float,
    lambda_rank: float,
    batch_size: int,
) -> float:
    model.train()
    total = 0.0
    num_samples = train_blobs["user_indices"].size(0)

    # 特徴をデバイス常駐
    user_feat = train_blobs["user_features"].to(device)    # (U, 686)
    movie_feat = train_blobs["movie_features"].to(device)  # (M, 1202)

    for batch in make_batches(train_blobs, batch_size):
        u_idx = batch["user_indices"].to(device)        # (B,)
        m_idx = batch["movie_indices"].to(device)       # (B,)
        r_x   = batch["review_signals"].to(device)      # (B, 22)
        y_norm= batch["ratings_user_norm"].to(device)   # (B,)

        # gather features
        u_x = user_feat[u_idx]      # (B, 686)
        m_x = movie_feat[m_idx]     # (B, 1202)
        aspect = r_x[:, :18]        # (B, 18)  # v3 の AspectAttention 用

        # forward
        y_hat = model(u_x, m_x, r_x, aspect)  # (B,)

        # losses
        loss_core = focal_loss(y_hat, y_norm) + lambda_reg * mse_loss(y_hat, y_norm)
        loss_rank = rank_loss_fn(y_hat.detach(), y_norm.detach(), u_idx)  # 勾配爆発を避けるなら detach せず小さめでも可
        # ※ ランク損を pred にも流したい場合は detach を外す:
        # loss_rank = rank_loss_fn(y_hat, y_norm, u_idx)

        loss = loss_core + lambda_rank * loss_rank

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total += float(loss_core.item()) * u_idx.size(0)  # ログ用はコア損失を記録

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
    ckpt_path = OUT_DIR / f"aaran_v3_full_ranked_{timestamp}.pt"

    # --- データ ---
    train_blobs = load_graph(DATA_DIR / "hetero_graph_train.pt")
    val_blobs   = load_graph(DATA_DIR / "hetero_graph_val.pt")
    test_blobs  = load_graph(DATA_DIR / "hetero_graph_test.pt")

    # --- μ,σ を train から推定（raw逆変換用） ---
    mu_map, std_map = build_user_stats(
        train_blobs["user_indices"],
        train_blobs["ratings_raw"]
    )

    # --- モデル & 最適化 ---
    cfg = AARANv3FullModel()  # 既定設定（v3 Full Fusion）
    model = AARANv3FullModel().to(DEVICE)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    focal_loss  = FocalRegressionLoss(alpha=0.25, gamma=2.0)
    mse_loss    = nn.MSELoss()
    rank_loss   = PairwiseRankLoss(margin=0.3, num_neg=2)

    # --- ハイパラ ---
    EPOCHS   = 200
    BATCH    = 1024
    LAMBDA_R = 0.10   # 回帰のMSE係数（Focalに加算）
    LAMBDA_K = 0.50   # Ranking loss 係数
    PATIENCE = 30

    best = {"mae": 1e9, "rmse": 1e9, "rho": -1.0}
    bad = 0

    print(f"Device: {DEVICE}")
    print("Start training AARAN v3 (Full Fusion + Ranking)…")

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model, optimizer, train_blobs, DEVICE,
            focal_loss, mse_loss, rank_loss,
            LAMBDA_R, LAMBDA_K, BATCH
        )

        # v3用の評価（aspect_signalsを内部で取り出して使う）
        val_mae, val_rmse, val_rho = evaluate_raw(model, val_blobs, mu_map, std_map, DEVICE)

        scheduler.step(val_mae)
        print(f"Ep {ep:03d} | Loss {tr_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f}")

        if val_mae < best["mae"] - 1e-4:
            best = {"mae": val_mae, "rmse": val_rmse, "rho": val_rho}
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"✅ Saved best checkpoint → {ckpt_path}")
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            print(f"Early stopping @ {ep}")
            break

    # --- テスト（rawスケール） ---
    test_mae, test_rmse, test_rho = evaluate_raw(model, test_blobs, mu_map, std_map, DEVICE)
    print("=== TEST ===")
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")


if __name__ == "__main__":
    main()
