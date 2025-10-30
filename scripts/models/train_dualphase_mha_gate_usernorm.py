#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dualphase_mha_gate_usernorm.py

AARAN Phase 2.6 + 2.7:
- Multi-Head CrossAttention (user → [movie_token, review_token])
- Review-Gate (review_signalsからのゲートで寄与制御)
- user_norm で学習し、評価は per-user で 0-10 スケールに復元

入出力前提（04_build_graph.py の最新仕様に対応）:
- hetero_graph_{train,val,test}.pt に以下が含まれる:
  'user_features' : (U, Du)
  'movie_features': (M, Dm)
  'user_indices'  : (E,)
  'movie_indices' : (E,)
  'review_signals': (E, Dr)  # 22D (18D + 4D)
  'ratings_raw'   : (E,)
  'ratings_user_norm': (E,)
  'user_names', 'movie_ids' などのメタ

実行例:
  /Users/watanabesaki/PycharmProjects/AARAN/.venv/bin/python \
    /Users/watanabesaki/PycharmProjects/AARAN/scripts/models/train_dualphase_mha_gate_usernorm.py
"""

import os
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# Config
# =====================
FILE_PATH = Path(__file__).resolve()
SCRIPT_DIR = FILE_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # .../AARAN
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 400
BATCH_SIZE = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 40

# モデル幅
H_USER = 256
H_MOV  = 256
H_REV  = 64
H_FUSE = 256
ATTN_HEADS = 4      # Phase 2.6: Multi-Head
D_MODEL   = 256     # Attentionの埋め込み次元（head_numで割り切れるように）
D_HEAD    = D_MODEL // ATTN_HEADS

# =====================
# Utils
# =====================
def load_graph(path: Path):
    data = torch.load(path, weights_only=False)

    user_feats   = data["user_features"]         # (U, Du)
    movie_feats  = data["movie_features"]        # (M, Dm)
    user_indices = data["user_indices"].long()   # (E,)
    movie_indices= data["movie_indices"].long()  # (E,)
    review_sigs  = data["review_signals"].float()# (E, Dr)
    y_raw        = data["ratings_raw"].float()   # (E,)
    y_norm       = data["ratings_user_norm"].float() # (E,)

    # テンソル化（CPU保持、バッチ時に.to(DEVICE)）
    return {
        "user_feats":   user_feats,
        "movie_feats":  movie_feats,
        "user_idx":     user_indices,
        "movie_idx":    movie_indices,
        "review_sigs":  review_sigs,
        "y_raw":        y_raw,
        "y_norm":       y_norm,
    }

def build_user_denorm_maps(train_dict):
    """
    0-10スケール復元用の per-user μ/σ を train から算出
    """
    user_idx = train_dict["user_idx"].cpu().numpy()
    y_raw    = train_dict["y_raw"].cpu().numpy()

    mu_map = {}
    std_map = {}
    df = defaultdict(list)
    for u, y in zip(user_idx, y_raw):
        df[int(u)].append(float(y))

    for u, ys in df.items():
        mu = float(np.mean(ys))
        sd = float(np.std(ys, ddof=0))
        if sd < 1e-6:  # 安全策
            sd = 1.0
        mu_map[u] = mu
        std_map[u] = sd
    return mu_map, std_map

def denorm_by_user(pred_norm: torch.Tensor, user_idx: torch.Tensor, mu_map, std_map):
    """
    y_hat_raw = y_hat_norm * σ_u + μ_u
    未知ユーザーは全体平均/標準偏差でデノーマライズ
    """
    # グローバル平均/分散の安全なfallback
    global_mu = float(np.mean(list(mu_map.values())))
    global_std = float(np.mean(list(std_map.values())))

    mus, sigmas = [], []
    for u in user_idx.cpu().tolist():
        mu = mu_map.get(int(u), global_mu)
        sd = std_map.get(int(u), global_std)
        mus.append(mu)
        sigmas.append(sd)

    mu = torch.tensor(mus, dtype=torch.float32, device=pred_norm.device)
    sd = torch.tensor(sigmas, dtype=torch.float32, device=pred_norm.device)
    return pred_norm * sd + mu

def metric_mae(pred, true):
    return float(torch.mean(torch.abs(pred - true)).item())

def metric_rmse(pred, true):
    return float(torch.sqrt(torch.mean((pred - true) ** 2)).item())

def metric_spearman(pred, true):
    # TorchでSpearman近似（降順rank）
    x = pred.detach().cpu().numpy()
    y = true.detach().cpu().numpy()
    # ランク化（同順位は平均順位）
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    xr = xr.astype(np.float32)
    yr = yr.astype(np.float32)
    xr = (xr - xr.mean()) / (xr.std() + 1e-8)
    yr = (yr - yr.mean()) / (yr.std() + 1e-8)
    return float((xr * yr).mean())

def batch_iter(idx, bs):
    for i in range(0, len(idx), bs):
        yield idx[i:i+bs]

# =====================
# Model
# =====================
class ReviewGate(nn.Module):
    """
    Phase 2.7: review_signals からスカラーgateを生成（0〜1）
    """
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, r):  # (B, Dr)
        return self.mlp(r).squeeze(-1)  # (B,)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
            nn.Dropout(p)
        )
    def forward(self, x): return self.net(x)

class MultiHeadCrossAttention(nn.Module):
    """
    Query: user_token (B, D)
    Key/Value: concat([movie_token, review_token], dim=1) => (B, 2, D)
    出力: (B, D)
    """
    def __init__(self, d_model=256, nhead=4, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv_tokens):
        """
        q: (B, D)
        kv_tokens: (B, T=2, D)
        """
        B, T, D = kv_tokens.shape
        # Project
        Q = self.Wq(q)            # (B, D)
        K = self.Wk(kv_tokens)    # (B, T, D)
        V = self.Wv(kv_tokens)    # (B, T, D)

        # Split heads
        Q = Q.view(B, self.nhead, self.d_head)                 # (B, H, Dh)
        K = K.view(B, T, self.nhead, self.d_head).transpose(1,2)  # (B, H, T, Dh)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1,2)  # (B, H, T, Dh)

        # Attention: (B, H, 1, Dh) x (B, H, T, Dh)^T -> (B, H, T)
        Q = Q.unsqueeze(2)  # (B, H, 1, Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, 1, T)
        attn = torch.softmax(scores, dim=-1)  # (B, H, 1, T)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)  # (B, H, 1, Dh)
        ctx = ctx.squeeze(2).contiguous().view(B, self.d_model)  # (B, D)
        out = self.out(ctx)  # (B, D)
        return out

class AARAN_DualPhase_MHAGate(nn.Module):
    """
    - user/movie をそれぞれエンコード（D_MODELへ）
    - review_signals を小MLPで token化して D_MODEL にマップ
    - CrossAttention(user → [movie, review])
    - ReviewGate で review_token の寄与を制御（movie_token: 1-g, review_token: g）
      -> kv_tokensの組成時に重み付け、さらに出力後の融合MLP
    - 最終は user_norm を予測（学習時MSE）
    """
    def __init__(self, du, dm, dr, d_model=256, nhead=4):
        super().__init__()
        self.user_enc  = MLP(du, H_USER, d_model)
        self.movie_enc = MLP(dm, H_MOV,  d_model)
        self.rev_enc   = nn.Sequential(
            nn.Linear(dr, H_REV), nn.ReLU(inplace=True),
            nn.Linear(H_REV, d_model)
        )
        self.attn = MultiHeadCrossAttention(d_model=d_model, nhead=nhead, dropout=0.0)
        self.rev_gate = ReviewGate(in_dim=dr, hidden=64)  # scalar gate in [0,1]

        # 融合後の回帰ヘッド（DualPhase: core + calibrator を1本化して安定性を優先）
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, H_FUSE), nn.ReLU(inplace=True),
            nn.Linear(H_FUSE, 1)
        )

    def forward(self, u_feat, m_feat, r_sig):
        # Encode
        u = self.user_enc(u_feat)      # (B, D)
        m = self.movie_enc(m_feat)     # (B, D)
        r = self.rev_enc(r_sig)        # (B, D)

        # Gate
        g = self.rev_gate(r_sig)       # (B,)
        g = g.unsqueeze(-1)            # (B, 1)

        # kv_tokensの重み付け（movie: 1-g, review: g）
        m_w = (1.0 - g) * m
        r_w = g * r
        kv = torch.stack([m_w, r_w], dim=1)  # (B, 2, D)

        # CrossAttention (user -> kv)
        ctx = self.attn(u, kv)         # (B, D)

        # user と ctx を結合して回帰
        z = torch.cat([u, ctx], dim=-1)
        y_hat_norm = self.fuse(z).squeeze(-1)  # (B,)
        return y_hat_norm, g.squeeze(-1)       # ついでにgateも返す


# =====================
# Training
# =====================
def forward_pass(model, batch, blobs, return_gate=False):
    u_idx = blobs["user_idx"][batch]
    m_idx = blobs["movie_idx"][batch]

    u_feat = blobs["user_feats"][u_idx].to(DEVICE)
    m_feat = blobs["movie_feats"][m_idx].to(DEVICE)
    r_sig  = blobs["review_sigs"][batch].to(DEVICE)

    y_norm = blobs["y_norm"][batch].to(DEVICE)
    y_raw  = blobs["y_raw"][batch].to(DEVICE)

    y_hat_norm, gate = model(u_feat, m_feat, r_sig)
    if return_gate:
        return y_hat_norm, y_norm, y_raw, u_idx.to(DEVICE), gate
    else:
        return y_hat_norm, y_norm, y_raw, u_idx.to(DEVICE)

def evaluate(model, blobs, mu_map, std_map, batch_size=8192):
    model.eval()
    idx = torch.arange(len(blobs["user_idx"]))
    preds_norm, truths_norm, preds_raw, truths_raw = [], [], [], []
    with torch.no_grad():
        for b in batch_iter(idx, batch_size):
            y_hat_norm, y_norm, y_raw, u_idx = forward_pass(model, b, blobs)
            y_hat_raw = denorm_by_user(y_hat_norm, u_idx, mu_map, std_map)

            preds_norm.append(y_hat_norm)
            truths_norm.append(y_norm)
            preds_raw.append(y_hat_raw)
            truths_raw.append(y_raw.to(DEVICE))

    ypn = torch.cat(preds_norm); ytn = torch.cat(truths_norm)
    ypr = torch.cat(preds_raw);  ytr = torch.cat(truths_raw)

    mae = metric_mae(ypr, ytr)
    rmse = metric_rmse(ypr, ytr)
    rho = metric_spearman(ypn, ytn)  # 順位は user_norm で評価
    return mae, rmse, rho

def main():
    print(f"Device: {DEVICE}")
    # --------------- Load ---------------
    train = load_graph(DATA_DIR / "hetero_graph_train.pt")
    val   = load_graph(DATA_DIR / "hetero_graph_val.pt")
    test  = load_graph(DATA_DIR / "hetero_graph_test.pt")

    # 共通辞書（学習効率のためCPU保持）
    blobs_train = train
    blobs_val   = val
    blobs_test  = test

    Du = blobs_train["user_feats"].shape[1]
    Dm = blobs_train["movie_feats"].shape[1]
    Dr = blobs_train["review_sigs"].shape[1]
    print(f"User:{Du}, Movie:{Dm}, Review:{Dr}")

    # --------------- User denorm maps ---------------
    mu_map, std_map = build_user_denorm_maps(blobs_train)

    # --------------- Model ---------------
    model = AARAN_DualPhase_MHAGate(du=Du, dm=Dm, dr=Dr, d_model=D_MODEL, nhead=ATTN_HEADS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

    best_val = float("inf")
    best_pack = None
    bad = 0

    train_idx = torch.randperm(len(blobs_train["user_idx"]))

    print(f"Train samples: {len(train_idx)}, Val: {len(blobs_val['user_idx'])}, Test: {len(blobs_test['user_idx'])}")

    for ep in range(1, EPOCHS+1):
        model.train()
        ep_loss = 0.0

        for b in batch_iter(train_idx, BATCH_SIZE):
            opt.zero_grad()
            y_hat_norm, y_norm, _, _ = forward_pass(model, b, blobs_train)
            loss = F.mse_loss(y_hat_norm, y_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item() * len(b)

        ep_loss /= len(train_idx)

        # ---- Eval ----
        val_mae, val_rmse, val_rho = evaluate(model, blobs_val, mu_map, std_map)
        test_mae, test_rmse, test_rho = evaluate(model, blobs_test, mu_map, std_map)  # モニタとして毎epoch測る

        scheduler.step(val_mae)

        improved = val_mae < best_val - 1e-4
        mark = "✓" if improved else " "
        print(f"Ep {ep:03d} | Loss {ep_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f}{mark}")

        if improved:
            best_val = val_mae
            bad = 0
            # ベスト保存パック
            best_pack = {
                "ep": ep,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val": {"mae": val_mae, "rmse": val_rmse, "rho": val_rho},
                "test": {"mae": test_mae, "rmse": test_rmse, "rho": test_rho},
                "config": {
                    "H_USER": H_USER, "H_MOV": H_MOV, "H_REV": H_REV, "H_FUSE": H_FUSE,
                    "D_MODEL": D_MODEL, "ATTN_HEADS": ATTN_HEADS
                }
            }
        else:
            bad += 1

        if bad >= PATIENCE:
            print(f"\nEarly stopping @ {ep}")
            break

    # --------------- Final Test ---------------
    if best_pack is not None:
        # ベストでロードし直して評価
        model.load_state_dict(best_pack["state_dict"])
        test_mae, test_rmse, test_rho = evaluate(model, blobs_test, mu_map, std_map)
        print("\n=== TEST ===")
        print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")

        # 保存
        save_path = OUTPUT_DIR / "v2_6_2_7_dualphase_mha_gate_best.pt"
        torch.save(best_pack, save_path)
        print(f"\nSaved best checkpoint → {save_path}")
    else:
        print("No improvement observed; model not saved.")

if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
