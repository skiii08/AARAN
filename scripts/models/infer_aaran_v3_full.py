# -*- coding: utf-8 -*-
"""
infer_aaran_v3_full.py
AARAN v3 Full Fusion 推論スクリプト
目的:
- テストデータでの推論 (rating予測)
- 中間表現 (h_gat, h_aspect, h_fused) の抽出
- 説明性評価や LLM連携の基盤データ生成
"""

import torch
from pathlib import Path
from collections import defaultdict

from scripts.models.aaran_v3_full import AARANv3FullModel
from scripts.models.utils import denorm_by_user, build_user_stats


def main():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / "processed"
    CKPT_PATH = ROOT / "outputs" / "aaran" / "checkpoints" / "aaran_v3_full_20251030_1720.pt"
    OUT_DIR = ROOT / "outputs" / "aaran" / "inference"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_PATH.name}")

    # --- モデル読込 ---
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = AARANv3FullModel().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- テストデータ読込 ---
    test = torch.load(DATA_DIR / "hetero_graph_test.pt", weights_only=False)
    u_feat = test["user_features"].to(DEVICE)
    m_feat = test["movie_features"].to(DEVICE)
    r_feat = test["review_signals"].to(DEVICE)
    u_idx = test["user_indices"].to(DEVICE)
    m_idx = test["movie_indices"].to(DEVICE)
    y_true = test["ratings_raw"].to(DEVICE)

    # --- アスペクト信号 (仮: 最初の18次元)
    aspect_signals = r_feat[:, :18]

    # --- μ,σ マップ構築 (denorm用) ---
    mu_map, std_map = build_user_stats(test["user_indices"], test["ratings_raw"])

    # --- フック登録 ---
    hooks = defaultdict(list)

    def register_hook(name):
        def hook(module, inp, out):
            # out が tuple (ctx, attn_weights) の場合があるため安全に処理
            if isinstance(out, tuple):
                # ctx 部分だけ保存
                hooks[name].append(out[0].detach().cpu())
            else:
                hooks[name].append(out.detach().cpu())

        return hook

    model.edge_gat.register_forward_hook(register_hook("h_gat"))
    model.aspect_att.register_forward_hook(register_hook("h_aspect"))
    model.fusion.register_forward_hook(register_hook("h_fused"))

    # --- 推論 ---
    BATCH = 512
    preds, users, movies = [], [], []

    model.eval()
    with torch.no_grad():
        for s in range(0, len(u_idx), BATCH):
            e = min(s + BATCH, len(u_idx))
            u_x = u_feat[u_idx[s:e]]
            m_x = m_feat[m_idx[s:e]]
            r_x = r_feat[s:e]
            asp = aspect_signals[s:e]
            y_hat = model(u_x, m_x, r_x, asp)

            preds.append(y_hat.cpu())
            users.append(u_idx[s:e].cpu())
            movies.append(m_idx[s:e].cpu())

    preds = torch.cat(preds)
    users = torch.cat(users)
    movies = torch.cat(movies)

    # --- Denorm (rawスケールに戻す) ---
    y_hat_raw = denorm_by_user(preds, users, mu_map, std_map)
    mae = torch.mean(torch.abs(y_hat_raw - y_true.cpu())).item()
    rmse = torch.sqrt(torch.mean((y_hat_raw - y_true.cpu()) ** 2)).item()

    print(f"✅ Inference done. Test MAE={mae:.3f}, RMSE={rmse:.3f}")

    # --- 保存 ---
    torch.save({
        "pred_raw": y_hat_raw,
        "pred_norm": preds,
        "user_idx": users,
        "movie_idx": movies,
        "h_gat": torch.cat(hooks["h_gat"]),
        "h_aspect": torch.cat(hooks["h_aspect"]),
        "h_fused": torch.cat(hooks["h_fused"]),
    }, OUT_DIR / "infer_v3_full_outputs.pt")

    print(f"💾 Saved inference results to: {OUT_DIR}/infer_v3_full_outputs.pt")


if __name__ == "__main__":
    main()
