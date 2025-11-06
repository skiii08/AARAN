#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_model_clustered.py (standalone)
- interaction_genre のクラスタリング統合版
- λ調整: review_aspects / user_aspect_sentiment 強化
- グループindex自動補正対応
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, entropy, skew
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs" / "linear_explainable"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

from scripts.linear_explainable.models.grouped_linear import GroupedLinearRegression  # noqa


# =============================================================
# Utils
# =============================================================
def load_features(split: str):
    path = DATA_DIR / f"linear_features_{split}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {split}: {data['X'].shape[0]} samples × {data['X'].shape[1]} dims")
    return data


def compute_user_stats(train_data):
    user_ids = train_data["user_indices"]
    y_raw = train_data["y_raw"]
    mu_map, std_map = {}, {}
    for uid in np.unique(user_ids):
        ratings = y_raw[user_ids == uid]
        mu = float(np.mean(ratings))
        sd = float(np.std(ratings)) if np.std(ratings) > 1e-6 else 1.0
        mu_map[int(uid)] = mu
        std_map[int(uid)] = sd
    mu_map[-1] = float(np.mean(list(mu_map.values())))
    std_map[-1] = float(np.mean(list(std_map.values())))
    return mu_map, std_map


def create_feature_groups_with_interaction(original_groups: dict):
    groups = {}
    for name, info in original_groups.items():
        groups[name] = {
            "start": info["start"],
            "end": info["end"],
            "regularization": info.get("regularization", "elastic"),
        }
    # interaction追加（元構造を拡張）
    interaction_start = 1910
    groups["interaction_genre"] = {
        "start": interaction_start,
        "end": interaction_start + 361,
        "regularization": "l1",
    }
    groups["interaction_matching"] = {
        "start": interaction_start + 361,
        "end": interaction_start + 364,
        "regularization": "l2",
    }
    return {"groups": groups}


def generate_edgy_lambda_schedule(group_names):
    cfg = {}
    for g in group_names:
        if g in ("user_aspect_zscore", "user_aspect_sentiment", "user_stats", "user_behavior"):
            cfg[g] = {"lambda1": 0.003, "lambda2": 0.0005, "alpha": 0.8}
        elif g in ("user_genre",):
            cfg[g] = {"lambda1": 0.002, "lambda2": 0.0002, "alpha": 0.9}
        elif g.startswith("interaction_genre"):
            cfg[g] = {"lambda1": 0.002, "lambda2": 0.0002, "alpha": 0.9}
        elif g.startswith("interaction_matching"):
            cfg[g] = {"lambda1": 0.001, "lambda2": 0.0001, "alpha": 0.5}
        elif g in ("movie_actor", "movie_director"):
            cfg[g] = {"lambda1": 0.015, "lambda2": 0.002, "alpha": 0.7}
        elif g == "movie_genre":
            cfg[g] = {"lambda1": 0.010, "lambda2": 0.001, "alpha": 0.8}
        elif g in ("movie_keyword", "movie_basic", "movie_tags", "movie_review_agg"):
            cfg[g] = {"lambda1": 0.010, "lambda2": 0.001, "alpha": 0.8}
        else:
            cfg[g] = {"lambda1": 0.010, "lambda2": 0.001, "alpha": 0.8}
    return cfg


# =============================================================
# Diagnostics
# =============================================================
def get_model_weights(model) -> np.ndarray:
    if hasattr(model, "coef_") and isinstance(model.coef_, np.ndarray):
        return model.coef_.copy()
    if hasattr(model, "linear"):
        w = model.linear.weight.detach().cpu().numpy().flatten()
        return w.copy()
    if hasattr(model, "weights"):
        return np.asarray(model.weights).copy()
    raise AttributeError("Cannot access model weights.")


def analyze_consistency_and_plausibility(model, X, y):
    w = get_model_weights(model)
    y_pred = model.predict(X)
    resid = y - y_pred

    corrs = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        xi = X[:, i]
        if np.std(xi) < 1e-6:
            corrs[i] = np.nan
            continue
        try:
            corrs[i] = pearsonr(xi, y)[0]
        except Exception:
            corrs[i] = np.nan

    mismatch = np.sign(w) * np.sign(corrs) < 0
    strong = np.abs(w) > 0.05
    mismatch_rate_strong = float(np.nanmean(mismatch[strong])) if np.any(strong) else np.nan
    corr_pred_resid = float(np.corrcoef(y_pred, resid)[0, 1])
    CI = (1 - (mismatch_rate_strong if np.isfinite(mismatch_rate_strong) else 0.5)) * (1 - abs(corr_pred_resid))

    abs_w = np.abs(w) + 1e-12
    w_norm = abs_w / np.sum(abs_w)
    ent = float(entropy(w_norm, base=np.e))
    skw = float(skew(abs_w))
    ent_norm = np.clip((ent - 4.5) / 2.5, 0, 1)
    skew_norm = np.clip((skw - 1.8) / 4.2, 0, 1)
    PI = float(0.6 * ent_norm + 0.4 * skew_norm)

    outlier_count = int(np.sum(abs_w > 1.0))
    return {
        "consistency": {
            "sign_mismatch_strong": mismatch_rate_strong,
            "corr_pred_resid": corr_pred_resid,
            "ConsistencyIndex": CI,
        },
        "plausibility": {
            "entropy": ent,
            "skewness": skw,
            "PlausibilityIndex": PI,
        },
        "personality": {"outlier_weight_count_>1.0": outlier_count},
    }


def summarize_group_contributions(model, feature_groups):
    w = get_model_weights(model)
    group_abs = {}
    for gname, g in feature_groups["groups"].items():
        idx = np.arange(g["start"], g["end"])
        idx = idx[idx < len(w)]  # out-of-bounds安全対策
        val = float(np.sum(np.abs(w[idx])))
        group_abs[gname] = val
    total = sum(group_abs.values()) + 1e-12
    group_percent = {k: v / total * 100 for k, v in group_abs.items()}
    genre_keys = [k for k in group_abs if k in ("user_genre", "movie_genre", "interaction_genre")]
    genre_contrib_percent = {k: group_percent.get(k, 0) for k in genre_keys}
    genre_contrib_percent["genre_total"] = sum(genre_contrib_percent.values())
    return {"group_percent": group_percent, "genre_contrib_percent": genre_contrib_percent}


# =============================================================
# Clustering
# =============================================================
def cluster_interaction_genre(X, feature_groups, corr_threshold=0.99):
    g = feature_groups["groups"]["interaction_genre"]
    idx = np.arange(g["start"], g["end"])
    X_sub = X[:, idx]
    print(f"[INFO] Clustering interaction_genre ({X_sub.shape[1]} dims)...")

    corr = np.corrcoef(X_sub, rowvar=False)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    clustering = AgglomerativeClustering(
        metric="precomputed", linkage="average",
        distance_threshold=1 - corr_threshold, n_clusters=None
    )
    clustering.fit(dist)
    labels = clustering.labels_
    n_clusters = len(np.unique(labels))
    rep_cols = []
    for label in np.unique(labels):
        members = np.where(labels == label)[0]
        if len(members) == 1:
            rep_cols.append(members[0])
        else:
            subcorr = np.abs(corr[np.ix_(members, members)])
            mean_corr = np.mean(subcorr, axis=1)
            rep_cols.append(members[int(np.argmin(mean_corr))])
    rep_cols = sorted(rep_cols)
    info = {
        "original_dim": int(X_sub.shape[1]),
        "clustered_dim": len(rep_cols),
        "reduction_ratio": round(len(rep_cols) / X_sub.shape[1], 3),
        "clusters": n_clusters,
    }
    print(f"[INFO] Reduced interaction_genre: {info}")
    return X_sub[:, rep_cols], rep_cols, info


# =============================================================
# Main
# =============================================================
def main():
    print("=" * 70)
    print("PHASE 2: TRAIN (interaction_genre clustered)")
    print("=" * 70)

    train = load_features("train")
    val = load_features("val")
    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]

    with open(DATA_DIR / "feature_groups.pkl", "rb") as f:
        original_fg = pickle.load(f)
    feature_groups = create_feature_groups_with_interaction(original_fg["groups"])

    # クラスタリング
    Xc_train, rep_cols, info = cluster_interaction_genre(X_train, feature_groups)
    start = feature_groups["groups"]["interaction_genre"]["start"]
    end = feature_groups["groups"]["interaction_genre"]["end"]
    Xc_val = X_val[:, start:end][:, rep_cols]
    X_train = np.concatenate([X_train[:, :start], Xc_train, X_train[:, end:]], axis=1)
    X_val = np.concatenate([X_val[:, :start], Xc_val, X_val[:, end:]], axis=1)
    print(f"[INFO] Feature matrix updated: {X_train.shape}")

    # ✅ グループindex補正
    old_end = end
    reduced = len(rep_cols)
    delta = old_end - (start + reduced)
    feature_groups["groups"]["interaction_genre"]["end"] = start + reduced
    for name, g in feature_groups["groups"].items():
        if g["start"] >= old_end:
            g["start"] -= delta
            g["end"] -= delta
    print(f"[INFO] Adjusted feature_groups for reduction Δ={delta} (new total {X_train.shape[1]})")

    # ✅ 上限安全クリップ（IndexError防止）
    max_dim = X_train.shape[1]
    for name, g in feature_groups["groups"].items():
        if g["end"] > max_dim:
            g["end"] = max_dim
        if g["start"] >= max_dim:
            g["start"] = max_dim - 1
    print(f"[INFO] Clipped group index ranges to <= {max_dim - 1}")

    # λスケジュール生成＋感情強化
    lambda_cfg = generate_edgy_lambda_schedule(list(feature_groups["groups"].keys()))
    for g in ("review_aspects", "user_aspect_sentiment"):
        if g in lambda_cfg:
            lambda_cfg[g]["lambda1"] *= 0.5
            lambda_cfg[g]["lambda2"] *= 0.5

    mu_map, std_map = compute_user_stats(train)

    # モデル定義
    group_indices = {n: np.arange(g["start"], g["end"]) for n, g in feature_groups["groups"].items()}
    group_regs = {n: g["regularization"] for n, g in feature_groups["groups"].items()}
    model = GroupedLinearRegression(
        group_indices=group_indices,
        group_regularizations=group_regs,
        lambda_l1=0.01, lambda_l2=0.01,
        lambda_elastic=0.01, alpha_elastic=0.5,
        use_focal=True,
    )
    # 次元合わせ
    model.linear = nn.Linear(X_train.shape[1], 1)

    # 学習
    model.fit(X_train, y_train, n_epochs=300, lr=0.001, batch_size=256, verbose=True)

    # 評価
    val_pred = model.predict(X_val)
    mae = float(np.mean(np.abs(y_val - val_pred)))
    rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
    diag = analyze_consistency_and_plausibility(model, X_val, y_val)
    group_summary = summarize_group_contributions(model, feature_groups)

    summary = {
        "val": {"mae": mae, "rmse": rmse},
        "consistency": diag["consistency"],
        "plausibility": diag["plausibility"],
        "personality": diag["personality"],
        "genre_contrib_percent": group_summary["genre_contrib_percent"],
        "cluster_info": info,
    }

    # ✅ モデル保存ブロック追加
    checkpoint = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "val_metrics": {"mae": mae, "rmse": rmse},
        "diagnostics": diag,
        "feature_groups": feature_groups,
        "lambda_schedule": lambda_cfg,
    }

    model_path = MODEL_DIR / "best_model_clustered.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"[INFO] ✅ Model checkpoint saved -> {model_path}")

    out_path = OUT_DIR / "model_quality_summary_clustered.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TRAINING SUMMARY (clustered) ===")
    print(f"Val MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"Consistency={diag['consistency']['ConsistencyIndex']:.3f}, Plausibility={diag['plausibility']['PlausibilityIndex']:.3f}")
    print(f"Genre contrib %: {group_summary['genre_contrib_percent']}")
    print(f"interaction_genre reduced {info['original_dim']}→{info['clustered_dim']} ({info['reduction_ratio']*100:.1f}%)")
    print("======================================")
    print(f"✅ Saved summary: {out_path}")
    print("✅ Phase 2 (clustered) complete!\n")


if __name__ == "__main__":
    main()
