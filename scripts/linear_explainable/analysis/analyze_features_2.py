#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_analyze_features.py

線形（Grouped Regularized Linear）モデルのフィーチャ徹底分析。
出力: outputs/linear_explainable/analysis/
- top_features.csv（上位特徴の一覧）
- group_importance.csv（グループ別の重要度集計）
- group_stats.csv（グループ別スパース率/スケール/寄与など）
- feature_stats.csv（各特徴の重み・スケール指標）
- scaling_report.json（スケーリングの健全性チェック）
- corr_top_{split}.csv（各splitでの特徴–目的相関 上位/下位）
- multicollinearity_{split}.json（簡易多重共線性指標）
- contributions_by_group_{split}.csv（データ貢献度ベースのグループ重要度）
"""

import sys
from pathlib import Path
import pickle
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# ==== Paths ====
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
OUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==== IO helpers ====
def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ==== Core ====
def safe_float(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def percent(x):
    return float(x) * 100.0

def compute_feature_std(X: np.ndarray) -> np.ndarray:
    # 数値安定のための ddof=0
    return np.std(X, axis=0, ddof=0)

def signed_abs(x):
    return np.sign(x) * np.abs(x)

def condition_number(X_std: np.ndarray) -> float:
    # ざっくりなコンディション数
    try:
        u, s, vh = np.linalg.svd(X_std, full_matrices=False)
        s = s[s > 1e-12]
        if len(s) == 0:
            return float("inf")
        return float(s.max() / s.min())
    except Exception:
        return float("inf")


def gather_group_meta(feature_groups):
    groups = feature_groups["groups"]
    rows = []
    for name, info in groups.items():
        start, end = int(info["start"]), int(info["end"])
        rows.append({
            "group": name,
            "start": start,
            "end": end,
            "size": end - start,
            "regularization": info.get("regularization", "none")
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def analyze_weights(model, groups_df, top_k=100):
    w = model.coef_
    absw = np.abs(w)
    # 上位特徴
    idx_sorted = np.argsort(absw)[::-1]
    top_idx = idx_sorted[:top_k]

    top_rows = []
    for rank, i in enumerate(top_idx, 1):
        gname = groups_df.loc[(groups_df["start"] <= i) & (i < groups_df["end"]), "group"]
        gname = gname.iloc[0] if len(gname) else "(unknown)"
        top_rows.append({
            "rank": rank,
            "dim": int(i),
            "group": gname,
            "weight": float(w[i]),
            "abs_weight": float(absw[i])
        })
    top_df = pd.DataFrame(top_rows)

    # グループ集計
    group_stats = []
    for _, row in groups_df.iterrows():
        s, e, g = int(row["start"]), int(row["end"]), row["group"]
        ww = w[s:e]
        abs_sum = float(np.sum(np.abs(ww)))
        l2_sum = float(np.sum(ww ** 2))
        size = int(e - s)
        # スパース率（閾値別）
        thr1 = float(np.mean(np.abs(ww) < 1e-6))
        thr2 = float(np.mean(np.abs(ww) < 1e-3))
        thr3 = float(np.mean(np.abs(ww) < 1e-2))
        group_stats.append({
            "group": g,
            "size": size,
            "abs_weight_sum": abs_sum,
            "l2_weight_sum": l2_sum,
            "mean_abs_weight": float(np.mean(np.abs(ww))) if size > 0 else 0.0,
            "sparse_rate_1e-6": thr1,
            "sparse_rate_1e-3": thr2,
            "sparse_rate_1e-2": thr3,
        })
    group_df = pd.DataFrame(group_stats).sort_values("abs_weight_sum", ascending=False)
    return top_df, group_df, w


def analyze_scaling(X_train: np.ndarray, groups_df: pd.DataFrame, w: np.ndarray):
    std_all = compute_feature_std(X_train)
    # 正規化の影響を受けにくい “スケール補正重要度”: |w| * std
    scale_inv_imp = np.abs(w) * std_all

    feat_rows = []
    for i in range(len(w)):
        gname = groups_df.loc[(groups_df["start"] <= i) & (i < groups_df["end"]), "group"]
        gname = gname.iloc[0] if len(gname) else "(unknown)"
        feat_rows.append({
            "dim": int(i),
            "group": gname,
            "weight": float(w[i]),
            "abs_weight": float(np.abs(w[i])),
            "std_train": float(std_all[i]),
            "scale_invariant_importance": float(scale_inv_imp[i]),
        })
    feat_df = pd.DataFrame(feat_rows)

    # グループ単位のスケール偏り指標
    grp_rows = []
    for _, row in groups_df.iterrows():
        s, e, g = int(row["start"]), int(row["end"]), row["group"]
        std_g = std_all[s:e]
        sii_g = scale_inv_imp[s:e]
        grp_rows.append({
            "group": g,
            "size": int(e - s),
            "std_mean": float(np.mean(std_g)) if len(std_g) else 0.0,
            "std_median": float(np.median(std_g)) if len(std_g) else 0.0,
            "std_max": float(np.max(std_g)) if len(std_g) else 0.0,
            "sii_sum": float(np.sum(sii_g)),
            "sii_mean": float(np.mean(sii_g)) if len(sii_g) else 0.0,
        })
    grp_df = pd.DataFrame(grp_rows).sort_values("sii_sum", ascending=False)

    scaling_report = {
        "std_global": {
            "mean": float(np.mean(std_all)),
            "median": float(np.median(std_all)),
            "max": float(np.max(std_all)),
            "min": float(np.min(std_all)),
            "p95": float(np.percentile(std_all, 95)),
            "p99": float(np.percentile(std_all, 99)),
        },
        "groups_by_sii_sum": grp_df.head(20).to_dict(orient="records")
    }

    return feat_df, grp_df, scaling_report


def analyze_correlations(split_name: str, X: np.ndarray, y: np.ndarray, groups_df: pd.DataFrame, top_k=50):
    # 各特徴と y の相関（Pearson）
    # y がユーザ正規化スケールの想定（train/val/testで渡されるものをそのまま使う）
    Xm = X - X.mean(axis=0, keepdims=True)
    ym = y - y.mean()
    denom = (np.sqrt((Xm ** 2).sum(axis=0)) * np.sqrt((ym ** 2).sum()))
    denom[denom == 0] = np.inf
    corr = (Xm * ym.reshape(-1, 1)).sum(axis=0) / denom

    idx_sorted = np.argsort(np.abs(corr))[::-1]
    top_idx = idx_sorted[:top_k]
    bot_idx = idx_sorted[-top_k:][::-1]

    def rows_from_index(indices):
        rows = []
        for i in indices:
            gname = groups_df.loc[(groups_df["start"] <= i) & (i < groups_df["end"]), "group"]
            gname = gname.iloc[0] if len(gname) else "(unknown)"
            rows.append({
                "dim": int(i),
                "group": gname,
                "corr": float(corr[i]),
                "abs_corr": float(abs(corr[i]))
            })
        return pd.DataFrame(rows)

    top_df = rows_from_index(top_idx)
    bot_df = rows_from_index(bot_idx)

    top_df.to_csv(OUT_DIR / f"corr_top_{split_name}.csv", index=False)
    bot_df.to_csv(OUT_DIR / f"corr_bottom_{split_name}.csv", index=False)

    # 簡易多重共線性: 上位相関特徴の行列でコンディション数を観測
    sel = top_idx[: min(200, len(top_idx))]
    X_sel = X[:, sel]
    # 標準化
    std = X_sel.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    X_std = (X_sel - X_sel.mean(axis=0)) / std
    cond = condition_number(X_std)
    with open(OUT_DIR / f"multicollinearity_{split_name}.json", "w") as f:
        json.dump({"split": split_name, "condition_number": float(cond)}, f, indent=2)

    return top_df, bot_df, cond


def analyze_contributions_by_group(split_name: str, model, X: np.ndarray, groups_df: pd.DataFrame):
    # 線形モデルなので “寄与 = w * x” で平均絶対寄与をグループ集計
    w = model.coef_
    contrib = X * w  # shape (N, D)
    abs_contrib_mean = np.mean(np.abs(contrib), axis=0)  # (D,)

    rows = []
    for _, row in groups_df.iterrows():
        s, e, g = int(row["start"]), int(row["end"]), row["group"]
        vals = abs_contrib_mean[s:e]
        rows.append({
            "group": g,
            "size": int(e - s),
            "mean_abs_contribution": float(np.mean(vals)) if len(vals) else 0.0,
            "sum_abs_contribution": float(np.sum(vals)) if len(vals) else 0.0
        })
    df = pd.DataFrame(rows).sort_values("sum_abs_contribution", ascending=False)
    df.to_csv(OUT_DIR / f"contributions_by_group_{split_name}.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    # === Load model & meta ===
    ckpt = load_pickle(MODEL_DIR / "best_model.pkl")
    model = ckpt["model"]
    feature_groups = ckpt.get("feature_groups", load_pickle(DATA_DIR / "feature_groups.pkl"))
    groups_df = gather_group_meta(feature_groups)

    # === 基本の重み分析 ===
    top_df, group_imp_df, w = analyze_weights(model, groups_df, top_k=args.top_k)
    top_df.to_csv(OUT_DIR / "top_features.csv", index=False)
    group_imp_df.to_csv(OUT_DIR / "group_importance.csv", index=False)

    # === trainでスケーリングとスケール不変重要度 ===
    train = load_pickle(DATA_DIR / "linear_features_train.pkl")
    Xtr, ytr = train["X"], train["y"]
    feat_df, grp_scale_df, scaling_report = analyze_scaling(Xtr, groups_df, w)
    feat_df.to_csv(OUT_DIR / "feature_stats.csv", index=False)
    grp_scale_df.to_csv(OUT_DIR / "group_stats.csv", index=False)
    with open(OUT_DIR / "scaling_report.json", "w") as f:
        json.dump(scaling_report, f, indent=2)

    # === splitごとの相関 & 多重共線性 & グループ寄与 ===
    for sp in args.splits:
        d = load_pickle(DATA_DIR / f"linear_features_{sp}.pkl")
        X, y = d["X"], d["y"]  # y はユーザ正規化スケール前提
        analyze_correlations(sp, X, y, groups_df, top_k=min(args.top_k, 200))
        analyze_contributions_by_group(sp, model, X, groups_df)

    print("\n==============================================")
    print("✅ Feature analysis complete.")
    print("Outputs at:", OUT_DIR)
    print("  - top_features.csv")
    print("  - group_importance.csv")
    print("  - feature_stats.csv")
    print("  - group_stats.csv")
    print("  - scaling_report.json")
    print("  - corr_top_{split}.csv / corr_bottom_{split}.csv")
    print("  - multicollinearity_{split}.json")
    print("  - contributions_by_group_{split}.csv")
    print("==============================================\n")


if __name__ == "__main__":
    main()
