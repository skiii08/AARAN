#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_lambda_scaler.py (safe version)
グループごとの寄与とスパース率に基づいてλを自動生成。
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = BASE_DIR / "outputs" / "linear_explainable" / "analysis"
OUT_PATH = BASE_DIR / "outputs" / "linear_explainable" / "lambda_schedule.json"

def auto_lambda_scaler(groups_df, base_l1=0.01, base_l2=0.001):
    mean_contrib = groups_df["mean_abs_contribution"].mean()
    mean_sparse = groups_df["sparse_rate_1e-3"].mean()
    mean_size = groups_df["size"].mean()
    result = {}

    for _, row in groups_df.iterrows():
        g = row["group"]
        size = row["size"]
        contrib = row["mean_abs_contribution"]
        sparse = row["sparse_rate_1e-3"]

        contrib_factor = np.clip(mean_contrib / (contrib + 1e-6), 0.5, 2.5)
        sparse_factor = np.clip(1.0 + (sparse - mean_sparse), 0.5, 2.5)
        size_factor = np.sqrt(mean_size / size)

        # === 寄与分散ブースト ===
        diversity_boost = 1.0 + 0.4 * (1.0 - contrib / (mean_contrib + 1e-6))
        if sparse > mean_sparse * 1.2:
            diversity_boost *= 1.3  # スパースが高いほど緩和

        l1 = base_l1 * contrib_factor * sparse_factor * size_factor * diversity_boost
        l2 = base_l2 * (2.0 / contrib_factor) * (1.0 / sparse_factor) * size_factor / diversity_boost

        alpha = np.clip(0.5 + 0.3 * (contrib / (mean_contrib + 1e-6)), 0.2, 0.95)
        result[g] = {
            "lambda_l1": round(float(l1), 6),
            "lambda_l2": round(float(l2), 6),
            "alpha": round(float(alpha), 3),
        }
    return result

def main():
    csv_path = ANALYSIS_DIR / "contributions_by_group_train.csv"
    df = pd.read_csv(csv_path)

    # --- スパース率の列を探索的に補完 ---
    s_path = ANALYSIS_DIR / "group_stats.csv"
    if s_path.exists():
        s_df = pd.read_csv(s_path)
        sparse_col = None
        for c in s_df.columns:
            if "sparse_rate" in c and "1e-3" in c:
                sparse_col = c
                break
        if sparse_col is None:
            # 他のスパース率列を代用
            alt_cols = [c for c in s_df.columns if "sparse_rate" in c]
            if alt_cols:
                sparse_col = alt_cols[0]
                print(f"[WARN] Using alternate sparse rate column: {sparse_col}")
            else:
                s_df["sparse_rate_1e-3"] = 0.0
                sparse_col = "sparse_rate_1e-3"
                print("[WARN] No sparse_rate columns found — defaulting to 0.0")

        if sparse_col != "sparse_rate_1e-3":
            s_df = s_df.rename(columns={sparse_col: "sparse_rate_1e-3"})
        df = df.merge(s_df[["group", "sparse_rate_1e-3"]], on="group", how="left")

    else:
        df["sparse_rate_1e-3"] = 0.0
        print("[WARN] group_stats.csv not found — using 0.0 for all sparse rates.")

    result = auto_lambda_scaler(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"✅ lambda_schedule.json saved at:\n{OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
