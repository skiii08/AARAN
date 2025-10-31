#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check for abnormal feature scales or outliers
"""
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = DATA_DIR / "linear_features_train.pkl"

def main():
    print("=== Feature Sanity Check ===")
    with open(TRAIN_PATH, "rb") as f:
        train = pickle.load(f)

    X = train["X"]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    maxv = X.max(axis=0)
    minv = X.min(axis=0)

    # 異常基準
    extreme_idx = np.where((np.abs(mean) > 5) | (std > 5) | (np.abs(maxv) > 20))[0]
    zero_var_idx = np.where(std < 1e-6)[0]

    print(f"Total dims: {X.shape[1]}")
    print(f"→ Extreme dims: {len(extreme_idx)}")
    print(f"→ Zero-var dims: {len(zero_var_idx)}")

    if len(extreme_idx):
        print("\n--- Potential Outliers ---")
        for i in extreme_idx[:20]:
            print(f"Dim {i:4d} | mean={mean[i]:+.3f} | std={std[i]:.3f} | range=({minv[i]:+.2f},{maxv[i]:+.2f})")
    else:
        print("✅ No extreme features found.")

    if len(zero_var_idx):
        print("\n--- Zero-Variance Features ---")
        print(zero_var_idx[:20])
    else:
        print("✅ No zero-variance features found.")

    print("\nDone.")

if __name__ == "__main__":
    main()
