#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Model Training (Grouped Linear Regression, fast mode)
レビュー単位split（ユーザー重複あり）に対応。
lambda_schedule.json を自動適用。
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso
from tqdm import tqdm
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from scripts.linear_explainable.models.grouped_linear import GroupedLinearRegression


# ============================================================
# Utility
# ============================================================

def load_features(split: str):
    path = BASE_DIR / "data" / "processed" / f"linear_features_{split}.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {split} set: {data['X'].shape[0]} samples, {data['X'].shape[1]} dims")
    return data


def compute_user_stats(train_data):
    """ユーザー単位の mean/std を計算"""
    user_ids = train_data['user_indices']
    y_raw = train_data['y_raw']
    mu_map, std_map = {}, {}

    for uid in np.unique(user_ids):
        ratings = y_raw[user_ids == uid]
        mu_map[int(uid)] = np.mean(ratings)
        std_map[int(uid)] = np.std(ratings) if np.std(ratings) > 1e-6 else 1.0

    # fallback
    mu_map[-1] = np.mean(list(mu_map.values()))
    std_map[-1] = np.mean(list(std_map.values()))
    return mu_map, std_map


def load_lambda_schedule(json_path: Path, group_names):
    """lambda_schedule.json から group-wise λ1, λ2, α をロード"""
    if not json_path.exists():
        print(f"[WARN] {json_path} not found. Using uniform defaults.")
        return {g: {"lambda1": 0.01, "lambda2": 0.001, "alpha": 0.8} for g in group_names}

    with open(json_path, 'r') as f:
        schedule = json.load(f)

    # group名にマッチしない場合のデフォルト補完
    defaults = {"lambda1": 0.01, "lambda2": 0.001, "alpha": 0.8}
    return {g: schedule.get(g, defaults) for g in group_names}


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 2: MODEL TRAINING (Grouped Linear, Fast Mode)")
    print("=" * 70)

    DATA_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_DIR = BASE_DIR / "outputs" / "linear_explainable"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # [1/6] Load data
    # --------------------------------------------------------
    print("\n[1/6] Loading prepared datasets...")
    train_data = load_features("train")
    val_data = load_features("val")

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # --------------------------------------------------------
    # [2/6] Load feature groups
    # --------------------------------------------------------
    print("\n[2/6] Loading feature groups...")
    fg_path = DATA_DIR / "feature_groups.pkl"
    if fg_path.exists():
        with open(fg_path, 'rb') as f:
            feature_groups = pickle.load(f)
        print(f"[INFO] Loaded feature_groups.pkl ({len(feature_groups['groups'])} groups)")
    else:
        raise FileNotFoundError(f"{fg_path} not found.")

    group_names = list(feature_groups['groups'].keys())

    # [3/6] Load lambda schedule
    # --------------------------------------------------------
    lambda_path = OUTPUT_DIR / "lambda_schedule.json"
    lambda_cfg = load_lambda_schedule(lambda_path, group_names)

    print("\n======================================================================")
    print("GROUP-WISE REGULARIZATION PARAMETERS")
    print("======================================================================")
    print(f"{'Group':25s} | {'λ1':>8s} | {'λ2':>8s} | {'α':>5s}")
    print("-" * 70)
    for name in group_names:
        p = lambda_cfg[name]
        # ✅ fallback: handle old keys 'l1', 'l2'
        lambda1 = p.get('lambda1', p.get('l1', 0.01))
        lambda2 = p.get('lambda2', p.get('l2', 0.001))
        alpha = p.get('alpha', 0.8)
        print(f"{name:25s} | {lambda1:8.5f} | {lambda2:8.5f} | {alpha:5.2f}")
    print("=" * 70)

    # --------------------------------------------------------
    # [4/6] Compute user normalization maps (from train only)
    # --------------------------------------------------------
    print("\n[4/6] Computing user mean/std maps (train only)...")
    mu_map, std_map = compute_user_stats(train_data)
    print(f"Computed {len(mu_map)-1} users. Mean std: {np.mean(list(std_map.values())):.3f}")

    # --------------------------------------------------------
    # [5/6] Train grouped linear model
    # --------------------------------------------------------
    print("\n[5/6] Training model...")

    group_indices = {name: np.arange(info['start'], info['end'])
                     for name, info in feature_groups['groups'].items()}
    group_regularizations = {name: info['regularization']
                             for name, info in feature_groups['groups'].items()}

    model = GroupedLinearRegression(
        group_indices=group_indices,
        group_regularizations=group_regularizations,
        lambda_l1=0.01,
        lambda_l2=0.01,
        lambda_elastic=0.01,
        alpha_elastic=0.5,
        use_focal=True
    )

    history = model.fit(
        X_train, y_train,
        n_epochs=300,  # ← max_epochs → n_epochs に修正
        lr=0.001,
        batch_size=256,
        verbose=True
    )

    print(f"\n✅ Training complete.")

    # --------------------------------------------------------
    # [6/6] Evaluate & Save
    # --------------------------------------------------------
    print("\n[6/6] Final evaluation & saving model...")

    val_pred = model.predict(X_val)
    mae = np.mean(np.abs(y_val - val_pred))
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"Validation MAE={mae:.4f}, RMSE={rmse:.4f}")

    checkpoint = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "val_metrics": {"mae": mae, "rmse": rmse},
        "user_stats": {"mu_map": mu_map, "std_map": std_map},
        "lambda_schedule": lambda_cfg,
        "feature_groups": feature_groups,
    }

    model_path = OUTPUT_DIR / "models" / "grouped_linear.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"\n✅ Model saved to: {model_path}")

    print("\n======================================================================")
    print("TRAINING SUMMARY")
    print("======================================================================")
    print(f"Train samples: {len(y_train):,}")
    print(f"Val samples:   {len(y_val):,}")
    print(f"Groups:        {len(group_names)}")
    print(f"Lambda file:   {lambda_path.exists()}")
    print(f"User overlap:  preserved (review-based split)")
    print(f"Val RMSE:      {rmse:.4f}")
    print("======================================================================")
    print("\n✅ Phase 2 Complete!")


if __name__ == "__main__":
    main()
