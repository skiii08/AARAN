#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Model Evaluation
学習済み GroupedLinear モデルの評価・誤差分析・特徴量重要度の可視化（完全版）
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from datetime import datetime

# パス設定
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "outputs" / "linear_explainable" / "models" / "best_model.pkl"


# ============================================================
# Utility functions
# ============================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def band_classification(y_true, y_pred):
    """低・中・高の分類精度を算出（閾値: <4=low, 4-7=mid, >=7=high）"""
    def to_band(y):
        if y < 4: return 'low'
        elif y < 7: return 'mid'
        else: return 'high'

    true_bands = np.array([to_band(v) for v in y_true])
    pred_bands = np.array([to_band(v) for v in y_pred])

    classes = ['low', 'mid', 'high']
    confusion = {t: {p: 0 for p in classes} for t in classes}

    for t, p in zip(true_bands, pred_bands):
        confusion[t][p] += 1

    accuracy = float(np.mean(true_bands == pred_bands))
    return accuracy, confusion

def plot_error_distribution(errors, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.title("Error Distribution (|y_pred - y_true|)")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 3: MODEL EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # [1/5] Load model
    # ------------------------------------------------------------
    print("\n[1/5] Loading model...")
    with open(MODEL_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    model = checkpoint["model"]
    mu_map = checkpoint["user_stats"]["mu_map"]
    std_map = checkpoint["user_stats"]["std_map"]
    print(f"  Model trained at: {checkpoint['timestamp']}")
    print(f"  Validation MAE: {checkpoint['val_metrics']['mae']:.4f}")

    # ------------------------------------------------------------
    # [2/5] Load test data
    # ------------------------------------------------------------
    print("\n[2/5] Loading test data...")
    with open(DATA_DIR / "linear_features_test.pkl", "rb") as f:
        test_data = pickle.load(f)

    X_test = test_data["X"]
    y_test_raw = test_data["y_raw"]         # Raw 1–10
    y_test_norm = test_data["y"]            # user-normalized
    user_indices = test_data["user_indices"]
    print(f"  Test samples: {len(y_test_raw):,}")

    # ------------------------------------------------------------
    # [3/5] Make predictions
    # ------------------------------------------------------------
    print("\n[3/5] Making predictions...")

    def denormalize_predictions(y_pred_norm, user_indices, mu_map, std_map):
        y_pred_raw = np.empty_like(y_pred_norm, dtype=float)
        for i, user_id in enumerate(user_indices):
            mu = mu_map.get(int(user_id), mu_map[-1])
            sd = std_map.get(int(user_id), std_map[-1])
            val = y_pred_norm[i] * sd + mu
            y_pred_raw[i] = np.clip(val, 1.0, 10.0)
        return y_pred_raw

    y_pred_norm = model.predict(X_test)
    y_pred_raw  = denormalize_predictions(y_pred_norm, user_indices, mu_map, std_map)

    # ------------------------------------------------------------
    # [4/5] Compute metrics (+ robust debug)
    # ------------------------------------------------------------
    print("\n[4/5] Computing metrics...")

    # === DEBUG: スケール/ID整合性チェック ===
    print("\n======================================================================")
    print("DEBUG: SCALE / ID CONSISTENCY CHECK")
    print("======================================================================")

    # 1) ユーザー空間の重なり
    train_user_ids = set(int(k) for k in mu_map.keys() if isinstance(k, int) or (isinstance(k, str) and k.isdigit()))
    test_user_ids = set(int(u) for u in np.unique(user_indices))
    overlap = len(train_user_ids & test_user_ids)
    unknown_users = len(test_user_ids - train_user_ids)
    tot_test_users = len(test_user_ids) if len(test_user_ids) > 0 else 1
    print(f"Train user count: {len(train_user_ids)}")
    print(f"Test user count:  {len(test_user_ids)}")
    print(f"Overlap users:    {overlap} ({overlap / tot_test_users * 100:.2f}%)")
    print(f"Unknown users:    {unknown_users} ({unknown_users / tot_test_users * 100:.2f}%)")

    # 2) Raw スケールの統計
    print("\n-- Value Stats (RAW) --")
    print(f"y_test_raw: mean={np.mean(y_test_raw):.3f}, std={np.std(y_test_raw):.3f}, "
          f"min={np.min(y_test_raw):.3f}, max={np.max(y_test_raw):.3f}")
    print(f"y_pred_raw: mean={np.mean(y_pred_raw):.3f}, std={np.std(y_pred_raw):.3f}, "
          f"min={np.min(y_pred_raw):.3f}, max={np.max(y_pred_raw):.3f}")

    # 3) Normalized スケールの統計
    print("\n-- Value Stats (NORMALIZED) --")
    print(f"y_test_norm: mean={np.mean(y_test_norm):.3f}, std={np.std(y_test_norm):.3f}, "
          f"min={np.min(y_test_norm):.3f}, max={np.max(y_test_norm):.3f}")
    print(f"y_pred_norm: mean={np.mean(y_pred_norm):.3f}, std={np.std(y_pred_norm):.3f}, "
          f"min={np.min(y_pred_norm):.3f}, max={np.max(y_pred_norm):.3f}")

    mae_norm_v = mae(y_test_norm, y_pred_norm)
    rmse_norm_v = rmse(y_test_norm, y_pred_norm)
    print(f"\nMAE/RMSE on normalized scale: {mae_norm_v:.4f} / {rmse_norm_v:.4f}")

    # 4) 逆正規化の範囲外チェック
    out_of_range = int(np.sum((y_pred_raw < 0) | (y_pred_raw > 10)))
    print(f"\nOut-of-range predictions: {out_of_range}/{len(y_pred_raw)}")
    print("======================================================================\n")

    # === 公式指標（RAW） ===
    test_mae = mae(y_test_raw, y_pred_raw)
    test_rmse = rmse(y_test_raw, y_pred_raw)
    rho = float(spearmanr(y_test_raw, y_pred_raw).correlation)

    print("\n" + "=" * 70)
    print("TEST RESULTS (Raw 1–10 scale)")
    print("=" * 70)
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"ρ:    {rho:.4f}")

    # バンド別
    print("\n" + "=" * 70)
    print("BY RATING BAND")
    print("=" * 70)
    bands = {
        "LOW": y_test_raw < 4,
        "MID": (y_test_raw >= 4) & (y_test_raw < 7),
        "HIGH": y_test_raw >= 7
    }
    for band_name, mask in bands.items():
        if np.any(mask):
            mae_band = mae(y_test_raw[mask], y_pred_raw[mask])
            rmse_band = rmse(y_test_raw[mask], y_pred_raw[mask])
            print(f"\n{band_name} ({int(mask.sum())} samples):")
            print(f"  MAE:  {mae_band:.4f}")
            print(f"  RMSE: {rmse_band:.4f}")
        else:
            print(f"\n{band_name} (0 samples):")
            print("  MAE:  n/a")
            print("  RMSE: n/a")

    # バンド分類
    acc, confusion = band_classification(y_test_raw, y_pred_raw)
    print("\n" + "=" * 70)
    print("BAND CLASSIFICATION ACCURACY")
    print("=" * 70)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print("        Predicted →")
    print("True ↓        low      mid     high")
    for t in ['low', 'mid', 'high']:
        row = confusion[t]
        print(f"     {t:<5}  {row['low']:8d} {row['mid']:8d} {row['high']:8d}")

    # ------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    errors = np.abs(y_pred_raw - y_test_raw)
    print(f"Mean absolute error: {float(errors.mean()):.4f}")
    print(f"Std of absolute error: {float(errors.std()):.4f}")
    print(f"Max absolute error: {float(errors.max()):.4f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {float(np.percentile(errors, p)):.4f}")

    large_err = int((errors > 2.0).sum())
    print(f"\nLarge errors (|error| > 2.0): {large_err} ({large_err / len(errors) * 100:.2f}%)")

    # Save histogram
    plot_error_distribution(errors, OUTPUT_DIR / "error_analysis.png")

    # ------------------------------------------------------------
    # [5/5] Feature Importance
    # ------------------------------------------------------------
    print("\n[5/5] Analyzing feature importance...")
    try:
        # 実装済みなら優先
        importance = model.get_feature_importance()
    except Exception:
        # フォールバック: |w|
        if hasattr(model, "linear") and hasattr(model.linear, "weight"):
            importance = np.abs(model.linear.weight.detach().cpu().numpy().ravel())
        else:
            importance = None

    if importance is not None:
        top_idx = np.argsort(importance)[::-1][:30]
        print("\n" + "=" * 70)
        print("TOP 30 IMPORTANT FEATURES")
        print("=" * 70)
        for rank, idx in enumerate(top_idx, 1):
            print(f"{rank:2d}. Dim {idx:4d} | |w| = {float(importance[idx]):.6f}")
    else:
        print("[WARN] Feature importance extraction failed: model has no accessible weights.")

    # ------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mae": test_mae,
        "rmse": test_rmse,
        "rho": rho,
        "band_accuracy": acc,
        "confusion_matrix": confusion,
        "error_stats": {
            "mean": float(errors.mean()),
            "std": float(errors.std()),
            "max": float(errors.max()),
        },
        "debug": {
            "users": {
                "train_user_count": len(train_user_ids),
                "test_user_count": len(test_user_ids),
                "overlap": overlap,
                "unknown": unknown_users
            },
            "raw_stats": {
                "y_test_raw": {
                    "mean": float(np.mean(y_test_raw)),
                    "std": float(np.std(y_test_raw)),
                    "min": float(np.min(y_test_raw)),
                    "max": float(np.max(y_test_raw)),
                },
                "y_pred_raw": {
                    "mean": float(np.mean(y_pred_raw)),
                    "std": float(np.std(y_pred_raw)),
                    "min": float(np.min(y_pred_raw)),
                    "max": float(np.max(y_pred_raw)),
                },
            },
            "norm_stats": {
                "y_test_norm": {
                    "mean": float(np.mean(y_test_norm)),
                    "std": float(np.std(y_test_norm)),
                    "min": float(np.min(y_test_norm)),
                    "max": float(np.max(y_test_norm)),
                },
                "y_pred_norm": {
                    "mean": float(np.mean(y_pred_norm)),
                    "std": float(np.std(y_pred_norm)),
                    "min": float(np.min(y_pred_norm)),
                    "max": float(np.max(y_pred_norm)),
                },
                "mae_norm": mae_norm_v,
                "rmse_norm": rmse_norm_v,
            },
            "out_of_range_pred_raw": out_of_range
        }
    }

    save_path = OUTPUT_DIR / "test_metrics.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved: {save_path}")
    print(f"✅ Error distribution plot saved: {OUTPUT_DIR / 'error_analysis.png'}")

    print("\n" + "=" * 70)
    print("✅ Phase 3 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
