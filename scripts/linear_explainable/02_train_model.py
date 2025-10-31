"""
Phase 2: Model Training
Group正則化付き線形モデルの学習とハイパーパラメータ探索
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from scripts.linear_explainable.models.grouped_linear import create_grouped_model_from_feature_groups


def load_features(path: Path) -> dict:
    """特徴量データを読み込み"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_user_stats_from_train(train_data):
    """Trainデータからユーザー別のμ,σを計算"""
    user_indices = train_data['user_indices']
    y_raw = train_data['y_raw']

    mu_map = {}
    std_map = {}

    unique_users = np.unique(user_indices)

    for user_id in unique_users:
        mask = (user_indices == user_id)
        ratings = y_raw[mask]

        mu_map[int(user_id)] = float(ratings.mean())
        std = float(ratings.std())
        std_map[int(user_id)] = std if std > 1e-6 else 1.0

    # Global fallback
    mu_map[-1] = float(y_raw.mean())
    std_map[-1] = float(y_raw.std()) if y_raw.std() > 1e-6 else 1.0

    return mu_map, std_map


def denormalize_predictions(y_pred_norm, user_indices, mu_map, std_map):
    """User正規化された予測をRawスケール(1-10)に戻す"""
    y_pred_raw = []

    for i, user_id in enumerate(user_indices):
        mu = mu_map.get(int(user_id), mu_map[-1])
        sd = std_map.get(int(user_id), std_map[-1])

        y_raw = y_pred_norm[i] * sd + mu
        y_pred_raw.append(np.clip(y_raw, 1.0, 10.0))

    return np.array(y_pred_raw)


def evaluate(model, X, y_norm, y_raw, user_indices, mu_map, std_map):
    """評価指標を計算（Raw scale 1-10で評価）"""
    # Predict in normalized scale
    y_pred_norm = model.predict(X)

    # Denormalize to raw scale
    y_pred_raw = denormalize_predictions(y_pred_norm, user_indices, mu_map, std_map)

    # Evaluate on raw scale
    mae = np.mean(np.abs(y_pred_raw - y_raw))
    rmse = np.sqrt(np.mean((y_pred_raw - y_raw) ** 2))

    # Spearman correlation
    rx = np.argsort(np.argsort(y_pred_raw))
    ry = np.argsort(np.argsort(y_raw))
    rho = np.corrcoef(rx, ry)[0, 1]

    # Also compute normalized metrics for reference
    mae_norm = np.mean(np.abs(y_pred_norm - y_norm))
    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_norm) ** 2))

    return {
        'mae': mae,
        'rmse': rmse,
        'rho': rho,
        'mae_norm': mae_norm,
        'rmse_norm': rmse_norm
    }


# NOTE: この関数はevaluate内に統合されているため、ここでは冗長です
# def denormalize_predictions(y_norm, user_indices, train_data):
#     """User正規化された予測をrawスケール(1-10)に戻す
#
#     注: 簡易実装（全体平均で代用）
#     正確にはユーザー別のμ,σを使うべき
#     """
#     # 簡易版: 全体平均とstdで逆変換
#     global_mean = train_data['y_raw'].mean()
#     global_std = train_data['y_raw'].std()
#
#     y_raw = y_norm * global_std + global_mean
#     return np.clip(y_raw, 1.0, 10.0)


def grid_search(X_train, y_train, y_train_raw, train_user_indices,
                X_val, y_val, y_val_raw, val_user_indices,
                feature_groups, mu_map, std_map, n_trials=30):
    """簡易的なグリッドサーチ（修正版）"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH")
    print("=" * 70)

    best_mae = float('inf')
    best_params = None
    best_model = None

    # Parameter grid (修正: より小さい値の範囲)
    lambda_candidates = [0.0001, 0.001, 0.01]  # 0.1を削除
    alpha_candidates = [0.3, 0.5, 0.7]
    focal_candidates = [False]  # まずはFocal Lossなし

    print(f"\nSearching {n_trials} random combinations...")
    print("(Lambda range: [0.0001, 0.001, 0.01])")
    print("(Focal Loss: disabled for stability)\n")

    results = []

    # Random search
    np.random.seed(42)
    for trial in tqdm(range(n_trials), desc="Trials"):
        # Random sampling
        lambda_l1 = np.random.choice(lambda_candidates)
        lambda_l2 = np.random.choice(lambda_candidates)
        lambda_elastic = np.random.choice(lambda_candidates)
        alpha_elastic = np.random.choice(alpha_candidates)
        use_focal = np.random.choice(focal_candidates)

        try:
            model = create_grouped_model_from_feature_groups(
                feature_groups=feature_groups,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                lambda_elastic=lambda_elastic,
                alpha_elastic=alpha_elastic,
                use_focal=use_focal
            )

            model.fit(X_train, y_train, verbose=False)

            # Evaluate on validation set (RAW SCALE)
            metrics = evaluate(model, X_val, y_val, y_val_raw, val_user_indices, mu_map, std_map)

            results.append({
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'lambda_elastic': lambda_elastic,
                'alpha_elastic': alpha_elastic,
                'use_focal': use_focal,
                'val_mae': metrics['mae'],
                'val_rmse': metrics['rmse'],
                'val_rho': metrics['rho'],
                'val_mae_norm': metrics['mae_norm']
            })

            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_params = {
                    'lambda_l1': lambda_l1,
                    'lambda_l2': lambda_l2,
                    'lambda_elastic': lambda_elastic,
                    'alpha_elastic': alpha_elastic,
                    'use_focal': use_focal
                }
                best_model = model

        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            continue

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)
    print(f"\nBest validation MAE (raw 1-10): {best_mae:.4f}")
    print(f"Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Show top 3 results
    print(f"\nTop 3 configurations:")
    results_sorted = sorted(results, key=lambda x: x['val_mae'])[:3]
    for i, r in enumerate(results_sorted, 1):
        print(f"\n{i}. MAE={r['val_mae']:.4f}, ρ={r['val_rho']:.4f}")
        print(f"   λ1={r['lambda_l1']}, λ2={r['lambda_l2']}, λe={r['lambda_elastic']}")

    return best_model, best_params, results


def main():
    print("=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)

    DATA_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # --- Load data ---
    print("\n[1/5] Loading features...")
    train_data = load_features(DATA_DIR / "linear_features_train.pkl")
    val_data = load_features(DATA_DIR / "linear_features_val.pkl")

    X_train = train_data['X']
    y_train = train_data['y']  # user-normalized
    y_train_raw = train_data['y_raw']  # raw 1-10
    train_user_indices = train_data['user_indices']

    X_val = val_data['X']
    y_val = val_data['y']
    y_val_raw = val_data['y_raw']
    val_user_indices = val_data['user_indices']

    print(f"  Train: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
    print(f"  Val:   {X_val.shape[0]:,} samples × {X_val.shape[1]:,} features")

    # --- Build user statistics ---
    print("\n[2/5] Building user statistics...")
    mu_map, std_map = build_user_stats_from_train(train_data)
    print(f"  ✓ User statistics computed for {len(mu_map)-1} users")

    # --- Load feature groups ---
    print("\n[3/5] Loading feature groups...")
    with open(DATA_DIR / "feature_groups.pkl", 'rb') as f:
        feature_groups = pickle.load(f)

    print(f"  Total groups: {len(feature_groups['groups'])}")

    # --- Hyperparameter search ---
    print("\n[4/5] Hyperparameter search...")

    best_model, best_params, search_results = grid_search(
        X_train, y_train, y_train_raw, train_user_indices,
        X_val, y_val, y_val_raw, val_user_indices,
        feature_groups, mu_map, std_map,
        n_trials=30  # 30回のランダムサーチ
    )

    # --- Retrain on train set with best params ---
    print("\n[5/5] Retraining with best parameters...")

    final_model = create_grouped_model_from_feature_groups(
        feature_groups=feature_groups,
        **best_params
    )

    final_model.fit(X_train, y_train, verbose=True)

    # --- Evaluate ---
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    train_metrics = evaluate(final_model, X_train, y_train, y_train_raw, train_user_indices, mu_map, std_map)
    val_metrics = evaluate(final_model, X_val, y_val, y_val_raw, val_user_indices, mu_map, std_map)

    print("\nTrain (raw scale 1-10):")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  ρ:    {train_metrics['rho']:.4f}")
    print(f"  [Normalized MAE: {train_metrics['mae_norm']:.4f}]")

    print("\nValidation (raw scale 1-10):")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  ρ:    {val_metrics['rho']:.4f}")
    print(f"  [Normalized MAE: {val_metrics['mae_norm']:.4f}]")

    # --- Regularization summary ---
    final_model.print_regularization_summary()

    # --- Save model ---
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    model_path = OUTPUT_DIR / "best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'params': best_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_groups': feature_groups,
            'user_stats': {
                'mu_map': mu_map,
                'std_map': std_map
            },
            'timestamp': datetime.now().isoformat()
        }, f)

    print(f"✅ Model saved: {model_path}")

    # --- Save training log ---
    log_path = OUTPUT_DIR / "training_log.json"

    # ❗❗❗ 修正箇所: JSONエラー回避のためのbool->str変換 ❗❗❗
    best_params_clean = {k: str(v) if isinstance(v, bool) else v for k, v in best_params.items()}

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params_clean, # 修正後の変数を使用
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'search_results': search_results,
        'data_info': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'n_features': int(X_train.shape[1]),
            'n_users': len(mu_map) - 1
        }
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"✅ Training log saved: {log_path}")

    print("\n" + "=" * 70)
    print("✅ Phase 2 Complete!")
    print("=" * 70)
    print("\n📊 FINAL RESULTS (Raw Scale 1-10):")
    print(f"   Train MAE:  {train_metrics['mae']:.4f}")
    print(f"   Val MAE:    {val_metrics['mae']:.4f}")
    print(f"   Val ρ:      {val_metrics['rho']:.4f}")
    print("\nNext step:")
    print("  python scripts/linear_explainable/03_evaluate.py")


if __name__ == "__main__":
    main()