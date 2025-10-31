#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Feature Preparation (improved)
- active_daysなど特定列をlogスケーリング
- 標準化をより安定化
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from scripts.linear_explainable.models.feature_groups import FeatureGroupManager
from scripts.linear_explainable.utils.dimension_metadata import DimensionMetadataGenerator


# ============================================================
# Utility
# ============================================================

def load_graph_data(split: str, data_dir: Path):
    path = data_dir / f"hetero_graph_{split}.pt"
    print(f"Loading {path}...")
    return torch.load(path, weights_only=False)


def merge_graphs(graph_list):
    print("Merging graphs...")
    user_features = graph_list[0]['user_features']
    movie_features = graph_list[0]['movie_features']

    user_idx = torch.cat([g['user_indices'] for g in graph_list])
    movie_idx = torch.cat([g['movie_indices'] for g in graph_list])
    review_signals = torch.cat([g['review_signals'] for g in graph_list])
    y_norm = torch.cat([g['ratings_user_norm'] for g in graph_list])
    y_raw = torch.cat([g['ratings_raw'] for g in graph_list])

    return {
        'user_features': user_features,
        'movie_features': movie_features,
        'user_indices': user_idx,
        'movie_indices': movie_idx,
        'review_signals': review_signals,
        'ratings_user_norm': y_norm,
        'ratings_raw': y_raw,
    }


# === ✨ manual scaling of extreme columns ===
def apply_manual_feature_scaling(user_features: torch.Tensor, feature_names: list):
    scaled = user_features.clone()
    scaler = MinMaxScaler()
    for i, name in enumerate(feature_names):
        if "active_days" in name:
            x = user_features[:, i].numpy()
            x = np.log1p(np.clip(x, 0, None))
            x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            scaled[:, i] = torch.tensor(x, dtype=torch.float32)
            print(f"[ManualScale] log1p+minmax applied: {name}")
    return scaled


def build_edge_features(graph_dict: dict) -> dict:
    user_features = graph_dict['user_features']
    movie_features = graph_dict['movie_features']
    review_signals = graph_dict['review_signals']
    user_idx = graph_dict['user_indices']
    movie_idx = graph_dict['movie_indices']
    y_norm = graph_dict['ratings_user_norm']
    y_raw = graph_dict['ratings_raw']

    print("  Building edge features...")
    X_list = []
    for i in tqdm(range(len(user_idx)), desc="  Edges"):
        u = user_features[user_idx[i]].numpy()
        m = movie_features[movie_idx[i]].numpy()
        r = review_signals[i].numpy()
        X_list.append(np.concatenate([u, m, r]))
    X = np.array(X_list)

    return {
        'X': X,
        'y': y_norm.numpy(),
        'y_raw': y_raw.numpy(),
        'user_indices': user_idx.numpy(),
        'movie_indices': movie_idx.numpy(),
    }


def standardize_split(train, val, test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train['X'])
    X_val_scaled = scaler.transform(val['X'])
    X_test_scaled = scaler.transform(test['X'])
    train['X'], val['X'], test['X'] = X_train_scaled, X_val_scaled, X_test_scaled
    return scaler


def save_features(data: dict, path: Path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 1: FEATURE PREPARATION (improved)")
    print("=" * 70)

    DATA_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_DIR = DATA_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graphs = [load_graph_data(s, DATA_DIR) for s in ["train", "val", "test"]]
    merged_graph = merge_graphs(graphs)

    # feature_names読み出し
    feature_meta_path = DATA_DIR / "dimension_metadata.json"
    feature_names = []
    if feature_meta_path.exists():
        import json
        feature_names = list(json.load(open(feature_meta_path)).keys())

    merged_graph['user_features'] = apply_manual_feature_scaling(
        merged_graph['user_features'], feature_names
    )

    all_data = build_edge_features(merged_graph)
    n = len(all_data['y'])
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.6, random_state=42)

    def subset(data, indices):
        return {k: (v[indices] if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    train_data = subset(all_data, train_idx)
    val_data = subset(all_data, val_idx)
    test_data = subset(all_data, test_idx)

    scaler = standardize_split(train_data, val_data, test_data)

    manager = FeatureGroupManager(
        user_dim=merged_graph['user_features'].shape[1],
        movie_dim=merged_graph['movie_features'].shape[1],
        review_dim=merged_graph['review_signals'].shape[1]
    )
    manager.print_summary()

    save_features(train_data, OUTPUT_DIR / "linear_features_train.pkl")
    save_features(val_data, OUTPUT_DIR / "linear_features_val.pkl")
    save_features(test_data, OUTPUT_DIR / "linear_features_test.pkl")

    print("\n✅ Phase 1 (improved) Complete!\n")


if __name__ == "__main__":
    main()
