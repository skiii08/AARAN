#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Feature Preparation (with Interaction features)
- active_daysなど特定列をlogスケーリング
- Interaction特徴を追加: genre, actor, director, keyword matching
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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


def compute_interaction_features(user_features, movie_features, user_idx, movie_idx):
    """
    Interaction特徴を計算:
    1. Genre interaction (19×19 = 361)
    2. Actor matching (1)
    3. Director matching (1)
    4. Keyword matching (1)
    """
    print("Computing interaction features...")

    n_samples = len(user_idx)

    # Feature indices (hardcoded - match dimension_metadata.py structure)
    # User: 0-17 aspect_zscore, 18-35 aspect_sentiment, 36-40 stats,
    #       41-341 fav_actor(300+1), 342-642 fav_director(300+1),
    #       643-681 genre(19+19), 682-686 behavior
    # Movie: 0-18 genre, 19-319 actor, 319-619 director, 619-919 keyword, ...

    user_genre_start = 643
    user_genre_end = 662  # 19 genres (rating)

    user_fav_actor_start = 41
    user_fav_actor_end = 341  # 300D embedding

    user_fav_director_start = 342
    user_fav_director_end = 642  # 300D embedding

    movie_genre_start = 0
    movie_genre_end = 19

    movie_actor_start = 19
    movie_actor_end = 319

    movie_director_start = 319
    movie_director_end = 619

    movie_keyword_start = 619
    movie_keyword_end = 919

    # Extract tensors
    user_genre_rating = user_features[:, user_genre_start:user_genre_end]  # (n_users, 19)
    user_fav_actor = user_features[:, user_fav_actor_start:user_fav_actor_end]  # (n_users, 300)
    user_fav_director = user_features[:, user_fav_director_start:user_fav_director_end]  # (n_users, 300)

    movie_genre_binary = movie_features[:, movie_genre_start:movie_genre_end]  # (n_movies, 19)
    movie_actor_emb = movie_features[:, movie_actor_start:movie_actor_end]  # (n_movies, 300)
    movie_director_emb = movie_features[:, movie_director_start:movie_director_end]  # (n_movies, 300)
    movie_keyword_emb = movie_features[:, movie_keyword_start:movie_keyword_end]  # (n_movies, 300)

    interaction_features = []

    for i in tqdm(range(n_samples), desc="  Computing interactions"):
        u_idx = user_idx[i]
        m_idx = movie_idx[i]

        # 1. Genre interaction (19×19 = 361)
        # user_genre_rating[u_idx]: (19,) - user's rating for each genre
        # movie_genre_binary[m_idx]: (19,) - movie's genre (0 or 1)
        # Outer product to get all pairwise interactions
        u_genre = user_genre_rating[u_idx].numpy()
        m_genre = movie_genre_binary[m_idx].numpy()
        genre_interact = np.outer(u_genre, m_genre).flatten()  # (361,)

        # 2. Actor matching (cosine similarity)
        u_actor = user_fav_actor[u_idx].numpy().reshape(1, -1)
        m_actor = movie_actor_emb[m_idx].numpy().reshape(1, -1)
        actor_match = cosine_similarity(u_actor, m_actor)[0, 0]

        # 3. Director matching
        u_director = user_fav_director[u_idx].numpy().reshape(1, -1)
        m_director = movie_director_emb[m_idx].numpy().reshape(1, -1)
        director_match = cosine_similarity(u_director, m_director)[0, 0]

        # 4. Keyword matching
        # Use user's fav_actor as proxy for keyword preference (简化版)
        # Better: compute from past high-rated movies' keywords
        # For now: cosine(user_fav_actor, movie_keyword) as placeholder
        keyword_match = cosine_similarity(u_actor, movie_keyword_emb[m_idx].numpy().reshape(1, -1))[0, 0]

        # Combine
        interact_vec = np.concatenate([
            genre_interact,  # 361
            [actor_match],  # 1
            [director_match],  # 1
            [keyword_match]  # 1
        ])  # Total: 364

        interaction_features.append(interact_vec)

    interaction_features = np.array(interaction_features)
    print(f"  Interaction features shape: {interaction_features.shape}")

    return interaction_features


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
    X_base = np.array(X_list)

    # Compute interaction features
    interaction_features = compute_interaction_features(
        user_features, movie_features, user_idx, movie_idx
    )

    # Concatenate
    X = np.concatenate([X_base, interaction_features], axis=1)
    print(f"  Final feature shape: {X.shape}")

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
    print("PHASE 1: FEATURE PREPARATION (with Interaction)")
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

    # Update FeatureGroupManager with new dimensions
    original_dim = merged_graph['user_features'].shape[1] + \
                   merged_graph['movie_features'].shape[1] + \
                   merged_graph['review_signals'].shape[1]
    new_dim = all_data['X'].shape[1]

    print(f"\n✅ Feature dimensions: {original_dim} → {new_dim} (+{new_dim - original_dim})")

    # ============================================================
    # 再構築: 正式な feature_groups.pkl（16グループ + interaction）
    # ============================================================
    import pickle as pkl
    from scripts.linear_explainable.models.feature_groups import FeatureGroupManager

    fg_path = DATA_DIR / "feature_groups.pkl"

    # --- 元構造を生成 ---
    fgm = FeatureGroupManager(user_dim=686, movie_dim=1202, review_dim=22)

    # --- 16既存グループを辞書化 ---
    groups_dict = {}
    for name, group in fgm.groups.items():
        groups_dict[name] = {
            "start": group.start_idx,
            "end": group.end_idx,
            "regularization": group.regularization,
            "description": group.description,
        }

    # --- interaction (364D) 追加 ---
    last_end = max(g["end"] for g in groups_dict.values())
    if new_dim - last_end == 364:
        groups_dict["interaction"] = {
            "start": last_end,
            "end": new_dim,
            "regularization": "l2",
            "description": "user×movie interaction features (genre×genre + actor/director/keyword match)"
        }
        print(f"✅ Added 'interaction' group: [{last_end}-{new_dim}] (364D)")
    else:
        raise ValueError(f"❌ Unexpected diff: {new_dim - last_end} (expected 364)")

    feature_groups = {"groups": groups_dict}

    # --- 保存 ---
    with open(fg_path, "wb") as f:
        pkl.dump(feature_groups, f)

    # ============================================================
    # ユーザー別 mean / std を保存 (for denormalization)
    # ============================================================
    print("\n[INFO] Computing per-user normalization stats...")

    # 各ユーザーの ratings_raw と ratings_user_norm の対応を利用
    user_indices = merged_graph["user_indices"].numpy()
    y_raw_all = merged_graph["ratings_raw"].numpy()
    y_norm_all = merged_graph["ratings_user_norm"].numpy()

    # ユーザー単位で mean, std を計算
    user_mean_map = {}
    user_std_map = {}
    for u in np.unique(user_indices):
        mask = user_indices == u
        y_u_raw = y_raw_all[mask]
        y_u_norm = y_norm_all[mask]
        # user_std = std(y_raw) = (y_raw - mean) / y_norm の逆から算出可
        mean_u = np.mean(y_u_raw)
        std_u = np.std(y_u_raw)
        if std_u < 1e-6:
            std_u = 1.0  # 安全対策
        user_mean_map[int(u)] = float(mean_u)
        user_std_map[int(u)] = float(std_u)

    # 各 split に埋め込む
    for split_name, data_dict in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data)
    ]:
        data_dict["user_mean_map"] = user_mean_map
        data_dict["user_std_map"] = user_std_map
        print(f"  ✓ Attached mean/std maps to {split_name}_data")


    print(f"✅ feature_groups.pkl saved with {len(feature_groups['groups'])} groups "
          f"({new_dim}D total)")
    save_features(train_data, OUTPUT_DIR / "linear_features_train.pkl")
    save_features(val_data, OUTPUT_DIR / "linear_features_val.pkl")
    save_features(test_data, OUTPUT_DIR / "linear_features_test.pkl")

    print("\n✅ Phase 1 (with Interaction) Complete!\n")


if __name__ == "__main__":
    main()