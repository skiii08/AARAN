#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Feature Preparation (Fixed Version)
- グループ別標準化でy（ユーザー内標準化）との整合性を確保
- user_aspect_sentimentはユーザー内標準化
- interaction特徴は外れ値処理後に全体標準化
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

    # Feature indices
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
    user_genre_rating = user_features[:, user_genre_start:user_genre_end]
    user_fav_actor = user_features[:, user_fav_actor_start:user_fav_actor_end]
    user_fav_director = user_features[:, user_fav_director_start:user_fav_director_end]

    movie_genre_binary = movie_features[:, movie_genre_start:movie_genre_end]
    movie_actor_emb = movie_features[:, movie_actor_start:movie_actor_end]
    movie_director_emb = movie_features[:, movie_director_start:movie_director_end]
    movie_keyword_emb = movie_features[:, movie_keyword_start:movie_keyword_end]

    interaction_features = []

    for i in tqdm(range(n_samples), desc="  Computing interactions"):
        u_idx = user_idx[i]
        m_idx = movie_idx[i]

        # 1. Genre interaction (19×19 = 361)
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

    # Verify user_aspect_sentiment is in correct range
    sentiment_start, sentiment_end = 18, 36
    sentiment_features = X_base[:, sentiment_start:sentiment_end]

    print(f"\n  [INFO] user_aspect_sentiment statistics:")
    print(f"    mean={np.mean(sentiment_features):.4f}, "
          f"range=[{np.min(sentiment_features):.2f}, {np.max(sentiment_features):.2f}]")

    if np.min(sentiment_features) < 0.5 or np.max(sentiment_features) > 5.5:
        print(f"    ⚠️  WARNING: Sentiment values outside expected [1,5] range")
    else:
        print(f"    ✓ Sentiment values in expected [1,5] range")

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


def apply_group_wise_standardization_fit(data_dict: dict, user_dim: int):
    """
    グループ別標準化: trainでfitしてscalerを返す

    NEW STRATEGY: user_aspect_sentimentは標準化しない（生の値を保持）
    理由: yはユーザー内標準化されているが、sentimentは元々意味のある値を持つ
          標準化すると情報が失われる

    Feature structure:
    [0-686]: user features
      [0-17]: user_aspect_zscore (skip - already normalized)
      [18-35]: user_aspect_sentiment (NO standardization - keep raw)
      [36-40]: user_stats (skip - preserve meaning)
      [41-686]: embeddings & genre (GLOBAL standardization)
    [687-1888]: movie features (GLOBAL standardization)
    [1889-1910]: review_signals (GLOBAL standardization)
    [1911-2274]: interaction features (GLOBAL standardization after clipping)
    """
    X = data_dict['X']
    user_indices = data_dict['user_indices']
    n_samples, n_features = X.shape

    X_standardized = X.copy()
    scalers = {}

    print("\n[Group-wise Standardization - FIT on TRAIN]")
    print("-" * 70)

    # 1. user_aspect_sentiment (18-35): NO STANDARDIZATION
    sentiment_start, sentiment_end = 18, 36
    print(f"1. user_aspect_sentiment [{sentiment_start}-{sentiment_end}]:")
    print(f"   Method: NO STANDARDIZATION (keep raw values)")
    print(f"   Reason: Preserve original sentiment scale for interpretability")

    sentiment_mean = np.mean(X[:, sentiment_start:sentiment_end])
    sentiment_std = np.std(X[:, sentiment_start:sentiment_end])
    sentiment_min = np.min(X[:, sentiment_start:sentiment_end])
    sentiment_max = np.max(X[:, sentiment_start:sentiment_end])

    print(f"   Raw statistics: mean={sentiment_mean:.4f}, std={sentiment_std:.4f}")
    print(f"                   range=[{sentiment_min:.4f}, {sentiment_max:.4f}]")
    print(f"   ✓ Keeping {sentiment_end - sentiment_start} dimensions as-is")

    # No changes to X_standardized for sentiment features
    scalers['sentiment'] = None

    # 2. user_aspect_zscore (0-17): SKIP (already normalized)
    zscore_start, zscore_end = 0, 18
    print(f"\n2. user_aspect_zscore [{zscore_start}-{zscore_end}]:")
    print(f"   Method: SKIP (already z-score normalized)")
    scalers['zscore'] = None

    # 3. user_stats (36-40): SKIP (preserve original meaning)
    stats_start, stats_end = 36, 41
    print(f"\n3. user_stats [{stats_start}-{stats_end}]:")
    print(f"   Method: SKIP (preserve rating statistics meaning)")
    scalers['stats'] = None

    # 4. Other user features (41-686): GLOBAL standardization
    user_other_indices = list(range(41, user_dim))
    if user_other_indices:
        print(f"\n4. user_embeddings & genre [{user_other_indices[0]}-{user_other_indices[-1] + 1}]:")
        print(f"   Method: GLOBAL standardization")
        scaler = StandardScaler()
        X_standardized[:, user_other_indices] = scaler.fit_transform(X[:, user_other_indices])
        scalers['user_other'] = scaler
        print(f"   ✓ Fitted and transformed {len(user_other_indices)} dimensions")

    # 5. Movie features: GLOBAL standardization
    movie_start = user_dim
    movie_end = movie_start + 1202
    print(f"\n5. movie_features [{movie_start}-{movie_end}]:")
    print(f"   Method: GLOBAL standardization")
    scaler = StandardScaler()
    X_standardized[:, movie_start:movie_end] = scaler.fit_transform(X[:, movie_start:movie_end])
    scalers['movie'] = scaler
    print(f"   ✓ Fitted and transformed {movie_end - movie_start} dimensions")

    # 6. Review signals: GLOBAL standardization
    review_start = movie_end
    review_end = review_start + 22
    print(f"\n6. review_signals [{review_start}-{review_end}]:")
    print(f"   Method: GLOBAL standardization")
    scaler = StandardScaler()
    X_standardized[:, review_start:review_end] = scaler.fit_transform(X[:, review_start:review_end])
    scalers['review'] = scaler
    print(f"   ✓ Fitted and transformed {review_end - review_start} dimensions")

    # 7. Interaction features: Outlier clipping + GLOBAL standardization
    interaction_start = review_end
    interaction_end = n_features
    print(f"\n7. interaction_features [{interaction_start}-{interaction_end}]:")
    print(f"   Method: Outlier clipping (1-99 percentile) + GLOBAL standardization")

    interaction_feats = X[:, interaction_start:interaction_end]

    # Clip outliers
    p1 = np.percentile(interaction_feats, 1, axis=0)
    p99 = np.percentile(interaction_feats, 99, axis=0)
    scalers['interaction_clip'] = {'p1': p1, 'p99': p99}

    interaction_clipped = np.clip(interaction_feats, p1, p99)

    # Global standardization
    scaler = StandardScaler()
    X_standardized[:, interaction_start:interaction_end] = scaler.fit_transform(interaction_clipped)
    scalers['interaction'] = scaler
    print(f"   ✓ Clipped and standardized {interaction_end - interaction_start} dimensions")

    print("-" * 70)

    data_dict['X'] = X_standardized
    return data_dict, scalers


def apply_group_wise_standardization_transform(data_dict: dict, scalers: dict, user_dim: int):
    """
    グループ別標準化: 学習済みscalerでtransform
    """
    X = data_dict['X']
    user_indices = data_dict['user_indices']
    n_samples, n_features = X.shape

    X_standardized = X.copy()

    print("\n[Group-wise Standardization - TRANSFORM]")
    print("-" * 70)

    # 1. user_aspect_sentiment: NO STANDARDIZATION
    sentiment_start, sentiment_end = 18, 36
    print(f"1. user_aspect_sentiment [{sentiment_start}-{sentiment_end}]: NO STANDARDIZATION")

    sentiment_mean = np.mean(X[:, sentiment_start:sentiment_end])
    print(f"   Raw statistics: mean={sentiment_mean:.4f}")
    print(f"   ✓ Keeping raw values")

    # 2-3. Skip groups
    print(f"2. user_aspect_zscore [0-18]: SKIP")
    print(f"3. user_stats [36-41]: SKIP")

    # 4. Other user features
    user_other_indices = list(range(41, user_dim))
    if user_other_indices and scalers['user_other'] is not None:
        print(f"4. user_embeddings & genre: TRANSFORM")
        X_standardized[:, user_other_indices] = scalers['user_other'].transform(X[:, user_other_indices])

    # 5. Movie features
    movie_start = user_dim
    movie_end = movie_start + 1202
    print(f"5. movie_features: TRANSFORM")
    X_standardized[:, movie_start:movie_end] = scalers['movie'].transform(X[:, movie_start:movie_end])

    # 6. Review signals
    review_start = movie_end
    review_end = review_start + 22
    print(f"6. review_signals: TRANSFORM")
    X_standardized[:, review_start:review_end] = scalers['review'].transform(X[:, review_start:review_end])

    # 7. Interaction features
    interaction_start = review_end
    interaction_end = n_features
    print(f"7. interaction_features: CLIP + TRANSFORM")

    interaction_feats = X[:, interaction_start:interaction_end]
    p1 = scalers['interaction_clip']['p1']
    p99 = scalers['interaction_clip']['p99']
    interaction_clipped = np.clip(interaction_feats, p1, p99)
    X_standardized[:, interaction_start:interaction_end] = scalers['interaction'].transform(interaction_clipped)

    print("-" * 70)

    data_dict['X'] = X_standardized
    return data_dict


def save_features(data: dict, path: Path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 1: FEATURE PREPARATION (Fixed - Group-wise Standardization)")
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

    # ============================================================
    # グループ別標準化を適用
    # ============================================================
    user_dim = 686
    movie_dim = 1202
    review_dim = 22

    fg_path = DATA_DIR / "feature_groups.pkl"
    if fg_path.exists():
        with open(fg_path, 'rb') as f:
            feature_groups = pickle.load(f)
    else:
        # Create default feature groups
        from scripts.linear_explainable.models.feature_groups import FeatureGroupManager
        fgm = FeatureGroupManager(user_dim=user_dim, movie_dim=movie_dim, review_dim=review_dim)
        groups_dict = {}
        for name, group in fgm.groups.items():
            groups_dict[name] = {
                "start": group.start_idx,
                "end": group.end_idx,
                "regularization": group.regularization,
                "description": group.description,
            }
        feature_groups = {"groups": groups_dict}

    print("\n" + "=" * 70)
    print("APPLYING GROUP-WISE STANDARDIZATION")
    print("=" * 70)

    # Train setでfitして、val/testはtransform
    print("\n=== TRAIN SET ===")
    train_data, scalers = apply_group_wise_standardization_fit(train_data, user_dim)

    print("\n=== VAL SET ===")
    val_data = apply_group_wise_standardization_transform(val_data, scalers, user_dim)

    print("\n=== TEST SET ===")
    test_data = apply_group_wise_standardization_transform(test_data, scalers, user_dim)

    # Update feature dimensions
    original_dim = user_dim + movie_dim + review_dim
    new_dim = all_data['X'].shape[1]

    print(f"\n✅ Feature dimensions: {original_dim} → {new_dim} (+{new_dim - original_dim})")

    # ============================================================
    # feature_groups.pklを更新（interaction追加 or 確認）
    # ============================================================
    groups_dict = feature_groups['groups']

    # interactionが既に存在するか確認
    if 'interaction' not in groups_dict:
        # interaction追加
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
    else:
        # 既に存在する場合は確認のみ
        interaction_info = groups_dict['interaction']
        interaction_size = interaction_info['end'] - interaction_info['start']
        if interaction_size == 364 and interaction_info['end'] == new_dim:
            print(
                f"✅ 'interaction' group already exists: [{interaction_info['start']}-{interaction_info['end']}] (364D)")
        else:
            print(f"⚠️  Warning: interaction group size mismatch. Expected 364D, got {interaction_size}D")

    feature_groups = {"groups": groups_dict}

    with open(fg_path, "wb") as f:
        pickle.dump(feature_groups, f)

  