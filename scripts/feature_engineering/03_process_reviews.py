#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1-3 (修正版): Review特徴量の生成 + ランダムsplit
- ユーザー重複を許可（既知ユーザーの未評価映画予測タスク）
- 時系列を考慮しないランダム分割
- 22D review signals（18D aspect + 4D person attention）

修正内容:
- ユーザー排他的split → ランダムsplitに変更
- train/val/test = 70/12/18% の比率
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==================== Config ====================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REVIEWS_CSV = RAW_DIR / "reviews.csv"
MOVIE_ENTITIES_JSON = PROCESSED_DIR / "movie_entities.json"
ENTITY_EMBEDDINGS_PT = PROCESSED_DIR / "entity_embeddings.pt"

ASPECT_NAMES = [
    'acting_performance', 'artistic_design', 'audio_music',
    'casting_choices', 'character_development', 'commercial_context',
    'comparative_analysis', 'editing_pacing', 'emotion',
    'expectation', 'filmmaking_direction', 'genre_style',
    'recommendation', 'story_plot', 'technical_visuals',
    'themes_messages', 'viewing_experience', 'writing_dialogue'
]

# ==================== Load Data ====================
print("=" * 70)
print("PHASE 1-3 FIXED: REVIEW FEATURE ENGINEERING (Random Split)")
print("=" * 70)
print("\nLoading data...")

reviews_df = pd.read_csv(REVIEWS_CSV)
print(f"✓ Reviews: {len(reviews_df)}")

with open(MOVIE_ENTITIES_JSON, 'r', encoding='utf-8') as f:
    movie_entities_list = json.load(f)
movie_entities = {m['movie_id']: m for m in movie_entities_list}

entity_embeddings = torch.load(ENTITY_EMBEDDINGS_PT, weights_only=False)

print(f"✓ Movies: {len(movie_entities)}")

# ==================== 1. ASPECT SIGNALS ====================
print("\n" + "=" * 70)
print("1. ASPECT SIGNALS (18D)")
print("=" * 70)

mention_cols = [f'mention_{aspect}' for aspect in ASPECT_NAMES]
sentiment_cols = [f'sentiment_{aspect}' for aspect in ASPECT_NAMES]

mention_values = reviews_df[mention_cols].values
sentiment_values = reviews_df[sentiment_cols].values

# Signal = mention × (sentiment - 3.0)
aspect_signals = mention_values * (sentiment_values - 3.0)

print(f"✓ Aspect signals: {aspect_signals.shape}")
print(f"  Mean: {aspect_signals.mean():.4f}")
print(f"  Std:  {aspect_signals.std():.4f}")

# ==================== 2. PERSON ATTENTION ====================
print("\n" + "=" * 70)
print("2. PERSON ATTENTION FEATURES (4D)")
print("=" * 70)


def get_embedding(name, emb_dict):
    """Get embedding for a person name"""
    if not name or pd.isna(name):
        return None

    name_lower = str(name).lower()

    for original_key in emb_dict.keys():
        if original_key.lower() == name_lower:
            emb = emb_dict[original_key]
            if isinstance(emb, torch.Tensor):
                return emb.numpy()
            return np.array(emb)

    return None


def compute_person_attention(review_idx):
    """Compute 4D person attention for a review"""
    row = reviews_df.iloc[review_idx]
    movie_id = int(row['movie_id'])

    if movie_id not in movie_entities:
        return np.zeros(4)

    movie = movie_entities[movie_id]
    movie_actors = movie.get('actors', [])
    movie_directors = movie.get('directors', [])

    # Parse mentioned persons
    mentioned = row.get('person_name_list', None)
    if mentioned is None or pd.isna(mentioned) or mentioned == '':
        return np.zeros(4)

    # ★ 修正: JSONまたはPythonリテラルとしてパース
    try:
        if isinstance(mentioned, str):
            import ast
            mentioned_list = ast.literal_eval(mentioned)
        else:
            mentioned_list = mentioned

        if not isinstance(mentioned_list, list):
            return np.zeros(4)

        mentioned_names = [str(n).strip().lower() for n in mentioned_list if n]
    except:
        return np.zeros(4)

    if len(mentioned_names) == 0:
        return np.zeros(4)

    # Actor matching
    actor_match = 0.0
    for actor in movie_actors:
        if actor.lower() in mentioned_names:
            actor_match = 1.0
            break

    # Director matching
    director_match = 0.0
    for director in movie_directors:
        if director.lower() in mentioned_names:
            director_match = 1.0
            break

    # Sentiment集約
    actor_sentiment = 0.0
    director_sentiment = 0.0
    actor_count = 0
    director_count = 0

    actor_emb_dict = entity_embeddings.get('actors', {})
    director_emb_dict = entity_embeddings.get('directors', {})

    for name in mentioned_names:
        # Check if actor
        if name in [a.lower() for a in movie_actors]:
            actor_emb = get_embedding(name, actor_emb_dict)
            if actor_emb is not None:
                sentiment = float(row.get('sentiment_acting_performance', 3.0)) - 3.0
                actor_sentiment += sentiment
                actor_count += 1

        # Check if director
        if name in [d.lower() for d in movie_directors]:
            director_emb = get_embedding(name, director_emb_dict)
            if director_emb is not None:
                sentiment = float(row.get('sentiment_filmmaking_direction', 3.0)) - 3.0
                director_sentiment += sentiment
                director_count += 1

    # 平均を取る
    if actor_count > 0:
        actor_sentiment /= actor_count
    if director_count > 0:
        director_sentiment /= director_count

    return np.array([actor_match, director_match, actor_sentiment, director_sentiment])


print("Computing person attention features...")
person_attention_features = []
for i in tqdm(range(len(reviews_df)), desc="Reviews"):
    features = compute_person_attention(i)
    person_attention_features.append(features)

person_attention_features = np.array(person_attention_features)

print(f"✓ Person attention: {person_attention_features.shape}")
print(f"  Actor match rate: {(person_attention_features[:, 0] > 0).sum() / len(reviews_df) * 100:.1f}%")
print(f"  Director match rate: {(person_attention_features[:, 1] > 0).sum() / len(reviews_df) * 100:.1f}%")

# ==================== 3. COMBINE FEATURES ====================
print("\n" + "=" * 70)
print("3. COMBINING REVIEW FEATURES (22D)")
print("=" * 70)

review_features = np.concatenate([
    aspect_signals,  # 18D
    person_attention_features  # 4D
], axis=1)

print(f"✓ Review features shape: {review_features.shape}")
print(f"  Aspect signals:    {aspect_signals.shape[1]:2d}D")
print(f"  Person attention:  {person_attention_features.shape[1]:2d}D")
print(f"  {'─' * 40}")
print(f"  TOTAL:             {review_features.shape[1]:2d}D")

# ==================== 4. RANDOM SPLIT (ユーザー重複OK) ====================
print("\n" + "=" * 70)
print("4. RANDOM SPLIT (User Overlap Allowed)")
print("=" * 70)

n_reviews = len(reviews_df)
n_train = int(n_reviews * 0.70)
n_val = int(n_reviews * 0.12)
n_test = n_reviews - n_train - n_val

print(f"Total reviews: {n_reviews:,}")
print(f"  Train: {n_train:,} ({n_train / n_reviews * 100:.1f}%)")
print(f"  Val:   {n_val:,} ({n_val / n_reviews * 100:.1f}%)")
print(f"  Test:  {n_test:,} ({n_test / n_reviews * 100:.1f}%)")

# Random shuffle
np.random.seed(42)
indices = np.random.permutation(n_reviews)

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]

splits = np.empty(n_reviews, dtype='U5')
splits[train_indices] = 'train'
splits[val_indices] = 'val'
splits[test_indices] = 'test'

# Verify user overlap
train_users = set(reviews_df.iloc[train_indices]['user_name'].unique())
val_users = set(reviews_df.iloc[val_indices]['user_name'].unique())
test_users = set(reviews_df.iloc[test_indices]['user_name'].unique())

print("\n✓ User Statistics:")
print(f"  Train unique users: {len(train_users)}")
print(f"  Val unique users:   {len(val_users)}")
print(f"  Test unique users:  {len(test_users)}")
print(f"\n✓ User Overlap (Allowed):")
print(f"  Train ∩ Val:  {len(train_users & val_users)}")
print(f"  Train ∩ Test: {len(train_users & test_users)}")
print(f"  Val ∩ Test:   {len(val_users & test_users)}")

# ==================== 5. RATING NORMALIZATION ====================
print("\n" + "=" * 70)
print("5. RATING NORMALIZATION")
print("=" * 70)

ratings_raw = reviews_df['rating_raw'].values

# User-wise normalization
user_stats = {}
for user_name in reviews_df['user_name'].unique():
    user_mask = (reviews_df['user_name'] == user_name).values
    user_ratings = ratings_raw[user_mask]

    mu = user_ratings.mean()
    std = user_ratings.std()
    if std < 1e-6:
        std = 1.0

    user_stats[user_name] = {'mu': float(mu), 'std': float(std)}

ratings_user_norm = np.zeros_like(ratings_raw, dtype=np.float32)
for i, user_name in enumerate(reviews_df['user_name']):
    mu = user_stats[user_name]['mu']
    std = user_stats[user_name]['std']
    ratings_user_norm[i] = (ratings_raw[i] - mu) / std

print(f"✓ Raw ratings:  μ={ratings_raw.mean():.2f}, σ={ratings_raw.std():.2f}")
print(f"✓ Normalized:   μ={ratings_user_norm.mean():.2f}, σ={ratings_user_norm.std():.2f}")

# ==================== 6. SAVE ====================
print("\n" + "=" * 70)
print("6. SAVING FEATURES")
print("=" * 70)

review_features_dict = {
    'features': torch.tensor(review_features, dtype=torch.float32),
    'ratings_raw': torch.tensor(ratings_raw, dtype=torch.float32),
    'ratings_user_norm': torch.tensor(ratings_user_norm, dtype=torch.float32),
    'user_names': reviews_df['user_name'].values,
    'movie_ids': reviews_df['movie_id'].values,
    'split': splits,
    'aspect_names': ASPECT_NAMES,
    'feature_dims': {
        'aspect_signals': aspect_signals.shape[1],
        'person_attention': person_attention_features.shape[1],
        'total': review_features.shape[1]
    },
    'split_meta': {
        'method': 'random',
        'user_overlap': 'allowed',
        'train_size': int(n_train),
        'val_size': int(n_val),
        'test_size': int(n_test),
        'random_seed': 42
    },
    'user_stats': user_stats
}

output_path = PROCESSED_DIR / "review_features.pt"
torch.save(review_features_dict, output_path)
print(f"✅ Saved: {output_path}")

# ==================== 7. SUMMARY ====================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for split_name in ['train', 'val', 'test']:
    mask = splits == split_name
    n = mask.sum()
    r = ratings_raw[mask]

    print(f"\n{split_name.upper()}:")
    print(f"  Reviews: {n:,}")
    print(f"  Rating - Mean: {r.mean():.2f}, Std: {r.std():.2f}")

    low = ((r >= 1) & (r < 4)).sum()
    mid = ((r >= 4) & (r < 7)).sum()
    high = (r >= 7).sum()

    print(f"  Low (1-4):   {low:6d} ({low / n * 100:5.1f}%)")
    print(f"  Mid (4-7):   {mid:6d} ({mid / n * 100:5.1f}%)")
    print(f"  High (7-10): {high:6d} ({high / n * 100:5.1f}%)")

print("\n" + "=" * 70)
print("🎉 PHASE 1-3 FIXED COMPLETE!")
print("=" * 70)
print("\n✅ Data split method: Random (user overlap allowed)")
print("✅ This evaluates: Known user + unseen movie prediction")
print("\nNext step:")
print("  python scripts/feature_engineering/04_build_graph.py")