"""
Phase 1-4 (改訂版): 異種グラフの構築
- User-Movie-Review グラフ
- PyTorch形式で保存（DGL不使用）
- 妥当性検証付き
- ✨ 22D review signals対応 (18D + 4D person attention)

配置: scripts/feature_engineering/04_build_graph.py
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== Config ====================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REVIEW_FEATURES = PROCESSED_DIR / "review_features.pt"
USER_FEATURES = PROCESSED_DIR / "user_features.pt"
MOVIE_FEATURES = PROCESSED_DIR / "movie_features.pt"

# ==================== Load Data ====================
print("="*70)
print("PHASE 1-4: GRAPH CONSTRUCTION (22D Review Signals)")
print("="*70)
print("\nLoading processed features...")

review_data = torch.load(REVIEW_FEATURES, weights_only=False)
user_data = torch.load(USER_FEATURES, weights_only=False)
movie_data = torch.load(MOVIE_FEATURES, weights_only=False)

print(f"✓ Users: {len(user_data['user_names'])}")
print(f"✓ Movies: {len(movie_data['movie_ids'])}")
print(f"✓ Reviews: {len(review_data['user_names'])}")

# ✨ Check review_signals dimension (修正: 'review_signals' -> 'features')
review_signals = review_data['features']
print(f"✓ Review signals shape: {review_signals.shape}")
if review_signals.shape[1] == 22:
    print("   🌟 22D review signals detected (18D + 4D person attention)")
elif review_signals.shape[1] == 18:
    print("   ⚠ WARNING: Only 18D review signals (missing person attention)")
else:
    print(f"   ⚠ UNEXPECTED: {review_signals.shape[1]}D review signals")

# ==================== 1. BUILD MAPPINGS ====================
print("\n" + "="*70)
print("1. BUILDING ID MAPPINGS")
print("="*70)

# User name -> index
user_name_to_idx = {name: idx for idx, name in enumerate(user_data['user_names'])}

# Movie ID -> index
movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(movie_data['movie_ids'])}

print(f"✓ User mapping: {len(user_name_to_idx)} users")
print(f"✓ Movie mapping: {len(movie_id_to_idx)} movies")

# ==================== 2. BUILD EDGES ====================
print("\n" + "="*70)
print("2. BUILDING GRAPH EDGES")
print("="*70)

splits = review_data['split'] # 'splits' -> 'split' に修正 (03_process_reviews.pyで保存されたキー名)
user_names = review_data['user_names']
movie_ids = review_data['movie_ids']

edges_by_split = defaultdict(lambda: {
    'user_indices': [],
    'movie_indices': [],
    'review_signals': [],
    'ratings_raw': [],              # ← rename
    'ratings_user_norm': [],        # ← 追加
    'review_indices': []
})

missing_users = 0
missing_movies = 0

print("Processing edges...")
for i in range(len(user_names)):
    user_name = user_names[i]
    movie_id = int(movie_ids[i])
    split = splits[i]

    # Check if user and movie exist in mappings
    if user_name not in user_name_to_idx:
        missing_users += 1
        continue
    if movie_id not in movie_id_to_idx:
        missing_movies += 1
        continue

    user_idx = user_name_to_idx[user_name]
    movie_idx = movie_id_to_idx[movie_id]

    edges_by_split[split]['user_indices'].append(user_idx)
    edges_by_split[split]['movie_indices'].append(movie_idx)

    # ✨ 修正: 'review_signals' -> 'features'
    edges_by_split[split]['review_signals'].append(review_data['features'][i])

    # 'rating_raw'はキー名が一致しているのでそのまま
    edges_by_split[split]['ratings_raw'].append(review_data['ratings_raw'][i])
    edges_by_split[split]['ratings_user_norm'].append(review_data['ratings_user_norm'][i])

    edges_by_split[split]['review_indices'].append(i)

print(f"✓ Edges processed")
print(f"  Missing users: {missing_users}")
print(f"  Missing movies: {missing_movies}")

# ==================== 3. VERIFICATION ====================
print("\n" + "=" * 60)
print("【EDGE CONSTRUCTION 検証】")
print("-" * 60)

total_edges = sum(len(edges_by_split[split]['user_indices']) for split in edges_by_split)
expected_edges = len(user_names) - missing_users - missing_movies

print(f"✅ Total edges created: {total_edges}")
print(f"✅ Expected edges: {expected_edges}")
print(f"✅ Match: {total_edges == expected_edges}")

if total_edges != expected_edges:
    print("   ⚠ WARNING: Edge count mismatch!")
else:
    print("   🌟 Edge count verified")

# Check for duplicates
for split in edges_by_split:
    edge_pairs = list(zip(edges_by_split[split]['user_indices'],
                          edges_by_split[split]['movie_indices']))
    unique_pairs = len(set(edge_pairs))
    total_pairs = len(edge_pairs)

    if unique_pairs != total_pairs:
        print(f"   ⚠ {split}: Found {total_pairs - unique_pairs} duplicate edges")
    else:
        print(f"   ✓ {split}: No duplicate edges")

print("=" * 60)

# ==================== 4. SAVE GRAPHS ====================
print("\n" + "="*70)
print("4. SAVING GRAPHS")
print("="*70)

# ✨ Detect review_signal dimension
review_signal_dim = review_signals.shape[1]

for split in ['train', 'val', 'test']:
    if split not in edges_by_split:
        print(f"⚠ Warning: Split '{split}' not found in data")
        continue

    edge_data = edges_by_split[split]

    graph_dict = {
        # Node features
        'user_features': user_data['features'],  # (num_users, user_dim)
        'movie_features': movie_data['features'],  # (num_movies, movie_dim)

        # Edge indices
        'user_indices': torch.tensor(edge_data['user_indices'], dtype=torch.long),
        'movie_indices': torch.tensor(edge_data['movie_indices'], dtype=torch.long),

        # Edge features ✨ 22D対応
        'review_signals': torch.stack(edge_data['review_signals']),
    'ratings_raw': torch.tensor(edge_data['ratings_raw'], dtype=torch.float32),  # ← rename
    'ratings_user_norm': torch.tensor(edge_data['ratings_user_norm'], dtype=torch.float32),  # ← 追加


        # Review indices (for lookup)
        'review_indices': torch.tensor(edge_data['review_indices'], dtype=torch.long),

        # Metadata
        'num_users': len(user_data['user_names']),
        'num_movies': len(movie_data['movie_ids']),
        'num_edges': len(edge_data['user_indices']),
        'split': split,

        # Mappings
        'user_name_to_idx': user_name_to_idx,
        'movie_id_to_idx': movie_id_to_idx,
        'user_names': user_data['user_names'],
        'movie_ids': movie_data['movie_ids'],
        'movie_titles': movie_data['movie_titles'],

        # Dimensions ✨ 動的に設定
        'user_dim': user_data['features'].shape[1],
        'movie_dim': movie_data['features'].shape[1],
        'review_signal_dim': review_signal_dim,
        # ✨ 修正: aspect_namesはreview_data['feature_dims']から取得するのが安全
        'aspect_names': [k for k, v in review_data['feature_dims'].items() if k not in ['person_attention', 'total']]
    }

    output_path = PROCESSED_DIR / f"hetero_graph_{split}.pt"
    torch.save(graph_dict, output_path)
    print(f"✅ Saved: {output_path}")
    print(f"   Nodes: {graph_dict['num_users']} users, {graph_dict['num_movies']} movies")
    print(f"   Edges: {graph_dict['num_edges']}")
    print(f"   Review signals: {review_signal_dim}D")

# ==================== 5. SUMMARY ====================
print("\n" + "="*70)
print("GRAPH CONSTRUCTION SUMMARY")
print("="*70)

for split in ['train', 'val', 'test']:
    if split not in edges_by_split:
        continue

    num_edges = len(edges_by_split[split]['user_indices'])
    ratings = np.array(edges_by_split[split]['ratings_raw'])

    print(f"\n{split.upper()}:")
    print(f"  Edges: {num_edges:,}")
    print(f"  Rating - Mean: {ratings.mean():.2f}, Std: {ratings.std():.2f}")

    # Band distribution
    low = ((ratings >= 1) & (ratings <= 4)).sum()
    mid = ((ratings >= 5) & (ratings <= 7)).sum()
    high = ((ratings >= 8) & (ratings <= 10)).sum()

    print(f"  Low (1-4):   {low:6d} ({low/num_edges*100:5.1f}%)")
    print(f"  Mid (5-7):   {mid:6d} ({mid/num_edges*100:5.1f}%)")
    print(f"  High (8-10): {high:6d} ({high/num_edges*100:5.1f}%)")

    # Check if distribution is consistent across splits
    high_pct = high / num_edges * 100
    if high_pct > 60:
        print(f"  ⚠ High rating imbalance: {high_pct:.1f}%")

# ==================== 6. FEATURE DIMENSION SUMMARY ====================
print("\n" + "=" * 60)
print("【FEATURE DIMENSIONS】")
print("-" * 60)

print(f"User features:   {user_data['features'].shape[1]:4d}D")
print(f"  - Zscore:      {user_data['feature_dims']['zscore']:4d}D")
print(f"  - Sentiment:   {user_data['feature_dims']['sentiment']:4d}D")
print(f"  - Stats:       {user_data['feature_dims']['stats']:4d}D")

print(f"\nMovie features:  {movie_data['features'].shape[1]:4d}D")
print(f"  - Genre:       {movie_data['feature_dims']['genre']:4d}D")
print(f"  - Actor emb:   {movie_data['feature_dims']['actor_emb']:4d}D")
print(f"  - Director:    {movie_data['feature_dims']['director_emb']:4d}D")
print(f"  - Keyword:     {movie_data['feature_dims']['keyword_emb']:4d}D")
print(f"  - Runtime/Yr:  {movie_data['feature_dims']['runtime_year']:4d}D")
print(f"  - Tags:        {movie_data['feature_dims']['tags']:4d}D")
print(f"  - Review agg:  {movie_data['feature_dims']['review_agg']:4d}D")

print(f"\n✨ Review signals: {review_signal_dim:4d}D")
if review_signal_dim == 22:
    print(f"  - Aspect signals:      18D")
    print(f"  - Person attention:     4D")

print("=" * 60)

print("\n" + "="*70)
print("🎉 PHASE 1-4 COMPLETE!")
print("="*70)
print("\nAll graph files saved:")
print(f"  - {PROCESSED_DIR / 'hetero_graph_train.pt'}")
print(f"  - {PROCESSED_DIR / 'hetero_graph_val.pt'}")
print(f"  - {PROCESSED_DIR / 'hetero_graph_test.pt'}")