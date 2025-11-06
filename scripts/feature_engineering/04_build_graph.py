#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1-4 (修正版): 異種グラフの構築
- User-Movie-Review グラフ（PyTorch辞書形式）
- 22D review signals（18D aspect + 4D person attention）対応
- ✅ ユーザー重複を許可（ランダムsplit対応）
- 充実したメタ情報の保存

修正内容:
- ユーザー排他性チェックを削除
- ランダムsplitに対応した検証ロジック
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ==================== Config ====================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REVIEW_FEATURES_PT = PROCESSED_DIR / "review_features.pt"
USER_FEATURES_PT   = PROCESSED_DIR / "user_features.pt"
MOVIE_FEATURES_PT  = PROCESSED_DIR / "movie_features.pt"

USE_USER_NORM_FOR_Y_ALIAS = False  # 'ratings' alias points to raw

# ==================== Utilities ====================
def check_duplicate_edges(pair_list):
    """(user_idx, movie_idx) ペアの重複件数を返す"""
    total = len(pair_list)
    unique = len(set(pair_list))
    return total - unique

def safe_tensor_stack(list_of_tensors, dtype=None):
    """list[Tensor] -> Tensor に安全に変換"""
    if len(list_of_tensors) == 0:
        return None
    t = torch.stack(list_of_tensors)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t

# ==================== Load Data ====================
print("="*70)
print("PHASE 1-4 FIXED: GRAPH CONSTRUCTION (Random Split)")
print("="*70)
print("\nLoading processed features...")

review_data = torch.load(REVIEW_FEATURES_PT, weights_only=False)
user_data   = torch.load(USER_FEATURES_PT,   weights_only=False)
movie_data  = torch.load(MOVIE_FEATURES_PT,  weights_only=False)

num_users  = len(user_data['user_names'])
num_movies = len(movie_data['movie_ids'])
num_edges  = len(review_data['user_names'])

print(f"✓ Users : {num_users}")
print(f"✓ Movies: {num_movies}")
print(f"✓ Reviews/Edges: {num_edges}")

# Review signals
review_signals = review_data['features']
if not isinstance(review_signals, torch.Tensor):
    review_signals = torch.as_tensor(review_signals, dtype=torch.float32)

print(f"✓ Review signals shape: {tuple(review_signals.shape)}")
if review_signals.shape[1] == 22:
    print("   🌟 22D review signals detected (18D aspect + 4D person attention)")

aspect_names = review_data.get('aspect_names', None)
if aspect_names is None:
    aspect_names = [f"aspect_{i}" for i in range(18)]
split_meta = review_data.get('split_meta', {})

print(f"✓ Split method: {split_meta.get('method', 'unknown')}")
print(f"✓ User overlap: {split_meta.get('user_overlap', 'unknown')}")

# ==================== 1. BUILD MAPPINGS ====================
print("\n" + "="*70)
print("1. BUILDING ID MAPPINGS")
print("="*70)

user_name_to_idx = {name: idx for idx, name in enumerate(user_data['user_names'])}
movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(movie_data['movie_ids'])}

print(f"✓ User mapping built:  {len(user_name_to_idx)} users")
print(f"✓ Movie mapping built: {len(movie_id_to_idx)} movies")

# ==================== 2. BUILD EDGES (by split) ====================
print("\n" + "="*70)
print("2. BUILDING GRAPH EDGES")
print("="*70)

splits     = np.asarray(review_data['split'])
rev_users  = np.asarray(review_data['user_names'])
rev_movies = np.asarray(review_data['movie_ids'])

ratings_raw       = review_data['ratings_raw']
ratings_user_norm = review_data['ratings_user_norm']

edges_by_split = defaultdict(lambda: {
    'user_indices': [],
    'movie_indices': [],
    'review_signals': [],
    'ratings_raw': [],
    'ratings_user_norm': [],
    'review_indices': []
})

missing_users  = 0
missing_movies = 0

print("Processing edges...")
for i in range(len(rev_users)):
    uname = rev_users[i]
    mid   = int(rev_movies[i])
    sp    = splits[i]

    uidx = user_name_to_idx.get(uname)
    if uidx is None:
        missing_users += 1
        continue
    midx = movie_id_to_idx.get(mid)
    if midx is None:
        missing_movies += 1
        continue

    edges_by_split[sp]['user_indices'].append(uidx)
    edges_by_split[sp]['movie_indices'].append(midx)
    edges_by_split[sp]['review_signals'].append(review_signals[i])
    edges_by_split[sp]['ratings_raw'].append(ratings_raw[i].item() if torch.is_tensor(ratings_raw[i]) else float(ratings_raw[i]))
    edges_by_split[sp]['ratings_user_norm'].append(ratings_user_norm[i].item() if torch.is_tensor(ratings_user_norm[i]) else float(ratings_user_norm[i]))
    edges_by_split[sp]['review_indices'].append(i)

print(f"✓ Edges processed")
print(f"  Missing users : {missing_users}")
print(f"  Missing movies: {missing_movies}")

# ==================== 3. VERIFICATION ====================
print("\n" + "=" * 60)
print("【EDGE CONSTRUCTION 検証】")
print("-" * 60)

total_edges = sum(len(edges_by_split[sp]['user_indices']) for sp in edges_by_split)
expected_edges = len(rev_users) - missing_users - missing_movies

print(f"✅ Total edges created: {total_edges}")
print(f"✅ Expected edges      : {expected_edges}")
print(f"✅ Match               : {total_edges == expected_edges}")

if total_edges != expected_edges:
    print("   ⚠ WARNING: Edge count mismatch!")

# Duplicates check (per split)
for sp in sorted(edges_by_split.keys()):
    pairs = list(zip(edges_by_split[sp]['user_indices'], edges_by_split[sp]['movie_indices']))
    dups = check_duplicate_edges(pairs)
    if dups > 0:
        print(f"   ⚠ {sp}: Found {dups} duplicate (user,movie) edges")
    else:
        print(f"   ✓ {sp}: No duplicate edges")

# ✅ User overlap check (not an error, just reporting)
print("\n✓ User Overlap Statistics:")
train_users = set()
val_users = set()
test_users = set()

if 'train' in edges_by_split:
    train_user_idx = set(edges_by_split['train']['user_indices'])
    train_users = {user_data['user_names'][i] for i in train_user_idx}
if 'val' in edges_by_split:
    val_user_idx = set(edges_by_split['val']['user_indices'])
    val_users = {user_data['user_names'][i] for i in val_user_idx}
if 'test' in edges_by_split:
    test_user_idx = set(edges_by_split['test']['user_indices'])
    test_users = {user_data['user_names'][i] for i in test_user_idx}

print(f"  Train users: {len(train_users)}")
print(f"  Val users:   {len(val_users)}")
print(f"  Test users:  {len(test_users)}")
print(f"  Train ∩ Test: {len(train_users & test_users)} (expected for known-user prediction)")

print("=" * 60)

# ==================== 4. SAVE GRAPHS ====================
print("\n" + "="*70)
print("4. SAVING GRAPHS")
print("="*70)

user_features  = user_data['features']
movie_features = movie_data['features']

user_dim  = user_features.shape[1]
movie_dim = movie_features.shape[1]
review_signal_dim = review_signals.shape[1]

movie_titles = movie_data.get('movie_titles', None)

for sp in ['train', 'val', 'test']:
    if sp not in edges_by_split:
        print(f"⚠ Warning: Split '{sp}' not found in data (skip saving).")
        continue

    e = edges_by_split[sp]

    e_user   = torch.tensor(e['user_indices'],  dtype=torch.long)
    e_movie  = torch.tensor(e['movie_indices'], dtype=torch.long)
    e_attr   = safe_tensor_stack(e['review_signals'])
    e_y_raw  = torch.tensor(e['ratings_raw'],       dtype=torch.float32)
    e_y_user = torch.tensor(e['ratings_user_norm'], dtype=torch.float32)
    e_ridx   = torch.tensor(e['review_indices'],    dtype=torch.long)

    ratings_alias = e_y_user if USE_USER_NORM_FOR_Y_ALIAS else e_y_raw

    graph_dict = {
        # Node features
        'user_features':  user_features,
        'movie_features': movie_features,

        # Edge indices/features/labels
        'user_indices':   e_user,
        'movie_indices':  e_movie,
        'review_signals': e_attr,
        'ratings_raw':       e_y_raw,          # y_raw → ratings_raw
        'ratings_user_norm': e_y_user,         # y_user_norm → ratings_user_norm
        'ratings':        ratings_alias,
        'review_indices': e_ridx,

        # Metadata
        'num_users':      num_users,
        'num_movies':     num_movies,
        'num_edges':      int(e_user.numel()),
        'split':          sp,

        # Mappings / vocab
        'user_name_to_idx': user_name_to_idx,
        'movie_id_to_idx':  movie_id_to_idx,
        'user_names':       user_data['user_names'],
        'movie_ids':        movie_data['movie_ids'],
        'movie_titles':     movie_titles,

        # Dimensions
        'user_dim':          int(user_dim),
        'movie_dim':         int(movie_dim),
        'review_signal_dim': int(review_signal_dim),

        # Aspects & split meta
        'aspect_names': aspect_names,
        'split_meta':   split_meta,
    }

    out_path = PROCESSED_DIR / f"hetero_graph_{sp}.pt"
    torch.save(graph_dict, out_path)
    print(f"✅ Saved: {out_path}")
    print(f"   Nodes: {graph_dict['num_users']} users, {graph_dict['num_movies']} movies")
    print(f"   Edges: {graph_dict['num_edges']}")
    print(f"   Review signals: {graph_dict['review_signal_dim']}D")

# ==================== 5. SUMMARY ====================
print("\n" + "="*70)
print("GRAPH CONSTRUCTION SUMMARY")
print("="*70)

for sp in ['train', 'val', 'test']:
    if sp not in edges_by_split:
        continue
    e = edges_by_split[sp]
    n = len(e['user_indices'])
    if n == 0:
        print(f"{sp.upper()}: (no edges)")
        continue

    r = np.array(e['ratings_raw'], dtype=np.float32)
    print(f"\n{sp.upper()}:")
    print(f"  Edges: {n:,}")
    print(f"  Raw rating - Mean: {r.mean():.2f}, Std: {r.std():.2f}")

    low  = ((r >= 1) & (r <= 4)).sum()
    mid  = ((r >= 5) & (r <= 7)).sum()
    high = ((r >= 8) & (r <= 10)).sum()

    print(f"  Low (1-4):   {low:6d} ({low/n*100:5.1f}%)")
    print(f"  Mid (5-7):   {mid:6d} ({mid/n*100:5.1f}%)")
    print(f"  High (8-10): {high:6d} ({high/n*100:5.1f}%)")

print("\n" + "=" * 60)
print("【FEATURE DIMENSIONS】")
print("-" * 60)
print(f"User features:   {user_dim:4d}D")
print(f"Movie features:  {movie_dim:4d}D")
print(f"✨ Review signals: {review_signal_dim:4d}D")
if review_signal_dim == 22:
    print(f"  - Aspect signals:      18D")
    print(f"  - Person attention:     4D")
print("=" * 60)

print("\n" + "="*70)
print("🎉 PHASE 1-4 FIXED COMPLETE!")
print("="*70)
print("\n✅ Split method: Random (user overlap allowed)")
print("✅ Task: Known user + unseen movie prediction")
print("\nAll graph files saved:")
for sp in ['train', 'val', 'test']:
    out_path = PROCESSED_DIR / f"hetero_graph_{sp}.pt"
    print(f"  - {out_path}")