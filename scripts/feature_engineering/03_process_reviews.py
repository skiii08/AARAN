# -*- coding: utf-8 -*-
"""
Phase 1-4 (改訂版): 異種グラフの構築
- User-Movie-Review グラフ（PyTorch辞書形式）
- 22D review signals（18D aspect + 4D person attention）対応
- ユーザー排他的 split の再検証
- 充実したメタ情報の保存（aspect_names, split_meta, dims, mappings など）

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

REVIEW_FEATURES_PT = PROCESSED_DIR / "review_features.pt"
USER_FEATURES_PT   = PROCESSED_DIR / "user_features.pt"
MOVIE_FEATURES_PT  = PROCESSED_DIR / "movie_features.pt"

# 学習時に参照される後方互換エイリアス:
# graph_dict['ratings'] は raw を指す
USE_USER_NORM_FOR_Y_ALIAS = False  # ここは保存時の 'ratings' エイリアスの方針（False: raw）

# ==================== Utilities ====================
def assert_user_disjoint_from_arrays(user_names: np.ndarray, splits: np.ndarray):
    """
    ユーザーが複数 split に跨っていないことをチェック。
    問題があれば RuntimeError。
    """
    df = pd.DataFrame({"user_name": user_names, "split": splits})
    g = df.groupby("user_name")["split"].nunique()
    offenders = g[g > 1]
    if len(offenders) > 0:
        # 上位のみ表示しても十分（全件は巨大）
        print("⚠ Users appearing in multiple splits (top 10):")
        print(offenders.head(10))
        raise RuntimeError(f"Leak detected: {len(offenders)} users appear in multiple splits.")


def check_duplicate_edges(pair_list):
    """
    (user_idx, movie_idx) ペアの重複件数を返す
    """
    total = len(pair_list)
    unique = len(set(pair_list))
    return total - unique  # dup count


def safe_tensor_stack(list_of_tensors, dtype=None):
    """
    list[Tensor] -> Tensor に安全に変換（空の場合は形状[0, ...]を推定して返す）
    """
    if len(list_of_tensors) == 0:
        return None
    t = torch.stack(list_of_tensors)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


# ==================== Load Data ====================
print("="*70)
print("PHASE 1-4: GRAPH CONSTRUCTION (22D Review Signals)")
print("="*70)
print("\nLoading processed features...")

review_data = torch.load(REVIEW_FEATURES_PT, weights_only=False)
user_data   = torch.load(USER_FEATURES_PT,   weights_only=False)
movie_data  = torch.load(MOVIE_FEATURES_PT,  weights_only=False)

# 基本形状
num_users  = len(user_data['user_names'])
num_movies = len(movie_data['movie_ids'])
num_edges  = len(review_data['user_names'])

print(f"✓ Users : {num_users}")
print(f"✓ Movies: {num_movies}")
print(f"✓ Reviews/Edges (raw): {num_edges}")

# ===== Review signals =====
review_signals = review_data['features']  # shape: (N, 22) 期待
if not isinstance(review_signals, torch.Tensor):
    review_signals = torch.as_tensor(review_signals, dtype=torch.float32)

print(f"✓ Review signals shape: {tuple(review_signals.shape)}")
if review_signals.shape[1] == 22:
    print("   🌟 22D review signals detected (18D aspect + 4D person attention)")
elif review_signals.shape[1] == 18:
    print("   ⚠ WARNING: Only 18D review signals (missing person attention)")
else:
    print(f"   ⚠ UNEXPECTED review signal dim: {review_signals.shape[1]}D")

# optional: aspect_names / split_meta
aspect_names = review_data.get('aspect_names', None)
if aspect_names is None:
    # フォールバック（名称までの再現は不可だが長さ18は保証）
    aspect_names = [f"aspect_{i}" for i in range(18)]
split_meta = review_data.get('split_meta', {})

# ==================== 1. BUILD MAPPINGS ====================
print("\n" + "="*70)
print("1. BUILDING ID MAPPINGS")
print("="*70)

# User name -> index
user_name_to_idx = {name: idx for idx, name in enumerate(user_data['user_names'])}

# Movie ID -> index（movie_id は int 化しているはず）
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

# ラベル（raw / user_norm 両方保持）
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

# Leak check (user-disjoint)
try:
    assert_user_disjoint_from_arrays(rev_users, splits)
    print("✓ User-disjoint confirmed across splits.")
except RuntimeError as e:
    print(str(e))
    # ここで止める（split は 03 の出力に準拠すべき）
    raise

print("=" * 60)

# ==================== 4. SAVE GRAPHS ====================
print("\n" + "="*70)
print("4. SAVING GRAPHS")
print("="*70)

user_features  = user_data['features']  # (num_users, user_dim)
movie_features = movie_data['features']  # (num_movies, movie_dim)

user_dim  = user_features.shape[1]
movie_dim = movie_features.shape[1]
review_signal_dim = review_signals.shape[1]

# 作品タイトル（存在しない環境もあるため安全に取得）
movie_titles = movie_data.get('movie_titles', None)

# 保存本体
for sp in ['train', 'val', 'test']:
    if sp not in edges_by_split:
        print(f"⚠ Warning: Split '{sp}' not found in data (skip saving).")
        continue

    e = edges_by_split[sp]

    # stack
    e_user   = torch.tensor(e['user_indices'],  dtype=torch.long)
    e_movie  = torch.tensor(e['movie_indices'], dtype=torch.long)
    e_attr   = safe_tensor_stack(e['review_signals'])
    e_y_raw  = torch.tensor(e['ratings_raw'],       dtype=torch.float32)
    e_y_user = torch.tensor(e['ratings_user_norm'], dtype=torch.float32)
    e_ridx   = torch.tensor(e['review_indices'],    dtype=torch.long)

    # 後方互換: ratings は raw を指す（設定で切替可）
    ratings_alias = e_y_user if USE_USER_NORM_FOR_Y_ALIAS else e_y_raw

    graph_dict = {
        # Node features
        'user_features':  user_features,    # (U, Du)
        'movie_features': movie_features,   # (M, Dm)

        # Edge indices/features/labels
        'user_indices':   e_user,           # (E,)
        'movie_indices':  e_movie,          # (E,)
        'review_signals': e_attr,           # (E, 22) 期待
        'y_raw':          e_y_raw,          # (E,)
        'y_user_norm':    e_y_user,         # (E,)
        'ratings':        ratings_alias,    # alias for backward-compat (default: raw)
        'review_indices': e_ridx,           # (E,)

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

    # Band distribution for raw ratings
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
ufd = user_data.get('feature_dims', {})
for k in ['zscore', 'sentiment', 'stats', 'fav_actor_emb', 'fav_actor_count', 'fav_director_emb']:
    if k in ufd:
        print(f"  - {k:15s}: {ufd[k]:4d}D")

print(f"\nMovie features:  {movie_dim:4d}D")
mfd = movie_data.get('feature_dims', {})
for k in ['genre', 'actor_emb', 'director_emb', 'keyword_emb', 'runtime_year', 'tags', 'review_agg']:
    if k in mfd:
        print(f"  - {k:15s}: {mfd[k]:4d}D")

print(f"\n✨ Review signals: {review_signal_dim:4d}D")
if review_signal_dim == 22:
    print(f"  - Aspect signals:      18D")
    print(f"  - Person attention:     4D")
print("=" * 60)

print("\n" + "="*70)
print("🎉 PHASE 1-4 COMPLETE!")
print("="*70)
print("\nAll graph files saved:")
for sp in ['train', 'val', 'test']:
    out_path = PROCESSED_DIR / f"hetero_graph_{sp}.pt"
    print(f"  - {out_path}")
