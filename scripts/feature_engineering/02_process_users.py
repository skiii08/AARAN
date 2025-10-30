"""
Phase 1-2 最終修正版: Person Embedding修正 (ロバスト処理適用)
配置: scripts/feature_engineering/02_process_users.py

最終修正内容:
- get_person_embedding関数はネスト構造と大文字小文字に対応済み。
- compute_top_k_embedding関数をロバスト化し、Top-K候補者リストから埋め込みがNoneの人名（OOV）を確実にフィルタリングし、有効な埋め込みのみで平均を計算する。
- デバッグ出力は環境依存で反映されないため、tqdm.writeを使用する形に修正しつつ、主要なロジックを修正。
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

# ==================== Config ====================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

USERS_CSV = RAW_DIR / "users.csv"
REVIEWS_CSV = RAW_DIR / "reviews.csv"
MOVIE_ENTITIES_JSON = PROCESSED_DIR / "movie_entities.json"
USER_PERSON_COUNTS_JSON = PROCESSED_DIR / "user_person_counts.json"
ENTITY_EMBEDDINGS_PT = PROCESSED_DIR / "entity_embeddings.pt"

ASPECT_NAMES = [
    'acting_performance', 'artistic_design', 'audio_music',
    'casting_choices', 'character_development', 'commercial_context',
    'comparative_analysis', 'editing_pacing', 'emotion',
    'expectation', 'filmmaking_direction', 'genre_style',
    'recommendation', 'story_plot', 'technical_visuals',
    'themes_messages', 'viewing_experience', 'writing_dialogue'
]

GENRE_NAMES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
    'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

# ==================== Load Data ====================
print("=" * 70)
print("PHASE 1-2 FIXED: USER FEATURE ENGINEERING (686D)")
print("=" * 70)
print("\nLoading data...")

users_df = pd.read_csv(USERS_CSV)
reviews_df = pd.read_csv(REVIEWS_CSV)

with open(MOVIE_ENTITIES_JSON, 'r', encoding='utf-8') as f:
    movie_entities_list = json.load(f)
movie_entities = {m['movie_id']: m for m in movie_entities_list}

with open(USER_PERSON_COUNTS_JSON, 'r', encoding='utf-8') as f:
    user_person_counts = json.load(f)

# weights_only=False は非推奨のため削除し、ここではそのままロードします
entity_embeddings = torch.load(ENTITY_EMBEDDINGS_PT)

print(f"✓ Users: {len(users_df)}")
print(f"✓ Reviews: {len(reviews_df)}")
print(f"✓ Movies: {len(movie_entities)}")
print(f"✓ User-Person counts: {len(user_person_counts)} users")
print(f"✓ Entity embeddings keys: {len(entity_embeddings)}")

# Check embedding structure
sample_keys = list(entity_embeddings.keys())[:5]
print(f"  Sample keys: {sample_keys}")

# ==================== 1. BUILD PERSON SETS ====================
print("\n" + "=" * 70)
print("1. BUILDING ACTOR/DIRECTOR SETS FROM METADATA")
print("=" * 70)

all_actors_set = set()
all_directors_set = set()

for movie in movie_entities.values():
    for actor in movie['actors']:
        all_actors_set.add(actor.lower())
    for director in movie['directors']:
        all_directors_set.add(director.lower())

print(f"✓ Unique actors: {len(all_actors_set)}")
print(f"✓ Unique directors: {len(all_directors_set)}")

overlap = all_actors_set & all_directors_set
print(f"✓ Overlap (actor & director): {len(overlap)}")

# ==================== 2. CLASSIFY PERSONS ====================
print("\n" + "=" * 70)
print("2. CLASSIFYING PERSON NAMES (Actor vs Director)")
print("=" * 70)

user_actors = {}
user_directors = {}

for user_name, person_dict in user_person_counts.items():
    actors_dict = {}
    directors_dict = {}

    for person_name, count in person_dict.items():
        person_lower = person_name.lower()

        is_actor = person_lower in all_actors_set
        is_director = person_lower in all_directors_set

        if is_actor and is_director:
            if overlap and person_lower in overlap:
                actors_dict[person_name] = count
                directors_dict[person_name] = count
        elif is_actor:
            actors_dict[person_name] = count
        elif is_director:
            directors_dict[person_name] = count

    if actors_dict:
        user_actors[user_name] = actors_dict
    if directors_dict:
        user_directors[user_name] = directors_dict

print(f"✓ Users with actor preferences: {len(user_actors)}")
print(f"✓ Users with director preferences: {len(user_directors)}")

# ==================== 3. COMPUTING PERSON PREFERENCE EMBEDDINGS (FIXED) ====================
print("\n" + "=" * 70)
print("3. COMPUTING PERSON PREFERENCE EMBEDDINGS (FINAL ROBUST FIX)")
print("=" * 70)

# ----------------- 🚨 OOV人名特定ロジック 🚨 -----------------
# 1. ユーザーに好まれているすべての人名（小文字）を抽出
all_user_persons = set()
for person_dict in user_person_counts.values():
    for person_name in person_dict.keys():
        all_user_persons.add(person_name.lower())

# 2. 埋め込みがあるすべての人名（小文字）を抽出
all_embedded_persons = set()
for person_key in ['actors', 'directors']:
    if person_key in entity_embeddings:
        for embedded_name in entity_embeddings[person_key].keys():
            all_embedded_persons.add(embedded_name.lower())

# 3. OOV人名（埋め込みがない人名）を特定
oov_persons = sorted(list(all_user_persons - all_embedded_persons))

print(f"\n🛑 CRITICAL DEBUG: Total OOV Persons (User-Preferred but No Embedding): {len(oov_persons)}")
if len(oov_persons) > 0:
    # OOV人名が多すぎる場合は、最初の20人のみ表示
    print(f"  Sample OOV Persons (First 20): {oov_persons[:20]}")
    print(f"  --- OOV Count Check: Actor (Users: {len(user_actors)}) / Director (Users: {len(user_directors)}) ---")
# ----------------- 🚨 OOV人名特定ロジック終了 🚨 -----------------

TOP_K_ACTORS = 5
TOP_K_DIRECTORS = 3
EMBEDDING_DIM = 300


def get_person_embedding(person_name, entity_type='actor'):
    """
    Get embedding for a person (actor/director) from entity_embeddings.
    FIXED: Handles nested dictionary structure and case inconsistency.
    """
    entity_key = entity_type + 's' # 'actor' -> 'actors', 'director' -> 'directors'

    # 1. トップレベルのキーで辞書を取得 (例: entity_embeddings['actors'])
    if entity_key not in entity_embeddings:
        return None

    person_map = entity_embeddings[entity_key]

    # 2. user_person_countsからのキーは小文字 (例: 'john goodman')
    person_name_lower = person_name.lower()

    # 3. 埋め込みマップ内のキー (例: 'John Goodman') と小文字で比較して、一致する元のキーを探す
    found_key = None
    for original_key in person_map.keys():
        if original_key.lower() == person_name_lower:
            found_key = original_key
            break

    if found_key:
        emb = person_map[found_key]
        if isinstance(emb, torch.Tensor):
            return emb.numpy()
        # 01_process_movies.pyでtolist()に変換されている可能性があるため
        return np.array(emb)

    return None


def compute_top_k_embedding(person_dict, entity_type, top_k):
    """
    [最終ロバスト版]
    Top-Kの候補者から、埋め込みが存在する人名のみを抽出し、その平均を計算する。
    埋め込みがない人名はサイレントに無視され、カウント数に反映される。
    """
    if len(person_dict) == 0:
        return np.zeros(EMBEDDING_DIM), 0, []

    # 1. 評価回数に基づいてTop-Kの人名を取得
    sorted_persons = sorted(person_dict.items(), key=lambda x: -x[1])[:top_k]

    embeddings = []
    missing_persons = []  # OOV/欠損人名を追跡

    # 2. Top-Kの各人名について埋め込みを取得し、None（埋め込みなし）をフィルタリング
    for person, count in sorted_persons:
        emb = get_person_embedding(person, entity_type)
        if emb is not None:
            embeddings.append(emb)
        else:
            missing_persons.append(person)  # 欠損人名をリストに追加

    if len(embeddings) == 0:
        # 埋め込みが見つからなかった場合、ゼロベクトルを返す
        return np.zeros(EMBEDDING_DIM), 0, missing_persons  # <-- 欠損人名リストを返す
    # ...
    # 3. 埋め込みが見つかった人名の平均を計算し、見つかった数(count)と欠損人名リストを返す
    return np.mean(embeddings, axis=0), len(embeddings), missing_persons  # <-- 欠損人名リストを返す


# Compute for all users
fav_actor_vectors = []
fav_actor_counts = []
fav_director_vectors = []
fav_director_counts = []

print(f"Computing Top-{TOP_K_ACTORS} actor embeddings...")
found_actor_users = 0
for user_name in tqdm(users_df['user_name'], desc="Actors"):
    if user_name in user_actors:
        # 3つの返り値を受け取る (vec, count, missing_persons)
        vec, count, missing_persons = compute_top_k_embedding(user_actors[user_name], 'actor', TOP_K_ACTORS)

        # --- 🚨 CRITICAL DEBUG INSERTION 🚨 ---
        # 完全に失敗した91人のユーザーを特定する
        if count == 0:
            tqdm.write(f"\n🛑 CRITICAL FAIL (Count=0): User '{user_name}'")
            tqdm.write(f"  Top-{TOP_K_ACTORS} persons are ALL missing embeddings: {missing_persons}")
        # --- 🚨 CRITICAL DEBUG INSERTION END 🚨 ---

        if count > 0:
            found_actor_users += 1
    else:
        vec, count = np.zeros(EMBEDDING_DIM), 0
    fav_actor_vectors.append(vec)
    fav_actor_counts.append(count)

print(f"  ✓ Found embeddings for {found_actor_users} users")

print(f"Computing Top-{TOP_K_DIRECTORS} director embeddings...")
found_director_users = 0
for user_name in tqdm(users_df['user_name'], desc="Directors"):
    if user_name in user_directors:
        # 3つの返り値を受け取るが、3つ目は使用しない ([]が返ってくる)
        vec, count, _ = compute_top_k_embedding(user_directors[user_name], 'director', TOP_K_DIRECTORS)

        # デバッグ出力は、デバッグの問題が解消されないため、今回は割愛します。

        if count > 0:
            found_director_users += 1
    else:
        vec, count = np.zeros(EMBEDDING_DIM), 0
    fav_director_vectors.append(vec)
    fav_director_counts.append(count)

print(f"  ✓ Found embeddings for {found_director_users} users")

fav_actor_vectors = np.array(fav_actor_vectors)
fav_actor_counts = np.array(fav_actor_counts)
fav_director_vectors = np.array(fav_director_vectors)
fav_director_counts = np.array(fav_director_counts)

print(f"\n✓ Actor vectors: {fav_actor_vectors.shape}")
print(f"✓ Director vectors: {fav_director_vectors.shape}")
# **この値が改善されるかを確認します**
print(f"✓ Avg actors per user: {fav_actor_counts.mean():.2f}")
print(f"✓ Avg directors per user: {fav_director_counts.mean():.2f}")
print(f"✓ Non-zero actor embeddings: {(fav_actor_counts > 0).sum()} / {len(fav_actor_counts)}")
print(f"✓ Non-zero director embeddings: {(fav_director_counts > 0).sum()} / {len(fav_director_counts)}")

# ==================== 続き: 元のコードと同じ ====================
# (Genre preferences, Temporal features など)

# Aspect features
aspect_zscore_cols = [f'zscore_{aspect}' for aspect in ASPECT_NAMES]
aspect_sentiment_cols = [f'sentiment_{aspect}' for aspect in ASPECT_NAMES]

aspect_zscore = users_df[aspect_zscore_cols].fillna(0).values
aspect_sentiment = users_df[aspect_sentiment_cols].fillna(3.0).values

# Review stats
user_stats = reviews_df.groupby('user_name').agg({
    'rating_raw': ['count', 'mean', 'std', 'min', 'max']
}).reset_index()
user_stats.columns = ['user_name', 'review_count', 'rating_mean', 'rating_std', 'rating_min', 'rating_max']
users_merged = users_df.merge(user_stats, on='user_name', how='left')

review_count = users_merged['review_count'].fillna(0).values
rating_mean = users_merged['rating_mean'].fillna(users_merged['rating_mean'].mean()).values
rating_std = users_merged['rating_std'].fillna(0).values
rating_min = users_merged['rating_min'].fillna(0).values
rating_max = users_merged['rating_max'].fillna(0).values

review_stats = np.column_stack([review_count, rating_mean, rating_std, rating_min, rating_max])

# Genre preferences
genre_rating_cols = [f'rating_{genre}' for genre in GENRE_NAMES]
genre_count_cols = [f'count_{genre}' for genre in GENRE_NAMES]

genre_ratings = users_df[genre_rating_cols].fillna(0).values
genre_counts = users_df[genre_count_cols].fillna(0).values
genre_counts_norm = genre_counts / (genre_counts.sum(axis=1, keepdims=True) + 1e-8)

# Genre diversity
genre_diversity = []
for i in range(len(users_df)):
    counts = genre_counts[i]
    if counts.sum() > 0:
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        div = entropy(probs)
    else:
        div = 0.0
    genre_diversity.append(div)
genre_diversity = np.array(genre_diversity)

# Temporal
if 'first_date' in users_df.columns and 'last_date' in users_df.columns:
    users_df['first_date'] = pd.to_datetime(users_df['first_date'], errors='coerce')
    users_df['last_date'] = pd.to_datetime(users_df['last_date'], errors='coerce')
    active_days = (users_df['last_date'] - users_df['first_date']).dt.days.fillna(0).values
    review_count_raw = users_df['review_count_raw'].fillna(users_df['review_count_raw'].mean()).values
    review_velocity = review_count_raw / (active_days + 1)
else:
    active_days = np.zeros(len(users_df))
    review_velocity = np.zeros(len(users_df))

# Rating vs global
global_mean = reviews_df['rating_raw'].mean()
rating_vs_global = rating_mean - global_mean

# Sentiment overall
sentiment_overall = aspect_sentiment.mean(axis=1)

# ==================== COMBINE FEATURES ====================
print("\n" + "=" * 70)
print("COMBINING ALL FEATURES")
print("=" * 70)

user_features = np.concatenate([
    aspect_zscore,           # 18D
    aspect_sentiment,        # 18D
    review_stats,            # 5D
    fav_actor_vectors,       # 300D ✨ FIXED
    fav_actor_counts.reshape(-1, 1),  # 1D
    fav_director_vectors,    # 300D ✨ FIXED
    fav_director_counts.reshape(-1, 1),  # 1D
    genre_ratings,           # 19D
    genre_counts_norm,       # 19D
    genre_diversity.reshape(-1, 1),   # 1D
    active_days.reshape(-1, 1),       # 1D
    review_velocity.reshape(-1, 1),   # 1D
    rating_vs_global.reshape(-1, 1),  # 1D
    sentiment_overall.reshape(-1, 1)  # 1D
], axis=1)

print(f"✓ Combined shape: {user_features.shape}")
print(f"  Expected: (N, 686)")

# ==================== SAVE ====================
output_dict = {
    'features': torch.tensor(user_features, dtype=torch.float32),
    'user_names': users_df['user_name'].values,
    'feature_dims': {
        'zscore': 18,
        'sentiment': 18,
        'stats': 5,
        'fav_actor_emb': 300,
        'fav_actor_count': 1,
        'fav_director_emb': 300,
        'fav_director_count': 1,
        'genre_ratings': 19,
        'genre_counts': 19,
        'genre_diversity': 1,
        'active_days': 1,
        'review_velocity': 1,
        'rating_vs_global': 1,
        'sentiment_overall': 1,
        'total': 686
    }
}

torch.save(output_dict, PROCESSED_DIR / "user_features.pt")
print(f"\n✅ Saved: {PROCESSED_DIR / 'user_features.pt'}")
print("\n🎉 PHASE 1-2 FIXED COMPLETE!")