"""
Phase 1-1 (改訂版): 映画特徴量の生成
- FastText埋め込み（actors/directors/keywords）
- Genre multi-hot
- Tag特徴量（sqrt → Z-score → L2正規化）
- Review集約（Mention-Weighted Mean）
- 各ステップで妥当性検証
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from gensim.models import KeyedVectors
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Config ====================
BASE_DIR = Path("/Users/watanabesaki/PycharmProjects/AARAN")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FASTTEXT_PATH = EXTERNAL_DIR / "cc.en.300.bin"
MOVIES_CSV = RAW_DIR / "movies_metadata.csv"
REVIEWS_CSV = RAW_DIR / "reviews.csv"

EMBEDDING_DIM = 300

ASPECT_NAMES = [
    'acting_performance', 'artistic_design', 'audio_music',
    'casting_choices', 'character_development', 'commercial_context',
    'comparative_analysis', 'editing_pacing', 'emotion',
    'expectation', 'filmmaking_direction', 'genre_style',
    'recommendation', 'story_plot', 'technical_visuals',
    'themes_messages', 'viewing_experience', 'writing_dialogue'
]

# ==================== Load Data ====================
print("="*70)
print("PHASE 1-1: MOVIE FEATURE ENGINEERING")
print("="*70)
print("\nLoading data...")

movies_df = pd.read_csv(MOVIES_CSV)
reviews_df = pd.read_csv(REVIEWS_CSV)

print(f"✓ Movies: {len(movies_df)}")
print(f"✓ Reviews: {len(reviews_df)}")
print(f"✓ Movie columns: {len(movies_df.columns)}")

# ==================== Load FastText ====================
print("\nLoading FastText model (this may take a few minutes)...")
import fasttext
import fasttext.util

try:
    # FastText .bin ファイルの正しい読み込み方法
    ft_model_raw = fasttext.load_model(str(FASTTEXT_PATH))
    print(f"✓ FastText loaded successfully")

    # KeyedVectors互換のラッパーを作成
    class FastTextWrapper:
        def __init__(self, model):
            self.model = model
            self.vector_size = model.get_dimension()

        def __contains__(self, word):
            return True  # FastTextは未知語にも対応

        def __getitem__(self, word):
            return self.model.get_word_vector(word)

        def __len__(self):
            return len(self.model.get_words())

    ft_model = FastTextWrapper(ft_model_raw)
    print(f"✓ FastText wrapper created: {ft_model.vector_size}D vectors")

except Exception as e:
    print(f"❌ Error loading FastText: {e}")
    print("Attempting alternative method with gensim...")

    # 代替: gensimのload_facebook_model
    from gensim.models.fasttext import load_facebook_model
    ft_model_raw = load_facebook_model(str(FASTTEXT_PATH))
    ft_model = ft_model_raw.wv
    print(f"✓ FastText loaded via gensim: {len(ft_model)} vectors")

# ==================== Helper Functions ====================
def parse_semicolon_list(text):
    """セミコロン区切りリストをパース"""
    if pd.isna(text) or text == '':
        return []
    return [s.strip() for s in str(text).split(';') if s.strip()]

def get_embedding(text, model):
    """テキストをFastText埋め込みに変換（単語平均）"""
    if pd.isna(text) or text == '':
        return np.zeros(EMBEDDING_DIM)

    words = str(text).lower().replace('-', ' ').replace('_', ' ').split()
    embeddings = []

    for word in words:
        if word in model:
            embeddings.append(model[word])

    if len(embeddings) == 0:
        return np.zeros(EMBEDDING_DIM)

    return np.mean(embeddings, axis=0)

def get_list_embedding(items, model):
    """リストの各要素を埋め込んで平均"""
    if len(items) == 0:
        return np.zeros(EMBEDDING_DIM)

    embeddings = [get_embedding(item, model) for item in items]
    embeddings = [e for e in embeddings if np.any(e != 0)]

    if len(embeddings) == 0:
        return np.zeros(EMBEDDING_DIM)

    return np.mean(embeddings, axis=0)

def zscore_normalize(series):
    """Z-score正規化（欠損値考慮）"""
    mean = series.mean()
    std = series.std()
    if std < 1e-8:
        return np.zeros_like(series)
    return (series - mean) / std

# ==================== 1. BASIC METADATA ====================
print("\n" + "="*70)
print("1. BASIC METADATA PROCESSING")
print("="*70)

# Parse lists
movies_df['genres_list'] = movies_df['genres'].apply(parse_semicolon_list)
movies_df['actors_list'] = movies_df['actors'].apply(parse_semicolon_list)
movies_df['directors_list'] = movies_df['directors'].apply(parse_semicolon_list)
movies_df['keywords_list'] = movies_df['keywords'].apply(parse_semicolon_list)

print(f"\n✓ Genres: {len(set([g for gs in movies_df['genres_list'] for g in gs]))} unique")
print(f"✓ Actors: {len(set([a for actors in movies_df['actors_list'] for a in actors]))} unique")
print(f"✓ Directors: {len(set([d for dirs in movies_df['directors_list'] for d in dirs]))} unique")
print(f"✓ Keywords: {len(set([k for kws in movies_df['keywords_list'] for k in kws]))} unique")

# Runtime & Year
movies_df['runtime'] = pd.to_numeric(movies_df['runtime'], errors='coerce')
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
movies_df['release_year'] = movies_df['release_date'].dt.year

runtime_mean = movies_df['runtime'].mean()
runtime_std = movies_df['runtime'].std()
year_mean = movies_df['release_year'].mean()
year_std = movies_df['release_year'].std()

movies_df['runtime_norm'] = (movies_df['runtime'] - runtime_mean) / (runtime_std + 1e-8)
movies_df['year_norm'] = (movies_df['release_year'] - year_mean) / (year_std + 1e-8)
movies_df['runtime_norm'].fillna(0, inplace=True)
movies_df['year_norm'].fillna(0, inplace=True)

print(f"\nRuntime: μ={runtime_mean:.1f}, σ={runtime_std:.1f}")
print(f"Year: μ={year_mean:.1f}, σ={year_std:.1f}")

# ==================== 2. GENRE FEATURES (Multi-hot) ====================
print("\n" + "="*70)
print("2. GENRE FEATURES (Multi-hot)")
print("="*70)

all_genres = sorted(set([g for gs in movies_df['genres_list'] for g in gs]))
genre_to_idx = {g: i for i, g in enumerate(all_genres)}

genre_features = []
for genres in movies_df['genres_list']:
    vec = np.zeros(len(all_genres))
    for g in genres:
        if g in genre_to_idx:
            vec[genre_to_idx[g]] = 1.0
    genre_features.append(vec)
genre_features = np.array(genre_features)

print(f"✓ Genre vocabulary: {len(all_genres)}")
print(f"✓ Genre feature shape: {genre_features.shape}")

# ==================== 3. ENTITY EMBEDDINGS ====================
print("\n" + "="*70)
print("3. ENTITY EMBEDDINGS (FastText)")
print("="*70)

print("Embedding actors...")
actor_embeddings = []
for actors in tqdm(movies_df['actors_list'], desc="Actors"):
    emb = get_list_embedding(actors, ft_model)
    actor_embeddings.append(emb)
actor_embeddings = np.array(actor_embeddings)

print("Embedding directors...")
director_embeddings = []
for directors in tqdm(movies_df['directors_list'], desc="Directors"):
    emb = get_list_embedding(directors, ft_model)
    director_embeddings.append(emb)
director_embeddings = np.array(director_embeddings)

print("Embedding keywords...")
keyword_embeddings = []
for keywords in tqdm(movies_df['keywords_list'], desc="Keywords"):
    emb = get_list_embedding(keywords, ft_model)
    keyword_embeddings.append(emb)
keyword_embeddings = np.array(keyword_embeddings)

print(f"\n✓ Actor embeddings: {actor_embeddings.shape}")
print(f"✓ Director embeddings: {director_embeddings.shape}")
print(f"✓ Keyword embeddings: {keyword_embeddings.shape}")

# ==================== 4. TAG FEATURES ====================
print("\n" + "="*70)
print("4. TAG FEATURES (A01-V52)")
print("="*70)

# Identify tag columns
BASIC_COLS = ['movie_id', 'movie_title', 'genres', 'release_date',
              'production_countries', 'runtime', 'original_language',
              'spoken_languages', 'directors', 'actors', 'keywords',
              'rating', 'num_raters', 'num_reviews',
              'genres_list', 'actors_list', 'directors_list', 'keywords_list',
              'runtime_norm', 'year_norm', 'release_year']

tag_cols = [c for c in movies_df.columns if c not in BASIC_COLS]
print(f"✓ Total tag columns: {len(tag_cols)}")

# Step 1: sqrt transformation
print("\nStep 1: sqrt transformation...")
tag_features_sqrt = []
tag_negatives = 0

for col in tag_cols:
    series = pd.to_numeric(movies_df[col], errors='coerce').fillna(0)

    if (series < 0).any():
        tag_negatives += 1
        transformed = series
    else:
        transformed = np.sqrt(series)

    tag_features_sqrt.append(transformed.values)

tag_features_sqrt = np.array(tag_features_sqrt).T  # (movies, tags)
print(f"  ✓ Transformed {len(tag_cols)} tags")
print(f"  ⚠ Columns with negative values (no sqrt): {tag_negatives}")

# Verify sqrt effect
print(f"\n  Before sqrt: mean={movies_df[tag_cols].mean().mean():.2f}, "
      f"max={movies_df[tag_cols].max().max():.1f}")
print(f"  After sqrt:  mean={tag_features_sqrt.mean():.2f}, "
      f"max={tag_features_sqrt.max():.1f}")

# Step 2: Z-score normalization
print("\nStep 2: Z-score normalization per tag...")
tag_features_zscore = np.zeros_like(tag_features_sqrt)

max_z_scores = []
for i in range(tag_features_sqrt.shape[1]):
    z = zscore_normalize(tag_features_sqrt[:, i])
    tag_features_zscore[:, i] = z
    if len(z) > 0:
        max_z_scores.append(np.abs(z).max())

print(f"  ✓ Z-score applied to {len(tag_cols)} tags")
print(f"  Max |Z-score|: {max(max_z_scores):.2f}")
print(f"  Tags with |Z| > 5: {sum(1 for z in max_z_scores if z > 5)}/{len(tag_cols)}")

# Step 3: L2 normalization
print("\nStep 3: L2 normalization (per movie)...")
epsilon = 1e-6
norms = np.linalg.norm(tag_features_zscore, axis=1, keepdims=True) + epsilon
tag_features_l2 = tag_features_zscore / norms

# Verification
final_norms = np.linalg.norm(tag_features_l2, axis=1)
max_value_after_l2 = np.abs(tag_features_l2).max()

print("\n" + "=" * 60)
print("【TAG FEATURES: L2正規化 検証】")
print("-" * 60)
print(f"✅ L2ノルム平均: {final_norms.mean():.6f} (目標: 1.000000)")
print(f"✅ L2ノルム標準偏差: {final_norms.std():.6f} (目標: 0.000000)")
print(f"✅ 正規化後の最大絶対値: {max_value_after_l2:.6f} (制約: ≤1.0)")

if final_norms.mean() > 0.999 and final_norms.std() < 1e-5:
    print("🌟 L2正規化は完全に成功しています。")
    print("   → タグ数による寄与の偏りは解消されました。")
else:
    print("❌ L2正規化に問題があります。")

print("=" * 60)

tag_features_final = tag_features_l2

# ==================== 5. REVIEW AGGREGATION ====================
print("\n" + "="*70)
print("5. REVIEW → MOVIE AGGREGATION (Mention-Weighted Mean)")
print("="*70)

# Calculate review signals
mention_cols = [f'mention_{aspect}' for aspect in ASPECT_NAMES]
sentiment_cols = [f'sentiment_{aspect}' for aspect in ASPECT_NAMES]

mention_values = reviews_df[mention_cols].values
sentiment_values = reviews_df[sentiment_cols].values
review_signals = mention_values * (sentiment_values - 3.0)

print(f"✓ Review signals: {review_signals.shape}")

# Build movie-level aggregated features
print("Computing mention-weighted mean per movie...")

movie_aggregated = {}

for movie_id in tqdm(movies_df['movie_id'].unique(), desc="Aggregating"):
    movie_reviews = reviews_df[reviews_df['movie_id'] == movie_id]

    if len(movie_reviews) == 0:
        movie_aggregated[movie_id] = np.zeros(len(ASPECT_NAMES))
        continue

    review_indices = movie_reviews.index
    signals = review_signals[review_indices]  # (N, 18)
    mentions = mention_values[review_indices]  # (N, 18)

    # Mention-weighted mean: Σ(mention × signal) / Σ(mention)
    weighted_sum = (mentions * signals).sum(axis=0)
    mention_sum = mentions.sum(axis=0) + 1e-8
    weighted_avg = weighted_sum / mention_sum

    movie_aggregated[movie_id] = weighted_avg

# Convert to array (align with movies_df order)
review_aggregated_features = []
for movie_id in movies_df['movie_id']:
    if movie_id in movie_aggregated:
        review_aggregated_features.append(movie_aggregated[movie_id])
    else:
        review_aggregated_features.append(np.zeros(len(ASPECT_NAMES)))

review_aggregated_features = np.array(review_aggregated_features)

print(f"✓ Review aggregated features: {review_aggregated_features.shape}")

# Verification
print("\n" + "=" * 60)
print("【REVIEW AGGREGATION 検証】")
print("-" * 60)
print(f"✅ 集約特徴量 shape: {review_aggregated_features.shape}")
print(f"✅ 平均値: {review_aggregated_features.mean():.4f}")
print(f"✅ 標準偏差: {review_aggregated_features.std():.4f}")
print(f"✅ 最小値: {review_aggregated_features.min():.4f}")
print(f"✅ 最大値: {review_aggregated_features.max():.4f}")
print(f"✅ ゼロベクトル数: {(np.abs(review_aggregated_features).sum(axis=1) == 0).sum()}/{len(movies_df)}")
print("=" * 60)

# ==================== 6. COMBINE ALL FEATURES ====================
print("\n" + "="*70)
print("6. COMBINING ALL FEATURES")
print("="*70)

movie_features = np.concatenate([
    genre_features,                    # 19D
    actor_embeddings,                  # 300D
    director_embeddings,               # 300D
    keyword_embeddings,                # 300D
    movies_df[['runtime_norm', 'year_norm']].values,  # 2D
    tag_features_final,                # 263D
    review_aggregated_features         # 18D
], axis=1)

print(f"✓ Final movie feature shape: {movie_features.shape}")
print(f"\nFeature composition:")
print(f"  - Genre (multi-hot):        {genre_features.shape[1]:4d}D")
print(f"  - Actor embedding:          {actor_embeddings.shape[1]:4d}D")
print(f"  - Director embedding:       {director_embeddings.shape[1]:4d}D")
print(f"  - Keyword embedding:        {keyword_embeddings.shape[1]:4d}D")
print(f"  - Runtime + Year:           {2:4d}D")
print(f"  - Tag features (L2-norm):   {tag_features_final.shape[1]:4d}D")
print(f"  - Review aggregation:       {review_aggregated_features.shape[1]:4d}D")
print(f"  {'─'*40}")
print(f"  TOTAL:                      {movie_features.shape[1]:4d}D")

# ==================== 7. SAVE FEATURES ====================
print("\n" + "="*70)
print("7. SAVING FEATURES")
print("="*70)

movie_features_dict = {
    'features': torch.tensor(movie_features, dtype=torch.float32),
    'movie_ids': movies_df['movie_id'].values,
    'movie_titles': movies_df['movie_title'].values,
    'genre_vocab': all_genres,
    'genre_to_idx': genre_to_idx,
    'tag_columns': tag_cols,
    'aspect_names': ASPECT_NAMES,
    'feature_dims': {
        'genre': genre_features.shape[1],
        'actor_emb': actor_embeddings.shape[1],
        'director_emb': director_embeddings.shape[1],
        'keyword_emb': keyword_embeddings.shape[1],
        'runtime_year': 2,
        'tags': tag_features_final.shape[1],
        'review_agg': review_aggregated_features.shape[1],
        'total': movie_features.shape[1]
    },
    'statistics': {
        'runtime_mean': float(runtime_mean),
        'runtime_std': float(runtime_std),
        'year_mean': float(year_mean),
        'year_std': float(year_std),
        'tag_max_zscore': float(max(max_z_scores)),
        'tag_l2_norm_mean': float(final_norms.mean()),
        'tag_l2_norm_std': float(final_norms.std())
    }
}

torch.save(movie_features_dict, PROCESSED_DIR / "movie_features.pt")
print(f"✅ Saved: {PROCESSED_DIR / 'movie_features.pt'}")

# ==================== 8. SAVE ENTITY DATA ====================
print("\nSaving entity data for explanations...")

# Movie entities (raw data)
movie_entities = []
for idx, row in movies_df.iterrows():
    movie_entities.append({
        'movie_id': int(row['movie_id']),
        'title': row['movie_title'],
        'genres': row['genres_list'],
        'actors': row['actors_list'][:10],
        'directors': row['directors_list'],
        'keywords': row['keywords_list'][:20],
        'runtime': float(row['runtime']) if not pd.isna(row['runtime']) else None,
        'release_year': int(row['release_year']) if not pd.isna(row['release_year']) else None
    })

with open(PROCESSED_DIR / "movie_entities.json", 'w', encoding='utf-8') as f:
    json.dump(movie_entities, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {PROCESSED_DIR / 'movie_entities.json'}")

# Entity embeddings
all_actors = set([a for actors in movies_df['actors_list'] for a in actors])
all_directors = set([d for dirs in movies_df['directors_list'] for d in dirs])
all_keywords = set([k for kws in movies_df['keywords_list'] for k in kws])

entity_embeddings_dict = {
    'actors': {},
    'directors': {},
    'keywords': {}
}

for actor in all_actors:
    emb = get_embedding(actor, ft_model)
    if np.any(emb != 0):
        entity_embeddings_dict['actors'][actor] = emb.tolist()

for director in all_directors:
    emb = get_embedding(director, ft_model)
    if np.any(emb != 0):
        entity_embeddings_dict['directors'][director] = emb.tolist()

for keyword in all_keywords:
    emb = get_embedding(keyword, ft_model)
    if np.any(emb != 0):
        entity_embeddings_dict['keywords'][keyword] = emb.tolist()

torch.save(entity_embeddings_dict, PROCESSED_DIR / "entity_embeddings.pt")
print(f"✅ Saved: {PROCESSED_DIR / 'entity_embeddings.pt'}")

print("\n" + "="*70)
print("🎉 PHASE 1-1 COMPLETE!")
print("="*70)