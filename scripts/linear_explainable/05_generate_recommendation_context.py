#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 5: Generate Recommendation Context for LLM
未視聴映画の予測と、関連視聴済み映画の分析

実行方法
python -m scripts.linear_explainable.05_generate_recommendation_context \
  --user_name 0231_Tweetienator \
  --num_candidates 1 \
  --top_related 4


"""

import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import fasttext
from collections import defaultdict
import fasttext

from scripts.linear_explainable.integrated_recommendation_pipeline import is_embedding_dimension

class GroupedLinear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
OUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTTEXT_MODEL_PATH = BASE_DIR / "data" / "external" / "cc.en.300.bin"



def load_fasttext_model():
    """FastTextモデルを読み込み"""
    if not FASTTEXT_MODEL_PATH.exists():
        print(f"⚠️ FastText model not found at {FASTTEXT_MODEL_PATH}")
        return None
    print("✅ Loaded FastText model")
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_bundle(path: Path):
    obj = _load_pickle(path)
    if isinstance(obj, dict) and "model" in obj:
        bundle = obj
    else:
        bundle = {"model": obj}
    bundle.setdefault("user_stats", {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}})
    return bundle


def load_dimension_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Add interaction metadata
    interaction_start = 1910
    genre_names = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
        'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
    ]

    dim_idx = interaction_start
    for i, g1 in enumerate(genre_names):
        for j, g2 in enumerate(genre_names):
            meta[str(dim_idx)] = {
                'type': 'interaction_genre',
                'name': f'interaction[user_{g1}×movie_{g2}]',
                'group': 'interaction_genre',
                'description': f'User preference for {g1} × Movie is {g2}'
            }
            dim_idx += 1

    meta[str(dim_idx)] = {'type': 'interaction_matching', 'name': 'actor_matching', 'group': 'interaction_matching'}
    dim_idx += 1
    meta[str(dim_idx)] = {'type': 'interaction_matching', 'name': 'director_matching', 'group': 'interaction_matching'}
    dim_idx += 1
    meta[str(dim_idx)] = {'type': 'interaction_matching', 'name': 'keyword_matching', 'group': 'interaction_matching'}

    return meta


def load_movie_metadata():
    """Load movies_metadata.csv with ratings"""
    df = pd.read_csv(RAW_DIR / "movies_metadata.csv")
    meta = {}
    for _, row in df.iterrows():
        meta[int(row['movie_id'])] = {
            'title': row['movie_title'],
            'rating': float(row['rating']) if pd.notna(row['rating']) else None,
            'num_raters': int(row['num_raters']) if pd.notna(row['num_raters']) else 0,
            'genres': row['genres'],
            'directors': row['directors'],
            'actors': row['actors']
        }
    return meta


def load_movie_entities():
    p = DATA_DIR / "movie_entities.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    if isinstance(data, list):
        for e in data:
            mid = int(e.get("movie_id", -1))
            idx[mid] = e
    elif isinstance(data, dict):
        for k, e in data.items():
            try:
                mid = int(e.get("movie_id", k))
            except Exception:
                mid = int(k)
            idx[mid] = e
    return idx


def as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def get_user_index_by_name(graph, user_name: str):
    users = [str(u) for u in as_list(graph.get("user_names", []))]
    try:
        return users.index(user_name)
    except ValueError:
        return None


def model_predict_norm(model, x: np.ndarray) -> float:
    """
    PyTorch GroupedLinear と scikit-learn LinearRegression 両対応
    """
    # --- scikit-learn 系 (predictメソッドが存在する場合)
    if hasattr(model, "predict"):
        return float(model.predict(x.reshape(1, -1))[0])

    # --- PyTorch GroupedLinear 系
    if hasattr(model, "linear") and hasattr(model.linear, "weight"):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
            y = model(x_t).cpu().numpy().flatten()[0]
        return float(y)

    raise AttributeError("Model has neither predict() nor linear.weight for inference.")

#一時デバッグ用
def model_predict_norm_v2(model, x: np.ndarray) -> float:
    print(">>> using model_predict_norm_v2 <<<")  # ←デバッグ印を追加
    """
    PyTorch GroupedLinear と scikit-learn LinearRegression 両対応
    """
    if hasattr(model, "predict"):
        return float(model.predict(x.reshape(1, -1))[0])

    if hasattr(model, "linear") and hasattr(model.linear, "weight"):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
            y = model(x_t).cpu().numpy().flatten()[0]
        return float(y)

    raise AttributeError("Model has neither predict() nor linear.weight for inference.")





def denormalize_userwise(y_norm: float, u_idx: int, user_stats: dict) -> float:
    mu = user_stats.get("mu_map", {}).get(int(u_idx), user_stats.get("mu_map", {}).get(-1, 0.0))
    sd = user_stats.get("std_map", {}).get(int(u_idx), user_stats.get("std_map", {}).get(-1, 1.0))
    return float(np.clip(y_norm * sd + mu, 1.0, 10.0))


def compute_contributions_linear(model, x: np.ndarray):
    w = np.asarray(getattr(model, "coef_", None))
    if w.ndim > 1:
        w = w.reshape(-1)
    return w * x


def compute_interaction_features(user_features, movie_features, u_idx, m_idx):
    """Compute interaction features for a single (user, movie) pair"""
    from sklearn.metrics.pairwise import cosine_similarity

    # Indices (same as 01_prepare_features.py)
    user_genre_start, user_genre_end = 643, 662
    user_fav_actor_start, user_fav_actor_end = 41, 341
    user_fav_director_start, user_fav_director_end = 342, 642

    movie_genre_start, movie_genre_end = 0, 19
    movie_actor_start, movie_actor_end = 19, 319
    movie_director_start, movie_director_end = 319, 619
    movie_keyword_start, movie_keyword_end = 619, 919

    u_genre = user_features[u_idx, user_genre_start:user_genre_end].cpu().numpy()
    m_genre = movie_features[m_idx, movie_genre_start:movie_genre_end].cpu().numpy()
    genre_interact = np.outer(u_genre, m_genre).flatten()

    u_actor = user_features[u_idx, user_fav_actor_start:user_fav_actor_end].cpu().numpy().reshape(1, -1)
    m_actor = movie_features[m_idx, movie_actor_start:movie_actor_end].cpu().numpy().reshape(1, -1)
    actor_match = cosine_similarity(u_actor, m_actor)[0, 0]

    u_director = user_features[u_idx, user_fav_director_start:user_fav_director_end].cpu().numpy().reshape(1, -1)
    m_director = movie_features[m_idx, movie_director_start:movie_director_end].cpu().numpy().reshape(1, -1)
    director_match = cosine_similarity(u_director, m_director)[0, 0]

    keyword_match = cosine_similarity(u_actor, movie_features[m_idx,
                                               movie_keyword_start:movie_keyword_end].cpu().numpy().reshape(1, -1))[
        0, 0]

    return np.concatenate([genre_interact, [actor_match, director_match, keyword_match]])


def find_related_watched_movies(candidate_contrib, candidate_id, watched_movies_data,
                                model, graphs, u_idx, dim_meta, movie_meta, top_k=5):
    """
    最終版:
    - embedding次元を除外
    - 弱い負寄与（|contrib| < 0.2）を除外
    """
    g = graphs["test"]
    user_features = g['user_features']
    movie_features = g['movie_features']
    review_signals_dim = g['review_signals'].shape[1]
    movie_ids_list = list(map(int, as_list(g.get("movie_ids", []))))

    user_specific_groups = {'user_stats', 'user_behavior', 'user_aspect_zscore',
                            'user_aspect_sentiment', 'user_genre', 'user_fav_actor',
                            'user_fav_director'}

    negative_cross_patterns = [
        'user_Crime×movie_Adventure',
        'user_Romance×movie_Adventure',
        'user_War×movie_Action',
        'user_Comedy×movie_Adventure',
        'user_Drama×movie_Adventure',
        'user_Comedy×movie_Action',
        'user_History×movie_Action',
        'user_Music×movie_Adventure',
        'user_Action×movie_Drama',
        'user_Family×movie_Drama',
        'user_Action×movie_Romance',
        'user_Romance×movie_Action',
        'user_Comedy×movie_Horror',
        'user_Horror×movie_Comedy',
        'user_Thriller×movie_Comedy',
        'user_Comedy×movie_Thriller'
    ]

    similarities = []

    for watched in tqdm(watched_movies_data, desc="  Comparing", leave=False):
        watched_id = watched['movie_id']

        if watched_id not in movie_ids_list:
            continue
        m_idx_watched = movie_ids_list.index(watched_id)

        watched_contrib_recomputed = compute_contribution_for_movie_pair(
            model, user_features, movie_features, u_idx, m_idx_watched, review_signals_dim
        )

        group_scores = defaultdict(list)
        group_priority = ['interaction_genre', 'interaction_matching', 'movie_actor',
                          'movie_director', 'movie_keyword', 'movie_genre', 'review_aspects']

        for i in range(len(candidate_contrib)):
            meta = dim_meta.get(str(i), {})
            group = meta.get('group', 'unknown')
            name = meta.get('name', '')

            if group in user_specific_groups:
                continue

            # Skip embedding dimensions
            if is_embedding_dimension(name):
                continue

            if group in group_priority:
                c_val = candidate_contrib[i]
                w_val = watched_contrib_recomputed[i]

                # Filter asymmetric genre interactions
                if 'interaction[user_' in name and '×movie_' in name:
                    parts = name.replace('interaction[', '').replace(']', '').split('×')
                    if len(parts) == 2:
                        user_genre = parts[0].replace('user_', '')
                        movie_genre = parts[1].replace('movie_', '')

                        if user_genre != movie_genre and abs(c_val) < 0.3:
                            continue

                        pattern = f'user_{user_genre}×movie_{movie_genre}'
                        if pattern in negative_cross_patterns and c_val < 0:
                            continue

                        # Filter weak negative cross-genre
                        if user_genre != movie_genre and c_val < 0 and abs(c_val) < 0.2:
                            continue

                if np.sign(c_val) == np.sign(w_val):
                    score = min(abs(c_val), abs(w_val))
                else:
                    score = -min(abs(c_val), abs(w_val))

                group_scores[group].append(score)

        group_scores_normalized = {}
        for group, scores in group_scores.items():
            if scores:
                scores_arr = np.array(scores)
                mean_s = scores_arr.mean()
                std_s = scores_arr.std() if scores_arr.std() > 1e-6 else 1.0
                normalized = (scores_arr - mean_s) / std_s
                group_scores_normalized[group] = float(normalized.sum())
            else:
                group_scores_normalized[group] = 0.0

        total_score_raw = sum(group_scores_normalized.values())

        # Shared dimensions (filtered)
        shared_dims_positive = []
        shared_dims_negative = []

        for i in range(len(candidate_contrib)):
            meta = dim_meta.get(str(i), {})
            group = meta.get('group', 'unknown')
            name = meta.get('name', '')

            if group in user_specific_groups:
                continue

            # Skip embedding dimensions
            if is_embedding_dimension(name):
                continue

            c_val = candidate_contrib[i]
            w_val = watched_contrib_recomputed[i]

            if abs(c_val) < 1e-4 or abs(w_val) < 1e-4:
                continue

            # Apply filters
            if 'interaction[user_' in name and '×movie_' in name:
                parts = name.replace('interaction[', '').replace(']', '').split('×')
                if len(parts) == 2:
                    user_genre = parts[0].replace('user_', '')
                    movie_genre = parts[1].replace('movie_', '')

                    if user_genre != movie_genre and abs(c_val) < 0.3:
                        continue

                    pattern = f'user_{user_genre}×movie_{movie_genre}'
                    if pattern in negative_cross_patterns and c_val < 0:
                        continue

                    # Filter weak negative
                    if user_genre != movie_genre and c_val < 0 and abs(c_val) < 0.2:
                        continue

            if np.sign(c_val) == np.sign(w_val):
                shared_score = min(abs(c_val), abs(w_val))
                dim_data = {
                    'dim': i,
                    'name': name,
                    'group': group,
                    'candidate_contribution': float(c_val),
                    'watched_contribution': float(w_val),
                    'shared_score': float(shared_score),
                    'effect_type': 'positive' if c_val > 0 else 'negative'
                }

                if c_val > 0:
                    shared_dims_positive.append(dim_data)
                else:
                    shared_dims_negative.append(dim_data)

        shared_dims_positive.sort(key=lambda x: x['shared_score'], reverse=True)
        shared_dims_negative.sort(key=lambda x: x['shared_score'], reverse=True)

        # why_similar
        why = "Similar pattern"
        if shared_dims_positive:
            non_genre = [d for d in shared_dims_positive if d['group'] in ['movie_keyword', 'review_aspects']]
            if non_genre and non_genre[0]['shared_score'] > 0.4:
                d = non_genre[0]
                why = f"Strong {d['group']}: {d['name']} ({d['shared_score']:.2f})"
            else:
                top_3 = shared_dims_positive[:3]
                if len(top_3) == 1:
                    d = top_3[0]
                    why = f"Both have {d['name']} ({d['shared_score']:.2f})"
                else:
                    parts = [f"{d['name']}({d['shared_score']:.2f})" for d in top_3]
                    why = f"Share: {', '.join(parts)}"

        similarities.append({
            'movie_id': watched_id,
            'title': watched['title'],
            'user_rating': watched['user_rating'],
            'similarity_score_raw': float(total_score_raw),
            'group_scores': group_scores_normalized,
            'shared_dimensions_positive': shared_dims_positive[:5],
            'shared_dimensions_negative': shared_dims_negative[:3],
            'why_similar': why
        })

    # Percentile normalization
    if similarities:
        from scipy.stats import rankdata
        raw_scores = [s['similarity_score_raw'] for s in similarities]
        ranks = rankdata(raw_scores, method='average')

        if len(ranks) > 1:
            percentile_scores = (ranks - 1) / (len(ranks) - 1)
        else:
            percentile_scores = [1.0]

        for i, sim in enumerate(similarities):
            sim['similarity_score'] = float(percentile_scores[i])
            del sim['similarity_score_raw']

    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Diversify
    selected = []
    used_groups = set()

    for sim in similarities:
        if len(selected) >= top_k:
            break

        dominant = max(sim['group_scores'].items(), key=lambda x: abs(x[1]))[0] if sim['group_scores'] else None

        if dominant not in used_groups or len(selected) < 2:
            selected.append(sim)
            if dominant:
                used_groups.add(dominant)

    for sim in similarities:
        if len(selected) >= top_k:
            break
        if sim not in selected:
            selected.append(sim)

    return selected[:top_k]


def compute_user_overall_preferences(watched_movies_data, dim_meta, top_k=10):
    """全視聴映画の平均寄与からユーザーの恒常的嗜好を抽出"""
    all_contribs = np.array([m['contribution'] for m in watched_movies_data])
    avg_contrib = all_contribs.mean(axis=0)

    idx = np.argsort(np.abs(avg_contrib))[::-1][:top_k]
    prefs = []
    for i in idx:
        if abs(avg_contrib[i]) < 1e-6:
            continue
        meta = dim_meta.get(str(i), {})
        prefs.append({
            'dim': int(i),
            'name': meta.get('name', f'dim_{i}'),
            'group': meta.get('group', 'unknown'),
            'avg_contribution': float(avg_contrib[i]),
            'frequency': float((np.abs(all_contribs[:, i]) > 1e-3).sum() / len(all_contribs))
        })
    return prefs


def extract_primary_genre_and_factors(contrib, dim_meta, movie_info):
    """候補映画の主要ジャンルと訴求要因を抽出"""
    # Movie genre contributions
    genre_contribs = {}
    genre_names = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
                   'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
                   'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

    for i, meta in dim_meta.items():
        if meta.get('group') == 'movie_genre':
            name = meta.get('name', '')
            if name in genre_names:
                genre_contribs[name] = float(contrib[int(i)])

    # Sort by absolute contribution
    sorted_genres = sorted(genre_contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    primary = sorted_genres[0][0] if sorted_genres else "Unknown"
    secondary = sorted_genres[1][0] if len(sorted_genres) > 1 else None

    # Main appeal factors (top contributors overall)
    idx = np.argsort(np.abs(contrib))[::-1][:5]
    factors = []
    for i in idx:
        if abs(contrib[i]) < 1e-6:
            continue
        meta = dim_meta.get(str(i), {})
        factors.append({
            'factor': meta.get('name', f'dim_{i}'),
            'group': meta.get('group', 'unknown'),
            'score': float(contrib[i])
        })

    return {
        'primary_genre': primary,
        'secondary_genre': secondary,
        'genre_contributions': {g: float(c) for g, c in sorted_genres[:3]},
        'main_appeal_factors': factors
    }


def find_person_connections(candidate_entities, watched_movies_data, movie_entities, movie_meta):
    """俳優・監督のつながりを抽出"""
    candidate_actors = set(candidate_entities.get('actors', []))
    candidate_directors = set(candidate_entities.get('directors', []))

    shared_actors = {}
    shared_directors = {}

    for watched in watched_movies_data:
        mid = watched['movie_id']
        ents = movie_entities.get(mid, {})

        # Actors
        for actor in candidate_actors:
            if actor in ents.get('actors', []):
                if actor not in shared_actors:
                    shared_actors[actor] = []
                shared_actors[actor].append({
                    'movie_id': mid,
                    'title': watched['title'],
                    'user_rating': watched['user_rating']
                })

        # Directors
        for director in candidate_directors:
            if director in ents.get('directors', []):
                if director not in shared_directors:
                    shared_directors[director] = []
                shared_directors[director].append({
                    'movie_id': mid,
                    'title': watched['title'],
                    'user_rating': watched['user_rating']
                })

    return {
        'shared_actors': [
            {
                'name': actor,
                'appears_in_watched': movies
            }
            for actor, movies in shared_actors.items()
        ],
        'shared_directors': [
            {
                'name': director,
                'appears_in_watched': movies
            }
            for director, movies in shared_directors.items()
        ]
    }


def reconstruct_features_for_pair(user_features, movie_features, u_idx, m_idx, review_signals_dim):
    """
    指定された(user, movie)ペアの特徴ベクトルを再構築
    review_signalsはゼロ埋め（未視聴前提）
    """
    u_feat = user_features[u_idx].cpu().numpy()
    m_feat = movie_features[m_idx].cpu().numpy()
    r_feat = np.zeros(review_signals_dim)
    base_feat = np.concatenate([u_feat, m_feat, r_feat])

    # Interaction features
    interact_feat = compute_interaction_features(
        user_features, movie_features, u_idx, m_idx
    )

    return np.concatenate([base_feat, interact_feat])


def compute_contribution_for_movie_pair(model, user_features, movie_features, u_idx, m_idx, review_signals_dim):
    """
    指定された(user, movie)ペアの寄与を完全再計算
    """
    x_full = reconstruct_features_for_pair(
        user_features, movie_features, u_idx, m_idx, review_signals_dim
    )
    return compute_contributions_linear(model, x_full)


def compute_novelty_score(movie_id, movie_meta):
    """
    映画の新規性スコア（人気度の逆数）
    """
    meta = movie_meta.get(movie_id, {})
    num_raters = meta.get('num_raters', 0)

    if num_raters < 10:
        return 1.0
    elif num_raters > 5000:
        return 0.0
    else:
        # Log-linear interpolation
        return float(1.0 - (np.log10(num_raters) - 1.0) / (np.log10(5000) - 1.0))


def compute_diversity_score(candidate_id, related_movies, movie_meta, graphs):
    """
    Cosine距離ベースの多様性スコア
    """
    g = graphs["test"]
    movie_ids_list = list(map(int, as_list(g.get("movie_ids", []))))

    if candidate_id not in movie_ids_list:
        return 0.5

    cand_idx = movie_ids_list.index(candidate_id)
    cand_feat = g['movie_features'][cand_idx].cpu().numpy()

    distances = []
    for rel in related_movies:
        rel_id = rel['movie_id']
        if rel_id not in movie_ids_list:
            continue

        rel_idx = movie_ids_list.index(rel_id)
        rel_feat = g['movie_features'][rel_idx].cpu().numpy()

        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([cand_feat], [rel_feat])[0, 0]
        distance = 1.0 - sim
        distances.append(distance)

    if not distances:
        return 0.5

    return float(np.mean(distances))


def find_person_connections_with_confidence(candidate_entities, watched_movies_data,
                                            movie_entities, movie_meta, model, graphs, u_idx):
    """
    最終版: top1のみ + representative example
    """
    candidate_actors = set(candidate_entities.get('actors', []))
    candidate_directors = set(candidate_entities.get('directors', []))

    # Exact matches
    shared_actors_exact = {}
    shared_directors_exact = {}

    for watched in watched_movies_data:
        mid = watched['movie_id']
        ents = movie_entities.get(mid, {})

        for actor in candidate_actors:
            if actor in ents.get('actors', []):
                if actor not in shared_actors_exact:
                    shared_actors_exact[actor] = []
                shared_actors_exact[actor].append({
                    'movie_id': mid,
                    'title': watched['title'],
                    'user_rating': watched['user_rating']
                })

        for director in candidate_directors:
            if director in ents.get('directors', []):
                if director not in shared_directors_exact:
                    shared_directors_exact[director] = []
                shared_directors_exact[director].append({
                    'movie_id': mid,
                    'title': watched['title'],
                    'user_rating': watched['user_rating']
                })

    # Fuzzy matches (top 1 only)
    g = graphs["test"]
    movie_ids_list = list(map(int, as_list(g.get("movie_ids", []))))

    best_actor_match = None
    best_director_match = None

    if candidate_entities.get('movie_id') in movie_ids_list:
        cand_idx = movie_ids_list.index(candidate_entities['movie_id'])
        cand_actor_emb = g['movie_features'][cand_idx, 19:319].cpu().numpy()
        cand_director_emb = g['movie_features'][cand_idx, 319:619].cpu().numpy()

        for watched in watched_movies_data:
            mid = watched['movie_id']
            if mid not in movie_ids_list:
                continue

            w_idx = movie_ids_list.index(mid)
            w_actor_emb = g['movie_features'][w_idx, 19:319].cpu().numpy()
            w_director_emb = g['movie_features'][w_idx, 319:619].cpu().numpy()

            from sklearn.metrics.pairwise import cosine_similarity
            actor_sim = cosine_similarity([cand_actor_emb], [w_actor_emb])[0, 0]
            director_sim = cosine_similarity([cand_director_emb], [w_director_emb])[0, 0]

            # Track best matches
            if actor_sim >= 0.5:
                if best_actor_match is None or actor_sim > best_actor_match['similarity']:
                    best_actor_match = {
                        'similarity': float(actor_sim),
                        'confidence_level': 'high' if actor_sim >= 0.75 else 'medium',
                        'movie_id': mid,
                        'title': watched['title'],
                        'user_rating': watched['user_rating']
                    }

            if director_sim >= 0.5:
                if best_director_match is None or director_sim > best_director_match['similarity']:
                    best_director_match = {
                        'similarity': float(director_sim),
                        'confidence_level': 'high' if director_sim >= 0.75 else 'medium',
                        'movie_id': mid,
                        'title': watched['title'],
                        'user_rating': watched['user_rating']
                    }

    result = {
        'shared_actors': [
            {
                'name': actor,
                'match_type': 'exact',
                'confidence': 1.0,
                'confidence_level': 'exact',
                'appears_in_watched': movies
            }
            for actor, movies in shared_actors_exact.items()
        ],
        'shared_directors': [
            {
                'name': director,
                'match_type': 'exact',
                'confidence': 1.0,
                'confidence_level': 'exact',
                'appears_in_watched': movies
            }
            for director, movies in shared_directors_exact.items()
        ]
    }

    # Add best fuzzy match as representative
    if best_actor_match:
        result['shared_actors'].append({
            'name': "similar_acting_style",
            'match_type': 'style_similarity',
            'confidence': best_actor_match['similarity'],
            'confidence_level': best_actor_match['confidence_level'],
            'representative_example': best_actor_match['title'],
            'appears_in_watched': [{
                'movie_id': best_actor_match['movie_id'],
                'title': best_actor_match['title'],
                'user_rating': best_actor_match['user_rating']
            }]
        })

    if best_director_match:
        result['shared_directors'].append({
            'name': "similar_directing_style",
            'match_type': 'style_similarity',
            'confidence': best_director_match['similarity'],
            'confidence_level': best_director_match['confidence_level'],
            'representative_example': best_director_match['title'],
            'appears_in_watched': [{
                'movie_id': best_director_match['movie_id'],
                'title': best_director_match['title'],
                'user_rating': best_director_match['user_rating']
            }]
        })

    return result


def get_novelty_tag(novelty_score):
    """Noveltyスコアをタグ化"""
    if novelty_score < 0.2:
        return "familiar"
    elif novelty_score < 0.5:
        return "moderately_novel"
    elif novelty_score < 0.8:
        return "novel"
    else:
        return "rare_discovery"




def extract_genre_effects(contrib, dim_meta):
    """
    interaction と movie_genre を統合してジャンルごとの効果をサマリ
    """
    genre_names = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
        'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
    ]

    genre_effects = {}

    for genre in genre_names:
        interaction_sum = 0.0
        genre_base = 0.0

        for i, meta in dim_meta.items():
            i_int = int(i)
            name = meta.get('name', '')

            # Interaction contribution (diagonal only)
            if f'interaction[user_{genre}×movie_{genre}]' == name:
                interaction_sum += contrib[i_int]

            # Base genre contribution
            if meta.get('group') == 'movie_genre' and name == genre:
                genre_base += contrib[i_int]

        # Only include if there's meaningful contribution
        net_effect = interaction_sum + genre_base
        if abs(net_effect) > 0.1 or abs(interaction_sum) > 0.3:
            interpretation = "positive" if net_effect > 0 else "negative"
            genre_effects[genre] = {
                'interaction_contribution': float(interaction_sum),
                'genre_base_contribution': float(genre_base),
                'net_effect': float(net_effect),
                'interpretation': interpretation
            }

    return genre_effects


def compute_user_relative_strength(user_prefs, all_users_stats):
    """
    全ユーザー統計に基づいてZ-scoreを計算

    Parameters:
    - user_prefs: compute_user_overall_preferences の出力
    - all_users_stats: {dim: {'mean': float, 'std': float}}

    Returns:
    - user_prefs に 'relative_strength' フィールドを追加
    """
    for pref in user_prefs:
        dim = pref['dim']
        user_contrib = pref['avg_contribution']

        global_mean = all_users_stats.get(dim, {}).get('mean', 0.0)
        global_std = all_users_stats.get(dim, {}).get('std', 1.0)

        if global_std < 1e-6:
            z_score = 0.0
        else:
            z_score = (user_contrib - global_mean) / global_std

        pref['relative_strength'] = float(z_score)
        pref['interpretation'] = f"{z_score:+.2f}σ from population mean"

    return user_prefs


def load_or_compute_global_stats(features_data, dim_meta):
    """
    全ユーザーの寄与統計を計算（キャッシュ可能）

    Returns:
    - {dim: {'mean': float, 'std': float}}
    """
    all_contribs = []

    for split in ['train', 'val', 'test']:
        X = features_data[split]['X']

        # モデルの重みを取得（ここでは簡易的に全特徴の平均を使用）
        # 実際には model.coef_ と掛け算した寄与が必要
        # ここでは特徴値の統計として代用
        all_contribs.append(X)

    all_contribs_stacked = np.vstack(all_contribs)

    stats = {}
    for dim in range(all_contribs_stacked.shape[1]):
        values = all_contribs_stacked[:, dim]
        stats[dim] = {
            'mean': float(values.mean()),
            'std': float(values.std() if values.std() > 1e-6 else 1.0)
        }

    return stats


def compute_person_influence(entities, contrib, dim_meta, graphs, candidate_id):
    """
    俳優・監督の寄与値を計算

    Returns:
    - {actor_name: influence_score, ...}
    """
    g = graphs["test"]
    movie_ids_list = list(map(int, as_list(g.get("movie_ids", []))))

    if candidate_id not in movie_ids_list:
        return {}, {}

    m_idx = movie_ids_list.index(candidate_id)

    # Actor embedding indices: 19-319
    actor_emb_start, actor_emb_end = 19, 319
    actor_contrib = contrib[1910 + actor_emb_start:1910 + actor_emb_end]  # Adjust for offset

    # Actually, in the base features:
    # movie_features start at dim 708
    # movie_actor: 708+19 to 708+319
    # But in X vector: user(708) + movie(1202) = positions 708 to 1910
    # movie_actor in X: 708+19 = 727 to 708+319 = 1027

    # Correct indices in contrib vector
    movie_start_in_X = 708
    actor_start_in_X = movie_start_in_X + 19
    actor_end_in_X = movie_start_in_X + 319
    director_start_in_X = movie_start_in_X + 319
    director_end_in_X = movie_start_in_X + 619

    actor_contrib_block = contrib[actor_start_in_X:actor_end_in_X]
    director_contrib_block = contrib[director_start_in_X:director_end_in_X]

    # Compute average influence
    actor_influence = float(np.mean(np.abs(actor_contrib_block)))
    director_influence = float(np.mean(np.abs(director_contrib_block)))

    # Map to specific people
    actors = entities.get('actors', [])
    directors = entities.get('directors', [])

    actor_influences = {actor: actor_influence for actor in actors}
    director_influences = {director: director_influence for director in directors}

    return actor_influences, director_influences


TAG_MAPPING = {
    'E01': 'fun', 'E02': 'sad', 'E03': 'scary', 'E04': 'funny', 'E05': 'tense',
    'E06': 'healing', 'E07': 'exciting', 'E08': 'heavy', 'E09': 'dark', 'E10': 'bright',
    'E11': 'bittersweet', 'E12': 'warm', 'E13': 'anxious', 'E14': 'refreshing',
    'E15': 'depressing', 'E16': 'nostalgic', 'E17': 'tearjerker', 'E18': 'touching',
    'E19': 'melancholic',

    'V01': 'easy_to_follow', 'V02': 'hard_to_understand', 'V03': 'requires_focus',
    'V04': 'thought_provoking', 'V05': 'immersive', 'V06': 'casual_watch',

    'T01': 'fast_paced', 'T02': 'slow_paced', 'T03': 'dragging', 'T04': 'picks_up',
    'T05': 'anti_climactic',

    'D01': 'plot_twist', 'D02': 'straightforward', 'D03': 'complex_timeline',
    'D04': 'ensemble', 'D05': 'happy_ending', 'D06': 'tragic_ending',

    'S01': 'commercial', 'S02': 'arthouse', 'S03': 'experimental', 'S04': 'mainstream',
    'S05': 'quirky', 'S06': 'blockbuster', 'S07': 'b_movie', 'S08': 'trashy',
    'S09': 'indie', 'S10': 'cult', 'S11': 'buzzworthy', 'S12': 'polished',

    'A01': 'family_friendly', 'A02': 'mature', 'A03': 'all_ages', 'A04': 'niche',
    'A05': 'crowd_pleaser', 'A06': 'male_oriented', 'A07': 'female_oriented',
    'A08': 'working_adult',

    'C01': 'solo_watch', 'C02': 'date_night', 'C03': 'family_time', 'C04': 'group_fun',
    'C05': 'late_night', 'C06': 'party_movie', 'C07': 'quiet_viewing',

    'J01': 'gory', 'J02': 'grotesque', 'J03': 'splatter', 'J04': 'slasher',
    'J05': 'disturbing', 'J06': 'suspenseful', 'J07': 'action_packed', 'J08': 'romantic',
    'J09': 'hardboiled', 'J10': 'parody', 'J11': 'dark_humor', 'J12': 'comedic',
    'J13': 'surreal', 'J14': 'absurd', 'J15': 'coming_of_age', 'J16': 'revenge',
    'J17': 'friendship', 'J18': 'family_love', 'J19': 'romance', 'J20': 'betrayal',
    'J21': 'growth_story', 'J22': 'mentor_student', 'J23': 'rivalry', 'J24': 'love_triangle',
    'J25': 'costume_design', 'J26': 'special_makeup', 'J27': 'cgi', 'J28': 'practical_effects',
    'J29': 'wire_action', 'J30': 'explosions', 'J31': 'car_chase', 'J32': 'gunfight',
    'J33': 'fight_scenes', 'J34': 'swordplay', 'J35': 'dance_sequence', 'J36': 'singing',
    'J37': 'musical_performance', 'J38': 'cooking_scenes', 'J39': 'sports_scenes',
    'J40': 'mockumentary', 'J41': 'first_person_pov', 'J42': 'static_camera',
    'J43': 'long_take', 'J44': 'single_shot', 'J45': 'time_loop', 'J46': 'real_time',
    'J47': 'confined_space', 'J48': 'post_apocalyptic', 'J49': 'psychological_horror',
    'J50': 'jump_scare', 'J51': 'occult', 'J52': 'retro', 'J53': 'class_divide',
    'J54': 'generational_conflict', 'J55': 'buddy_movie',

    'P01': 'satisfying', 'P02': 'disappointing', 'P03': 'shocking', 'P04': 'memorable',
    'P05': 'confusing', 'P06': 'overwhelming', 'P07': 'fresh', 'P08': 'predictable',
    'P09': 'bitter_aftertaste', 'P10': 'escapist', 'P11': 'challenging',
    'P12': 'guilty_pleasure', 'P13': 'engaging', 'P14': 'divisive',

    'W01': 'binge_worthy', 'W02': 'rewatch_value', 'W03': 'time_killer', 'W04': 'time_waster',

    'L01': 'educational', 'L02': 'culturally_significant', 'L03': 'must_see_classic',
    'L04': 'relevant'
}


def apply_tag_mapping(name):
    """タグコードを人間可読名に変換"""
    for tag_code, readable_name in TAG_MAPPING.items():
        if tag_code in name:
            return name.replace(tag_code, f"{tag_code}({readable_name})")
    return name


def extract_movie_tags_with_contributions(contrib, dim_meta, top_k=5):
    """
    映画が持つtag（非ゼロ）の寄与を上位K個抽出
    """
    tag_contribs = []

    for dim_str, meta in dim_meta.items():
        if meta.get('group') == 'movie_tags':
            dim_idx = int(dim_str)
            tag_value = contrib[dim_idx]

            if abs(tag_value) > 1e-6:
                name = meta.get('name', f'tag_{dim_idx}')

                # Apply tag mapping (修正: 必ず適用)
                if name in TAG_MAPPING:
                    readable_name = f"{name}({TAG_MAPPING[name]})"
                else:
                    readable_name = name

                tag_contribs.append({
                    'tag': readable_name,
                    'dim': dim_idx,
                    'contribution': float(tag_value),
                    'effect_type': 'positive' if tag_value > 0 else 'negative'
                })

    tag_contribs.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return tag_contribs[:top_k]


def extract_suggested_tags(contrib, dim_meta, existing_tag_dims, top_k=3):
    """
    映画が持たないが重要なtagを抽出
    """
    suggested = []

    for dim_str, meta in dim_meta.items():
        if meta.get('group') == 'movie_tags':
            dim_idx = int(dim_str)

            if dim_idx in existing_tag_dims:
                continue

            contrib_value = contrib[dim_idx]

            if abs(contrib_value) > 1e-6:
                name = meta.get('name', f'tag_{dim_idx}')

                # Apply tag mapping
                if name in TAG_MAPPING:
                    readable_name = f"{name}({TAG_MAPPING[name]})"
                else:
                    readable_name = name

                suggested.append({
                    'tag': readable_name,
                    'dim': dim_idx,
                    'contribution': float(contrib_value),
                    'effect_type': 'positive' if contrib_value > 0 else 'negative',
                    'reason': 'high_model_contribution'
                })

    suggested.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return suggested[:top_k]


def extract_movie_keywords_with_contributions(candidate_id, movie_entities, contrib,
                                              dim_meta, ft_model, graphs):
    """
    映画が持つkeywordの寄与をFastTextで計算
    """
    entities = movie_entities.get(candidate_id, {})
    keywords = entities.get('keywords', [])

    if not keywords:
        return []

    # Find keyword embedding block in contrib
    keyword_start, keyword_end = None, None
    for dim_str, meta in dim_meta.items():
        if meta.get('group') == 'movie_keyword':
            dim_idx = int(dim_str)
            if keyword_start is None:
                keyword_start = dim_idx
            keyword_end = dim_idx + 1

    if keyword_start is None:
        return []

    # Get weight vector for keyword block
    keyword_contrib_block = contrib[keyword_start:keyword_end]

    keyword_contribs = []
    for keyword in keywords:
        try:
            # Get FastText embedding for this keyword
            keyword_emb = ft_model.get_word_vector(keyword)

            # Compute contribution as dot product
            contribution = float(np.dot(keyword_emb, keyword_contrib_block))

            keyword_contribs.append({
                'keyword': keyword,
                'contribution': contribution,
                'effect_type': 'positive' if contribution > 0 else 'negative'
            })
        except Exception as e:
            # Skip if keyword not in FastText vocab
            continue

    # Sort by absolute contribution
    keyword_contribs.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return keyword_contribs


def extract_suggested_keywords(candidate_id, movie_entities, contrib, dim_meta,
                               ft_model, all_keywords_pool, top_k=3):
    """
    映画が持たないが、寄与が高いkeywordを抽出

    Parameters:
    - all_keywords_pool: 全映画のkeywordセット（事前に構築）
    """
    entities = movie_entities.get(candidate_id, {})
    existing_keywords = set(entities.get('keywords', []))

    # Find keyword embedding block
    keyword_start, keyword_end = None, None
    for dim_str, meta in dim_meta.items():
        if meta.get('group') == 'movie_keyword':
            dim_idx = int(dim_str)
            if keyword_start is None:
                keyword_start = dim_idx
            keyword_end = dim_idx + 1

    if keyword_start is None:
        return []

    keyword_contrib_block = contrib[keyword_start:keyword_end]

    suggested = []
    for keyword in all_keywords_pool:
        if keyword in existing_keywords:
            continue

        try:
            keyword_emb = ft_model.get_word_vector(keyword)
            contribution = float(np.dot(keyword_emb, keyword_contrib_block))

            if abs(contribution) > 1e-6:
                suggested.append({
                    'keyword': keyword,
                    'contribution': contribution,
                    'effect_type': 'positive' if contribution > 0 else 'negative',
                    'reason': 'high_model_contribution'
                })
        except Exception:
            continue

    suggested.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return suggested[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_name", required=True)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--top_related", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print(f"RECOMMENDATION CONTEXT GENERATION for {args.user_name}")
    print("=" * 70)

    print("\nLoading data...")
    bundle = load_model_bundle(MODEL_DIR / "best_model.pkl")
    model, user_stats = bundle["model"], bundle["user_stats"]
    dim_meta = load_dimension_metadata(DATA_DIR / "dimension_metadata.json")
    movie_meta = load_movie_metadata()
    movie_entities = load_movie_entities()
    ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

    print("Loading graphs...")
    graphs = {split: torch.load(DATA_DIR / f"hetero_graph_{split}.pt", weights_only=False)
              for split in ["train", "val", "test"]}

    features_data = {}
    for split in ["train", "val", "test"]:
        with open(DATA_DIR / f"linear_features_{split}.pkl", "rb") as f:
            features_data[split] = pickle.load(f)

    g_ref = graphs["test"]
    u_idx = get_user_index_by_name(g_ref, args.user_name)
    if u_idx is None:
        print(f"❌ User not found")
        return

    print(f"User index: {u_idx}")

    print("\nCollecting watched movies...")
    watched_movies_data = []
    for split in ["train", "val", "test"]:
        feat_data = features_data[split]
        mask = (feat_data['user_indices'] == u_idx)
        edge_indices = np.where(mask)[0]

        g = graphs[split]
        movie_ids = list(map(int, as_list(g.get("movie_ids", []))))

        for edge_idx in edge_indices:
            m_idx = int(feat_data['movie_indices'][edge_idx])
            movie_id = movie_ids[m_idx]

            x_full = feat_data['X'][edge_idx]
            y_raw = feat_data['y_raw'][edge_idx]
            contrib = compute_contributions_linear(model, x_full)

            watched_movies_data.append({
                'movie_id': movie_id,
                'title': movie_meta.get(movie_id, {}).get('title', f'Movie {movie_id}'),
                'user_rating': float(y_raw),
                'contribution': contrib
            })

    watched_movie_ids = set(m['movie_id'] for m in watched_movies_data)
    print(f"Found {len(watched_movie_ids)} watched movies")

    # Compute user overall preferences
    print("Computing user overall preferences...")
    user_prefs = compute_user_overall_preferences(watched_movies_data, dim_meta, top_k=10)

    # Compute global statistics for relative strength
    print("Computing global statistics...")
    global_stats = load_or_compute_global_stats(features_data, dim_meta)
    user_prefs = compute_user_relative_strength(user_prefs, global_stats)

    # Build keyword pool from all movies
    print("Building keyword pool...")
    all_keywords_pool = set()
    for mid, entities in movie_entities.items():
        keywords = entities.get('keywords', [])
        all_keywords_pool.update(keywords)

    print(f"Total unique keywords: {len(all_keywords_pool)}")

    # Find unwatched movies
    all_movie_ids = set(movie_meta.keys())
    unwatched_ids = list(all_movie_ids - watched_movie_ids)

    if len(unwatched_ids) == 0:
        print("❌ No unwatched movies")
        return

    num_candidates = min(args.num_candidates, len(unwatched_ids))
    candidate_ids = np.random.choice(unwatched_ids, size=num_candidates, replace=False)

    print(f"\nProcessing {num_candidates} candidate movies...")

    results = []
    for candidate_id in tqdm(candidate_ids, desc="Candidates"):
        g = graphs["test"]
        movie_ids_list = list(map(int, as_list(g.get("movie_ids", []))))

        if candidate_id not in movie_ids_list:
            continue

        m_idx = movie_ids_list.index(candidate_id)

        # Build candidate features
        x_full = reconstruct_features_for_pair(
            g['user_features'], g['movie_features'], u_idx, m_idx, g['review_signals'].shape[1]
        )

        # Predict
        y_norm = model_predict_norm_v2(model, x_full)


        y_raw = denormalize_userwise(y_norm, u_idx, user_stats)
        contrib = compute_contributions_linear(model, x_full)

        # Genre effects
        genre_effects = extract_genre_effects(contrib, dim_meta)

        movie_info = movie_meta.get(candidate_id, {})
        genre_analysis = extract_primary_genre_and_factors(contrib, dim_meta, movie_info)

        # Find related watched movies
        related = find_related_watched_movies(
            contrib, candidate_id, watched_movies_data,
            model, graphs, u_idx, dim_meta, movie_meta, top_k=args.top_related
        )

        # Extract keyword/tag contributions
        movie_keywords = extract_movie_keywords_with_contributions(
            candidate_id, movie_entities, contrib, dim_meta, ft_model, graphs
        )

        movie_tags = extract_movie_tags_with_contributions(
            contrib, dim_meta, top_k=5
        )

        existing_tag_dims = {t['dim'] for t in movie_tags}

        suggested_keywords = extract_suggested_keywords(
            candidate_id, movie_entities, contrib, dim_meta, ft_model,
            all_keywords_pool, top_k=3
        )

        suggested_tags = extract_suggested_tags(
            contrib, dim_meta, existing_tag_dims, top_k=3
        )

        # Top contributors (exclude keyword/tag groups)
        user_specific_groups = {'user_stats', 'user_behavior', 'user_aspect_zscore',
                                'user_aspect_sentiment', 'user_genre', 'user_fav_actor',
                                'user_fav_director'}

        excluded_groups = user_specific_groups | {'movie_keyword', 'movie_tags'}

        priority_groups = ['review_aspects', 'interaction_matching']

        top_contrib = []
        idx_sorted = np.argsort(np.abs(contrib))[::-1]

        for i in idx_sorted:
            if abs(contrib[i]) < 1e-6:
                continue
            meta = dim_meta.get(str(i), {})
            group = meta.get('group', 'unknown')
            name = meta.get('name', '')

            # Exclude user-specific, embedding, keyword, tag
            if group in excluded_groups or is_embedding_dimension(name):
                continue

            readable_name = apply_tag_mapping(name)

            if group in priority_groups and abs(contrib[i]) > 0.3 and len(top_contrib) < 3:
                top_contrib.append({
                    'dim': int(i),
                    'name': readable_name,
                    'group': group,
                    'contribution': float(contrib[i]),
                    'effect_type': 'positive' if contrib[i] > 0 else 'negative'
                })
            elif len(top_contrib) < 20 and not any(c['dim'] == int(i) for c in top_contrib):
                top_contrib.append({
                    'dim': int(i),
                    'name': readable_name,
                    'group': group,
                    'contribution': float(contrib[i]),
                    'effect_type': 'positive' if contrib[i] > 0 else 'negative'
                })

        # Person connections with influence
        entities = movie_entities.get(candidate_id, {})
        entities['movie_id'] = candidate_id

        actor_influences, director_influences = compute_person_influence(
            entities, contrib, dim_meta, graphs, candidate_id
        )

        person_connections = find_person_connections_with_confidence(
            entities, watched_movies_data, movie_entities, movie_meta, model, graphs, u_idx
        )

        # Add influence scores
        for actor_data in person_connections['shared_actors']:
            actor_name = actor_data['name']
            if actor_name in actor_influences:
                actor_data['influence_on_score'] = actor_influences[actor_name]
            elif actor_name == "similar_acting_style":
                avg_influence = np.mean(list(actor_influences.values())) if actor_influences else 0.0
                actor_data['influence_on_score'] = float(avg_influence)

        for director_data in person_connections['shared_directors']:
            director_name = director_data['name']
            if director_name in director_influences:
                director_data['influence_on_score'] = director_influences[director_name]
            elif director_name == "similar_directing_style":
                avg_influence = np.mean(list(director_influences.values())) if director_influences else 0.0
                director_data['influence_on_score'] = float(avg_influence)

        # Novelty & Diversity
        novelty = compute_novelty_score(candidate_id, movie_meta)
        novelty_tag = get_novelty_tag(novelty)
        diversity = compute_diversity_score(candidate_id, related, movie_meta, graphs)

        results.append({
            'candidate_movie': {
                'movie_id': int(candidate_id),
                'title': movie_info.get('title', f'Movie {candidate_id}'),
                'predicted_rating': float(y_raw),
                'movie_avg_rating': movie_info.get('rating'),
                'num_raters': movie_info.get('num_raters', 0),
                'novelty_score': float(novelty),
                'novelty_tag': novelty_tag,
                'diversity_score': float(diversity),
                'genres': movie_info.get('genres', ''),
                'directors': entities.get('directors', []),
                'actors': entities.get('actors', []),
                'keywords': entities.get('keywords', []),
                'genre_effects': genre_effects,
                'movie_keywords_with_contributions': movie_keywords,
                'movie_tags_with_contributions': movie_tags,
                'suggested_keywords': suggested_keywords,
                'suggested_tags': suggested_tags,
                **genre_analysis
            },
            'related_watched_movies': related,
            'top_contributors': top_contrib,
            'person_connections': person_connections
        })

    output = {
        'user_name': args.user_name,
        'user_overall_preferences': user_prefs,
        'recommendations': results
    }

    out_path = OUT_DIR / f"{args.user_name}_recommendations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {out_path}")
    print(f"✅ Generated {len(results)} recommendations")
    print("🎉 Done!")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()