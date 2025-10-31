"""
Dimension Metadata Generator
各次元のメタデータを生成し、生データへの遡及を可能にする
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np


class DimensionMetadataGenerator:
    """次元メタデータの生成クラス"""

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

    def __init__(self,
                 movie_features_path: Path,
                 user_features_path: Path,
                 movie_entities_path: Path):
        """
        Args:
            movie_features_path: movie_features.ptのパス
            user_features_path: user_features.ptのパス
            movie_entities_path: movie_entities.jsonのパス
        """
        # Load data
        self.movie_data = torch.load(movie_features_path, weights_only=False)
        self.user_data = torch.load(user_features_path, weights_only=False)

        with open(movie_entities_path, 'r', encoding='utf-8') as f:
            self.movie_entities = json.load(f)

        self.metadata = {}

    def generate_all(self) -> Dict[int, Dict[str, Any]]:
        """全次元のメタデータを生成"""
        self.metadata = {}

        # User features
        self._generate_user_metadata()

        # Movie features
        offset = self.user_data['features'].shape[1]
        self._generate_movie_metadata(offset)

        # Review features
        offset += self.movie_data['features'].shape[1]
        self._generate_review_metadata(offset)

        return self.metadata

    def _generate_user_metadata(self):
        """ユーザー特徴のメタデータ"""
        dim = 0

        # Aspect zscore (18D)
        for i, aspect in enumerate(self.ASPECT_NAMES):
            self.metadata[dim] = {
                'type': 'user_aspect_zscore',
                'name': aspect,
                'group': 'user_aspect_zscore',
                'dimension_in_group': i,
                'source': 'users.csv (zscore_* columns)',
                'description': f'ユーザーの{aspect}重視度（Z-score）',
                'traceable': False
            }
            dim += 1

        # Aspect sentiment (18D)
        for i, aspect in enumerate(self.ASPECT_NAMES):
            self.metadata[dim] = {
                'type': 'user_aspect_sentiment',
                'name': aspect,
                'group': 'user_aspect_sentiment',
                'dimension_in_group': i,
                'source': 'users.csv (sentiment_* columns)',
                'description': f'ユーザーの{aspect}評価傾向',
                'traceable': False
            }
            dim += 1

        # Review stats (5D)
        stat_names = ['review_count', 'rating_mean', 'rating_std', 'rating_min', 'rating_max']
        for i, stat in enumerate(stat_names):
            self.metadata[dim] = {
                'type': 'user_stats',
                'name': stat,
                'group': 'user_stats',
                'dimension_in_group': i,
                'source': 'aggregated from reviews.csv',
                'description': f'ユーザーの{stat}',
                'traceable': False
            }
            dim += 1

        # Favorite actor embedding (300D)
        for i in range(300):
            self.metadata[dim] = {
                'type': 'user_fav_actor_emb',
                'name': f'fav_actor_emb_{i}',
                'group': 'user_fav_actor',
                'dimension_in_group': i,
                'embedding_dim': i,
                'source': 'Top-5 actors FastText average',
                'description': f'お気に入り俳優埋め込み（次元{i}）',
                'traceable': True,
                'traceable_to': 'user_features.person_name_list (actors)'
            }
            dim += 1

        # Favorite actor count (1D)
        self.metadata[dim] = {
            'type': 'user_fav_actor_count',
            'name': 'fav_actor_count',
            'group': 'user_fav_actor',
            'dimension_in_group': 300,
            'source': 'computed from person_name_list',
            'description': 'お気に入り俳優の数',
            'traceable': False
        }
        dim += 1

        # Favorite director embedding (300D)
        for i in range(300):
            self.metadata[dim] = {
                'type': 'user_fav_director_emb',
                'name': f'fav_director_emb_{i}',
                'group': 'user_fav_director',
                'dimension_in_group': i,
                'embedding_dim': i,
                'source': 'Top-3 directors FastText average',
                'description': f'お気に入り監督埋め込み（次元{i}）',
                'traceable': True,
                'traceable_to': 'user_features.person_name_list (directors)'
            }
            dim += 1

        # Favorite director count (1D)
        self.metadata[dim] = {
            'type': 'user_fav_director_count',
            'name': 'fav_director_count',
            'group': 'user_fav_director',
            'dimension_in_group': 300,
            'source': 'computed from person_name_list',
            'description': 'お気に入り監督の数',
            'traceable': False
        }
        dim += 1

        # Genre ratings (19D)
        for i, genre in enumerate(self.GENRE_NAMES):
            self.metadata[dim] = {
                'type': 'user_genre_rating',
                'name': genre,
                'group': 'user_genre',
                'dimension_in_group': i,
                'source': f'users.csv (rating_{genre})',
                'description': f'{genre}ジャンルの平均評価',
                'traceable': False
            }
            dim += 1

        # Genre counts (19D)
        for i, genre in enumerate(self.GENRE_NAMES):
            self.metadata[dim] = {
                'type': 'user_genre_count',
                'name': genre,
                'group': 'user_genre',
                'dimension_in_group': i + 19,
                'source': f'users.csv (count_{genre})',
                'description': f'{genre}ジャンルの視聴回数',
                'traceable': False
            }
            dim += 1

        # Genre diversity (1D)
        self.metadata[dim] = {
            'type': 'user_genre_diversity',
            'name': 'genre_diversity',
            'group': 'user_behavior',
            'dimension_in_group': 0,
            'source': 'computed entropy of genre distribution',
            'description': 'ジャンル嗜好の多様性',
            'traceable': False
        }
        dim += 1

        # Active days (1D)
        self.metadata[dim] = {
            'type': 'user_active_days',
            'name': 'active_days',
            'group': 'user_behavior',
            'dimension_in_group': 1,
            'source': 'computed from first_date and last_date',
            'description': 'アクティブ日数',
            'traceable': False
        }
        dim += 1

        # Review velocity (1D)
        self.metadata[dim] = {
            'type': 'user_review_velocity',
            'name': 'review_velocity',
            'group': 'user_behavior',
            'dimension_in_group': 2,
            'source': 'review_count / active_days',
            'description': 'レビュー投稿速度',
            'traceable': False
        }
        dim += 1

        # Rating vs global (1D)
        self.metadata[dim] = {
            'type': 'user_rating_vs_global',
            'name': 'rating_vs_global',
            'group': 'user_behavior',
            'dimension_in_group': 3,
            'source': 'user_mean - global_mean',
            'description': '全体平均との評価差',
            'traceable': False
        }
        dim += 1

        # Sentiment overall (1D)
        self.metadata[dim] = {
            'type': 'user_sentiment_overall',
            'name': 'sentiment_overall',
            'group': 'user_behavior',
            'dimension_in_group': 4,
            'source': 'mean of aspect sentiments',
            'description': '総合的なセンチメント',
            'traceable': False
        }
        dim += 1

    def _generate_movie_metadata(self, offset: int):
        """映画特徴のメタデータ"""
        dim = offset

        # Genre (19D)
        for i, genre in enumerate(self.GENRE_NAMES):
            self.metadata[dim] = {
                'type': 'movie_genre',
                'name': genre,
                'group': 'movie_genre',
                'dimension_in_group': i,
                'source': 'movies_metadata.csv (genres)',
                'description': f'{genre}ジャンル（binary）',
                'traceable': False
            }
            dim += 1

        # Actor embedding (300D)
        for i in range(300):
            self.metadata[dim] = {
                'type': 'movie_actor_emb',
                'name': f'actor_emb_{i}',
                'group': 'movie_actor',
                'dimension_in_group': i,
                'embedding_dim': i,
                'source': 'actors FastText average',
                'description': f'俳優埋め込み（次元{i}）',
                'traceable': True,
                'traceable_to': 'movie_entities.json (actors)'
            }
            dim += 1

        # Director embedding (300D)
        for i in range(300):
            self.metadata[dim] = {
                'type': 'movie_director_emb',
                'name': f'director_emb_{i}',
                'group': 'movie_director',
                'dimension_in_group': i,
                'embedding_dim': i,
                'source': 'directors FastText average',
                'description': f'監督埋め込み（次元{i}）',
                'traceable': True,
                'traceable_to': 'movie_entities.json (directors)'
            }
            dim += 1

        # Keyword embedding (300D)
        for i in range(300):
            self.metadata[dim] = {
                'type': 'movie_keyword_emb',
                'name': f'keyword_emb_{i}',
                'group': 'movie_keyword',
                'dimension_in_group': i,
                'embedding_dim': i,
                'source': 'keywords FastText average',
                'description': f'キーワード埋め込み（次元{i}）',
                'traceable': True,
                'traceable_to': 'movie_entities.json (keywords)'
            }
            dim += 1

        # Runtime & Year (2D)
        self.metadata[dim] = {
            'type': 'movie_runtime',
            'name': 'runtime_norm',
            'group': 'movie_basic',
            'dimension_in_group': 0,
            'source': 'movies_metadata.csv (runtime)',
            'description': '上映時間（正規化）',
            'traceable': False
        }
        dim += 1

        self.metadata[dim] = {
            'type': 'movie_year',
            'name': 'year_norm',
            'group': 'movie_basic',
            'dimension_in_group': 1,
            'source': 'movies_metadata.csv (release_date)',
            'description': '公開年（正規化）',
            'traceable': False
        }
        dim += 1

        # Tags (263D)
        if 'tag_columns' in self.movie_data:
            tag_cols = self.movie_data['tag_columns']
            for i, tag in enumerate(tag_cols):
                self.metadata[dim] = {
                    'type': 'movie_tag',
                    'name': tag,
                    'group': 'movie_tags',
                    'dimension_in_group': i,
                    'source': f'movies_metadata.csv ({tag})',
                    'description': f'タグ特徴 {tag}',
                    'traceable': False,
                    'transform': 'sqrt → zscore → L2norm'
                }
                dim += 1
        else:
            # Fallback: 263 tags
            for i in range(263):
                self.metadata[dim] = {
                    'type': 'movie_tag',
                    'name': f'tag_{i}',
                    'group': 'movie_tags',
                    'dimension_in_group': i,
                    'source': 'movies_metadata.csv (A01-V52)',
                    'description': f'タグ特徴（次元{i}）',
                    'traceable': False,
                    'transform': 'sqrt → zscore → L2norm'
                }
                dim += 1

        # Review aggregation (18D)
        for i, aspect in enumerate(self.ASPECT_NAMES):
            self.metadata[dim] = {
                'type': 'movie_review_agg',
                'name': aspect,
                'group': 'movie_review_agg',
                'dimension_in_group': i,
                'source': 'aggregated from reviews (mention-weighted)',
                'description': f'{aspect}の集約評価',
                'traceable': False
            }
            dim += 1

    def _generate_review_metadata(self, offset: int):
        """レビュー特徴のメタデータ"""
        dim = offset

        # Aspect signals (18D)
        for i, aspect in enumerate(self.ASPECT_NAMES):
            self.metadata[dim] = {
                'type': 'review_aspect_signal',
                'name': aspect,
                'group': 'review_aspects',
                'dimension_in_group': i,
                'source': 'mention × (sentiment - 3.0)',
                'description': f'{aspect}のシグナル強度',
                'traceable': False
            }
            dim += 1

        # Person attention (4D) - if available
        person_attention_names = ['actor_match', 'director_match', 'actor_sentiment', 'director_sentiment']
        for i, name in enumerate(person_attention_names):
            self.metadata[dim] = {
                'type': 'review_person_attention',
                'name': name,
                'group': 'review_person',
                'dimension_in_group': i,
                'source': 'computed person attention',
                'description': f'人物注目度: {name}',
                'traceable': False
            }
            dim += 1

    def save(self, output_path: Path):
        """メタデータをJSONで保存"""
        # Convert keys to strings for JSON
        metadata_str_keys = {str(k): v for k, v in self.metadata.items()}

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_str_keys, f, indent=2, ensure_ascii=False)

        print(f"✅ Dimension metadata saved: {output_path}")
        print(f"   Total dimensions: {len(self.metadata)}")

    def get_traceable_dimensions(self) -> Dict[int, Dict[str, Any]]:
        """生データに遡及可能な次元のみを取得"""
        return {dim: meta for dim, meta in self.metadata.items() if meta.get('traceable', False)}

    def print_summary(self):
        """メタデータのサマリーを表示"""
        print("=" * 70)
        print("DIMENSION METADATA SUMMARY")
        print("=" * 70)

        type_counts = {}
        group_counts = {}
        traceable_count = 0

        for meta in self.metadata.values():
            type_counts[meta['type']] = type_counts.get(meta['type'], 0) + 1
            group_counts[meta['group']] = group_counts.get(meta['group'], 0) + 1
            if meta.get('traceable', False):
                traceable_count += 1

        print(f"\nTotal dimensions: {len(self.metadata)}")
        print(f"Traceable dimensions: {traceable_count}")

        print(f"\nDimensions by group:")
        for group, count in sorted(group_counts.items()):
            print(f"  {group:30s}: {count:4d}D")

        print("=" * 70)


if __name__ == "__main__":
    # テスト実行
    from pathlib import Path

    base_dir = Path("/Users/watanabesaki/PycharmProjects/AARAN")
    processed_dir = base_dir / "data" / "processed"

    generator = DimensionMetadataGenerator(
        movie_features_path=processed_dir / "movie_features.pt",
        user_features_path=processed_dir / "user_features.pt",
        movie_entities_path=processed_dir / "movie_entities.json"
    )

    metadata = generator.generate_all()
    generator.print_summary()

    # Save
    output_path = processed_dir / "dimension_metadata.json"
    generator.save(output_path)