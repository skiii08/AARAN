"""
Feature Groups Definition
特徴量のグループ分けと正則化戦略の定義
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class FeatureGroup:
    """特徴グループの定義"""
    name: str
    start_idx: int
    end_idx: int
    regularization: str  # 'none', 'l1', 'l2', 'elastic'
    description: str

    def size(self) -> int:
        return self.end_idx - self.start_idx

    def indices(self) -> np.ndarray:
        return np.arange(self.start_idx, self.end_idx)


class FeatureGroupManager:
    """特徴量グループの管理クラス"""

    def __init__(self, user_dim: int = 686, movie_dim: int = 1202, review_dim: int = 22):
        self.user_dim = user_dim
        self.movie_dim = movie_dim
        self.review_dim = review_dim
        self.total_dim = user_dim + movie_dim + review_dim

        self.groups = self._define_groups()

    def _define_groups(self) -> Dict[str, FeatureGroup]:
        """特徴グループの定義"""

        offset_user = 0
        offset_movie = self.user_dim
        offset_review = self.user_dim + self.movie_dim

        groups = {}

        # ===== User Features (低相関対策: L1を積極適用し、説明の安定化を図る) =====
        groups['user_aspect_zscore'] = FeatureGroup(
            name='user_aspect_zscore',
            start_idx=offset_user + 0,
            end_idx=offset_user + 18,
            regularization='l2',  # アスペクト嗜好はL2で安定化
            description='ユーザーのアスペクト重視度（Z-score）'
        )

        groups['user_aspect_sentiment'] = FeatureGroup(
            name='user_aspect_sentiment',
            start_idx=offset_user + 18,
            end_idx=offset_user + 36,
            regularization='l2',  # アスペクト評価はL2で安定化
            description='ユーザーのアスペクト評価傾向'
        )

        groups['user_stats'] = FeatureGroup(
            name='user_stats',
            start_idx=offset_user + 36,
            end_idx=offset_user + 41,
            regularization='l2',
            description='ユーザーのレビュー統計'
        )

        # 変更: L2 -> L1 (不要な埋め込み次元を強制的にゼロ化)
        groups['user_fav_actor'] = FeatureGroup(
            name='user_fav_actor',
            start_idx=offset_user + 41,
            end_idx=offset_user + 341,
            regularization='l1',
            description='ユーザーのお気に入り俳優埋め込み'
        )

        # 変更: L2 -> L1 (不要な埋め込み次元を強制的にゼロ化)
        groups['user_fav_director'] = FeatureGroup(
            name='user_fav_director',
            start_idx=offset_user + 342,
            end_idx=offset_user + 642,
            regularization='l1',
            description='ユーザーのお気に入り監督埋め込み'
        )

        groups['user_genre'] = FeatureGroup(
            name='user_genre',
            start_idx=offset_user + 643,
            end_idx=offset_user + 681,
            regularization='elastic',  # 変更なし: Elastic Netを維持
            description='ユーザーのジャンル嗜好'
        )

        # 変更: L2 -> L1 (行動特徴はノイズが多く相関が低いためL1で厳選)
        groups['user_behavior'] = FeatureGroup(
            name='user_behavior',
            start_idx=offset_user + 681,
            end_idx=offset_user + 686,
            regularization='l1',
            description='ユーザーの行動特徴'
        )

        # ===== Movie Features (重要だがノイズが多い特徴にL1/Elasticを適用) =====
        groups['movie_genre'] = FeatureGroup(
            name='movie_genre',
            start_idx=offset_movie + 0,
            end_idx=offset_movie + 19,
            regularization='none',
            description='映画のジャンル（multi-hot）'
        )

        groups['movie_actor'] = FeatureGroup(
            name='movie_actor',
            start_idx=offset_movie + 19,
            end_idx=offset_movie + 319,
            regularization='l2',
            description='映画の俳優埋め込み'
        )

        groups['movie_director'] = FeatureGroup(
            name='movie_director',
            start_idx=offset_movie + 319,
            end_idx=offset_movie + 619,
            regularization='l2',
            description='映画の監督埋め込み'
        )

        groups['movie_keyword'] = FeatureGroup(
            name='movie_keyword',
            start_idx=offset_movie + 619,
            end_idx=offset_movie + 919,
            regularization='l2',
            description='映画のキーワード埋め込み'
        )

        groups['movie_basic'] = FeatureGroup(
            name='movie_basic',
            start_idx=offset_movie + 919,
            end_idx=offset_movie + 921,
            regularization='none',
            description='映画の基本情報（runtime, year）'
        )

        # 変更: L1 -> Elastic Net (L1を維持しつつ、ノイズ安定化のL2要素を追加)
        groups['movie_tags'] = FeatureGroup(
            name='movie_tags',
            start_idx=offset_movie + 921,
            end_idx=offset_movie + 1184,
            regularization='elastic',
            description='映画のタグ特徴（A01-V52）'
        )

        groups['movie_review_agg'] = FeatureGroup(
            name='movie_review_agg',
            start_idx=offset_movie + 1184,
            end_idx=offset_movie + 1202,
            regularization='l2', # none -> L2に変更し、重みの安定化を図る
            description='映画のレビュー集約（アスペクト別）'
        )

        # ===== Review Features (高相関だが低スケール対策: L2で重み安定化) =====
        # 変更: none -> L2 (相関が最も高いため、L2で重みを安定させ、過剰適合を防ぐ)
        groups['review_aspects'] = FeatureGroup(
            name='review_aspects',
            start_idx=offset_review + 0,
            end_idx=offset_review + 18,
            regularization='l2', # <-- 過学習対策としてL2を適用
            description='レビューのアスペクトシグナル'
        )

        groups['review_person'] = FeatureGroup(
            name='review_person',
            start_idx=offset_review + 18,
            end_idx=offset_review + 22,
            regularization='l2',
            description='レビューの人物注目度'
        )

        return groups


    def get_group(self, name: str) -> FeatureGroup:
        """グループ名からFeatureGroupを取得"""
        return self.groups[name]

    def get_regularization_groups(self) -> Dict[str, List[FeatureGroup]]:
        """正則化タイプ別にグループをまとめる"""
        reg_groups = {
            'none': [],
            'l1': [],
            'l2': [],
            'elastic': []
        }

        for group in self.groups.values():
            reg_groups[group.regularization].append(group)

        return reg_groups

    def get_all_indices_by_regularization(self, reg_type: str) -> np.ndarray:
        """指定した正則化タイプの全インデックスを取得"""
        indices = []
        for group in self.groups.values():
            if group.regularization == reg_type:
                indices.extend(group.indices())
        return np.array(indices)

    def print_summary(self):
        """特徴グループのサマリーを表示"""
        print("=" * 70)
        print("FEATURE GROUPS SUMMARY")
        print("=" * 70)

        reg_groups = self.get_regularization_groups()

        for reg_type in ['none', 'l1', 'l2', 'elastic']:
            groups = reg_groups[reg_type]
            if not groups:
                continue

            print(f"\n[{reg_type.upper()} REGULARIZATION]")
            total_dims = sum(g.size() for g in groups)
            print(f"Total dimensions: {total_dims}")

            for group in groups:
                print(f"  - {group.name:25s}: [{group.start_idx:4d}-{group.end_idx:4d}] "
                      f"({group.size():3d}D) {group.description}")

        print(f"\n{'=' * 70}")
        print(f"TOTAL DIMENSIONS: {self.total_dim}")
        print("=" * 70)


def create_feature_group_manager(graph_dict: dict) -> FeatureGroupManager:
    """グラフ辞書から特徴グループマネージャーを作成"""
    user_dim = graph_dict['user_features'].shape[1]
    movie_dim = graph_dict['movie_features'].shape[1]
    review_dim = graph_dict['review_signals'].shape[1]

    return FeatureGroupManager(user_dim, movie_dim, review_dim)


if __name__ == "__main__":
    # テスト
    manager = FeatureGroupManager()

    # 既存のサマリーを表示
    manager.print_summary()

    # --- NEW: 正則化戦略のデバッグログ（ご要望の全グループ確認） ---
    print("\n" + "=" * 70)
    print("STRATEGY CHECK: GROUP REGULARIZATION STATUS")
    print("=" * 70)

    header = f"{'Group Name':30s} | {'Defined Reg':12s} | {'Dimensions':10s} | {'Start Index':10s}"
    print(header)
    print("-" * len(header))

    total_dims = 0
    none_dims = 0

    for name, group in manager.groups.items():
        print(f"{group.name:30s} | {group.regularization:12s} | {group.size():<10d} | {group.start_idx:<10d}")
        total_dims += group.size()
        if group.regularization == 'none':
            none_dims += group.size()

    print("-" * len(header))
    print(f"{'TOTAL':30s} | {'-':12s} | {total_dims:<10d} | {'-':10s}")

    # 全体サマリー
    print("\n" + "=" * 70)
    print("REGULARIZATION COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Total Features:      {manager.total_dim:4d} D")
    print(f"Total Regularized:   {manager.total_dim - none_dims:4d} D ({(manager.total_dim - none_dims) / manager.total_dim * 100:.1f}%)")
    print(f"Total NONE Reg:      {none_dims:4d} D ({none_dims / manager.total_dim * 100:.1f}%)")

    if none_dims > 0:
        print("\n❗ NOTE: 'NONE' Reg Groups (意図的に正則化を外したグループ):")
        for group in manager.groups.values():
            if group.regularization == 'none':
                 print(f"  - {group.name} ({group.size()}D): {group.description}")
    else:
        print("\n✅ GREAT: 全てのグループに何らかの正則化が定義されています。")

    print("=" * 70)