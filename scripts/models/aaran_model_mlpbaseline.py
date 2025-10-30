#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AARANMLPBaseline
非線形統合版（User/Movie/Review特徴の多層結合）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AARANMLPBaseline(nn.Module):
    def __init__(self, user_dim, movie_dim, review_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        # 個別エンコーダ
        self.user_enc = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.movie_enc = nn.Sequential(
            nn.Linear(movie_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.review_enc = nn.Sequential(
            nn.Linear(review_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 統合MLP
        total_dim = hidden_dim * 2 + hidden_dim // 4  # 256 + 256 + 64 = 576
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, g):
        u = self.user_enc(g['user_features'][g['user_indices']])
        m = self.movie_enc(g['movie_features'][g['movie_indices']])
        r = self.review_enc(g['review_signals'])
        x = torch.cat([u, m, r], dim=1)
        rating = self.mlp(x).squeeze(1)
        return {'rating': rating}
