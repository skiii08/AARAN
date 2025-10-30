#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AARAN Linear Baseline (no GAT, no attention)
→ Simple MLP fusion model for PCA features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AARANLinearBaseline(nn.Module):
    def __init__(self, user_dim, movie_dim, review_dim, dropout=0.3):
        super().__init__()
        self.user_enc = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.movie_enc = nn.Sequential(
            nn.Linear(movie_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.review_enc = nn.Sequential(
            nn.Linear(review_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.predictor = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, g):
        u = self.user_enc(g['user_features'][g['user_indices']])
        m = self.movie_enc(g['movie_features'][g['movie_indices']])
        r = self.review_enc(g['review_signals'])
        x = torch.cat([u, m, r], dim=-1)
        rating = self.predictor(x).squeeze(-1)
        return {'rating': rating}
