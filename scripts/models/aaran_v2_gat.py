# -*- coding: utf-8 -*-
"""
aaran_v2_gat.py
AARAN v2: v1-min に EdgeGATBlock を導入。
- Encoder: User(686)->256, Movie(1202)->256, Review(22)->64
- Bi-Edge GAT 融合 (heads=4, hidden=256)
- 最終 MLP で user-norm rating を予測
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# 相対 → 絶対 import に変更（パッケージ文脈がなくてもOK）
from scripts.models.gat_layers import EdgeGATBlock


@dataclass
class AARANV2Config:
    user_dim: int = 686
    movie_dim: int = 1202
    review_dim: int = 22

    user_hidden: int = 256
    movie_hidden: int = 256
    review_hidden: int = 64

    gat_hidden: int = 256
    gat_heads: int = 4

    mlp_hidden: int = 128
    dropout: float = 0.2


class AARANV2GAT(nn.Module):
    def __init__(self, cfg: AARANV2Config = AARANV2Config()):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.user_enc = nn.Sequential(
            nn.Linear(cfg.user_dim, cfg.user_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.movie_enc = nn.Sequential(
            nn.Linear(cfg.movie_dim, cfg.movie_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.review_enc = nn.Sequential(
            nn.Linear(cfg.review_dim, cfg.review_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Edge-level Bi-GAT-like fusion
        self.edge_gat = EdgeGATBlock(
            user_dim=cfg.user_hidden,
            movie_dim=cfg.movie_hidden,
            review_dim=cfg.review_hidden,
            hidden=cfg.gat_hidden,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )

        # Prediction head（edge表現 → user-norm rating）
        self.pred = nn.Sequential(
            nn.Linear(cfg.gat_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(self, u_x: torch.Tensor, m_x: torch.Tensor, r_x: torch.Tensor) -> torch.Tensor:
        """
        u_x: (B, 686)
        m_x: (B, 1202)
        r_x: (B, 22)
        return: y_hat (B,)  # user-normalized scale
        """
        hu = self.user_enc(u_x)
        hm = self.movie_enc(m_x)
        hr = self.review_enc(r_x)

        h_edge = self.edge_gat(hu, hm, hr)  # (B, 256)
        y_hat = self.pred(h_edge).squeeze(-1)  # (B,)
        return y_hat


# ===== 互換エイリアス（import 名のブレ対策） =====
AARANv2GAT = AARANV2GAT  # ← これで `from scripts.models.aaran_v2_gat import AARANv2GAT` が通る

__all__ = ["AARANV2Config", "AARANV2GAT", "AARANv2GAT"]
