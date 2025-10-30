# -*- coding: utf-8 -*-
"""
AARAN v3 Full Model
Phase 2-3: Full Fusion (GAT × Aspect × MHA × Gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_layers import EdgeGATBlock
from .attention_modules import AspectAttention


class AARANFusionBlock(nn.Module):
    """GAT × Aspect × MHA × Gate 融合ブロック"""
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)
        # gateはconcat(3H)に合わせて3H次元を出力
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 3),
            nn.ReLU(),
            nn.Linear(hidden * 3, hidden * 3),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(hidden * 3, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_gat, h_aspect):
        src = (h_gat + h_aspect).unsqueeze(1)  # (B,1,H)
        h_mha, _ = self.mha(src, src, src)
        h_mha = h_mha.squeeze(1)

        concat = torch.cat([h_gat, h_aspect, h_mha], dim=-1)  # (B, 3H)
        gate = self.gate(concat)                              # (B, 3H)
        fused = self.proj(concat * gate)                      # (B, H)
        out = self.norm(h_gat + self.dropout(fused))
        return out



class AARANv3FullModel(nn.Module):
    """AARAN v3: Encoder + Core(GAT+Aspect) + Fusion(MHA+Gate) + Pred Head"""
    def __init__(self, user_dim=686, movie_dim=1202, review_dim=22,
                 hidden=256, heads=4, dropout=0.2):
        super().__init__()

        # --- Encoder ---
        self.user_enc = nn.Sequential(nn.Linear(user_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.movie_enc = nn.Sequential(nn.Linear(movie_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.review_enc = nn.Sequential(nn.Linear(review_dim, hidden), nn.ReLU(), nn.Dropout(dropout))

        # --- Core ---
        self.edge_gat = EdgeGATBlock(
            user_dim=hidden,
            movie_dim=hidden,
            review_dim=hidden,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
        )
        self.aspect_att = AspectAttention(query_dim=hidden, key_dim=hidden, num_aspects=18)

        # --- Fusion ---
        self.fusion = AARANFusionBlock(hidden=hidden, heads=heads, dropout=dropout)

        # --- Prediction Head ---
        self.pred_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, u_x, m_x, r_x, aspect_signals):
        """
        u_x: (B,686)
        m_x: (B,1202)
        r_x: (B,22)
        aspect_signals: (B,18)
        """
        hu = self.user_enc(u_x)
        hm = self.movie_enc(m_x)
        hr = self.review_enc(r_x)

        h_gat = self.edge_gat(hu, hm, hr)
        ctx, _ = self.aspect_att(h_gat, hr, aspect_signals)
        h_aspect = h_gat + ctx
        h_fused = self.fusion(h_gat, h_aspect)

        y_hat = self.pred_head(h_fused)
        return y_hat.squeeze(-1)
