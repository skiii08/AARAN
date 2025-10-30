import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBiGAT(nn.Module):
    """Bipartite per-edge multi-head gated interaction (GAT-inspired, no softmax over neighbors).

    For each edge (u, m):
      head_i:  a_i = (W_u^i u) ⊙ (W_m^i m)  → gate_i = σ(w_i^T a_i)
      concat over heads → linear projection to dim_out
    """

    def __init__(self, dim_u: int, dim_m: int, dim_out: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.dim_u = dim_u
        self.dim_m = dim_m
        self.dim_out = dim_out

        d_head = dim_out // heads
        self.Wu = nn.Parameter(torch.Tensor(heads, dim_u, d_head))
        self.Wm = nn.Parameter(torch.Tensor(heads, dim_m, d_head))
        self.att = nn.Parameter(torch.Tensor(heads, d_head))  # vector per head
        self.proj = nn.Linear(heads * d_head, dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.heads):
            nn.init.xavier_uniform_(self.Wu[h])
            nn.init.xavier_uniform_(self.Wm[h])
            nn.init.zeros_(self.att[h])
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B = u.size(0)
        heads_out = []
        for h in range(self.heads):
            u_h = u @ self.Wu[h]          # (B, d_h)
            m_h = m @ self.Wm[h]          # (B, d_h)
            a_h = u_h * m_h               # (B, d_h) elementwise interaction
            gate = torch.sigmoid(a_h @ self.att[h])  # (B,)
            z_h = a_h * gate.unsqueeze(-1)           # gated features
            heads_out.append(z_h)
        z = torch.cat(heads_out, dim=-1)             # (B, heads*d_h)
        z = F.elu(z)
        z = self.proj(z)                              # (B, dim_out)
        return z


class EdgeGATBlock(nn.Module):
    """
    Bipartite(User, Movie)向けの簡易GATブロック。
    エッジ(u,m)ごとに attention を計算し、融合表現 h_um を出力。
    """
    def __init__(self, user_dim: int, movie_dim: int, review_dim: int, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (hidden // heads) ** 0.5

        # 線形射影（マルチヘッド用に hidden を heads に分割）
        self.Wu = nn.Linear(user_dim, hidden, bias=False)
        self.Wm = nn.Linear(movie_dim, hidden, bias=False)
        self.Wr = nn.Linear(review_dim, hidden, bias=False)  # レビュー（22D）をバイアス的に利用

        # ヘッド毎の attention パラメータ（簡易：ドット積 + ゲート）
        self.att_gate = nn.Linear(hidden * 3, heads, bias=True)

        # エッジ出力の統合
        self.out = nn.Linear(hidden * 3, hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # 残差
        self.res_fc = nn.Linear(user_dim + movie_dim + review_dim, hidden) if (user_dim + movie_dim + review_dim) != hidden else nn.Identity()

    def forward(
        self,
        h_u: torch.Tensor,         # (B, user_dim)
        h_m: torch.Tensor,         # (B, movie_dim)
        h_r: torch.Tensor,         # (B, review_dim)
    ) -> torch.Tensor:             # (B, hidden)
        """
        B はエッジ数（= ミニバッチの辺数）
        """
        Bu = self.Wu(h_u)  # (B, H)
        Bm = self.Wm(h_m)  # (B, H)
        Br = self.Wr(h_r)  # (B, H)

        # ヘッド次元に分割
        def split_heads(x):
            B, H = x.shape
            x = x.view(B, self.heads, H // self.heads)  # (B, heads, d_k)
            return x

        U = split_heads(Bu)
        M = split_heads(Bm)
        R = split_heads(Br)

        # スケールド・ドット積（近似的な注意スコア）
        # score = <U, M> / sqrt(d) + <U, R> / sqrt(d) + <M, R> / sqrt(d)
        score = (U * M).sum(-1) / self.scale
        score = score + (U * R).sum(-1) / self.scale
        score = score + (M * R).sum(-1) / self.scale  # (B, heads)

        # シグモイドでゲート化（0〜1）
        gate = torch.sigmoid(score)  # (B, heads)

        # 3者結合した特徴をヘッド毎に線形でゲート
        cat_full = torch.cat([Bu, Bm, Br], dim=-1)           # (B, 3H)
        head_logits = self.att_gate(cat_full)                # (B, heads)
        head_gate = torch.sigmoid(head_logits) * gate        # (B, heads)

        # ヘッドを結合した表現（単純に concat ではなく、cat_full をゲートで縮約）
        # （近似として、各ヘッドの重み平均を取り、cat_full をスケール）
        alpha = head_gate.mean(dim=1, keepdim=True)          # (B, 1)
        fused = self.out(self.dropout(torch.cat([Bu, Bm, Br], dim=-1)))  # (B, H)
        fused = fused * alpha

        # 残差
        res = self.res_fc(torch.cat([h_u, h_m, h_r], dim=-1))  # (B, H) or Identity
        out = self.act(fused + res)
        return out
