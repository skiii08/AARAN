import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectAttention(nn.Module):
    """Attention over 18 aspect signals using the review embedding as KV and query from edge features.

    * aspect_signals: (B, 18) used to gate keys
    * review_emb: (B, d) provides K/V
    * query: (B, q) attends to gated K to produce context of size d
    """

    def __init__(self, query_dim: int, key_dim: int, num_aspects: int = 18):
        super().__init__()
        self.num_aspects = num_aspects
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.val_proj = nn.Linear(key_dim, key_dim)
        self.out_ln = nn.LayerNorm(key_dim)

    def forward(self, query: torch.Tensor, review_emb: torch.Tensor, aspect_signals: torch.Tensor):
        # Build pseudo-keys for each aspect by gating the same review_emb
        # K_i = tanh(Wk r) * g_i, where g_i = sigmoid(signal_i)
        B, d = review_emb.shape
        r_k = torch.tanh(self.key_proj(review_emb))            # (B, d)
        r_v = self.val_proj(review_emb)                        # (B, d)
        gates = torch.sigmoid(aspect_signals).unsqueeze(-1)    # (B, 18, 1)
        K = (r_k.unsqueeze(1) * gates).contiguous()            # (B, 18, d)
        V = (r_v.unsqueeze(1) * gates).contiguous()            # (B, 18, d)

        q = self.query_proj(query).unsqueeze(1)                # (B, 1, d)
        att = torch.softmax((q * K).sum(-1) / (d ** 0.5), dim=-1)  # (B, 18)
        ctx = (att.unsqueeze(-1) * V).sum(1)                   # (B, d)
        ctx = self.out_ln(ctx)
        return ctx, att
