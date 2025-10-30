import torch
from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

# =========================================================
# Load Graph Blobs
# =========================================================
def load_graph_blobs(data_dir):
    """
    Load graph blobs (dict形式) safely, allowing pickle ops.
    """
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    full_dir = Path(data_dir)
    train = torch.load(full_dir / "hetero_graph_train.pt", weights_only=False)
    val = torch.load(full_dir / "hetero_graph_val.pt", weights_only=False)
    test = torch.load(full_dir / "hetero_graph_test.pt", weights_only=False)

    user_mu_map = train.get("user_mu_map", {})
    user_std_map = train.get("user_std_map", {})

    return train, val, test, user_mu_map, user_std_map


# =========================================================
# Dataloader Utilities (dict構造対応)
# =========================================================
def make_dataloaders(train, val, test, batch_size=512):
    """dict形式 (edge-level構造) に対応したDataLoader"""

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x

    def make_loader(blob, shuffle=False):
        # --- ノード特徴 ---
        user_all = to_tensor(blob["user_features"])
        movie_all = to_tensor(blob["movie_features"])

        # --- エッジ情報 ---
        review_x = to_tensor(blob["review_signals"])
        y = to_tensor(blob.get("ratings_user_norm", blob.get("ratings")))
        u_idx = blob["user_indices"].long()
        m_idx = blob["movie_indices"].long()

        # --- 各edgeに対応する特徴をgather ---
        user_x = user_all[u_idx]
        movie_x = movie_all[m_idx]

        assert (
            user_x.shape[0] == movie_x.shape[0] == review_x.shape[0] == y.shape[0]
        ), f"Shape mismatch: {user_x.shape}, {movie_x.shape}, {review_x.shape}, {y.shape}"

        dataset = TensorDataset(user_x, movie_x, review_x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return (
        make_loader(train, shuffle=True),
        make_loader(val, shuffle=False),
        make_loader(test, shuffle=False),
    )


# =========================================================
# Utility: Correlation
# =========================================================
def spearman_corr(x, y):
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))
    return torch.corrcoef(torch.stack([rx.float(), ry.float()]))[0, 1]


def build_user_stats(user_indices: torch.Tensor, ratings_raw: torch.Tensor) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute per-user mean/std from TRAIN split for denorm.
    user_indices: (N,) contiguous user indices used in train graph
    ratings_raw:  (N,) raw ratings in [1,10]
    """
    mu_map, std_map = {}, {}
    user_indices = user_indices.cpu().numpy()
    ratings = ratings_raw.cpu().numpy()
    for u in np.unique(user_indices):
        mask = (user_indices == u)
        r = ratings[mask]
        mu_map[int(u)] = float(r.mean())
        std_map[int(u)] = float(r.std() if r.std() > 1e-6 else 1.0)
    # global fallback
    g_mu = float(ratings.mean())
    g_std = float(ratings.std() if ratings.std() > 1e-6 else 1.0)
    mu_map[-1] = g_mu
    std_map[-1] = g_std
    return mu_map, std_map


def denorm_by_user(pred_norm: torch.Tensor, user_idx: torch.Tensor, mu_map: Dict[int, float], std_map: Dict[int, float]) -> torch.Tensor:
    mu = torch.tensor([mu_map.get(int(u), mu_map[-1]) for u in user_idx.cpu().tolist()], dtype=torch.float32, device=pred_norm.device)
    sd = torch.tensor([std_map.get(int(u), std_map[-1]) for u in user_idx.cpu().tolist()], dtype=torch.float32, device=pred_norm.device)
    out = pred_norm * sd + mu
    return out.clamp(1.0, 10.0)


def spearmanr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    # Per-batch Spearman (global). For robust eval, compute per-user later.
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))
    rx = rx.float()
    ry = ry.float()
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = (vx.pow(2).sum() * vy.pow(2).sum()).sqrt() + 1e-8
    return float((vx * vy).sum() / denom)


def mae_rmse(pred: torch.Tensor, true: torch.Tensor) -> Tuple[float, float]:
    err = pred - true
    mae = float(err.abs().mean())
    rmse = float((err.pow(2).mean()).sqrt())
    return mae, rmse


def extract_aspect_signals(blob):
    """review_signals から 18D のアスペクト信号を抽出（簡易平均など）
       今後: mention/sentiment融合に置き換え予定
    """
    review_x = blob["review_signals"]
    if review_x.shape[1] >= 36:
        # 先頭18=mention, 後半18=sentiment → 平均
        aspect = (review_x[:, :18] + review_x[:, 18:36]) / 2
    else:
        aspect = review_x[:, :18]
    return aspect
