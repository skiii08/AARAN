#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_graph_pca.py
Build PCA-based graphs for train/val/test (reuse edges/ratings/review_signals)
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
USER_PCA  = DATA_DIR / "user_features_pca.pt"
MOVIE_PCA = DATA_DIR / "movie_features_pca.pt"

def load_tensor(path):
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ["features","data","tensor","X"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                print(f"  ⚙ Loaded '{path.name}' via key='{k}'")
                return obj[k]
    raise TypeError(f"Unsupported type for {path}: {type(obj)}")

def build(split, user_t, movie_t):
    base = torch.load(DATA_DIR / f"hetero_graph_{split}.pt", weights_only=False)
    newg = {
        "user_features": user_t,
        "movie_features": movie_t,
        "review_signals": base["review_signals"],   # (E,22)
        "user_indices": base["user_indices"],
        "movie_indices": base["movie_indices"],
        "ratings": base["ratings"],
        "num_users": user_t.shape[0],
        "num_movies": movie_t.shape[0],
        "num_edges": len(base["ratings"]),
    }
    torch.save(newg, DATA_DIR / f"hetero_graph_{split}_pca.pt")
    print(f"  💾 hetero_graph_{split}_pca.pt  edges={newg['num_edges']}")

if __name__ == "__main__":
    print("="*70)
    print("PHASE 1.6: GRAPH CONSTRUCTION (PCA v3)")
    print("="*70)
    user_t  = load_tensor(USER_PCA)
    movie_t = load_tensor(MOVIE_PCA)
    print(f"User: {tuple(user_t.shape)} | Movie: {tuple(movie_t.shape)}")
    for sp in ["train","val","test"]:
        build(sp, user_t, movie_t)
    print("✅ Done.")
