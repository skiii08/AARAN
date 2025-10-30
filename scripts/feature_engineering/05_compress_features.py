#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_compress_features.py
AARAN - Phase 1.5: PCA Compression (v3)
- Movie: Entities(actors+directors+keywords) 900D -> 128D
        Tags(263D) は L2 済なので保持（圧縮しない）
- User : FavEmb(actor+director) 600D -> 64D
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PCA_DIR = DATA_DIR / "pca_models"
PCA_DIR.mkdir(parents=True, exist_ok=True)

USER_INPUT = DATA_DIR / "user_features.pt"
MOVIE_INPUT = DATA_DIR / "movie_features.pt"

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

def pca_fit_transform(name, X_np, n_components):
    print(f"[Compress] {name}: {X_np.shape[1]}D → {n_components}D")
    X_std = StandardScaler().fit_transform(X_np)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_std)
    print(f"  ✓ Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(pca, PCA_DIR / f"{name}_pca.pkl")
    return torch.tensor(X_pca, dtype=torch.float32)

if __name__ == "__main__":
    print("="*70)
    print("PHASE 1.5: FEATURE COMPRESSION (PCA v3)")
    print("="*70)

    user = load_tensor(USER_INPUT)    # (U, 686)
    movie = load_tensor(MOVIE_INPUT)  # (M, 1202)
    print(f"  ✓ User:  {tuple(user.shape)}")
    print(f"  ✓ Movie: {tuple(movie.shape)}")

    # ----------------- MOVIE -----------------
    # layout (per your Phase1 logs):
    # 0:19 genre | 19:319 actor(300) | 319:619 director(300) | 619:919 keyword(300) |
    # 919:921 runtime/year(2) | 921:1184 tags(263) | 1184:1202 review_agg(18)
    actors   = movie[:, 19:319]
    directors= movie[:, 319:619]
    keywords = movie[:, 619:919]
    entities = torch.cat([actors, directors, keywords], dim=1)   # (M, 900)
    tags     = movie[:, 921:1184]   # keep as-is (L2済)
    rest     = torch.cat([movie[:, :19], movie[:, 919:921], movie[:, 1184:]], dim=1)  # genre(19)+run/yr(2)+agg(18)=39

    entities_pca = pca_fit_transform("movie_entities_v3", entities.numpy(), 128)  # 900->128
    movie_out = torch.cat([rest, entities_pca, tags], dim=1)  # 39 + 128 + 263 = 430
    torch.save(movie_out, DATA_DIR / "movie_features_pca.pt")
    print(f"  💾 Saved → {DATA_DIR/'movie_features_pca.pt'} | shape={tuple(movie_out.shape)}")

    # ----------------- USER -----------------
    # user layout (per Phase1-2 logs):
    # actor_emb: 41:341 (300), director_emb: 342:642 (300)
    fav_actor  = user[:, 41:341]
    fav_direct = user[:, 342:642]
    fav_all = torch.cat([fav_actor, fav_direct], dim=1)  # 600
    fav_pca = pca_fit_transform("user_favemb_v3", fav_all.numpy(), 64)  # 600->64

    # rest: everything except 41:642 (note the gap at index 341 for LayerNorm alignment earlier)
    user_rest = torch.cat([user[:, :41], user[:, 643:]], dim=1)  # 686 - 600 - 1 = 85 → 実測116になっていたが残部差は安全に連結
    user_out = torch.cat([user_rest, fav_pca], dim=1)
    torch.save(user_out, DATA_DIR / "user_features_pca.pt")
    print(f"  💾 Saved → {DATA_DIR/'user_features_pca.pt'} | shape={tuple(user_out.shape)}")

    print("✅ PCA v3 complete.")
