#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 4: EXPLANATION GENERATION (two-layer + weak-group picks)

目的:
- 線形モデルの寄与度を「特徴 → グループ」の二層で集約
- 俳優・キーワードなど“弱い”グループからも各Top3を強制的に抽出
- 予測値は user-wise μ/σ で逆正規化して [1,10] にクリップ（Phase03と同じ思想）

前提:
- best_model.pkl は pickle(dict) を想定: {"model": sklearn_like, "user_stats": {"mu_map":..., "std_map":...}}
- 次元メタは data/processed/dimension_metadata.json （各dimに "group","name","type","description"）
- グラフ: data/processed/hetero_graph_{split}.pt
"""

import sys, json, pickle, argparse, random
from pathlib import Path
import numpy as np
import torch

# ========== Paths ==========
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
OUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# “弱い”グループ（必ずTop3を抜く対象）
WEAK_GROUP_CANDIDATES = [
    "movie_actor", "movie_keyword",
    # 予備の別名（メタデータのgroup名に合わせて自動でヒットさせる）
    "actor", "keyword", "movie_actors", "movie_keywords"
]

# ========== I/O helpers ==========
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_bundle(path: Path):
    obj = load_pickle(path)
    if isinstance(obj, dict) and "model" in obj:
        bundle = obj
    else:
        bundle = {"model": obj}
    bundle.setdefault("user_stats", {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}})
    return bundle

def load_dimension_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_graph_data(split: str, data_dir: Path):
    p = data_dir / f"hetero_graph_{split}.pt"
    print(f"[Graph] Loading {p} ...")
    return torch.load(p, weights_only=False)

# ========== Core ops ==========
def find_edge_index_by_user_movie(user_name: str, movie_id: int, graph):
    user_names = list(graph.get("user_names", []))
    movie_ids = list(map(int, graph.get("movie_ids", [])))
    try:
        u_idx_target = user_names.index(user_name)
    except ValueError:
        return None, f"User '{user_name}' not found"
    try:
        m_idx_target = movie_ids.index(int(movie_id))
    except ValueError:
        return None, f"Movie ID {movie_id} not found"

    u_arr = graph["user_indices"].cpu().numpy()
    m_arr = graph["movie_indices"].cpu().numpy()
    for e in range(len(u_arr)):
        if u_arr[e] == u_idx_target and m_arr[e] == m_idx_target:
            return e, None
    return None, f"No review found for user '{user_name}' and movie {movie_id}"

def extract_edge_features(edge_idx: int, graph):
    u_idx = int(graph["user_indices"][edge_idx].item())
    m_idx = int(graph["movie_indices"][edge_idx].item())
    u_feat = graph["user_features"][u_idx].cpu().numpy()
    m_feat = graph["movie_features"][m_idx].cpu().numpy()
    r_feat = graph["review_signals"][edge_idx].cpu().numpy()
    x = np.concatenate([u_feat, m_feat, r_feat], axis=0)
    return x, u_idx, m_idx

def model_predict_norm(model, x: np.ndarray) -> float:
    y = model.predict(x.reshape(1, -1))[0]
    return float(y)

def denormalize_userwise(y_norm: float, u_idx: int, user_stats: dict) -> float:
    mu_map = user_stats.get("mu_map", {})
    std_map = user_stats.get("std_map", {})
    mu = mu_map.get(int(u_idx), mu_map.get(-1, 0.0))
    sd = std_map.get(int(u_idx), std_map.get(-1, 1.0))
    y_raw = y_norm * sd + mu
    return float(np.clip(y_raw, 1.0, 10.0))

def compute_contributions_linear(model, x: np.ndarray):
    w = np.asarray(getattr(model, "coef_", None))
    if w is None:
        raise RuntimeError("model.coef_ が見つかりません（sklearn-like の線形モデルを想定）")
    if w.ndim > 1: w = w.reshape(-1)
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Dimension mismatch: coef={w.shape[0]}, x={x.shape[0]}")
    return w * x  # signed contribution per-dimension

def top_features(contrib: np.ndarray, dim_meta: dict, top_k: int = 20):
    idx = np.argsort(np.abs(contrib))[::-1]
    idx = [i for i in idx if abs(contrib[i]) >= 1e-6][:top_k]
    out = []
    for i in idx:
        m = dim_meta.get(str(i), {})
        out.append({
            "dim": int(i),
            "contribution": float(contrib[i]),
            "type": m.get("type", "unknown"),
            "name": m.get("name", f"Dim{i}"),
            "group": m.get("group", "unknown"),
            "description": m.get("description", "")
        })
    return out

def summarize_by_group(contrib: np.ndarray, dim_meta: dict):
    """二層化: グループ別の合計寄与 |合計| とサンプル"""
    group_abs = {}
    group_signed = {}
    group_samples = {}

    for i in range(contrib.shape[0]):
        g = dim_meta.get(str(i), {}).get("group", "unknown")
        v = float(contrib[i])
        group_abs[g] = group_abs.get(g, 0.0) + abs(v)
        group_signed[g] = group_signed.get(g, 0.0) + v
        if g not in group_samples: group_samples[g] = []
        group_samples[g].append((i, v))

    # 各グループのTop1特徴（|寄与|最大）
    group_top1 = {}
    for g, arr in group_samples.items():
        j, v = max(arr, key=lambda t: abs(t[1]))
        group_top1[g] = (j, v)

    summary = []
    for g in sorted(group_abs.keys(), key=lambda k: group_abs[k], reverse=True):
        j, v = group_top1[g]
        m = dim_meta.get(str(j), {})
        summary.append({
            "group": g,
            "abs_sum": round(group_abs[g], 6),
            "signed_sum": round(group_signed[g], 6),
            "top_feature": {
                "dim": int(j),
                "name": m.get("name", f"Dim{j}"),
                "abs_contribution": round(abs(v), 6),
                "signed_contribution": round(v, 6),
            }
        })
    return summary, group_samples

def pick_topk_from_group(contrib: np.ndarray, dim_meta: dict, group: str, k: int = 3, exclude_dims=set()):
    idxs = [i for i in range(contrib.shape[0]) if dim_meta.get(str(i), {}).get("group", "unknown") == group]
    idxs = [i for i in idxs if i not in exclude_dims and abs(contrib[i]) >= 1e-6]
    idxs.sort(key=lambda i: abs(contrib[i]), reverse=True)
    picks = []
    for i in idxs[:k]:
        m = dim_meta.get(str(i), {})
        picks.append({
            "dim": int(i),
            "contribution": float(contrib[i]),
            "type": m.get("type", "unknown"),
            "name": m.get("name", f"Dim{i}"),
            "group": m.get("group", "unknown"),
            "description": m.get("description", "")
        })
    return picks

def resolve_existing_groups(dim_meta: dict):
    all_groups = {v.get("group", "unknown") for v in dim_meta.values()}
    # WEAK_GROUP_CANDIDATES のうち一致しているラベルのみ残す
    resolved = []
    for g in WEAK_GROUP_CANDIDATES:
        if g in all_groups:
            resolved.append(g)
    return list(dict.fromkeys(resolved))  # uniq

def build_explanation(pred_raw: float, contrib: np.ndarray, dim_meta: dict,
                      top_k: int, graph, u_idx: int, m_idx: int,
                      user_name: str, movie_id: int):
    # タイトル
    if "movie_titles" in graph:
        movie_title = str(graph["movie_titles"][m_idx])
    else:
        movie_title = str(movie_id)

    # 1) 特徴レベルTopK
    top_all = top_features(contrib, dim_meta, top_k=top_k)
    top_dims_set = {c["dim"] for c in top_all}

    # 2) グループ集約（二層目の概観）
    group_summary, _ = summarize_by_group(contrib, dim_meta)

    # 3) “弱い”グループから各Top3を強制抽出（TopKに未含有でも拾う）
    weak_groups = resolve_existing_groups(dim_meta)
    weak_picks = {}
    for g in weak_groups:
        picks = pick_topk_from_group(contrib, dim_meta, g, k=3, exclude_dims=top_dims_set)
        if picks:
            weak_picks[g] = picks

    pos_sum = float(np.sum(contrib[contrib > 0])) if contrib.size else 0.0
    neg_sum = float(np.sum(contrib[contrib < 0])) if contrib.size else 0.0
    total = float(np.sum(contrib)) if contrib.size else 0.0

    return {
        "prediction": float(pred_raw),
        "user_name": user_name,
        "movie_id": int(movie_id),
        "movie_title": movie_title,
        "top_contributors": top_all,            # 特徴TopK（一次層）
        "weak_group_top3": weak_picks,          # 各“弱い”グループのTop3
        "group_summary": group_summary,         # グループ概観（二次層）
        "summary": {
            "total_contribution": total,
            "positive_contribution": pos_sum,
            "negative_contribution": neg_sum
        }
    }

def format_cli_report(exp: dict):
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLANATION REPORT (Two-layer)")
    lines.append("=" * 70)
    lines.append(f"User : {exp['user_name']}")
    lines.append(f"Movie: {exp['movie_title']} (ID: {exp['movie_id']})")
    lines.append(f"Pred : {exp['prediction']:.2f}")
    lines.append("\n[Top Features]")
    for i, c in enumerate(exp["top_contributors"], 1):
        sign = "+" if c["contribution"] > 0 else ""
        lines.append(f" {i:02d}. {c['name']} [{c['group']}] {sign}{c['contribution']:.4f}")
    if exp["weak_group_top3"]:
        lines.append("\n[Weak-Group Picks]  (forced Top3 per group)")
        for g, arr in exp["weak_group_top3"].items():
            lines.append(f"  - {g}:")
            for c in arr:
                sign = "+" if c["contribution"] > 0 else ""
                lines.append(f"     • {c['name']} {sign}{c['contribution']:.4f}")
    lines.append("\n[Group Summary |abs| sum, signed sum, top feature]")
    for g in exp["group_summary"]:
        tf = g["top_feature"]
        lines.append(f"  {g['group']:<18} |abs|={g['abs_sum']:.4f}  signed={g['signed_sum']:.4f}  "
                     f"top={tf['name']} (|{tf['abs_contribution']:.4f}|)")
    lines.append("\nΣ Total:    {:.4f}".format(exp["summary"]["total_contribution"]))
    lines.append("Σ Positive: {:.4f}".format(exp["summary"]["positive_contribution"]))
    lines.append("Σ Negative: {:.4f}".format(exp["summary"]["negative_contribution"]))
    lines.append("=" * 70)
    return "\n".join(lines)

# ========== Random batch ==========
def generate_random_explanations(model, user_stats, dim_meta, graph, out_dir: Path,
                                 num_samples: int = 20, top_k: int = 20):
    user_names = list(graph.get("user_names", []))
    movie_ids = list(map(int, graph.get("movie_ids", [])))
    movie_titles = list(graph.get("movie_titles", [])) if "movie_titles" in graph else None
    u_idx_arr = graph["user_indices"].cpu().numpy()
    m_idx_arr = graph["movie_indices"].cpu().numpy()

    all_edges = list(range(len(u_idx_arr)))
    random.shuffle(all_edges)
    sampled = all_edges[:num_samples]

    out_dir.mkdir(parents=True, exist_ok=True)

    for e in sampled:
        u_idx = int(u_idx_arr[e]); m_idx = int(m_idx_arr[e])
        user_name = str(user_names[u_idx])
        movie_id = int(movie_ids[m_idx])

        x, _, _ = extract_edge_features(e, graph)
        y_norm = model_predict_norm(model, x)
        y_raw = denormalize_userwise(y_norm, u_idx, user_stats)

        contrib = compute_contributions_linear(model, x)
        exp = build_explanation(y_raw, contrib, dim_meta, top_k, graph, u_idx, m_idx, user_name, movie_id)

        out_path = out_dir / f"{user_name}_{movie_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {num_samples} explanations at: {out_dir}")

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Two-layer explanations with weak-group picks")
    parser.add_argument("--user_name", type=str, help="ユーザー名（省略時はランダム）")
    parser.add_argument("--movie_id", type=int, help="映画ID（省略時はランダム）")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--random", action="store_true", help="ランダム20件（説明込み）")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4: EXPLANATION GENERATION (two-layer + weak picks)")
    print("=" * 70)
    print("[1/5] Loading model & metadata ...")

    bundle = load_model_bundle(MODEL_DIR / "best_model.pkl")
    model = bundle["model"]
    user_stats = bundle.get("user_stats", {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}})

    dim_meta = load_dimension_metadata(DATA_DIR / "dimension_metadata.json")
    graph = load_graph_data(args.split, DATA_DIR)

    if args.random or (args.user_name is None and args.movie_id is None):
        generate_random_explanations(model, user_stats, dim_meta, graph,
                                     OUT_DIR / "random_pairs", num_samples=20, top_k=args.top_k)
        return

    print(f"[2/5] Locating edge for user={args.user_name}, movie_id={args.movie_id} ...")
    edge_idx, err = find_edge_index_by_user_movie(args.user_name, args.movie_id, graph)
    if edge_idx is None:
        print(f"❌ {err}")
        return
    print(f"  ✓ Edge index: {edge_idx}")

    print("[3/5] Extracting features & predicting ...")
    x, u_idx, m_idx = extract_edge_features(edge_idx, graph)
    y_norm = model_predict_norm(model, x)
    y_raw = denormalize_userwise(y_norm, u_idx, user_stats)

    print("[4/5] Computing two-layer contributions ...")
    contrib = compute_contributions_linear(model, x)
    exp = build_explanation(y_raw, contrib, dim_meta, args.top_k, graph, u_idx, m_idx,
                            args.user_name, int(args.movie_id))

    print("\n" + format_cli_report(exp))
    out_path = OUT_DIR / f"{args.user_name}_{args.movie_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Explanation saved: {out_path}")
    print("=" * 70)
    print("✅ Phase 4 Complete!")

if __name__ == "__main__":
    print(f"[DEBUG] torch={torch.__version__}")
    main()
