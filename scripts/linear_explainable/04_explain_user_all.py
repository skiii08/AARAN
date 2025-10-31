#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 4: USER-WIDE EXPLANATION REPORT (standalone, predictable, entity-named weak groups)
-----------------------------------------------------------------------------------------
指定ユーザーが「視聴済み」の全映画について:
- 二層寄与分析（特徴→グループ）
- 俳優/監督/キーワードの “弱グループ Top3” を、emb_番号ではなく実名で表示
- モデルはそのまま（再学習なし）。FastText×重みベクトルで実体寄与を復元
- 1ユーザー分を JSON/TXT に一括保存

使い方:
  python scripts/linear_explainable/04_explain_user_all.py \
    --user_name 3505_januaryman-1 \
    --split test
"""

import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import fasttext

# ========= Paths =========
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
OUT_DIR  = BASE_DIR / "outputs" / "linear_explainable" / "explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTTEXT_MODEL_PATH = BASE_DIR / "data" / "external" / "cc.en.300.bin"

# “弱い”グループ（ここからは必ずTop3を抜いて表示）
WEAK_GROUP_TARGETS = ["movie_actor", "movie_keyword", "movie_director"]

# ========= I/O helpers =========
def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_bundle(path: Path):
    obj = _load_pickle(path)
    if isinstance(obj, dict) and "model" in obj:
        bundle = obj
    else:
        bundle = {"model": obj}
    bundle.setdefault("user_stats", {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}})
    return bundle

def load_dimension_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_graph(split: str):
    p = DATA_DIR / f"hetero_graph_{split}.pt"
    print(f"[Graph] Loading {p} ...")
    return torch.load(p, weights_only=False)

def load_movie_entities():
    p = DATA_DIR / "movie_entities.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # movie_id -> entity dict に変換（高速化）
    idx = {}
    if isinstance(data, list):
        for e in data:
            mid = int(e.get("movie_id", -1))
            idx[mid] = e
    elif isinstance(data, dict):
        for k, e in data.items():
            try:
                mid = int(e.get("movie_id", k))
            except Exception:
                mid = int(k)
            idx[mid] = e
    return idx

# ========= Safe access utils =========
def as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def get_user_index(graph, user_name: str):
    users = [str(u) for u in as_list(graph.get("user_names", []))]
    try:
        return users.index(user_name)
    except ValueError:
        return None

# ========= Model ops =========
def model_predict_norm(model, x: np.ndarray) -> float:
    return float(model.predict(x.reshape(1, -1))[0])

def denormalize_userwise(y_norm: float, u_idx: int, user_stats: dict) -> float:
    mu = user_stats.get("mu_map", {}).get(int(u_idx), user_stats.get("mu_map", {}).get(-1, 0.0))
    sd = user_stats.get("std_map", {}).get(int(u_idx), user_stats.get("std_map", {}).get(-1, 1.0))
    return float(np.clip(y_norm * sd + mu, 1.0, 10.0))

def compute_contributions_linear(model, x: np.ndarray):
    w = np.asarray(getattr(model, "coef_", None))
    if w.ndim > 1:
        w = w.reshape(-1)
    return w * x

# ========= Feature building =========
def extract_edge_features(edge_idx: int, graph):
    u_idx = int(graph["user_indices"][edge_idx])
    m_idx = int(graph["movie_indices"][edge_idx])
    u = graph["user_features"][u_idx].cpu().numpy()
    m = graph["movie_features"][m_idx].cpu().numpy()
    r = graph["review_signals"][edge_idx].cpu().numpy()
    return np.concatenate([u, m, r], axis=0), u_idx, m_idx

# ========= Two-layer helpers =========
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
    group_abs, group_signed, group_samples = {}, {}, {}
    # dim_meta のキーは "0","1",...。インデックス順に辿る
    for i in range(contrib.shape[0]):
        g = dim_meta.get(str(i), {}).get("group", "unknown")
        v = float(contrib[i])
        group_abs[g] = group_abs.get(g, 0.0) + abs(v)
        group_signed[g] = group_signed.get(g, 0.0) + v
        group_samples.setdefault(g, []).append((i, v))
    group_top1 = {g: max(arr, key=lambda t: abs(t[1])) for g, arr in group_samples.items()}
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
    return summary

# ========= Entity-level contribution (names instead of emb_xxx) =========
def find_block_indices(dim_meta: dict, group_name: str):
    """dim_meta から、指定グループの連続ブロック開始・終了(含まず)を推定"""
    idxs = [int(k) for k, v in dim_meta.items() if v.get("group") == group_name]
    if not idxs:
        return None, None
    idxs.sort()
    return idxs[0], idxs[-1] + 1  # [start, end)

def compute_entity_contributions(ft_model, names, weight_block: np.ndarray, topn: int = 5):
    """FastText(name) · weight_block をスコア化（再学習なしの後付け推定）"""
    if not names:
        return []
    # weight_block は (300,) 想定。保険で平均化
    if weight_block.ndim > 1:
        w = weight_block.mean(axis=0)
    else:
        w = weight_block
    out = []
    for n in names:
        try:
            v = ft_model.get_word_vector(n)
            score = float(np.dot(v, w))
            out.append({"name": n, "contribution": score})
        except Exception:
            continue
    out.sort(key=lambda z: abs(z["contribution"]), reverse=True)
    return out[:topn]

def weak_group_top3_as_names(model, dim_meta, ft_model, movie_entities_by_id, movie_id: int):
    """弱グループ（俳優/監督/キーワード）を “実名Top3” で返す"""
    # モデル重み
    w = np.asarray(getattr(model, "coef_", None))
    if w.ndim > 1:
        w = w.reshape(-1)

    result = {}
    ents = movie_entities_by_id.get(int(movie_id), {})
    actors   = ents.get("actors", []) or []
    directors= ents.get("directors", []) or []
    keywords = ents.get("keywords", []) or []

    # 各ブロックの weight を切り出し → 実名寄与に変換
    # movie_actor
    a_s, a_e = find_block_indices(dim_meta, "movie_actor")
    if a_s is not None:
        wa = w[a_s:a_e]
        a_list = compute_entity_contributions(ft_model, actors, wa, topn=3)
        if a_list:
            result["movie_actor"] = a_list

    # movie_keyword
    k_s, k_e = find_block_indices(dim_meta, "movie_keyword")
    if k_s is not None:
        wk = w[k_s:k_e]
        k_list = compute_entity_contributions(ft_model, keywords, wk, topn=3)
        if k_list:
            result["movie_keyword"] = k_list

    # movie_director
    d_s, d_e = find_block_indices(dim_meta, "movie_director")
    if d_s is not None:
        wd = w[d_s:d_e]
        d_list = compute_entity_contributions(ft_model, directors, wd, topn=3)
        if d_list:
            result["movie_director"] = d_list

    return result

# ========= Build one-movie explanation =========
def build_explanation(pred_raw, contrib, dim_meta, graph, u_idx, m_idx, user_name, movie_id,
                      model, ft_model, movie_entities_by_id, top_k: int):
    movie_title = str(graph["movie_titles"][m_idx]) if "movie_titles" in graph else str(movie_id)

    # 1) 特徴レベルTopK
    top_all = top_features(contrib, dim_meta, top_k=top_k)

    # 2) グループ要約
    group_summary = summarize_by_group(contrib, dim_meta)

    # 3) 弱グループ Top3（実名に変換）
    weak_named = weak_group_top3_as_names(model, dim_meta, ft_model, movie_entities_by_id, movie_id)

    # 合計
    pos_sum = float(np.sum(contrib[contrib > 0])) if contrib.size else 0.0
    neg_sum = float(np.sum(contrib[contrib < 0])) if contrib.size else 0.0
    total   = float(np.sum(contrib)) if contrib.size else 0.0

    return {
        "prediction": float(pred_raw),
        "user_name": user_name,
        "movie_id": int(movie_id),
        "movie_title": movie_title,
        "top_contributors": top_all,
        "weak_group_top3_named": weak_named,   # ← 実名表示
        "group_summary": group_summary,
        "summary": {
            "total_contribution": total,
            "positive_contribution": pos_sum,
            "negative_contribution": neg_sum
        }
    }

# ========= CLI formatting =========
def format_cli_report(exp: dict):
    lines = []
    lines.append("=" * 70)
    lines.append(f"{exp['movie_title']} ({exp['movie_id']})")
    lines.append(f"Pred: {exp['prediction']:.2f}")
    lines.append("[Top Features]")
    for c in exp["top_contributors"][:10]:
        sign = "+" if c["contribution"] > 0 else ""
        lines.append(f"  {c['name']} [{c['group']}] {sign}{c['contribution']:.4f}")

    if exp.get("weak_group_top3_named"):
        lines.append("[Weak Groups (converted to names)]")
        for g in WEAK_GROUP_TARGETS:
            if g in exp["weak_group_top3_named"]:
                parts = []
                for item in exp["weak_group_top3_named"][g]:
                    sign = "+" if item["contribution"] > 0 else ""
                    parts.append(f"{item['name']}({sign}{item['contribution']:.4f})")
                lines.append(f"  {g}: " + ", ".join(parts))
    lines.append("=" * 70)
    return "\n".join(lines)

# ========= MAIN =========
def main():
    parser = argparse.ArgumentParser(description="User-wide explanations with entity-named weak groups (standalone)")
    parser.add_argument("--user_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print(f"USER-WIDE EXPLANATION for {args.user_name}")
    print("=" * 70)

    # Load everything
    bundle = load_model_bundle(MODEL_DIR / "best_model.pkl")
    model, user_stats = bundle["model"], bundle["user_stats"]
    dim_meta = load_dimension_metadata(DATA_DIR / "dimension_metadata.json")
    graph    = load_graph(args.split)
    ents_by_id = load_movie_entities()
    ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

    # Resolve user index
    u_idx = get_user_index(graph, args.user_name)
    if u_idx is None:
        print(f"❌ user_name '{args.user_name}' not found.")
        return

    # Collect watched edges for the user
    u_arr = graph["user_indices"].cpu().numpy()
    m_arr = graph["movie_indices"].cpu().numpy()
    edges = [e for e, u in enumerate(u_arr) if int(u) == int(u_idx)]
    if not edges:
        print(f"❌ No watched movies for {args.user_name}")
        return

    movie_ids = list(map(int, as_list(graph.get("movie_ids", []))))
    movie_titles = [str(t) for t in as_list(graph.get("movie_titles", []))]

    print(f"Found {len(edges)} edges (reviews) for {args.user_name}")

    # Some datasets may include multiple reviews per same (user, movie). Deduplicate by movie.
    seen_movie_edge = {}
    for e in edges:
        midx = int(m_arr[e])
        mid  = int(movie_ids[midx])
        # keep first (or latest) edge
        if mid not in seen_movie_edge:
            seen_movie_edge[mid] = e

    results, reports = [], []
    for mid, e in tqdm(seen_movie_edge.items(), desc="Processing movies"):
        x, u_i, m_i = extract_edge_features(e, graph)
        y_norm = model_predict_norm(model, x)
        y_raw  = denormalize_userwise(y_norm, u_i, user_stats)
        contrib = compute_contributions_linear(model, x)

        exp = build_explanation(
            pred_raw=y_raw,
            contrib=contrib,
            dim_meta=dim_meta,
            graph=graph,
            u_idx=u_i,
            m_idx=m_i,
            user_name=args.user_name,
            movie_id=mid,
            model=model,
            ft_model=ft_model,
            movie_entities_by_id=ents_by_id,
            top_k=args.top_k
        )
        results.append(exp)
        reports.append(format_cli_report(exp))

    # Save
    out_json = OUT_DIR / f"{args.user_name}_all_with_names.json"
    out_txt  = OUT_DIR / f"{args.user_name}_all_with_names.txt"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))

    print(f"\n✅ Saved JSON: {out_json}")
    print(f"✅ Saved TXT : {out_txt}")
    print("🎉 Done!")

if __name__ == "__main__":
    main()
