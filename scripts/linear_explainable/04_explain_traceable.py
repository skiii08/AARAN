#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 4: EXPLANATION GENERATION (two-layer + weak-group + traceable recovery)

目的:
- 二層寄与分析（特徴→グループ）
- 俳優・監督・キーワードなど traceable 特徴を元データに復元
- 弱グループ（movie_actor / movie_keyword）もTop3を強制抽出
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

WEAK_GROUP_CANDIDATES = ["movie_actor", "movie_keyword", "movie_director"]

# ========== I/O ==========
def load_pickle(p):
    with open(p, "rb") as f: return pickle.load(f)

def load_model_bundle(path: Path):
    obj = load_pickle(path)
    return obj if isinstance(obj, dict) and "model" in obj else {"model": obj, "user_stats": {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}}}

def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

# ========== Core utils ==========
def compute_contributions(model, x):
    w = np.asarray(getattr(model, "coef_", None)).reshape(-1)
    return w * x

def top_features(contrib, meta, top_k=20):
    idx = np.argsort(np.abs(contrib))[::-1][:top_k]
    out = []
    for i in idx:
        m = meta.get(str(i), {})
        out.append({
            "dim": int(i),
            "group": m.get("group", "unknown"),
            "name": m.get("name", f"Dim{i}"),
            "desc": m.get("description", ""),
            "value": float(contrib[i]),
            "traceable": m.get("traceable", False)
        })
    return out

# ---------- traceable補助 ----------
def traceable_lookup(group: str, entities: list):
    """グループ名に対応する候補を返す"""
    key = None
    if "actor" in group: key = "actors"
    elif "director" in group: key = "directors"
    elif "keyword" in group: key = "keywords"
    if not key: return []
    names = set()
    for m in entities:
        names.update(m.get(key, []))
    return list(names)

def attach_trace_names(entries, entities):
    """Top特徴にtraceable情報を追加"""
    for e in entries:
        g = e["group"]
        if e.get("traceable", False):
            cand = traceable_lookup(g, entities)
            e["trace_detail"] = random.sample(cand, k=min(3, len(cand))) if cand else []
        else:
            e["trace_detail"] = []
    return entries

# ========== Explanation構築 ==========
def build_explanation(pred, contrib, meta, top_k, graph, u_idx, m_idx, user_name, movie_id, entities):
    top_all = attach_trace_names(top_features(contrib, meta, top_k), entities)

    # group summary
    group_abs = {}
    for i, v in enumerate(contrib):
        g = meta.get(str(i), {}).get("group", "unknown")
        group_abs[g] = group_abs.get(g, 0.0) + abs(v)
    group_summary = sorted(group_abs.items(), key=lambda kv: kv[1], reverse=True)

    # weak-group picks
    weak_groups = [g for g in WEAK_GROUP_CANDIDATES if g in {m.get("group") for m in meta.values()}]
    weak_picks = {}
    for g in weak_groups:
        idxs = [i for i, m in meta.items() if m.get("group") == g]
        idxs = sorted(idxs, key=lambda i: abs(contrib[int(i)]), reverse=True)[:3]
        picks = []
        for i in idxs:
            m = meta[str(i)]
            picks.append({
                "name": m.get("name", f"Dim{i}"),
                "value": float(contrib[int(i)]),
                "trace_detail": random.sample(traceable_lookup(g, entities), k=2)
            })
        weak_picks[g] = picks

    movie_title = str(graph["movie_titles"][m_idx]) if "movie_titles" in graph else str(movie_id)
    return {
        "user_name": user_name,
        "movie_title": movie_title,
        "movie_id": movie_id,
        "prediction": float(pred),
        "top_contributors": top_all,
        "weak_group_top3": weak_picks,
        "group_summary": [{"group": g, "abs_sum": v} for g, v in group_summary]
    }

# ========== CLI出力 ==========
def format_report(exp):
    s = [f"\n{'='*70}\nEXPLANATION REPORT\n{'='*70}",
         f"User: {exp['user_name']}", f"Movie: {exp['movie_title']} ({exp['movie_id']})",
         f"Pred: {exp['prediction']:.2f}\n\n[Top Features]"]
    for i, c in enumerate(exp["top_contributors"], 1):
        sign = "+" if c["value"] > 0 else ""
        trace = f" → {', '.join(c['trace_detail'])}" if c["trace_detail"] else ""
        s.append(f" {i:02d}. {c['name']} [{c['group']}] {sign}{c['value']:.4f}{trace}")
    if exp["weak_group_top3"]:
        s.append("\n[Weak-Group Picks]")
        for g, arr in exp["weak_group_top3"].items():
            s.append(f"  {g}:")
            for a in arr:
                trace = f" → {', '.join(a['trace_detail'])}" if a['trace_detail'] else ""
                s.append(f"     • {a['name']} {a['value']:+.4f}{trace}")
    return "\n".join(s)

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    print("="*70)
    print("PHASE 4: Two-layer + Traceable Recovery")
    print("="*70)

    bundle = load_model_bundle(MODEL_DIR / "best_model.pkl")
    model = bundle["model"]
    dim_meta = load_json(DATA_DIR / "dimension_metadata.json")
    graph = torch.load(DATA_DIR / f"hetero_graph_{args.split}.pt", weights_only=False)
    entities = load_json(DATA_DIR / "movie_entities.json")

    # ランダム1件
    e = random.randint(0, len(graph["user_indices"]) - 1)
    u_idx = int(graph["user_indices"][e]); m_idx = int(graph["movie_indices"][e])
    user_name = str(graph["user_names"][u_idx]); movie_id = int(graph["movie_ids"][m_idx])

    x = np.concatenate([graph["user_features"][u_idx], graph["movie_features"][m_idx], graph["review_signals"][e]])
    contrib = compute_contributions(model, x)
    pred = float(model.predict(x.reshape(1, -1))[0])

    exp = build_explanation(pred, contrib, dim_meta, args.top_k, graph, u_idx, m_idx, user_name, movie_id, entities)
    print(format_report(exp))

    out = OUT_DIR / f"{user_name}_{movie_id}_traceable.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {out}")

if __name__ == "__main__":
    main()
