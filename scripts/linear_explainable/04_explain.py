"""
Phase 4: Explanation Generation
予測の寄与度分析と説明生成 (linear_explainable v2)
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import json
import argparse
import torch
import random

# プロジェクトルートを追加
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

# ========= 修正版ユーティリティ =========

def load_model(path: Path):
    """Pickle保存されたモデルを読み込み"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "model" in data:
        return data["model"]
    return data


def load_dimension_metadata(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def load_graph_data(split: str, data_dir: Path):
    """hetero_graph_train.pt / val.pt / test.pt 読み込み"""
    path = data_dir / f"hetero_graph_{split}.pt"
    print(f"  Loading {path} ...")
    data = torch.load(path, weights_only=False)
    # torch.loadで得た辞書の内容をそのまま使う
    return data


def find_user_movie_pair(user_name: str, movie_id: int, graph_data):
    """ユーザーと映画のエッジを探索"""
    user_names = graph_data.get("user_names", [])
    movie_ids = graph_data.get("movie_ids", [])
    user_indices = graph_data["user_indices"].numpy()
    movie_indices = graph_data["movie_indices"].numpy()

    # ✅ NumPy配列をPythonリスト化してindexを使えるようにする
    user_names = list(user_names)
    movie_ids = list(map(int, movie_ids))

    # ユーザー検索
    try:
        user_idx = user_names.index(user_name)
    except ValueError:
        return None, f"User '{user_name}' not found"

    # 映画検索
    try:
        movie_idx = movie_ids.index(int(movie_id))
    except ValueError:
        return None, f"Movie ID {movie_id} not found"

    # エッジ探索
    for edge_idx in range(len(user_indices)):
        if user_indices[edge_idx] == user_idx and movie_indices[edge_idx] == movie_idx:
            return edge_idx, None

    return None, f"No review found for user '{user_name}' and movie {movie_id}"



def extract_features(edge_idx: int, graph_data):
    """線形特徴を抽出"""
    u_idx = graph_data["user_indices"][edge_idx]
    m_idx = graph_data["movie_indices"][edge_idx]

    u_feat = graph_data["user_features"][u_idx].numpy()
    m_feat = graph_data["movie_features"][m_idx].numpy()
    r_feat = graph_data["review_signals"][edge_idx].numpy()

    x = np.concatenate([u_feat, m_feat, r_feat])
    return x, u_idx, m_idx


def compute_contributions(model, x: np.ndarray):
    """各特徴の寄与度"""
    w = model.coef_
    if len(w) != len(x):
        raise ValueError(f"Dimension mismatch: weight={len(w)}, x={len(x)}")
    return w * x


def get_top_contributors(contributions: np.ndarray, dimension_metadata, top_k=20):
    """寄与度上位の特徴を抽出"""
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
    contributors = []
    for idx in top_indices:
        contrib = float(contributions[idx])
        if abs(contrib) < 1e-4:
            continue
        meta = dimension_metadata.get(str(idx), {"name": f"Dim{idx}", "group": "unknown"})
        contributors.append({
            "dim": int(idx),
            "contribution": contrib,
            "type": meta.get("type", "unknown"),
            "name": meta.get("name", f"Dim{idx}"),
            "group": meta.get("group", "unknown"),
            "description": meta.get("description", "")
        })
    return contributors


def generate_explanation(pred, contributions, top_contribs, graph_data, u_idx, m_idx, user_name, movie_id):
    """説明データ構築"""
    movie_title = None
    if "movie_titles" in graph_data:
        movie_title = graph_data["movie_titles"][m_idx]
    elif "movie_ids" in graph_data:
        movie_title = str(graph_data["movie_ids"][m_idx])

    return {
        "prediction": float(pred),
        "user_name": user_name,
        "movie_id": int(movie_id),
        "movie_title": movie_title,
        "top_contributors": top_contribs,
        "summary": {
            "total_contribution": float(np.sum(contributions)),
            "positive_contribution": float(np.sum(contributions[contributions > 0])),
            "negative_contribution": float(np.sum(contributions[contributions < 0]))
        }
    }


def format_explanation_text(exp):
    """CLI出力整形"""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLANATION REPORT")
    lines.append("=" * 70)
    lines.append(f"User: {exp['user_name']}")
    lines.append(f"Movie: {exp['movie_title']} (ID: {exp['movie_id']})")
    lines.append(f"Predicted Rating: {exp['prediction']:.2f}")
    lines.append("")
    lines.append("Top contributors:")
    for i, c in enumerate(exp["top_contributors"][:20], 1):
        sign = "+" if c["contribution"] > 0 else ""
        lines.append(f"{i:02d}. {c['name']} ({c['group']})  {sign}{c['contribution']:.4f}")
        if c["description"]:
            lines.append(f"    {c['description']}")
    lines.append("")
    lines.append(f"Σ Total: {exp['summary']['total_contribution']:.4f}")
    lines.append(f"Σ Positive: {exp['summary']['positive_contribution']:.4f}")
    lines.append(f"Σ Negative: {exp['summary']['negative_contribution']:.4f}")
    lines.append("=" * 70)
    return "\n".join(lines)

import random
import json

def generate_random_explanations(model, dim_meta, graph_data, out_dir: Path, num_samples: int = 20, top_k: int = 20):
    """ランダムに user-movie ペアを選び、explanation 付き JSON を出力"""
    import random

    user_names = list(graph_data.get("user_names", []))
    movie_titles = list(graph_data.get("movie_titles", []))
    movie_ids = list(map(int, graph_data.get("movie_ids", [])))
    u_idx = graph_data["user_indices"].numpy()
    m_idx = graph_data["movie_indices"].numpy()

    all_edges = list(range(len(u_idx)))
    random.shuffle(all_edges)
    sampled = all_edges[:num_samples]

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- ユーザー統計をロードして正規化解除に使う ---
    ckpt_path = BASE_DIR / "outputs" / "linear_explainable" / "models" / "best_model.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    mu_map = ckpt["user_stats"]["mu_map"]
    std_map = ckpt["user_stats"]["std_map"]

    for edge_idx in sampled:
        u = user_names[u_idx[edge_idx]]
        m_title = movie_titles[m_idx[edge_idx]]
        m_id = movie_ids[m_idx[edge_idx]]

        # 特徴抽出
        x, u_id, m_id = extract_features(edge_idx, graph_data)

        # 予測 + 寄与計算
        y_pred_norm = model.predict(x.reshape(1, -1))[0]
        mu = mu_map.get(int(u_id), mu_map[-1])
        sd = std_map.get(int(u_id), std_map[-1])
        y_pred_raw = float(np.clip(y_pred_norm * sd + mu, 1.0, 10.0))
        contrib = compute_contributions(model, x)
        top = get_top_contributors(contrib, dim_meta, top_k)
        exp = generate_explanation(y_pred_raw, contrib, top, graph_data, u_id, m_id, u, m_id)

        # 保存
        out_path = out_dir / f"{u}_{m_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {num_samples} explanations at: {out_dir}")




def main():
    parser = argparse.ArgumentParser(description="Generate linear explanation (v2)")
    parser.add_argument("--user_name", type=str, help="ユーザー名（省略時はランダムモード）")
    parser.add_argument("--movie_id", type=int, help="映画ID（省略時はランダムモード）")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--random", action="store_true", help="ランダムに20件出力モード")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4: EXPLANATION GENERATION (v2)")
    print("=" * 70)

    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
    OUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "explanations"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. モデル読込 ---
    print("[1/5] Loading model...")
    model = load_model(MODEL_DIR / "best_model.pkl")

    # --- 2. メタデータ読込 ---
    print("[2/5] Loading dimension metadata...")
    dim_meta = load_dimension_metadata(DATA_DIR / "dimension_metadata.json")

    # --- 3. グラフ読込 ---
    print(f"[3/5] Loading {args.split} graph data...")
    graph = load_graph_data(args.split, DATA_DIR)
    if args.random or (args.user_name is None and args.movie_id is None):
        generate_random_explanations(model, dim_meta, graph, OUT_DIR / "random_pairs", num_samples=20)
        return



    # --- 4. ペア探索 ---
    print(f"[4/5] Finding user={args.user_name}, movie_id={args.movie_id}...")
    edge_idx, err = find_user_movie_pair(args.user_name, args.movie_id, graph)
    if edge_idx is None:
        print(f"❌ {err}")
        return
    print(f"  ✓ Edge index found: {edge_idx}")

    # --- 5. 特徴抽出 & 予測 ---
    print("[5/5] Computing contributions...")
    x, u_idx, m_idx = extract_features(edge_idx, graph)
    pred = model.predict(x.reshape(1, -1))[0]

    # ✅ add: denormalize using user stats
    model_dir = MODEL_DIR / "best_model.pkl"
    with open(model_dir, "rb") as f:
        ckpt = pickle.load(f)
    mu_map = ckpt["user_stats"]["mu_map"]
    std_map = ckpt["user_stats"]["std_map"]
    mu = mu_map.get(int(u_idx), mu_map.get(-1, 0))
    sd = std_map.get(int(u_idx), std_map.get(-1, 1))
    pred = float(np.clip(pred * sd + mu, 1.0, 10.0))
    contrib = compute_contributions(model, x)
    top = get_top_contributors(contrib, dim_meta, args.top_k)
    explanation = generate_explanation(pred, contrib, top, graph, u_idx, m_idx, args.user_name, args.movie_id)

    # --- 出力 ---
    print("\n" + format_explanation_text(explanation))
    out_path = OUT_DIR / f"{args.user_name}_{args.movie_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Explanation saved: {out_path}")
    print("=" * 70)
    print("✅ Phase 4 Complete!")


if __name__ == "__main__":
    main()
