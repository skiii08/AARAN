#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 4: EXPLANATION GENERATION (with pseudo prediction)
- レビューが存在しない user × movie の組み合わせでも予測可能
"""

import sys, json, pickle, argparse, random
from pathlib import Path
import numpy as np
import torch
import fasttext

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "linear_explainable" / "models"
OUT_DIR = BASE_DIR / "outputs" / "linear_explainable" / "explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTTEXT_MODEL_PATH = BASE_DIR / "data" / "external" / "cc.en.300.bin"

# ===== I/O =====
def load_pickle(p): return pickle.load(open(p, "rb"))
def load_model_bundle(p):
    obj = load_pickle(p)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "user_stats": {"mu_map": {-1: 0.0}, "std_map": {-1: 1.0}}}
def load_graph(split): return torch.load(DATA_DIR / f"hetero_graph_{split}.pt", weights_only=False)
def load_dim_meta(): return json.load(open(DATA_DIR / "dimension_metadata.json"))
def load_entities(): return json.load(open(DATA_DIR / "movie_entities.json"))

# ===== Core =====
def find_edge(user, mid, g):
    u, m = list(g["user_names"]), list(map(int, g["movie_ids"]))
    try:
        ui, mi = u.index(user), m.index(int(mid))
    except ValueError:
        return None
    arr_u, arr_m = g["user_indices"].numpy(), g["movie_indices"].numpy()
    for e, (uu, mm) in enumerate(zip(arr_u, arr_m)):
        if uu == ui and mm == mi: return e
    return None

def extract_features(idx, g):
    u_i = int(g["user_indices"][idx]); m_i = int(g["movie_indices"][idx])
    u = g["user_features"][u_i].numpy(); m = g["movie_features"][m_i].numpy()
    r = g["review_signals"][idx].numpy(); return np.concatenate([u, m, r]), u_i, m_i

def model_predict(model, x): return float(model.predict(x.reshape(1,-1))[0])
def denorm(y, u_i, stats):
    mu = stats.get("mu_map",{}).get(int(u_i),0.0); sd = stats.get("std_map",{}).get(int(u_i),1.0)
    return float(np.clip(y*sd+mu,1.0,10.0))
def compute_contrib(model,x):
    w=np.asarray(getattr(model,"coef_",None));
    if w.ndim>1:w=w.reshape(-1)
    return w*x

def compute_entity_contrib(ft, names, w_blk):
    if not names: return []
    w_mean=np.mean(w_blk,axis=0) if w_blk.ndim>1 else w_blk
    res=[]
    for n in names:
        v=ft.get_word_vector(n)
        res.append({"name":n,"contribution":float(np.dot(v,w_mean))})
    return sorted(res,key=lambda z:abs(z["contribution"]),reverse=True)[:5]

# ===== Build Explanation =====
def build_explanation(pred, contrib, dim_meta, g, u_i, m_i, user, mid, model, ft, ents):
    title=str(g["movie_titles"][m_i]) if "movie_titles" in g else str(mid)
    groups=list(dim_meta.values())
    a_s=next(i for i,v in enumerate(groups) if v["group"]=="movie_actor")
    d_s=next(i for i,v in enumerate(groups) if v["group"]=="movie_director")
    k_s=next(i for i,v in enumerate(groups) if v["group"]=="movie_keyword")
    a_e=a_s+300; d_e=d_s+300; k_e=k_s+300
    w=np.asarray(getattr(model,"coef_",None))
    wa,wd,wk=w[a_s:a_e],w[d_s:d_e],w[k_s:k_e]
    e = next((e for e in ents if int(e.get("movie_id",-1))==int(mid)),{})
    a,b,c=e.get("actors",[]),e.get("directors",[]),e.get("keywords",[])
    return {
        "prediction":pred,"user_name":user,"movie_id":int(mid),"movie_title":title,
        "actor_detail":compute_entity_contrib(ft,a,wa),
        "director_detail":compute_entity_contrib(ft,b,wd),
        "keyword_detail":compute_entity_contrib(ft,c,wk)
    }

def report(exp):
    l=[]
    l.append("="*70);l.append("EXPLANATION REPORT (with Actor/Director/Keyword details)")
    l.append("="*70)
    l.append(f"User : {exp['user_name']}");l.append(f"Movie: {exp['movie_title']} ({exp['movie_id']})")
    l.append(f"Pred : {exp['prediction']:.2f}")
    l.append("\n[Actor Contributions]")
    for a in exp["actor_detail"]: s="+" if a["contribution"]>0 else ""; l.append(f"  {s}{a['contribution']:.4f}  {a['name']}")
    l.append("\n[Director Contributions]")
    for d in exp["director_detail"]: s="+" if d["contribution"]>0 else ""; l.append(f"  {s}{d['contribution']:.4f}  {d['name']}")
    l.append("\n[Keyword Contributions]")
    for k in exp["keyword_detail"]: s="+" if k["contribution"]>0 else ""; l.append(f"  {s}{k['contribution']:.4f}  {k['name']}")
    l.append("="*70);return "\n".join(l)

# ===== Main =====
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--user_name",required=True)
    parser.add_argument("--movie_id",type=int,required=True)
    parser.add_argument("--split",default="test")
    args=parser.parse_args()

    print("="*70)
    print("PHASE 4: EXPLANATION (predictable version)")
    print("="*70)

    bundle=load_model_bundle(MODEL_DIR/"best_model.pkl")
    model,user_stats=bundle["model"],bundle["user_stats"]
    dim_meta,graph,ents=load_dim_meta(),load_graph(args.split),load_entities()
    ft=fasttext.load_model(str(FASTTEXT_MODEL_PATH))

    edge=find_edge(args.user_name,args.movie_id,graph)
    if edge is None:
        print(f"⚠ No review found for {args.user_name} × {args.movie_id}")
        print("→ Generating pseudo-edge (prediction only)")
        if edge is None:
            print(f"⚠ No review found for {args.user_name} × {args.movie_id}")
            print("→ Generating pseudo-edge (prediction only)")
            u_idx = int(np.where(np.array(graph["user_names"]) == args.user_name)[0][0])
            m_idx = int(np.where(np.array(graph["movie_ids"]) == int(args.movie_id))[0][0])
            u_feat = graph["user_features"][u_idx].numpy()
            m_feat = graph["movie_features"][m_idx].numpy()
            r_feat = np.zeros_like(graph["review_signals"][0].numpy())
            x = np.concatenate([u_feat, m_feat, r_feat])
        else:
            x, u_idx, m_idx = extract_features(edge, graph)
        u_feat=graph["user_features"][u_idx].numpy()
        m_feat=graph["movie_features"][m_idx].numpy()
        r_feat=np.zeros_like(graph["review_signals"][0].numpy())
        x=np.concatenate([u_feat,m_feat,r_feat])
    else:
        x,u_idx,m_idx=extract_features(edge,graph)

    y_norm=model_predict(model,x)
    y_raw=denorm(y_norm,u_idx,user_stats)
    contrib=compute_contrib(model,x)
    exp=build_explanation(y_raw,contrib,dim_meta,graph,u_idx,m_idx,
                          args.user_name,args.movie_id,model,ft,ents)
    print(report(exp))
    out=OUT_DIR/f"{args.user_name}_{args.movie_id}_predictable.json"
    json.dump(exp,open(out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n✅ Saved: {out}")

if __name__=="__main__":
    main()
