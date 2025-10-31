# -*- coding: utf-8 -*-
"""
infer_aaran_v3_full_explain.py
AARAN v3 FullModel の説明可能性拡張版
- attention/gate/grad×input を同時に抽出
- 各特徴群に対する寄与率を保存
"""

import torch
from pathlib import Path
from collections import defaultdict
from scripts.models.aaran_v3_full import AARANv3FullModel

# -------------------------------
# Utility
# -------------------------------
def safe_mean(tensor, dim=None):
    return tensor.mean(dim) if tensor.numel() > 0 else torch.tensor(0.0)

# -------------------------------
# Main
# -------------------------------
def main():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / "processed"
    OUT_DIR = ROOT / "outputs" / "aaran" / "inference"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT / "outputs" / "aaran" / "checkpoints" / "aaran_v3_full_20251030_1720.pt"
    blobs = torch.load(DATA_DIR / "hetero_graph_test.pt")

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = AARANv3FullModel().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -----------------------
    # Hooks for intermediate activations
    # -----------------------
    hooks = defaultdict(list)

    def hook_aspect(module, inp, out):
        ctx, attn = out
        hooks["aspect_attn"].append(attn.detach().cpu())

    def hook_gate(module, inp, out):
        # out = fused output (B, H)
        # We want gate value: stored in module.gate output before proj
        x = inp[0] if len(inp) > 0 else None
        if x is not None:
            concat = torch.cat([x, x, x], dim=-1)  # dummy to match shape
            with torch.no_grad():
                g = module.gate(concat).mean(0)
            hooks["gate"].append(g.detach().cpu())

    model.aspect_att.register_forward_hook(hook_aspect)
    model.fusion.register_forward_hook(hook_gate)

    # -----------------------
    # Grad × Input attribution
    # -----------------------
    u_x = blobs["user_features"][blobs["user_indices"]].to(DEVICE).requires_grad_(True)
    m_x = blobs["movie_features"][blobs["movie_indices"]].to(DEVICE).requires_grad_(True)
    r_x = blobs["review_signals"].to(DEVICE).requires_grad_(True)
    aspect = torch.rand(r_x.size(0), 18, device=DEVICE)

    with torch.enable_grad():
        y_hat = model(u_x, m_x, r_x, aspect)
        y_mean = y_hat.mean()
        y_mean.backward()

    grad_user = (u_x.grad * u_x).detach().cpu()
    grad_movie = (m_x.grad * m_x).detach().cpu()
    grad_review = (r_x.grad * r_x).detach().cpu()

    print("✅ Collected Grad×Input attributions.")

    # -----------------------
    # Aggregate results
    # -----------------------
    results = {
        "user_idx": blobs["user_indices"],
        "movie_idx": blobs["movie_indices"],
        "pred": y_hat.detach().cpu(),
        "grad_user": grad_user,
        "grad_movie": grad_movie,
        "grad_review": grad_review,
        "aspect_attn": torch.cat(hooks["aspect_attn"], dim=0) if hooks["aspect_attn"] else None,
        "gate_mean": safe_mean(torch.stack(hooks["gate"])) if hooks["gate"] else None,
    }

    out_path = OUT_DIR / "explain_v3_full_contrib.pt"
    torch.save(results, out_path)
    print(f"💾 Saved explainability outputs → {out_path}")

    # Summary
    if results["aspect_attn"] is not None:
        print(f"Aspect weights shape: {results['aspect_attn'].shape}")
    print("Done.")


if __name__ == "__main__":
    main()
