
import torch
from pathlib import Path
from datetime import datetime

from scripts.models.aaran_model import AARANModel, AARANConfig
from scripts.models.loss_functions import FocalRegressionLoss
from scripts.models.utils import build_user_stats
from scripts.models.eval import evaluate_raw


def load_graph(path: Path):
    data = torch.load(path, weights_only=False)
    # Assemble a simple dict of tensors for training/eval
    blobs = {
        "user_features": data["user_features"],
        "movie_features": data["movie_features"],
        "user_indices": data["user_indices"],
        "movie_indices": data["movie_indices"],
        "review_signals": data["review_signals"],
        "ratings_user_norm": data["ratings_user_norm"],
        "ratings_raw": data["ratings_raw"],
    }
    return blobs


def make_batches(blobs, batch_size: int):
    N = blobs["user_indices"].size(0)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        idx = slice(s, e)
        yield {
            "user_indices": blobs["user_indices"][idx],
            "movie_indices": blobs["movie_indices"][idx],
            "review_signals": blobs["review_signals"][idx],
            "ratings_user_norm": blobs["ratings_user_norm"][idx],
        }


def train_one_epoch(model, opt, train_blobs, device, loss_focal, loss_mse, lambda_reg: float, batch_size: int):
    model.train()
    total = 0.0
    for batch in make_batches(train_blobs, batch_size):
        u_idx = batch["user_indices"].to(device)
        m_idx = batch["movie_indices"].to(device)
        u_feat = train_blobs["user_features"].to(device)[u_idx]
        m_feat = train_blobs["movie_features"].to(device)[m_idx]
        r_feat = batch["review_signals"].to(device)
        y_norm = batch["ratings_user_norm"].to(device)

        y_hat = model(u_feat, m_feat, r_feat)
        loss = loss_focal(y_hat, y_norm) + lambda_reg * loss_mse(y_hat, y_norm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * u_idx.size(0)
    return total / train_blobs["user_indices"].size(0)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(__file__).resolve().parents[2] / "data" / "processed"

    # ====== 出力ディレクトリの自動生成 ======
    OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "aaran" / "checkpoints"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 出力ファイル名（日時付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path = OUTPUT_ROOT / f"aaran_v1_min_{timestamp}.pt"

    # ====== データロード ======
    train_blobs = load_graph(root / "hetero_graph_train.pt")
    val_blobs = load_graph(root / "hetero_graph_val.pt")
    test_blobs = load_graph(root / "hetero_graph_test.pt")

    mu_map, std_map = build_user_stats(train_blobs["user_indices"], train_blobs["ratings_raw"])

    model = AARANModel(AARANConfig()).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loss_focal = FocalRegressionLoss(alpha=0.25, gamma=2.0)
    loss_mse = torch.nn.MSELoss()

    best = {"mae": 1e9, "rmse": 1e9, "rho": -1.0}
    patience = 40
    bad = 0
    EPOCHS = 200
    BATCH = 1024
    LAMBDA = 0.1

    print(f"Device: {device}")
    print("Start training AARAN v1 (minimal)…")

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, opt, train_blobs, device, loss_focal, loss_mse, LAMBDA, BATCH)
        val_mae, val_rmse, val_rho = evaluate_raw(model, val_blobs, mu_map, std_map, device)
        sched.step(val_mae)
        print(f"Ep {ep:03d} | Loss {tr_loss:.3f} | Val MAE {val_mae:.3f} RMSE {val_rmse:.3f} ρ {val_rho:.3f}")

        improved = val_mae < best["mae"] - 1e-4
        if improved:
            best = {"mae": val_mae, "rmse": val_rmse, "rho": val_rho}
            torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, ckpt_path)
            print(f"✅ Saved best checkpoint → {ckpt_path}")
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stopping @ {ep}")
            break

    # ====== Final Test ======
    test_mae, test_rmse, test_rho = evaluate_raw(model, test_blobs, mu_map, std_map, device)
    print("=== TEST ===")
    print(f"Test → MAE {test_mae:.3f} | RMSE {test_rmse:.3f} | ρ {test_rho:.3f}")

if __name__ == "__main__":
    main()