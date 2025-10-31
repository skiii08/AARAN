"""
Grouped Regularized Linear Model
グループ別正則化を適用した線形回帰モデル (PyTorch / CPU固定・正則化完全制御版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

# ⚙️ CPU固定
DEVICE = torch.device("cpu")
print(f"[DEBUG:DEVICE] ✅ PyTorch device set to CPU (FORCE) in grouped_linear.py.")
print(f"[DEBUG:DEVICE] Current PyTorch version: {torch.__version__}")


class GroupedLinearRegression(nn.Module):
    """
    グループ別正則化を適用した線形回帰モデル (正則化制御強化版)
    """

    def __init__(self,
                 group_indices: Dict[str, np.ndarray],
                 group_regularizations: Dict[str, str],
                 lambda_l1: float = 0.01,
                 lambda_l2: float = 0.01,
                 lambda_elastic: float = 0.01,
                 alpha_elastic: float = 0.5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_focal: bool = True):
        super().__init__()

        # 入力構造
        self.group_indices_np = group_indices
        self.group_indices = {name: torch.from_numpy(indices) for name, indices in group_indices.items()}
        self.group_regularizations = group_regularizations

        # ハイパーパラメータ
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_elastic = lambda_elastic
        self.alpha_elastic = alpha_elastic
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

        # 全特徴数を決定
        if len(self.group_indices_np) == 0:
            raise ValueError("Feature groups cannot be empty.")
        total_dim = np.concatenate(list(self.group_indices_np.values())).max() + 1

        # 線形層定義
        self.linear = nn.Linear(total_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.to(DEVICE)
        self.group_indices = {name: idx.to(DEVICE) for name, idx in self.group_indices.items()}
        self.first_forward = True

    # ---- Properties ----
    @property
    def coef_(self) -> np.ndarray:
        return self.linear.weight.data.cpu().numpy().flatten()

    @property
    def intercept_(self) -> float:
        return float(self.linear.bias.data.cpu().item())

    # ---- Forward ----
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.first_forward:
            self.first_forward = False
        return self.linear(X).squeeze(1)

    # ---- Regularization ----
    def regularization_term(self) -> torch.Tensor:
        """グループ別正則化項 (λ=0は完全スキップ)"""
        reg_loss = torch.tensor(0.0, device=DEVICE)
        w = self.linear.weight.squeeze(0)

        for name, reg_type in self.group_regularizations.items():
            indices = self.group_indices[name]
            w_g = w[indices]

            # λ=0ならスキップ（autogradにも影響しない）
            if reg_type == "l1" and self.lambda_l1 > 0:
                reg_loss += self.lambda_l1 * torch.sum(torch.abs(w_g))
            elif reg_type == "l2" and self.lambda_l2 > 0:
                reg_loss += self.lambda_l2 * torch.sum(w_g.pow(2))
            elif reg_type == "elastic" and self.lambda_elastic > 0:
                l1_term = self.alpha_elastic * torch.sum(torch.abs(w_g))
                l2_term = (1.0 - self.alpha_elastic) * torch.sum(w_g.pow(2))
                reg_loss += self.lambda_elastic * (l1_term + l2_term)

        return reg_loss

    # ---- Loss ----
    def focal_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """回帰用Focal Loss"""
        mse = (y_pred - y_true).pow(2)
        abs_error = (y_pred - y_true).abs()
        modulating_factor = abs_error.pow(self.focal_gamma)
        return (modulating_factor * mse).mean()

    def loss_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """総合損失関数: MSE/Focal + 正則化"""
        if self.use_focal:
            data_loss = self.focal_loss(y_pred, y_true)
        else:
            data_loss = F.mse_loss(y_pred, y_true)

        reg_loss = self.regularization_term()
        return data_loss + reg_loss

    # ---- Fit ----
    def fit(self, X: np.ndarray, y: np.ndarray,
            n_epochs: int = 1000, lr: float = 0.001,
            batch_size: int = 256, verbose: bool = True):
        """モデルの訓練 (CPU固定 / Adam)"""

        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        y_tensor = torch.from_numpy(y).float().to(DEVICE)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, n_epochs + 1):
            total_loss = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

                if epoch == 1 and batch_idx == 0:
                    print(f"[DEBUG:LOOP] Batch 0 completed. (Loss: {loss.item():.6f})")

            avg_loss = total_loss / len(X_tensor)

            if epoch % 100 == 0:
                print(f"[DEBUG:LOOP] Epoch {epoch:4d}/{n_epochs} END. Loss = {avg_loss:.6f}")
            if epoch == 1:
                print(f"[DEBUG:LOOP_EPOCH] 🚀 Epoch 1 completed. Avg Loss: {avg_loss:.6f}. Proceeding to Trial 02...")

    # ---- Predict ----
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測処理（evalモード固定）"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(DEVICE)
            y_pred = self(X_tensor).cpu().numpy()
        return y_pred

    # ---- Summary ----
    def print_regularization_summary(self):
        """グループ別の正則化寄与を表示"""
        w = self.linear.weight.data.cpu().numpy().flatten()
        print("\n" + "=" * 70)
        print("FINAL WEIGHTS REGULARIZATION SUMMARY (Absolute Sum)")
        print("=" * 70)
        for name, reg_type in self.group_regularizations.items():
            indices = self.group_indices_np[name]
            w_g = w[indices]
            abs_sum = np.sum(np.abs(w_g))
            l2_sum = np.sum(w_g ** 2)
            print(f"  - {name:25s} ({reg_type.upper():7s}): L1 Sum = {abs_sum:.4f}, L2 Sum = {l2_sum:.4f}")
        print("=" * 70)

    def get_feature_importance(self) -> np.ndarray:
        """
        各特徴量の重要度（重みの絶対値）を返す。
        """
        w = self.linear.weight.data.cpu().numpy().flatten()
        importance = np.abs(w)
        return importance


# ---- Factory ----
def create_grouped_model_from_feature_groups(
        feature_groups: Dict,
        lambda_l1: float = 0.01,
        lambda_l2: float = 0.01,
        lambda_elastic: float = 0.01,
        alpha_elastic: float = 0.5,
        use_focal: bool = True,
) -> GroupedLinearRegression:
    """feature_groups.pkl からモデルを構築"""
    group_indices = {}
    group_regularizations = {}

    for name, info in feature_groups["groups"].items():
        indices = np.arange(info["start"], info["end"])
        group_indices[name] = indices
        group_regularizations[name] = info["regularization"]

    return GroupedLinearRegression(
        group_indices=group_indices,
        group_regularizations=group_regularizations,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_elastic=lambda_elastic,
        alpha_elastic=alpha_elastic,
        use_focal=use_focal
    )
