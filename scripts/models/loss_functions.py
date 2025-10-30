import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalRegressionLoss(nn.Module):
    """Focal-style loss for regression on normalized targets.
    Emphasizes hard examples (large residuals).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = torch.abs(pred - target)
        mod = (self.alpha + (1 - self.alpha) * (err / (err.detach().mean() + 1e-6))) ** self.gamma
        loss = mod * (err ** 2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss