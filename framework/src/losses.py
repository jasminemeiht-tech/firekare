from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 0.5,
        neg_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.register_buffer("neg_weight", torch.tensor(float(neg_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        sample_weight = targets * self.pos_weight + (1.0 - targets) * self.neg_weight
        loss = sample_weight * focal_factor * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _resolve_focal_alpha(alpha: float | str | None, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if alpha is None:
        return 0.5
    if isinstance(alpha, str) and alpha.lower() == "auto":
        pos_count = int(labels.sum())
        neg_count = int(len(labels) - pos_count)
        total = pos_count + neg_count
        if total == 0:
            return 0.5
        return float(neg_count / total)
    alpha_value = float(alpha)
    return float(np.clip(alpha_value, 1e-6, 1.0 - 1e-6))


def _class_balanced_weights(labels: np.ndarray, beta: float) -> Tuple[float, float]:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.zeros(2, dtype=np.float64)
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        effective_num = 1.0 - np.power(beta, count)
        weights[idx] = (1.0 - beta) / max(effective_num, 1e-12)
    if weights.sum() <= 0:
        return 1.0, 1.0
    weights = weights / weights.sum() * 2.0
    neg_weight, pos_weight = weights.tolist()
    return float(neg_weight), float(pos_weight)


def build_binary_classification_loss(
    config: dict,
    train_labels: np.ndarray,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float | str]]:
    train_cfg = config.get("training", {})
    loss_cfg = train_cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "bce")).lower()

    pos_count = int(np.asarray(train_labels).astype(int).sum())
    neg_count = int(len(train_labels) - pos_count)

    if loss_name == "bce":
        if pos_count == 0:
            pos_weight_value = 1.0
        elif str(train_cfg.get("positive_class_weight", "auto")).lower() == "auto":
            pos_weight_value = max(neg_count / max(pos_count, 1), 1.0)
        else:
            pos_weight_value = float(train_cfg["positive_class_weight"])
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        )
        metadata: Dict[str, float | str] = {
            "loss_name": "bce",
            "loss_gamma": 0.0,
            "loss_pos_weight": float(pos_weight_value),
            "loss_neg_weight": 1.0,
        }
        return criterion, metadata

    gamma = float(loss_cfg.get("gamma", 2.0))
    if loss_name == "focal":
        alpha = _resolve_focal_alpha(loss_cfg.get("alpha", "auto"), train_labels)
        criterion = BinaryFocalLoss(
            gamma=gamma,
            pos_weight=alpha,
            neg_weight=1.0 - alpha,
        ).to(device)
        metadata = {
            "loss_name": "focal",
            "loss_gamma": gamma,
            "loss_pos_weight": float(alpha),
            "loss_neg_weight": float(1.0 - alpha),
        }
        return criterion, metadata

    if loss_name in {"cb_focal", "class_balanced_focal"}:
        beta = float(loss_cfg.get("cb_beta", loss_cfg.get("beta", 0.999)))
        neg_weight, pos_weight = _class_balanced_weights(train_labels, beta=beta)
        criterion = BinaryFocalLoss(
            gamma=gamma,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        ).to(device)
        metadata = {
            "loss_name": "cb_focal",
            "loss_gamma": gamma,
            "loss_pos_weight": float(pos_weight),
            "loss_neg_weight": float(neg_weight),
            "loss_cb_beta": beta,
        }
        return criterion, metadata

    raise ValueError(f"unsupported training loss: {loss_name}")
