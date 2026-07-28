from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegmentationHead"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def sigmoid_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute batch-level soft Dice loss for one semantic class.

    The class channel is aggregated across the complete ``N x H x W`` batch
    so sparse positives receive a useful gradient. If the class is absent from
    the whole batch, Dice contributes zero and focal loss still supervises the
    negative pixels.
    """
    inputs = inputs.float()
    targets = targets.float()
    probabilities = torch.sigmoid(inputs)

    intersection = (probabilities * targets).sum()
    denominator = probabilities.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (denominator + smooth)

    has_positive = (targets.sum() > 0).to(dtype=inputs.dtype)
    return (1 - dice) * has_positive


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        focal_alpha: float = -1,
        focal_gamma: float = 2,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth

        if self.loss not in ("xent", "focal", "focal_dice"):
            raise ValueError(f"unsupported loss: {self.loss}")
        if self.focal_weight < 0 or self.dice_weight < 0:
            raise ValueError("focal_weight and dice_weight must be non-negative")
        if self.dice_smooth <= 0:
            raise ValueError("dice_smooth must be positive")

        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        x = self.classifier(x)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    loss = sigmoid_xent_loss(x[:, index], target[:, index])
                elif self.loss == "focal":
                    loss = sigmoid_focal_loss(
                        x[:, index],
                        target[:, index],
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                    )
                elif self.loss == "focal_dice":
                    focal_loss = sigmoid_focal_loss(
                        x[:, index],
                        target[:, index],
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                    )
                    dice_loss = sigmoid_dice_loss(
                        x[:, index],
                        target[:, index],
                        smooth=self.dice_smooth,
                    )
                    loss = (
                        self.focal_weight * focal_loss
                        + self.dice_weight * dice_loss
                    )
                losses[f"{name}/{self.loss}"] = loss
            return losses
        else:
            return torch.sigmoid(x)
