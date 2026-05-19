import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch import nn


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None):
    num_classes = pred.size(1)
    target = target.long()
    valid_mask = (target >= 0) & (target < num_classes)

    target_one_hot = pred.new_zeros(pred.shape)
    if valid_mask.any():
        target_one_hot[valid_mask, target[valid_mask]] = 1

    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target_one_hot + pred_sigmoid * (1 - target_one_hot)
    focal_weight = (alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction="none") * focal_weight

    if weight is not None:
        if weight.dim() == 1:
            weight = weight[:, None]
        loss = loss * weight

    return weight_reduce_loss(loss, None, reduction, avg_factor)


@LOSSES.register_module()
class PyTorchFocalLoss(nn.Module):
    """Sigmoid focal loss implemented with native PyTorch ops.

    This avoids depending on mmcv.ops.sigmoid_focal_loss_forward, which may be
    unavailable when the installed MMCV binary was built without the matching
    CUDA extension.
    """

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        super().__init__()
        if not use_sigmoid:
            raise NotImplementedError("PyTorchFocalLoss only supports sigmoid focal loss.")
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override if reduction_override else self.reduction
        loss = py_sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * loss
