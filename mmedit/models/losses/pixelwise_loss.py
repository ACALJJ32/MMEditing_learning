# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import distance

from ..registry import LOSSES
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)


@masked_loss
def focal_loss(pred, target, eps=1e-12, alpha=1.0):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    
    # 2D DFT with orthonnomalization
    pred_fft = torch.fft.fft2(pred, norm='ortho')
    target_fft = torch.fft.fft2(target, norm='ortho')

    x_dist = (target_fft.real - pred_fft.real) ** 2
    y_dist = (target_fft.imag - pred_fft.imag) ** 2

    distance_mat = torch.sqrt(x_dist + y_dist + eps)
    squared_mat = distance_mat ** 2

    weight_mat = distance_mat ** alpha

    # normalization weight mat to [0, 1]
    norm_weight_mat = (weight_mat - torch.min(weight_mat)) / (torch.max(weight_mat) - torch.min(weight_mat))
    prod = torch.mul(squared_mat, norm_weight_mat)

    return prod

@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MaskedTVLoss(L1Loss):
    """Masked TV loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def forward(self, pred, mask=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor, optional): Tensor with shape of (n, 1, h, w).
                Defaults to None.

        Returns:
            [type]: [description]
        """
        y_diff = super().forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super().forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


@LOSSES.register_module()
class PCFLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, eps=1e-12):
        super().__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        b, t, c, h, w = pred.size()
        pred_reshape = pred.contiguous().clone().view(-1, c, h, w)
        target_reshape = target.contiguous().clone().view(-1, c, h, w)

        # L1
        charbonnier_loss_l1 = charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        # focal_loss_l1 = focal_loss(pred, target)

        # L2
        pred_l2 = F.interpolate(pred_reshape, (h // 2, w // 2), mode='bilinear', align_corners=False)
        target_l2 = F.interpolate(target_reshape, (h // 2, w // 2), mode='bilinear', align_corners=False)
        
        charbonnier_loss_l2 = charbonnier_loss(
            pred_l2,
            target_l2,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        # focal_loss_l2 = focal_loss(pred_l2, target_l2)

        # L3
        pred_l3 = F.interpolate(pred_reshape, (h // 4, w // 4), mode='bilinear', align_corners=False)
        target_l3 = F.interpolate(target_reshape, (h // 4, w // 4), mode='bilinear', align_corners=False)
        
        charbonnier_loss_l3 = charbonnier_loss(
            pred_l3,
            target_l3,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        # focal_loss_l3 = focal_loss(pred_l3, target_l3)


        # l1 = self.loss_weight * charbonnier_loss_l1 + self.loss_weight * 0.20 * focal_loss_l1
        # l2 = self.loss_weight * charbonnier_loss_l2 + self.loss_weight * 0.20 * focal_loss_l2
        # l3 = self.loss_weight * charbonnier_loss_l3 + self.loss_weight * 0.20 * focal_loss_l3

        l1 = self.loss_weight * charbonnier_loss_l1 + self.loss_weight
        l2 = self.loss_weight * charbonnier_loss_l2 + self.loss_weight 
        l3 = self.loss_weight * charbonnier_loss_l3 + self.loss_weight


        # return l1 + 0.10 * l2 + 0.10 * l3
        # return l1 + 0.10 * l2 
        return l1 + l2 + l3