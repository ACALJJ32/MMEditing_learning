# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class GaussModulation(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        
        self.conv_first = nn.Conv2d(1, mid_channels, 1,1,0, bias=False)
        
        self.conv_middle1 = nn.Conv2d(mid_channels, mid_channels, 1,1,0, bias=False)
        self.conv_middle2 = nn.Conv2d(mid_channels, mid_channels, 1,1,0, bias=False)
        self.conv_middle3 = nn.Conv2d(mid_channels, mid_channels, 1,1,0, bias=False)

        feature_extractor = []
        feature_extractor.append(
            make_layer(
                ResidualBlockNoBN, num_blocks=5, mid_channels=mid_channels))  # modified
        self.feature_extractor = nn.Sequential(*feature_extractor)
        
         # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if res_scale == 1.0:
            self.init_weights()
    
    def init_weights(self):
        for m in [self.conv_first, self.conv_middle1, self.conv_middle2, self.conv_middle3]:
            default_init_weights(m, 0.1)

    def forward(self, gauss_key):
        feat = self.conv_first(gauss_key)
        feat = self.conv_middle1(feat)
        feat = self.conv_middle2(feat)
        feat = self.conv_middle3(feat)
        feat = self.feature_extractor(feat)

        return feat