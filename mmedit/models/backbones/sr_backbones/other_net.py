import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.models.restorers.encoder_decoder import EncoderDecoder
from mmedit.utils import get_root_logger
from mmedit.models.common.sr_backbone_utils import default_init_weights, GaussModulation

class CRANV2(nn.Conv2d):
    def __init__(self, in_channels=64, mid_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(CRANV2, self).__init__(in_channels, mid_channels, kernel_size, stride, padding, dilation, groups, bias)
       
        self.stride = stride
        self.padding= padding
        self.dilation = dilation
        self.groups = groups
        self.mid_channel = mid_channels
        self.kernel_size = kernel_size

        # weight & bias for content-gated-convolution
        self.weight_conv = Parameter(torch.zeros(mid_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.bias_conv = Parameter(torch.zeros(mid_channels), requires_grad=True)
       
        # init weight_conv layer
        nn.init.kaiming_normal_(self.weight_conv)

        # target spatial size of the pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))

        # the dimension of latent representation
        self.num_latent = int((kernel_size * kernel_size) / 2 + 1)

        # the context encoding module
        self.context_encoding = nn.Linear(kernel_size*kernel_size, self.num_latent, False)
        self.context_encoding_bn = nn.BatchNorm1d(in_channels)

        # relu function
        self.relu = nn.ReLU(inplace=True)

        # the number of groups in the channel interaction module
        if in_channels // 16: self.g = 16
        else: self.g = in_channels
       
        # the channel interacting module
        self.channel_interact = nn.Linear(self.g, mid_channels // (in_channels // self.g), bias=False)
        self.channel_interact_bn = nn.BatchNorm1d(mid_channels)
        self.channel_interact_bn2 = nn.BatchNorm1d(in_channels)

        # the gate decoding module (spatial interaction)
        self.gate_decode = nn.Linear(self.num_latent, kernel_size * kernel_size, False)
        self.gate_decode2 = nn.Linear(self.num_latent, kernel_size * kernel_size, False)

        # used to prepare the input feature map to patches
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

        # sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()              
        weight = self.weight_conv

        # allocate global information and context-encoding module
        out = self.context_encoding(self.avg_pool(x).view(b, c, -1))          

        # use different bn for following two branches
        context_encoding2 = out.clone()                                  
        out = self.relu(self.context_encoding_bn(out))                  

        # gate decoding branch 1 (spatial interaction)
        out = self.gate_decode(out)                      # out: batch x n_feat x 9 (5 --> 9 = 3x3)

        # channel interacting module
        oc = self.channel_interact(self.relu(self.channel_interact_bn2(context_encoding2).view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
        oc = self.relu(self.channel_interact_bn(oc.view(b, self.mid_channel, -1)))                       # oc: batch x n_feat x 5 (after grouped linear layer)

        # gate decoding branch 2 (spatial interaction)
        oc = self.gate_decode2(oc)                    
       
        # produce gate (equation (4) in the CRAN paper)
        out = self.sigmoid(out.view(b, 1, c, self.kernel_size, self.kernel_size)
            + oc.view(b, self.mid_channel, 1, self.kernel_size, self.kernel_size))

        # unfolding input feature map to patches
        x_unfold = self.unfold(x)
        b, _, l = x_unfold.size()

        # gating
        out = (out * weight.unsqueeze(0)).view(b, self.mid_channel, -1)

        return torch.matmul(out, x_unfold).view(-1, c, h, w)

class CRANResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.residual_block = ResidualBlockNoBN(mid_channels=mid_channels)
        self.cran_block = CRANV2(mid_channels=mid_channels)

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1)
    
    def forward(self, feat):
        cran_feat = self.cran_block(feat)

        feat = torch.cat([cran_feat, feat], dim=1)
        feat = self.fusion(feat)
        
        feat_prop = self.residual_block(feat)
        return feat_prop
