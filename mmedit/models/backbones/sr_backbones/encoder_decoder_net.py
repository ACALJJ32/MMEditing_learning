# Copyright (c) OpenMMLab. All rights reserved.
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

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        deform_groups (int): Deformable groups. Defaults: 8.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 deform_groups=8,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=deform_groups)

            if i < 3:
                act_cfg_ = act_cfg if i == 2 else None
                self.feat_conv[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg_)

        # Cascading DCN
        self.cas_offset_conv1 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(
            mid_channels,
            mid_channels,
            3,
            padding=1,
            deform_groups=deform_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        """Forward function for PCDAlignment.

        Align neighboring frames to the reference frame in the feature level.

        Args:
            neighbor_feats (list[Tensor]): List of neighboring features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
            ref_feats (list[Tensor]): List of reference features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # The number of pyramid levels is 3.
        assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
            'The length of neighbor_feats and ref_feats must be both 3, '
            f'but got {len(neighbor_feats)} and {len(ref_feats)}')

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]],
                               dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](
                    torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            if i == 3:
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob

        # fusion
        feat = self.feat_fusion(aligned_feat)

        # spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat

class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.

    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".

    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
        pretrained (str): The pretrained model path. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 out_channel=3,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True,
                 pretrained=None):

        super().__init__()

        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN,
            num_blocks_extraction,
            mid_channels=mid_channels)

        # generate pyramid features
        self.feat_l2_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        # pcd alignment
        self.pcd_alignment = PCDAlignment(
            mid_channels=mid_channels, deform_groups=deform_groups)
        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)

        # CRAC module
        # self.crac_module = CRACV2(in_channels=mid_channels, mid_channels=mid_channels)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """

        n, t, c, h, w = x.size()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        # l1_feat = self.crac_module(l1_feat)
        
        l1_feat = self.feature_extraction(l1_feat)

        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))

        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx, :, :, :].clone(),
            l2_feat[:, self.center_frame_idx, :, :, :].clone(),
            l3_feat[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        return feat

class DftFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, num_blocks=5, with_gauss=True, guass_key = 1.0):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        main = []
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks=num_blocks, mid_channels=mid_channels))  # modified
        self.main = nn.Sequential(*main)

        self.conv_middle = nn.Conv2d(3 * mid_channels, mid_channels, 3, 1, 1, bias=True)

        last_feature_extractor = []
        last_feature_extractor.append(
            make_layer(
                ResidualBlockNoBN, num_blocks=num_blocks, mid_channels=mid_channels))  # modified
        self.last_feature_extractor = nn.Sequential(*last_feature_extractor)

        # fusion
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1)

        self.conv_last = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        # Gauss 
        self.with_gauss = with_gauss
        self.guass_key = guass_key
        
        self.modulation = GaussModulation(mid_channels=mid_channels)
    
    def modulation_block(self, feat, x_proj):
        b, c, h, w = feat.size()

        gauss_b = torch.randn((h,h)).to(feat.device)
        
        # modulate the gauss mat
        gauss_key =  torch.ones((b,1,h,h)) * self.guass_key
        gauss_key = gauss_key.to(feat.device)
        gauss_key = self.modulation(gauss_key)
        gauss_b = gauss_b * gauss_key

        x_proj = torch.matmul(gauss_b, x_proj)

        feat = torch.cat((torch.sin(x_proj), torch.cos(x_proj), feat), dim=1)  # [b, 2 * mid_channels, h, w]
        feat = self.conv_middle(feat)

        return feat

    def init_weights(self):
        for m in [self.conv_first, self.conv_middle, self.fusion, self.conv_last]:
            default_init_weights(m, 0.1)

    def forward(self, lr):
        feat = self.conv_first(lr) # [b, mid_channels, h, w]
        feat = self.main(feat)   # [b, mid_channels, h, w]

        x_proj = (2 * math.pi * feat)

        if self.with_gauss:
            feat_prop = self.modulation_block(feat, x_proj)
            feat = torch.cat([feat_prop, feat], dim=1)
            feat = self.fusion(feat)

        feat = self.last_feature_extractor(feat)

        return feat

class Encoder(nn.Module):
    def __init__(self,
                in_channels=3,
                out_channel=64,
                mid_channels=64
                ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # dft encoder blocks
        self.dft_feature_extractor = DftFeatureExtractor(mid_channels=mid_channels)

        # residual blocks
        self.propagation_resblocks = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 20)

        # fusion module
        self.fusion = nn.Conv2d(mid_channels * 2 + 6, mid_channels, 3, 1, 1)
        
        # downsample
        self.img_downsample = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)

        # activate function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        

        # init weights
        self.init_weights()
    
    def init_weights(self):
        for m in [self.conv_first, self.fusion]:
            default_init_weights(m, 0.1)

    def forward(self, lr_curr, gts, lr_feat):
        base = self.img_downsample(gts)
        dft_feat = self.dft_feature_extractor(base)
        
        feat_prop = torch.cat([lr_curr, base, dft_feat, lr_feat], dim=1)
        feat = self.fusion(feat_prop)

        feat = self.propagation_resblocks(feat)

        return feat

class Decoder(nn.Module):
    def __init__(self,
                mid_channels=64,
                pretrained=None):
        super().__init__()
        # dft decoder blocks
        # self.dft_feature_extractor = DftFeatureExtractor(in_channels=mid_channels ,mid_channels=mid_channels)

        self.propagation_resblocks = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 30)

        # fusion module
        # self.fusion = nn.Conv2d(mid_channels * 2, mid_channels * 2, 3, 1, 1)
    
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
    
    def forward(self, feat):
        feat = self.propagation_resblocks(feat)

        return feat
        
@BACKBONES.register_module()
class EncoderDecoderNet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 padding=2,
                 decoder_pretrained=None,
                 edvr_pretrained=None):
        super().__init__()

        self.encoder = Encoder(in_channels=3, out_channel=64, mid_channels=64)
        self.decoder = Decoder(mid_channels=mid_channels,pretrained=decoder_pretrained)
        self.padding = padding

        self.edvr_feature_extractor = EDVRFeatureExtractor(pretrained=edvr_pretrained)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
    
    def forward(self, lrs, gts):
        b, t, c, h, w = lrs.size()
        lr_curr = lrs[:, t // 2, :, :, :].clone()  # center frame

        # edvr feature
        lr_feat = self.edvr_feature_extractor(lrs)

        # encoder propagation
        gt_feat = self.encoder(lr_curr, gts, lr_feat)

        # decoder propagation
        feat = self.decoder(gt_feat)

        # upsample construct
        out = self.lrelu(self.upsample1(feat))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = self.img_upsample(lr_curr)
        out += base                                  # [b, c, h, w]

        return out