from os import replace
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmedit.models.registry import BACKBONES, MODELS
from mmedit.utils import get_root_logger
from .edvr_net import PCDAlignment, TSAFusion
import math
from .raft_net import SmallUpdateBlock, SmallEncoder, CorrBlock, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


@BACKBONES.register_module()
class BasicVSRGaussModulation(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=30,
                 keyframe_stride=5,
                 padding=2,
                 raftnet_pretrained=None,
                 edvr_pretrained=None,
                 with_dft=False):

        super().__init__()

        self.mid_channels = mid_channels
        self.padding = padding
        self.keyframe_stride = keyframe_stride

        # optical flow network for feature alignment
        self.raftnet = RAFTNet(pretrained=raftnet_pretrained)

        # information-refill
        self.edvr = EDVRFeatureExtractor(
            num_frames=padding * 2 + 1,
            center_frame_idx=padding,
            pretrained=edvr_pretrained)
        self.backward_fusion = nn.Conv2d(
            2 * mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.forward_fusion = nn.Conv2d(
            2 * mid_channels, mid_channels, 3, 1, 1, bias=True)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            2 * mid_channels + 3, mid_channels, num_blocks)

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

        # DFT feature extractor
        self.with_dft_feature_extractor = with_dft
        self.dft_feature_extractor = DftFeatureExtractor(mid_channels, num_blocks=15, with_gauss=True)

        self.dft_fusion_backward = nn.Conv2d(2 * mid_channels + 3, mid_channels + 3, 3, 1, 1, bias=True)
        self.dft_fusion_forward = nn.Conv2d(3 * mid_channels + 3, 2 * mid_channels + 3, 3, 1, 1, bias=True)

        # Pyramid attention module
        self.with_pyramid_attention = True

        # embedding layer
        self.embedding_refill_l1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.embedding_refill_l2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.embedding_refill_l3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)

        self.embedding_prop_l1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.embedding_prop_l2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.embedding_prop_l3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        
        # pooling layer
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.level2_att_fusion = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.level3_att_fusion = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=False)

    def spatial_padding(self, lrs):
        """ Apply pdding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).

        """
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_refill_features(self, lrs, keyframe_idx):
        """ Compute keyframe features for information-refill.
        Since EDVR-M is used, padding is performed before feature computation.
        Args:
            lrs (Tensor): Input LR images with shape (n, t, c, h, w)
            keyframe_idx (list(int)): The indices specifying the keyframes.
        Return:
            dict(Tensor): The keyframe features. Each key corresponds to the
                indices in keyframe_idx.
        """

        if self.padding == 2:
            lrs = [lrs[:, [4, 3]], lrs, lrs[:, [-4, -5]]]  # padding
        elif self.padding == 3:
            lrs = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]  # padding
        lrs = torch.cat(lrs, dim=1)

        num_frames = 2 * self.padding + 1
        feats_refill = {}
        for i in keyframe_idx:
            feats_refill[i] = self.edvr(lrs[:, i:i + num_frames].contiguous())
        return feats_refill
    
    def compute_flow_with_raft(self, lrs):
        """Compute optical flow using RAFTNet for feature warping.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.raftnet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.raftnet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def pyramid_attention(self, prop, refill):
        """Compute attention map between feature prop and refill.

        Args:
            prop (tensor): Input prop feature map with shape (n, c, h, w)
            refill (tensor): Input refill feature map with shape (n, c, h, w)

        Return:
            Tensor: Output attention map with shape (n, c, h, w).
        """
        b, c, h, w = prop.size()

        # L1 level attention
        prop_l1 = self.lrelu(self.embedding_prop_l1(prop))
        refill_l1 = self.lrelu(self.embedding_refill_l1(refill))
        att_l1 = F.softmax(torch.cat((prop_l1, refill_l1), dim=1), dim=1) # [b, 2 * mid_channels, h, w]

        # L2 level attention
        prop_level2 = F.interpolate(prop, (h // 2, w // 2), mode='bilinear', align_corners=False)
        refill_level2 = F.interpolate(refill, (h // 2, w // 2), mode='bilinear', align_corners=False)
        prop_l2 = self.lrelu(self.embedding_prop_l2(prop_level2))
        refill_l2 = self.lrelu(self.embedding_refill_l2(refill_level2))

        prop_l2_max, prop_l2_avg = self.max_pool(prop_l2), self.avg_pool(prop_l2)
        refill_l2_max, refill_l2_avg = self.max_pool(refill_l2), self.avg_pool(refill_l2)

        prop_l2 = self.level2_att_fusion(torch.cat((prop_l2_max, prop_l2_avg), dim=1))
        refill_l2 = self.level2_att_fusion(torch.cat((refill_l2_max, refill_l2_avg), dim=1))

        att_l2 = F.softmax(torch.cat((prop_l2, refill_l2), dim=1), dim=1)
        att_l2 = F.interpolate(att_l2, (h, w), mode='bilinear', align_corners=False)

        # L3 level attention
        prop_level3 = F.interpolate(prop, (h // 4, w // 4), mode='bilinear', align_corners=False)
        refill_level3 = F.interpolate(refill, (h //4, w // 4), mode='bilinear', align_corners=False)
        prop_l3 = self.lrelu(self.embedding_prop_l3(prop_level3))
        refill_l3 = self.lrelu(self.embedding_refill_l3(refill_level3))

        prop_l3_max, prop_l3_avg = self.max_pool(prop_l3), self.avg_pool(prop_l3)
        refill_l3_max, refill_l3_avg = self.max_pool(refill_l3), self.avg_pool(refill_l3)

        prop_l3 = self.level3_att_fusion(torch.cat((prop_l3_max, prop_l3_avg), dim=1))
        refill_l3 = self.level3_att_fusion(torch.cat((refill_l3_max, refill_l3_avg), dim=1))

        att_l3 = F.softmax(torch.cat((prop_l3, refill_l3), dim=1), dim=1) # [b, 2 * mid_channels, h, w]
        att_l3 = F.interpolate(att_l3, (h, w), mode='bilinear', align_corners=False)

        feat_prop = prop * (att_l1[:, :c, :, :] + att_l2[:, :c, :, :] +  att_l3[:, :c, :, :]) * 2

        return feat_prop

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h_input, w_input = lrs.size()

        assert h_input >= 64 and w_input >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h_input} and {w_input}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs = self.spatial_padding(lrs)
        h, w = lrs.size(3), lrs.size(4)

        # get the keyframe indices for information-refill
        keyframe_idx = list(range(0, t, self.keyframe_stride))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # the last frame must be a keyframe

        # compute optical flow and compute features for information-refill  
        flows_forward, flows_backward = self.compute_flow_with_raft(lrs)  
        feats_refill = self.compute_refill_features(lrs, keyframe_idx)   # dict; feats_refill[0] shape: [b, mid_channels, h, w]

        # compute dft feature
        dft_features = [self.dft_feature_extractor(feats_refill[i]) for i in range(t)]

        # backward-time propgation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i, :, :, :]
            if i < t - 1:  # no warping for the last timestep
                flow = flows_backward[:, i, :, :, :]   # [b, 2, h, w]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            if i in keyframe_idx:
                # Pyramid attention module
                if self.with_pyramid_attention:
                    feat_prop = self.pyramid_attention(feat_prop.clone(), feats_refill[i].clone())

                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)  # [b, 2 * mid_channles, h, w]
                feat_prop = self.backward_fusion(feat_prop)                 # [b, mid_channels, h, w]

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)          # [b, mid_channel + 3, h, w]

            # DFT feature extractor
            if self.with_dft_feature_extractor:
                dft_feature = dft_features[i]                           # [b, mid_channels, h, w]
                feat_prop = torch.cat((dft_feature, feat_prop), dim=1)  # [b, 2 * mid_channles + 3, h, w]
                feat_prop = self.dft_fusion_backward(feat_prop)  
             
            feat_prop = self.backward_resblocks(feat_prop)
                        
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            if i in keyframe_idx:  # information-refill
                # Pyramid attention module
                if self.with_pyramid_attention:
                    feat_prop = self.pyramid_attention(feat_prop.clone(), feats_refill[i].clone())

                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([lr_curr, outputs[i], feat_prop], dim=1)  # [b, 2 * mid_channels + 3, h, w]

            # DFT feature extractor
            if self.with_dft_feature_extractor:
                dft_feature = dft_features[i]
                feat_prop = torch.cat((dft_feature, feat_prop), dim=1)
                feat_prop = self.dft_fusion_forward(feat_prop)

            feat_prop = self.forward_resblocks(feat_prop)  # [b, mid_channel, h, w]

            out = self.lrelu(self.upsample1(feat_prop))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base                                  # [b, c, h, w]
            outputs[i] = out

        return torch.stack(outputs, dim=1)[:, :, :, :4 * h_input, :4 * w_input]

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
                 pretrained=None,
                 with_homography_align=False):

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
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,
                                    1)

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

class RAFTNet(nn.Module):
    def __init__(self, pretrained, dropout = 0, mixed_precision=True):
        super(RAFTNet, self).__init__()

        self.hidden_dim = 96
        self.context_dim = 64
        self.corr_levels = 4
        self.corr_radius = 3

        self.dropout = dropout
        self.mixed_precision = mixed_precision

        # feature network, context network, and update block
        self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)        
        self.cnet = SmallEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='none', dropout=self.dropout)
        self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')
        
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0
        
        Args:
            img (Tensor): Input image sequence with shape (n, t, c, h, w).

        """
        n, c, h, w = img.shape
        coords0 = coords_grid(n, h//8, w//8).to(img.device)
        coords1 = coords_grid(n, h//8, w//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.size()
        mask = mask.view(n, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8*h, 8*w)

    def compute_flow(self, ref, supp, iters=12):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
            ref  <== supp

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        n, _, h, w = ref.size()

        # normalize the input images
        ref = (ref - self.mean) / self.std
        supp = (supp - self.mean) / self.std


        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([supp, ref])        
        
        fmap1 = fmap1.float()   
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(supp)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(supp) 

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume  

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
        
        flow_up = flow_predictions[-1]

        return flow_up

    def forward(self, ref, supp):
        """ Estimate optical flow between pair of frames """

        t, c, h, w = ref.size()

        # upsize to a multiple of 32
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)

        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)  
        
        supp = F.interpolate(
            input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)
        
        flow_up = self.compute_flow(ref, supp)  # [2 * (t-1), 2, h, w]

        flow_up = F.interpolate(
            input=flow_up,
            size=(h,w),
            mode="bilinear",
            align_corners=False)
        
        # adjust the flow values
        flow_up[:, 0, :, :] *= float(w) / float(w_up)
        flow_up[:, 1, :, :] *= float(h) / float(h_up)

        return flow_up

class DftFeatureExtractor(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=20, with_gauss=False, guass_key = 1.0):
        super().__init__()
        self.conv_first = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        main = []
        main.append(
            make_layer(
                ResidualBlockNoBN, 6, mid_channels=mid_channels))
        self.main = nn.Sequential(*main)

        self.conv_middle = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=True)

        feature_extractor = []
        feature_extractor.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=mid_channels))
        self.feature_extractor = nn.Sequential(*feature_extractor)

        self.conv_last = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        # Gauss 
        self.with_gauss = with_gauss
        self.guass_key = guass_key

        modulations = [
            nn.Conv2d(1, mid_channels, 1,1,0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(mid_channels, mid_channels, 1,1,0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(mid_channels, mid_channels, 1,1,0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ]

        modulations.append(
            make_layer(
                ResidualBlockNoBN, 5, mid_channels=mid_channels))
        
        self.modulation = nn.Sequential(*modulations)
        
    def forward(self, lr):
        """
        Args
            lr: low resolution images. 

        Returns  
            DFT feature maps of lr image.              
        """
        assert isinstance(lr, torch.Tensor), (
            print("lr must be Torch.Tensor!"))

        b, c, h, w = lr.size()

        x = self.conv_first(lr) # [b, mid_channels, h, w]
        x = self.main(x)   # [b, mid_channels, h, w]

        x_proj = (2 * math.pi * x)

        if self.with_gauss:
            gauss_b = torch.randn((h,h)).to(lr.device)

            # modulate the gauss mat
            gauss_key =  torch.ones((b,1,h,h)) * self.guass_key
            gauss_key = gauss_key.to(lr.device)
            gauss_key = self.modulation(gauss_key)
            gauss_b = gauss_b * gauss_key

            x_proj = torch.matmul(gauss_b, x_proj)
            
        dft_feature = torch.cat((torch.sin(x_proj), torch.cos(x_proj)),dim=1)  # [b, 2 * mid_channels, h, w]

        dft_feature = self.conv_middle(dft_feature)
        dft_feature = self.feature_extractor(dft_feature)

        return dft_feature