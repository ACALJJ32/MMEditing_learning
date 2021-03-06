import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from torch.nn.parameter import Parameter

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from .edvr_net import PCDAlignment, TSAFusion
import math


@BACKBONES.register_module()
class BasicVSRGaussModulationV2(nn.Module):
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
                 spynet_pretrained=None,
                 edvr_pretrained=None,
                 with_dft=False):

        super().__init__()

        self.mid_channels = mid_channels
        self.padding = padding
        self.keyframe_stride = keyframe_stride

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

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
        self.dft_feature_extractor = DftFeatureExtractor(mid_channels, num_blocks=10, with_gauss=True)

        self.crac_module = CRACV2(in_channels=mid_channels, mid_channels = mid_channels)

        self.dft_fusion_backward = nn.Conv2d(2 * mid_channels + 3, mid_channels + 3, 3, 1, 1, bias=True)
        self.dft_fusion_forward = nn.Conv2d(3 * mid_channels + 3, 2 * mid_channels + 3, 3, 1, 1, bias=True)
        
        # pooling layer
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

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
    
    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

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

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, gts):
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
        flows_forward, flows_backward = self.compute_flow(lrs)  
        feats_refill = self.compute_refill_features(lrs, keyframe_idx) # dict; feats_refill[0] shape: [b, mid_channels, h, w]

        # backward-time propgation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i, :, :, :]
            if i < t - 1:  # no warping for the last timestep
                flow = flows_backward[:, i, :, :, :]   # [b, 2, h, w]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)  # [b, 2 * mid_channles, h, w]
                feat_prop = self.backward_fusion(feat_prop)                 # [b, mid_channels, h, w]

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)          # [b, mid_channel + 3, h, w]

            # DFT feature extractor
            if self.with_dft_feature_extractor and i in keyframe_idx:
                dft_feature = self.dft_feature_extractor(feats_refill[i])   # [b, mid_channels, h, w]
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
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([lr_curr, outputs[i], feat_prop], dim=1)  # [b, 2 * mid_channels + 3, h, w]

            # DFT feature extractor
            if self.with_dft_feature_extractor and i in keyframe_idx:
                dft_feature = self.dft_feature_extractor(feats_refill[i])
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

        return torch.stack(outputs, dim=1)[:, :, :, :4 * h_input, :4 * w_input], gts

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

class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

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

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

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
    def __init__(self, mid_channels=64, num_blocks=20, with_gauss=False, guass_key = 2.0):
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

class CRACV2(nn.Conv2d):
    def __init__(self, in_channels=64, mid_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(CRACV2, self).__init__(in_channels, mid_channels, kernel_size, stride, padding, dilation, groups, bias)
       
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
        print(out.size())

        # channel interacting module
        oc = self.channel_interact(self.relu(self.channel_interact_bn2(context_encoding2).view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
        oc = self.relu(self.channel_interact_bn(oc.view(b, self.mid_channel, -1)))                       # oc: batch x n_feat x 5 (after grouped linear layer)

        # gate decoding branch 2 (spatial interaction)
        oc = self.gate_decode2(oc)                       # oc: batch x n_feat x 9 (5 --> 9 = 3x3)
       
        # produce gate (equation (4) in the CRAN paper)
        out = self.sigmoid(out.view(b, 1, c, self.kernel_size, self.kernel_size)
            + oc.view(b, self.mid_channel, 1, self.kernel_size, self.kernel_size))  # out: batch x out_channel x in_channel x kernel_size x kernel_size (same dimension as conv2d weight)

        # unfolding input feature map to patches
        x_unfold = self.unfold(x)
        b, _, l = x_unfold.size()

        # gating
        out = (out * weight.unsqueeze(0)).view(b, self.mid_channel, -1)

        return torch.matmul(out, x_unfold).view(-1, c, h, w)