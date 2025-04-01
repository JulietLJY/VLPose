# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from .heatmap_head import HeatmapHead

OptIntSeq = Optional[Sequence[int]]

class LastAddDeconvBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        out_channels = cfg['out_channels']
        self.deconv0 = build_upsample_layer(cfg)
        self.deconv1 = build_upsample_layer(cfg)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels) \
            if i==0 else nn.Identity()
        self.add = i==1
        self.act = nn.ReLU(inplace=True)
                
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.deconv1(image_text_feat)
        output = self.deconv0(image_feat) + image_text_output \
            if self.add else self.deconv0(image_feat)
        output = self.act(self.bn0(output))
        image_text_output = self.act(self.bn1(image_text_output))
        return output, image_text_output

class LastCatDeconvBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        out_channels = cfg['out_channels']
        in_channels = cfg['in_channels']
        if i==0:
            self.deconv1 = build_upsample_layer(cfg)
            self.deconv0 = build_upsample_layer(cfg)
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        else:
            cfg['in_channels'] = 2 * in_channels
            self.deconv0 = build_upsample_layer(cfg)
            self.deconv1 = nn.Identity()
            self.bn1 = nn.Identity()
        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.add = i==1
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, feats):
        image_feat, image_text_feat = feats
        inputs = torch.cat([image_feat, image_text_feat], dim=1) \
            if self.add else image_feat 
        output = self.act(self.bn0(self.deconv0(inputs)))
        image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
        return output, image_text_output

class SingleAddDeconvBlock(LastAddDeconvBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.deconv1(image_text_feat)
        output = self.deconv0(image_feat) + image_text_output
        output = self.act(self.bn0(output))
        image_text_output = self.act(self.bn1(image_text_output))
        return output, image_text_output

class DoubleAddDeconvBlock(LastAddDeconvBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.deconv1(image_feat + image_text_feat)
        output = self.deconv0(image_feat) + image_text_output
        output = self.act(self.bn0(output))
        image_text_output = self.act(self.bn1(image_text_output))
        return output, image_text_output
 
class SingleCatDeconvBlock(nn.Module):
    def __init__(self, cfg, i=False):
        super().__init__()
        out_channels = cfg['out_channels']
        in_channels = cfg['in_channels']
        if i==0:
            self.deconv1 = build_upsample_layer(cfg)
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.deconv1 = nn.Identity()
            self.bn1 = nn.Identity()
        cfg['in_channels'] = 2 * in_channels
        self.deconv0 = build_upsample_layer(cfg)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.add = i==1
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.deconv1(image_text_feat)
        output = self.deconv0(torch.cat([image_feat, image_text_output], dim=1))
        output = self.act(self.bn0(output))
        image_text_output = self.act(self.bn1(image_text_output))
        return output, image_text_output

class MiddleFinalAddBlock(nn.Module):
    def __init__(self, cfg, i=None, freeze_main_branch=False):
        super().__init__()
        out_channels = cfg['out_channels']
        self.deconv0 = build_upsample_layer(cfg)
        self.deconv1 = build_upsample_layer(cfg)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)
        self.i = i
        if freeze_main_branch:
            for param in self.bn0.parameters():
                param.requires_grad = False
            for param in self.deconv0.parameters():
                param.requires_grad = False
        
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
        output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class FirstFinalAddBlock(MiddleFinalAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
        if self.i == 0:
            output = self.act(self.bn0(self.deconv0(image_feat + image_text_feat)))
        else:
            output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class FirstMiddleAddBlock(nn.Module):
    def __init__(self, cfg, i=None):
        super().__init__()
        out_channels = cfg['out_channels']
        self.deconv0 = build_upsample_layer(cfg)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.i = i
        if self.i == 0:
            self.deconv1 = build_upsample_layer(cfg)
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.deconv1 = nn.Identity()
            self.bn1 = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        
        
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
        if self.i == 0:
            output = self.act(self.bn0(self.deconv0(image_feat + image_text_feat))) + image_text_output
        else:
            output = self.act(self.bn0(self.deconv0(image_feat)))
        return output, image_text_output

class FirstMiddleLastAddBlock(MiddleFinalAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.act(self.bn1(self.deconv1(image_feat)))
        if self.i == 0:
            output = self.act(self.bn0(self.deconv0(image_feat + image_text_feat))) + image_text_output
        else:
            output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class FirstAMiddleLastAddBlock(MiddleFinalAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        image_text_output = self.act(self.bn1(self.deconv1(image_feat)))
        if self.i == 0:
            output = self.act(self.bn0(self.deconv0(image_feat + image_text_feat))) + image_text_output
        else:
            output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class AFirstMiddleLastAddBlock(MiddleFinalAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        if self.i == 0:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat + image_feat)))
        else:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
        output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class AFirstLastAddBlock(MiddleFinalAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        if self.i == 0:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat + image_feat)))
            output = self.act(self.bn0(self.deconv0(image_feat)))
        else:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat)))
            output = self.act(self.bn0(self.deconv0(image_feat))) + image_text_output
        return output, image_text_output

class AFirstMiddleAddBlock(FirstMiddleAddBlock):
    def forward(self, feats):
        image_feat, image_text_feat = feats
        if self.i == 0:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat + image_feat)))
            output = self.act(self.bn0(self.deconv0(image_feat)))
        else:
            image_text_output = self.act(self.bn1(self.deconv1(image_text_feat))) + image_text_feat
            output = self.act(self.bn0(self.deconv0(image_feat)))
        return output, image_text_output

@MODELS.register_module()
class VisionLanguageHeatmapHead(HeatmapHead):
    def __init__(self,
                 deconv_type: str,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 freeze_main_branch: bool = True, 
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 ):
        self.freeze_main_branch = freeze_main_branch

        if deconv_type == 'single_add':
            self.deconv_block = SingleAddDeconvBlock
        elif deconv_type == 'single_cat':
            self.deconv_block = SingleCatDeconvBlock
        elif deconv_type == 'double_add':
            self.deconv_block = DoubleAddDeconvBlock
        elif deconv_type == 'last_add':
            self.deconv_block = LastAddDeconvBlock
        elif deconv_type == 'last_cat':
            self.deconv_block = LastCatDeconvBlock
        elif deconv_type == 'middle_final_add':
            self.deconv_block = MiddleFinalAddBlock
        elif deconv_type == 'first_final_add':
            self.deconv_block = FirstFinalAddBlock
        elif deconv_type == 'first_middle_add':
            self.deconv_block = FirstMiddleAddBlock
        elif deconv_type == 'first_middle_last_add':
            self.deconv_block = FirstMiddleLastAddBlock
        elif deconv_type == 'afirst_middle_last_add':
            self.deconv_block = AFirstMiddleLastAddBlock
        elif deconv_type == 'afirst_last_add':
            self.deconv_block = AFirstLastAddBlock
        elif deconv_type == 'afirst_middle_add':
            self.deconv_block = AFirstMiddleAddBlock
        elif deconv_type == 'first_amiddle_last_add':
            self.deconv_block = FirstAMiddleLastAddBlock
        else:
            raise NotImplementError(deconv_type)
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer=final_layer,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        layers = []
        for i, (out_channels, kernel_size) in enumerate(zip(layer_out_channels, layer_kernel_sizes)):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(self.deconv_block(cfg=cfg, i=i, freeze_main_branch=self.freeze_main_branch))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, image_feats: Tuple[Tensor], image_text_feats: Tuple[Tensor]) -> Tensor:
        x, image_text_feats = self.deconv_layers([image_feats[-1], image_text_feats[-1]])
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x

    def predict(self,
                image_feats: Features,
                image_text_feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> image_feats = [orig, flipped]
            assert isinstance(image_feats, list) and len(image_feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _image_feats, _image_feats_flip = image_feats
            _image_text_feat, _image_text_feat_flip = image_text_feats
            _batch_heatmaps = self.forward(_image_feats, _image_text_feat)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_image_feats_flip, _image_text_feat_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(image_feats, image_text_feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             image_feats: Tuple[Tensor],
             image_text_feats: Tensor,
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        pred_fields = self.forward(image_feats, image_text_feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses
