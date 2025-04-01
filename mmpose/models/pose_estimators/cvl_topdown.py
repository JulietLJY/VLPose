# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import Optional
import yaml
import os
import torch
from torch import Tensor, nn
from mmpose.registry import MODELS
from mmcv.cnn import build_norm_layer
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
from mmcv.cnn.bricks.transformer import MultiheadAttention
from lavis.models import load_model
from .vl_topdown import VisionLanguageTopdownPoseEstimator
 
   
@MODELS.register_module()
class ConditionalVisionLanguageTopdownPoseEstimator(VisionLanguageTopdownPoseEstimator):
    def __init__(self,
                 image_encoder: ConfigType,
                 text_encoder: ConfigType,
                 image_text_matcher: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            image_encoder=image_encoder,
            text_encoder=text_encoder, 
            image_text_matcher=image_text_matcher,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
         
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        image_feats, image_text_feats = self.extract_feat(inputs, data_samples)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(image_feats, image_text_feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _image_feats, _feats = self.extract_feat(inputs, data_samples)
            _image_feats_flip, _feats_flip = self.extract_feat(inputs.flip(-1), data_samples)
            image_feats = [_image_feats, _image_feats_flip]
            feats = [_feats, _feats_flip]
        else:
            image_feats, feats = self.extract_feat(inputs, data_samples)

        preds = self.head.predict(image_feats, feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results
