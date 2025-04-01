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

class AttentionMatcher(nn.Module):
    def __init__(self, embed_dims, num_heads=8, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 norm_cfg=dict(type='LN'),):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward_attention(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        
        return x

    def count_parameters(self, require_grad=False):
        if require_grad:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        xq = image_embed
        xk = text_embed
        xv = text_embed

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        return out

class ConcatAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        xq = image_embed
        xk = xv = torch.cat([image_embed, text_embed], axis=1)

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        return out
    
class CosMaxAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        B, L, C = image_embed.shape
        cos = torch.einsum('blc,btc->blt', image_embed, text_embed) # [64, 192, 13*17]
        cos_max, cos_idx = torch.max(cos, dim=-1)                     # [64, 192]
        cos_repeat = cos_max.unsqueeze(dim=2).repeat_interleave(C, axis=-1)   # [blc]
        
        xq = image_embed
        xk = xv = cos_repeat

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        
        return out
    
class CosMeanAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        B, L, C = image_embed.shape
        cos = torch.einsum('blc,btc->blt', image_embed, text_embed) # [64, 192, 13*17]
        cos_max, cos_idx = torch.mean(cos, dim=-1)                     # [64, 192]
        cos_repeat = cos_max.unsqueeze(dim=2).repeat_interleave(C, axis=-1)   # [blc]
        
        xq = image_embed
        xk = xv = cos_repeat

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        
        return out
 
class CatCatAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        xq = image_embed
        xk = xv = torch.cat([image_embed, text_embed], axis=1)

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        out = torch.cat([image_embed, out], axis=1)
        return out
    
class CosCatAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        B, L, C = image_embed.shape
        cos = torch.einsum('blc,btc->btc', image_embed, text_embed) # [64, 13*17, 384]
        
        xq = image_embed
        xk = xv = torch.cat([image_embed, cos], axis=1)

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        
        return out
    
class CosCatCatAttentionMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        B, L, C = image_embed.shape
        cos = torch.einsum('blc,btc->btc', image_embed, text_embed) # [64, 13*17, 384]
        
        xq = image_embed
        xk = xv = torch.cat([image_embed, cos], axis=1)

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        out = torch.cat([image_embed, out], axis=1)
        return out
    
class RandomMatcher(AttentionMatcher):
    def forward(self, image_embed, text_embed):
        '''
            image_embed: torch.Size([64, 192, 384])
            text_embed: torch.Size([64, 13*17, 384])
        '''
        B, L, C = image_embed.shape
        cos = torch.einsum('blc,btc->btc', image_embed, text_embed) # [64, 13*17, 384]
        
        xq = image_embed
        xk = xv = torch.cat([image_embed, cos], axis=1)

        attn = self.forward_attention(xq, xk, xv)
        out = self.norm(attn + image_embed)    # residual
        out = torch.cat([image_embed, out], axis=1)
        return out
    
@MODELS.register_module()
class VisionLanguageTopdownPoseEstimator(BasePoseEstimator):
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
            backbone=image_encoder,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        self._init_text_encoder(**text_encoder)
        self._init_image_text_matcher(**image_text_matcher)
        if self.num_keywords >= 5:
            self._init_text_prompt()
 
    def _init_text_prompt(self, ):
        return
        raise NotImplementError(self.text_prompt)
        self.text_prompt = nn.Parameter()
        
    def _init_text_encoder(self, text_embed=768, image_embed=768, 
                           name='blip_feature_extractor', model_type='base',
                           num_keywords=1, text_prompt_file='', keypoints_anno_file=''):
        self.text_encoder_name = name
        self.text_encoder = load_model(
            name=name, model_type=model_type, is_eval=True, device='cuda')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.num_keywords = num_keywords
        
        if self.num_keywords >= 0:
            assert os.path.exists(text_prompt_file), f'{text_prompt_file} not find'
            with open(text_prompt_file, "r") as f:
                self.text_prompt = yaml.safe_load(f)
            # print(f'text prompt: {self.text_prompt}')
        elif self.num_keywords == -1:
            self.text_prompt = 'a photo of a human'
            # print(f'text prompt: {self.text_prompt}')
        elif self.num_keywords == -2:   # for ablation
            self.text_prompt = 'None'
            # print(f'text prompt: {self.text_prompt}')
        elif self.num_keywords == -3:   # for ablation
            text_embed = image_embed
            self.text_embed = torch.rand((10, text_embed))
            # print(f'text embed: {self.text_embed.shape}')
        elif self.num_keywords == -4:   # for ablation
            text_embed = image_embed
            self.text_embed_shape = (10, text_embed)
            # print(f'text embed shape: {self.text_embed_shape}')
        else:
            raise NotImplementError
        
        keypoints_anno_file = keypoints_anno_file
        self.keypoints = None
        if keypoints_anno_file:
            assert os.path.exists(keypoints_anno_file), f'{keypoints_anno_file} not find'
            with open(keypoints_anno_file, "r") as f:
                self.keypoints = yaml.safe_load(f)
        assert self.num_keywords <= 1 or keypoints_anno_file, \
            f'Need keypoints_anno_file when num_keywords > 1'
        
        self.text_embed_dim = text_embed
        self.text_embed_proj = nn.Linear(text_embed, image_embed) \
            if text_embed != image_embed else nn.Identity()

    def _init_image_text_matcher(self, matcher_type='attn', **kwargs):
        self.matcher = None
        if matcher_type == 'attn':
            self.matcher = AttentionMatcher(**kwargs)
        elif matcher_type == 'concat_attn':
            self.matcher = ConcatAttentionMatcher(**kwargs)
        elif matcher_type == 'cosmax_attn':
            self.matcher = CosMaxAttentionMatcher(**kwargs)
        elif matcher_type == 'cosmean_attn':
            self.matcher = CosMeanAttentionMatcher(**kwargs)
        elif matcher_type == 'coscat_attn':
            self.matcher = CosCatAttentionMatcher(**kwargs)
        elif matcher_type == 'catcat_attn':
            self.matcher = CatCatAttentionMatcher(**kwargs)
        elif matcher_type == 'coscatcat_attn':
            self.matcher = CosCatCatAttentionMatcher(**kwargs)
        else:
            raise NotImplementError(matcher_type)

    def extract_text_feat(self, sample: SampleList) -> Tensor:
        '''sample: {"image": image, "text_input": text_input}'''
        if 'clip' in self.text_encoder_name:
            sample["image"] = None
            text_embed = self.text_encoder.extract_features(sample).unsqueeze(1)
        else:
            text_embed = self.text_encoder.extract_features(sample, mode="text").text_embeds
        return text_embed

    def extract_image_feat(self, inputs: Tensor) -> Tensor: 
        return super().extract_feat(inputs)

    def extract_feat(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        # text feats
        B = inputs.shape[0]
        styles = [sample.img_path.split('/')[-2] for sample in data_samples]
        
        if self.num_keywords == -4:
            text_embed = torch.rand(self.text_embed_shape).expand(B, -1, -1).cuda() # torch.Size([64, 10, 768])
        
        elif self.num_keywords == -3:
            text_embed = self.text_embed.expand(B, -1, -1).cuda() # torch.Size([64, 10, 768])
        
        elif self.num_keywords in [-1, -2]:
            assert isinstance(self.text_prompt, str), \
                f'num_keywords = {self.text_prompt} < 0 is only for simple text prompt'
            text_input = [self.text_prompt for _ in data_samples]
            sample = {'image': inputs, 'text_input': text_input}
            text_embed = self.extract_text_feat(sample)               # torch.Size([64, 10, 768])
        
        elif self.num_keywords == 0:
            assert isinstance(self.text_prompt, dict), 'Not support when text_prompt_file==None'
            text_input = [sample.description for sample in data_samples]
            sample = {'image': inputs, 'text_input': text_input}
            text_embed = self.extract_text_feat(sample)               # torch.Size([64, 10, 768])
            
        elif self.num_keywords == 1:
            if isinstance(self.text_prompt, str):
                text_input = [self.text_prompt for _ in data_samples]
            elif isinstance(self.text_prompt, dict):
                text_input = [self.text_prompt[style] for style in styles]
            else:
                raise NotImplementError(self.text_prompt)
            sample = {'image': inputs, 'text_input': text_input}
            text_embed = self.extract_text_feat(sample)               # torch.Size([64, 10, 768])
            
        elif self.num_keywords == 2:
            assert isinstance(self.text_prompt, dict), 'Not support when text_prompt_file==None'
            text_input = []
            for style in styles:
                for k in self.keypoints:
                    text_input.append(self.text_prompt[style].replace('human', f'human\'s {k}'))
            sample = {'image': inputs, 'text_input': text_input}
            text_embed = self.extract_text_feat(sample)               # torch.Size([64*17, 13, 768])
            N, L, C = text_embed.shape
            text_embed = text_embed.reshape(-1, len(self.keypoints), L, C)  # torch.Size([64, 17, 13, 768])
            text_embed = text_embed.reshape(-1, len(self.keypoints)*L, C)   # torch.Size([64, 17*13, 768])
            
        elif self.num_keywords >= 5:
            text_input = styles
            sample = {'image': inputs, 'text_input': text_input}
            text_embed = self.extract_text_feat(sample)               # torch.Size([64*17, 13, 768])
            
        else:
            raise NotImplementError(f'num_keywords={self.num_keywords}')

        text_embed = self.text_embed_proj(text_embed)              # torch.Size([64, 10, 384])
        
        # image feats
        image_feats = self.extract_image_feat(inputs)  # [torch.Size([64, 384, 16, 12])]
        feats = []
        for image_feat in image_feats:
            B, C, H, W = image_feat.shape
            image_embed = image_feat.reshape(B, C, H*W).permute(0, 2, 1) # torch.Size([64, 192, 384])
            
            # image-text feats
            feat = self.matcher(image_embed=image_embed, text_embed=text_embed) # torch.Size([64, 192, 384])
            feat = feat.permute(0, 2, 1).reshape(B, -1, H, W)    # [torch.Size([64, 384, 16, 12])]
            feats.append(feat)
        return image_feats, feats
         
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        image_feats, feats = self.extract_feat(inputs, data_samples)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            image_feats, _feats = self.extract_feat(inputs, data_samples)
            image_feats_flip, _feats_flip = self.extract_feat(inputs.flip(-1), data_samples)
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs, data_samples)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
