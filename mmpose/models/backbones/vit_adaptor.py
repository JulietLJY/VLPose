# Copyright (c) OpenMMLab. All rights reserved.
# this code is copy from https://github.com/open-mmlab/mmpretrain/blob/1.x/mmcls/models/backbones/vision_transformer.py
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpose.registry import MODELS
from mmpretrain.models.utils import MultiheadAttention, resize_pos_embed, to_2tuple
from .base_backbone import BaseBackbone
from .vit import TransformerEncoderLayer, VisionTransformer


class TransformerEncoderLayer_ADPT(TransformerEncoderLayer):
    def __init__(self, adapter_config=dict(reduction_factor=8), **kwargs):
        super(TransformerEncoderLayer_ADPT, self).__init__(**kwargs)
        self.reduction_factor = adapter_config['reduction_factor']
        self.adapter_downsample = nn.Linear(
                self.embed_dims,
                self.embed_dims // self.reduction_factor
            )
        self.adapter_upsample = nn.Linear(
            self.embed_dims // self.reduction_factor,
            self.embed_dims
        )
        self.adapter_act_fn = build_activation_layer(dict(type='GELU'))
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        x = x + self.adapter_upsample(self.adapter_act_fn(self.adapter_downsample(x)))
        return x


@MODELS.register_module()
class ViTADPT(VisionTransformer):
    def __init__(self, 
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 avg_token=False,
                 frozen_stages=-1,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None, 
                 adapter_config=dict(reduction_factor=8)):
        super(ViTADPT, self).__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            with_cls_token=with_cls_token,
            avg_token=avg_token,
            frozen_stages=frozen_stages,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            pre_norm=pre_norm,
            init_cfg=init_cfg,
        )
        
        assert self.frozen_stages == self.num_layers, \
            'Please set \'frozen_stages = num_layers\''
            
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        del self.layers
        
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
            
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                adapter_config=adapter_config)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer_ADPT(**_layer_cfg))

        if self.frozen_stages > 0:
            self._freeze_stages()

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.attn.eval()
            m.norm1.eval()
            m.ffn.eval()
            m.norm2.eval()
            for param in m.attn.parameters():
                param.requires_grad = False
            for param in m.norm1.parameters():
                param.requires_grad = False
            for param in m.ffn.parameters():
                param.requires_grad = False
            for param in m.norm2.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.norm1.eval()
            for param in self.norm1.parameters():
                param.requires_grad = False
