# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpose.registry import MODELS
from .vit import VisionTransformer
from .base_backbone import BaseBackbone
from mmpretrain.models.utils import MultiheadAttention, resize_pos_embed, to_2tuple

    

@MODELS.register_module()
class ViTPrompt(VisionTransformer):
    def __init__(self, prompt_config=dict(deep=False, num_tokens=5), **kwargs):
        super(ViTPrompt, self).__init__(**kwargs)
        self.prompt_config = prompt_config
        self.num_tokens = self.prompt_config['num_tokens']
        self.prompt_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dims))
        trunc_normal_(self.prompt_embed, std=.02)
        if self.prompt_config['deep']:
            self.deep_prompt_embed = nn.Parameter(torch.zeros(
                self.num_layers - 1, self.num_tokens, self.embed_dims))
            trunc_normal_(self.deep_prompt_embed, std=.02)
        assert self.with_cls_token == False, \
            'We do not support with_cls_token. Please set \'with_cls_token = False\''
        if self.frozen_stages != len(self.layers):
            print('Please set \'self.frozen_stages = len(self.layers)\' to freeze the orignal layers')
        self._freeze_stages()
        
    def count_parameters(self, require_grad=False):
        if require_grad:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
  
    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)
        x = self.pre_norm(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]
            
        # Add prompt after adding position embedding
        x_ = torch.cat((
                self.prompt_embed.expand(B, -1, -1),
                x, 
            ), dim=1)
                    
        outs = []
        for i, layer in enumerate(self.layers):
            # Add deep prompt before each intermediate layer
            if self.prompt_config['deep']:
                if i > 0:
                    x_ = torch.cat((
                        self.deep_prompt_embed[i-1].expand(B, -1, -1),
                        x_, 
                    ), dim=1)     
                x_ = layer(x_)
                x = x_ = x_[:, self.num_tokens:, :]
            else:
                x_ = layer(x_)
                x = x_[:, self.num_tokens:, :]
                
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.avg_token:
                    patch_token = patch_token.permute(0, 2, 3, 1)
                    patch_token = patch_token.reshape(
                        B, patch_resolution[0] * patch_resolution[1],
                        C).mean(dim=1)
                    patch_token = self.norm2(patch_token)
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)
        return tuple(outs)

    # def train(self, mode=True):
    #     if mode:
    #         # training: disable all but the prompt-related modules
    #         # since we do not use norm or dropout layer here, there is no prompt-related modules
    #         self.patch_embed.eval()
    #         self.drop_after_pos.eval()
    #         self.layers.eval()
    #         if self.final_norm:
    #             self.ln1.eval()
    #         if self.avg_token:
    #             self.ln2.eval()
    #         self.pre_norm.eval()
               
    #     else:
    #         # eval:
    #         for module in self.children():
    #             module.train(mode)