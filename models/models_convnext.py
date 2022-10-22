from functools import partial

import timm.models.convnext
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ConvNeXt(timm.models.convnext.ConvNeXt):
    """Vision Transformer with support for semantic seg."""
    def __init__(self, patch_size=4, **kwargs):
        norm_layer = kwargs.pop('norm_layer')
        super(ConvNeXt, self).__init__(**kwargs)
        assert self.num_classes > 0

        del self.head
        del self.norm_pre

        self.patch_size = patch_size
        self.depths = kwargs['depths']
        self.num_layers = sum(self.depths) + len(self.depths)
        self.rf_change = []

        self.seg_norm = norm_layer(self.num_features)
        self.seg_head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Conv2d(self.num_features, self.num_classes * (self.patch_size**2), 1))
        ]))

        trunc_normal_(self.seg_head[1].weight, std=.02)
        torch.nn.init.zeros_(self.seg_head[1].bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.seg_norm(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

    def forward_head(self, x):
        x = self.seg_head.drop(x)
        x = self.seg_head.fc(x)
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h, w, self.patch_size, self.patch_size, self.num_classes)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.contiguous().view(b, self.num_classes, h * self.patch_size, w * self.patch_size)
        return x
    
    def get_layer_id(self, name):
        if name in ("cls_token", "mask_token", "pos_embed"):
            return 0, 0
        elif name.startswith("stem"):
            return 0, 0
        elif name.startswith("stages") and 'downsample' in name:
            stage_id = int(name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            else:
                layer_id = sum(self.depths[:stage_id]) + stage_id
            return 0, layer_id
        elif name.startswith("stages") and 'downsample' not in name:
            stage_id = int(name.split('.')[1])
            block_id = int(name.split('.')[3])
            if stage_id == 0:
                layer_id = block_id + 1
            else:
                layer_id = sum(self.depths[:stage_id]) + stage_id + block_id + 1
            return 0, layer_id
        else:
            return 0, self.num_layers


def convnext_tiny(args):
    model = ConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768), 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        num_classes=getattr(args, 'nb_classes', 920),
        drop_path_rate=getattr(args, 'drop_path', 0),
        patch_size=getattr(args, 'patch_size', 4)
    )
    return model
