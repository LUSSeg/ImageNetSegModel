from functools import partial

import timm.models.rfconvnext
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class RFConvNeXt(timm.models.rfconvnext.RFConvNeXt):
    """Vision Transformer with support for semantic seg."""
    def __init__(self, patch_size=4, **kwargs):
        norm_layer = kwargs.pop('norm_layer')
        super(RFConvNeXt, self).__init__(**kwargs)
        assert self.num_classes > 0

        del self.head
        del self.norm_pre

        self.patch_size = patch_size
        self.depths = kwargs['depths']
        self.num_layers = sum(self.depths) + len(self.depths)
        # The layers whose dilation rates are changed in RF-Next.
        # These layers use different hyper-parameters in training.
        self.rf_change = []
        self.rf_change_name = []

        self.seg_norm = norm_layer(self.num_features)
        self.seg_head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Conv2d(self.num_features, self.num_classes * (self.patch_size**2), 1))
        ]))
    
        trunc_normal_(self.seg_head[1].weight, std=.02)
        torch.nn.init.zeros_(self.seg_head[1].bias)

        self.get_kernel_size_changed()
        
    
    def get_kernel_size_changed(self):
        """
        To get rfconvs whose dilate rates are changed. 
        """
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage.blocks):
                if block.conv_dw.dilation[0] > 1 or block.conv_dw.kernel_size[0] > 13:
                    self.rf_change_name.extend(
                        [
                            'stages.{}.blocks.{}.conv_dw.weight'.format(i, j),
                            'stages.{}.blocks.{}.conv_dw.bias'.format(i, j),
                            'stages.{}.blocks.{}.conv_dw.sample_weights'.format(i, j)
                        ]
                    )
                    self.rf_change.append(self.stages[i].blocks[j].conv_dw)

    def freeze(self):
        """
        In the mode of rfmerge, 
        we initilize the model with weights in rfmultiple and 
        only finetune seg_norm, seg_head and rfconvs whose dilate rates are changed. 
        The other parts of the network are freezed during funetuning.

        Note that this freezing operation may be not required for other tasks.
        """
        if len(self.rf_change_name) == 0:
            self.get_kernel_size_changed()
        # finetune the rfconvs whose dilate rates are changed
        for n, p in self.named_parameters():
            p.requires_grad = True if n in self.rf_change_name else False
        # finetune the seg_norm, seg_head
        for n, p in self.seg_head.named_parameters():
            p.requires_grad = True
        for n, p in self.seg_norm.named_parameters():
            p.requires_grad = True

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
            return (0, 0)
        elif name.startswith("stem"):
            return (0, 0)
        elif name.startswith("stages") and 'downsample' in name:
            stage_id = int(name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            else:
                layer_id = sum(self.depths[:stage_id]) + stage_id
            
            if name.endswith('sample_weights') or name in self.rf_change_name:
                return (1, layer_id)
            return (0, layer_id)
        elif name.startswith("stages") and 'downsample' not in name:
            stage_id = int(name.split('.')[1])
            block_id = int(name.split('.')[3])
            if stage_id == 0:
                layer_id = block_id + 1
            else:
                layer_id = sum(self.depths[:stage_id]) + stage_id + block_id + 1

            if name.endswith('sample_weights') or name in self.rf_change_name:
                return (1, layer_id)
            return (0, layer_id)
        else:
            return (0, self.num_layers)


def rfconvnext_tiny_rfsearch(args):
    search_cfgs = dict(
        samples=3,
        expand_rate=0.5,
        max_dilation=None,
        min_dilation=1,
        init_weight=0.01,
        search_interval=getattr(args, 'iteration_one_epoch', 1250) * 10, # step every 10 epochs
        max_search_step=3, # search for 3 steps
    )
    model = RFConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768), 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        rf_mode='rfsearch', 
        search_cfgs=search_cfgs, 
        num_classes=getattr(args, 'nb_classes', 920),
        drop_path_rate=getattr(args, 'drop_path', 0),
        pretrained_weights=getattr(args, 'pretrained_rfnext', None),
        patch_size=getattr(args, 'patch_size', 4)
    )
    return model


def rfconvnext_tiny_rfmultiple(args):
    search_cfgs = dict(
        samples=3,
        expand_rate=0.5,
        max_dilation=None,
        min_dilation=1,
        init_weight=0.01,
    )
    model = RFConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768), 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        rf_mode='rfmultiple', 
        search_cfgs=search_cfgs,  
        num_classes=getattr(args, 'nb_classes', 920),
        drop_path_rate=getattr(args, 'drop_path', 0),
        pretrained_weights=getattr(args, 'pretrained_rfnext', None),
        patch_size=getattr(args, 'patch_size', 4)
    )
    return model

def rfconvnext_tiny_rfsingle(args):
    search_cfgs = dict(
        samples=3,
        expand_rate=0.5,
        max_dilation=None,
        min_dilation=1,
        init_weight=0.01,
    )
    model = RFConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768), 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        rf_mode='rfsingle', 
        search_cfgs=search_cfgs, 
        num_classes=getattr(args, 'nb_classes', 920),
        drop_path_rate=getattr(args, 'drop_path', 0),
        pretrained_weights=getattr(args, 'pretrained_rfnext', None),
        patch_size=getattr(args, 'patch_size', 4)
    )
    return model

def rfconvnext_tiny_rfmerge(args):
    search_cfgs = dict(
        samples=3,
        expand_rate=0.5,
        max_dilation=None,
        min_dilation=1,
        init_weight=0.01
    )
    model = RFConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768), 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        rf_mode='rfmerge', 
        search_cfgs=search_cfgs, 
        num_classes=getattr(args, 'nb_classes', 920),
        drop_path_rate=getattr(args, 'drop_path', 0),
        pretrained_weights=getattr(args, 'pretrained_rfnext', None),
        patch_size=getattr(args, 'patch_size', 4)
    )
    # freeze layers except for seg_norm, seg_head and the rfconvs whose dialtion rates are changed.
    model.freeze()
    return model
