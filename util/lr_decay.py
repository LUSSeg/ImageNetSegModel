# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


def param_groups_lrd(model,
                     weight_decay=0.05,
                     no_weight_decay_list=[],
                     layer_decay=[.75],
                     layer_multiplier=[1.0]):
    """Parameter groups for layer-wise lr decay Following BEiT: https://github.

    com/microsoft/unilm/blob/master/beit/optim_factory.py#L58.
    """
    param_group_names = {}
    param_groups = {}

    num_layers = model.num_layers

    if isinstance(layer_decay, (float, int)):
        layer_decay = [layer_decay]

    layer_scales = [
        list(decay**(num_layers - i) for i in range(num_layers + 1)) for decay in layer_decay]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = 'no_decay'
            this_decay = 0.
        else:
            g_decay = 'decay'
            this_decay = weight_decay

        """
        For each layer, the get_layer_id returns (layer_group, layer_id). 
        According to the layer_group, different parameters are grouped, 
        and layers in different groups use different decay rates.

        If only a layer_id is returned, the layer_group are set to 0 by default.
        """
        layer_group_id = model.get_layer_id(n)
        if isinstance(layer_id, (list, tuple)):
            layer_group, layer_id = layer_group_id
        elif isinstance(layer_id, int):
            layer_group, layer_id = 0, layer_group_id
        else:
            raise NotImplementedError()
        group_name = 'layer_%d_%d_%s' % (layer_group, layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_group][layer_id] * layer_multiplier[layer_group]

            param_group_names[group_name] = {
                'lr_scale': this_scale,
                'weight_decay': this_decay,
                'params': [],
            }
            param_groups[group_name] = {
                'lr_scale': this_scale,
                'weight_decay': this_decay,
                'params': [],
            }

        param_group_names[group_name]['params'].append(n)
        param_groups[group_name]['params'].append(p)

    return list(param_groups.values())
