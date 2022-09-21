# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import warnings
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ReduceOp

import util.lr_sched as lr_sched
import util.misc as misc
from util.metric import FMeasureGPU, IoUGPU


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            outputs = torch.nn.functional.interpolate(outputs,
                                                      scale_factor=2,
                                                      align_corners=False,
                                                      mode='bilinear')
            targets = torch.nn.functional.interpolate(
                targets.unsqueeze(1),
                size=(outputs.shape[2], outputs.shape[3]),
                mode='nearest').squeeze(1)
            loss = criterion(outputs, targets.long())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group['lr'])

        metric_logger.update(lr=max_lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, max_res=1000):
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Test:'

    T = torch.zeros(size=(num_classes, )).cuda()
    P = torch.zeros(size=(num_classes, )).cuda()
    TP = torch.zeros(size=(num_classes, )).cuda()
    IoU = torch.zeros(size=(num_classes, )).cuda()
    FMeasure = 0.

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(images)

            # process an image with a large resolution
            H, W = target.shape[1], target.shape[2]
            if (H > W and H * W > max_res * max_res
                    and max_res > 0):
                output = F.interpolate(output, (max_res, int(max_res * W / H)),
                                       mode='bilinear',
                                       align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)
                output = F.interpolate(output.float(), (H, W),
                                       mode='nearest').long()
            elif (H <= W and H * W > max_res * max_res
                  and max_res > 0):
                output = F.interpolate(output, (int(max_res * H / W), max_res), 
                                       mode='bilinear', align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)
                output = F.interpolate(output.float(), (H, W),
                                       mode='nearest').long()
            else:
                output = F.interpolate(output, (H, W),
                                       mode='bilinear',
                                       align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)

        target = target.view(-1)
        output = output.view(-1)
        mask = target != 1000
        target = target[mask]
        output = output[mask]

        area_intersection, area_output, area_target = IoUGPU(
            output, target, num_classes)
        f_score = FMeasureGPU(output, target)

        T += area_output
        P += area_target
        TP += area_intersection
        FMeasure += f_score

    metric_logger.synchronize_between_processes()

    # gather the stats from all processes
    dist.barrier()
    dist.all_reduce(T, op=ReduceOp.SUM)
    dist.all_reduce(P, op=ReduceOp.SUM)
    dist.all_reduce(TP, op=ReduceOp.SUM)
    dist.all_reduce(FMeasure, op=ReduceOp.SUM)

    IoU = TP / (T + P - TP + 1e-10) * 100
    FMeasure = FMeasure / len(data_loader.dataset)

    mIoU = torch.mean(IoU).item()
    FMeasure = FMeasure.item() * 100

    log = {}
    log['mIoU'] = mIoU
    log['IoUs'] = IoU.tolist()
    log['FMeasure'] = FMeasure

    print('* mIoU {mIoU:.3f} FMeasure {FMeasure:.3f}'.format(
        mIoU=mIoU, FMeasure=FMeasure))

    return log
