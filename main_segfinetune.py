# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from timm.models.layers import trunc_normal_

import models
import util.lr_decay as lrd
import util.misc as misc
from engine_segfinetune import evaluate, train_one_epoch
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
from timm.models.convnext import checkpoint_filter_fn


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Semi-supervised fine-tuning for '
        'the semantic segmentation on the ImageNet-S dataset',
        add_help=False)
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch size per GPU '
        '(effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations '
        '(for increasing the effective batch size under memory constraints)')
    parser.add_argument('--saveckp_freq',
                        default=20,
                        type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--eval_freq',
                        default=20,
                        type=int,
                        help='Evaluate the model every x epochs.')
    parser.add_argument(
        '--max_res',
        default=1000,
        type=int,
        help='Maximum resolution for evaluation. 0 for disable.')

    # Model parameters
    parser.add_argument('--model',
                        default='vit_small_patch16',
                        type=str,
                        metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path',
                        type=float,
                        default=0.1,
                        metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--patch_size',
                        type=int,
                        default=4,
                        help='For convnext/rfconvnext, the numnber of output channels is '
                        'nb_classes * patch_size * patch_size.'
                        'https://arxiv.org/pdf/2111.06377.pdf')

    # Optimizer parameters
    parser.add_argument('--clip_grad',
                        type=float,
                        default=None,
                        metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr',
                        type=float,
                        default=None,
                        metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr',
                        type=float,
                        default=1e-3,
                        metavar='LR',
                        help='base learning rate: '
                        'absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay',
                        type=float,
                        default=[0.75],
                        nargs="+",
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--layer_multiplier',
                        type=float,
                        default=[1.0],
                        nargs="+",
                        help='multiplier of learning rate for each group of layers')
    parser.add_argument('--min_lr',
                        type=float,
                        default=1e-6,
                        metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=5,
                        metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=None,
        metavar='PCT',
        help='Color jitter factor (enabled only when not using Auto/RandAug)')

    # * Finetuning params
    parser.add_argument('--finetune',
                        default='',
                        help='finetune from checkpoint')
    parser.add_argument('--pretrained_rfnext',
                        default='',
                        help='pretrained weights for RF-Next')

    # Dataset parameters
    parser.add_argument('--data_path',
                        default='/datasets01/imagenet_full_size/061417/',
                        type=str,
                        help='dataset path')
    parser.add_argument('--iteration_one_epoch',
                        default=-1,
                        type=int,
                        help='number of iterations in one epoch')
    parser.add_argument('--nb_classes',
                        default=1000,
                        type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir',
                        default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval',
                        action='store_true',
                        default=False,
                        help='Enabling distributed evaluation '
                        '(recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True)
        print('Sampler_train = %s' % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation '
                    'with an eval dataset not divisible by process number. '
                    'This will slightly alter validation '
                    'results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  sampler=sampler_val,
                                                  batch_size=1,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  drop_last=False)
    args.iteration_one_epoch = len(data_loader_train)
    model = models.__dict__[args.model](args)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print('Load pre-trained checkpoint from: %s' % args.finetune)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {
            k.replace('module.', ''): v
            for k, v in checkpoint.items()
        }
        checkpoint = {
            k.replace('backbone.', ''): v
            for k, v in checkpoint.items()
        }

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint.keys():
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint[k]

        if 'vit' in args.model:
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint)
        elif 'convnext' in args.model:
            checkpoint = checkpoint_filter_fn(checkpoint, model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint, strict=False)
        print('Missing: {}'.format(msg.missing_keys))

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print('Model = %s' % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)

    print('accumulate grad iterations: %d' % args.accum_iter)
    print('effective batch size: %d' % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        layer_multiplier=args.layer_multiplier)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=1000)  # 1000 denotes the ignored region in ImageNet-S.
    print('criterion = %s' % str(criterion))

    misc.load_model(args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.nb_classes)
        print(f'mIoU of the network on the {len(dataset_val)} '
              f"test images: {test_stats['mIoU']:.1f}%")
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation '
                  'with an eval dataset not divisible by process number. '
                  'This will slightly alter validation '
                  'results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        exit(0)

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      criterion,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      args.clip_grad,
                                      args=args)
        if args.output_dir and (epoch + 1) % args.saveckp_freq == 0:
            misc.save_model(args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            test_stats = evaluate(data_loader_val,
                                  model,
                                  device,
                                  args.nb_classes,
                                  max_res=args.max_res)
        print(f'mIoU of the network on the {len(dataset_val)} '
              f"test images: {test_stats['mIoU']:.3f}%")
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation '
                  'with an eval dataset not divisible by process number. '
                  'This will slightly alter validation '
                  'results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        max_accuracy = max(max_accuracy, test_stats['mIoU'])
        print(f'Max mIoU: {max_accuracy:.2f}%')

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'),
                      mode='a',
                      encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
