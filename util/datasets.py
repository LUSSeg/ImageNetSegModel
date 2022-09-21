# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

import util.transforms as custom_transforms


def build_dataset(is_train, args):
    transform = build_transform(is_train)
    data_root = os.path.join(args.data_path,
                             'train-semi' if is_train else 'validation')
    gt_root = os.path.join(
        args.data_path,
        'train-semi-segmentation' if is_train else 'validation-segmentation')
    dataset = SegDataset(data_root, gt_root, transform, is_train)
    return dataset


def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        randomresizedcrop = custom_transforms.RandomResizedCrop(
            224,
            scale=(0.14, 1),
        )
        transform = custom_transforms.Compose([
            randomresizedcrop,
            custom_transforms.RandomHorizontalFlip(p=0.5),
            transforms.Compose(color_transform),
            custom_transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform

    # eval transform
    t = []
    t.append(transforms.Resize(256))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class SegDataset(datasets.ImageFolder):
    def __init__(self, data_root, gt_root=None, transform=None, is_train=True):
        super(SegDataset, self).__init__(data_root)
        assert gt_root is not None
        self.gt_root = gt_root
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        segmentation = self.load_segmentation(path)

        if self.is_train:
            image, segmentation = self.transform(image, segmentation)
        else:
            image = self.transform(image)
            segmentation = torch.from_numpy(np.array(segmentation))
            segmentation = segmentation.long()

        segmentation = segmentation[:, :, 1] * 256 + segmentation[:, :, 0]
        return image, segmentation

    def load_segmentation(self, path):
        cate, name = path.split('/')[-2:]
        name = name.replace('JPEG', 'png')
        path = os.path.join(self.gt_root, cate, name)
        segmentation = Image.open(path)
        return segmentation


class PILRandomGaussianBlur(object):
    """Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)))


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
