#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:48:08 2019

@author: sumche
"""
import random
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision import transforms


class RandomScale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        assert isinstance(scale, tuple)
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return self.random_scale(img)

    def random_scale(self, image):

        scale = random.randint(int(self.scale[0]*2), int(self.scale[1]*2))/2.0
        h, w = int(image.size[-2]*scale), int(image.size[-1]*scale)
        # print(scale,(w,h))
        image = TF.resize(image, (w, h), self.interpolation)
        return image


def get_transforms(cfg):
    all_transforms = {}
    for k, v in cfg.items():
        all_transforms[k] = {}
        for key in v.keys():
            all_transforms[k][key] = transforms.Compose(
                get_transforms_list(cfg[k][key]))

    return all_transforms


def get_transforms_list(cfg):
    # cfg['data']['transforms']['train']['input']
    transform_list = []
    for transform_name, params in cfg.items():
        if params['flag']:
            transform_list.append(
                get_transform_from_name(transform_name, params))
    return transform_list


def get_transform_from_name(transform_name, params):

    if transform_name == 'Resize':
        return transforms.Resize(size=params['size'], interpolation=Image.NEAREST)

    elif transform_name == 'RandomScale':
        return RandomScale(scale=params['scale'])

    elif transform_name == 'RandomCrop':
        if isinstance(params['size'], tuple):
            return transforms.RandomCrop(size=params['size'])
        else:
            return transforms.RandomCrop(size=(params['size'], 2*params['size']))

    elif transform_name == 'RandomHorizontalFlip':
        return transforms.RandomHorizontalFlip(p=0.5)

    elif transform_name == 'FiveCrop':
        return transforms.FiveCrop(size=params['size'])

    elif transform_name == 'RandomResizedCrop':
        return transforms.RandomResizedCrop(size=params['size'],
                                            scale=params['scale'])

    elif transform_name == 'ColorJitter':
        return transforms.ColorJitter(brightness=params['brightness'],
                                      contrast=params['contrast'],
                                      saturation=params['saturation'],
                                      hue=params['hue'])

    elif transform_name == 'ToTensor':
        return transforms.ToTensor()

    elif transform_name == 'Normalize':
        return transforms.Normalize(mean=params['mean'], std=params['mean'])
