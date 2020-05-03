#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:55:03 2019

@author: sumche
"""
# Source: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
# from utils.data_utils import up_scale_tensors


def flatten_data(input, target):
    target = target.long()
    n, c, h, w = input.size()
    if len(target.size()) == 2 and n == 1:
        ht, wt = target.size()
    elif len(target.size()) == 3 and n > 1:
        nt, ht, wt = target.size()
    else:
        raise ValueError('Check size of inputs and targets')

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt),
                              mode="bilinear", align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    return input, target


def cross_entropy2d(input, target, weight=None, size_average=True):
    input, target = flatten_data(input, target)
    loss = F.cross_entropy(input, target,
                           weight=weight, ignore_index=255)
    return loss


def weighted_binary_cross_entropy(input, target):
    input, target = flatten_data(input, target)
    n_classes = input.shape[1]
    mean_loss = 0.0
    for class_id in range(n_classes):
        label = (target == class_id).float()
        mask = get_weight_mask(label)
        prediction = input[:, class_id, ...]
        loss = F.binary_cross_entropy_with_logits(prediction, label, mask)
        mean_loss += loss

    #nmsloss = StealNMSLoss()
    #ln = nmsloss.__call__(target_onehot, input_soft)

    return mean_loss  # +ln


def get_weight_mask(target):
    mask = (target != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / mask.numel()
    mask[mask == 0] = num_positive / mask.numel()
    return mask


def make_one_hot(labels, num_classes=10):
    n, h, w = labels.size()
    one_hot = torch.zeros((n, num_classes, h, w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id, ...] = (labels == class_id+1)
    return one_hot.to(labels.get_device())


def dice_loss(input_soft, target, eps=1e-8):
    if not torch.is_tensor(input_soft):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input_soft)))

    if not len(input_soft.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input_soft.shape))

    if not input_soft.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} instead of {}"
                         .format(input_soft.shape, target.shape))

    if not input_soft.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input_soft.device, target.device))

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target, dims)
    cardinality = torch.sum(input_soft + target, dims)

    dice_score = 2. * intersection/(cardinality + eps)
    return torch.mean(-dice_score + 1.)


def huber_loss(input, target, weight=None, size_average=True):
    if input.size() != target.size():
        input = input.permute(0, 2, 3, 1).squeeze()
    return F.smooth_l1_loss(input, target)


def mae_loss(input, target, weight=None, size_average=True):
    if input.size() != target.size():
        input = input.permute(0, 2, 3, 1).squeeze()
    return F.l1_loss(input, target)


def mse_loss(input, target, weight=None, size_average=True):
    if input.size() != target.size():
        input = input.permute(0, 2, 3, 1).squeeze()
    return F.mse_loss(input, target)
