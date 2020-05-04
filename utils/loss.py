#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:55:03 2019

@author: sumche
"""
# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import logging

from models import StealNMSLoss
from utils.constants import LENGTH


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
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
    target = target.view(-1)

    return input, target


def cross_entropy2d(input, target, weights=None, size_average=True):
    input, target = flatten_data(input, target)
    loss = F.cross_entropy(input, target,
                           weight=weights, ignore_index=255)
    return loss


def weighted_binary_cross_entropy(input, target, weights=None):
    input, target = flatten_data(input, target)
    n_classes = input.shape[1]
    mean_loss = 0.0
    for class_id in range(n_classes):
        label = (target == class_id).float()
        mask = get_weight_mask(label)
        prediction = input[:, class_id, ...]
        loss = F.binary_cross_entropy_with_logits(prediction, label, mask)
        mean_loss += loss * weights[class_id] if weights is not None else loss
    return mean_loss


class CrossEntropy2D(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        super(CrossEntropy2D, self).__init__()
        self.weight = class_weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input, target = flatten_data(input, target)
        weight = self.weight.cuda()
        return F.cross_entropy(input, target, weight,
                               ignore_index=self.ignore_index)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, input, target, weight):
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weight)
        return loss


class MultiLabelLoss(nn.Module):
    def __init__(self, class_weights, nms=False):
        super(MultiLabelLoss, self).__init__()
        self.pos_weight = class_weights
        self.bce = WeightedBCELoss()
        if nms:
            self.nms = StealNMSLoss()

    def forward(self, input, target):
        n_classes = input.shape[1]
        target = make_one_hot(target, n_classes)
        mask = self.get_weight_mask(target, n_classes)
        bce_loss = self.bce(input.view(-1), target.view(-1), mask)
        if self.nms:
            input = torch.softmax(input, dim=1)
            nms_loss = self.nms(input, target)
        logging.info('bce_loss: {}, nms_loss: {}'
                     .format(bce_loss, nms_loss).center(LENGTH, '='))
        return 10*bce_loss + nms_loss

    def get_weight_mask(self, target, n_classes):
        n, _, h, w = target.size()
        beta = torch.zeros((n, n_classes, h, w))
        device = target.get_device()
        for i in range(n_classes):
            mask = (target[:, i, ...] != 0).float()
            num_positive = torch.sum(mask).float()
            num_negative = mask.numel() - num_positive
            mask[mask != 0] = \
                (num_negative * self.pos_weight[i]) / mask.numel()
            mask[mask == 0] = num_positive / mask.numel()
            beta[:, i, ...] = mask
        beta = beta.view(-1).cuda()
        return beta


def weighted_binary_cross_entropy_with_nms(input, target, weights=None):
    mean_loss = weighted_binary_cross_entropy(input, target, weights=weights)
    nmsloss = StealNMSLoss()
    n_classes = input.shape[1]
    target = make_one_hot(target, n_classes)
    input = torch.softmax(input, dim=1)
    mean_ln = nmsloss.__call__(input, target)
    return mean_loss + mean_ln


def make_one_hot(labels, num_classes=10):
    n, h, w = labels.size()
    one_hot = torch.zeros((n, num_classes, h, w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id, ...] = (labels == class_id+1)
    return one_hot.cuda()


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
