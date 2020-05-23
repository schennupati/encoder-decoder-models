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
import numpy as np
import time
import logging
from kornia.filters import spatial_gradient, filter2D
from torch.autograd import Variable


from utils.constants import LENGTH


class WeightedMultiClassBinaryCrossEntropy(nn.Module):

    def __init__(self, huber_active=True):
        super(WeightedMultiClassBinaryCrossEntropy, self).__init__()
        self.bce = WeightedBinaryCrossEntropy(huber_active)

    def forward(self, input, target):
        loss = 0.0
        num_classes = input.shape[1]
        for i in range(num_classes):
            weighted = True if i != 0 else False
            loss += self.bce(input[:, i, ...], (target == i), weighted)
        return loss


class WeightedBinaryCrossEntropy(nn.Module):

    def __init__(self, huber_active=True):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.huber_active = huber_active

    def forward(self, input, target, weighted=True):
        n_batch = input.shape[0]
        mean_bce = 0.0
        mean_l1 = 0.0
        for _id in range(n_batch):
            if weighted:
                bce_loss = \
                    F.binary_cross_entropy_with_logits(input[_id].squeeze(),
                                                       target[_id].float(),
                                                       get_weight_mask(
                                                       target[_id].float()),
                                                       size_average=True)
            else:
                bce_loss = \
                    F.binary_cross_entropy_with_logits(input[_id].squeeze(),
                                                       target[_id].float(),
                                                       size_average=True)
            mean_bce += bce_loss

            if self.huber_active:
                l1_loss = huber_loss(torch.sigmoid(input[_id].squeeze()),
                                     target[_id].float(), delta=0.3)
                mean_l1 += l1_loss

        mean_bce /= n_batch
        mean_l1 /= n_batch
        return mean_l1 + 10*mean_bce


class FocalLoss(nn.Module):

    def __init__(self, ignore_index=255, gamma=1):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, input, target):
        n, c, h, w = input.size()
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target, weight=None,
                                  ignore_index=self.ignore_index.cuda(),
                                  reduction='none')
        focal_loss = ((1 - F.softmax(input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, c))**self.gamma).squeeze() * ce_loss
        return focal_loss.mean()


class DualityCELoss(nn.Module):

    def __init__(self, weights=None, ignore_index=255):
        super(DualityCELoss, self).__init__()
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target, weight=self.weights.cuda(),
                                  ignore_index=self.ignore_index)
        smooth_l1_loss = huber_loss((torch.argmax(input, dim=1) == input.size(1)).float(),
                                    (target == 19).float())
        return ce_loss + 50*smooth_l1_loss


class DualityFocalLoss(nn.Module):

    def __init__(self, ignore_index=255, gamma=1):
        super(DualityFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, input, target):
        n, c, h, w = input.size()
        input, target = flatten_data(input, target)
        target = target.long()
        ce_loss = F.cross_entropy(input, target,
                                  weight=None,
                                  ignore_index=self.ignore_index,
                                  reduction='none')
        focal_loss = ((1 - F.softmax(input, dim=1).permute(
            0, 2, 3, 1).contiguous().view(-1, c))**self.gamma).squeeze() * ce_loss

        smooth_l1_loss = huber_loss((torch.argmax(input, dim=1) == input.size(1)).float(),
                                    (target == input.size(1)).float())
        return focal_loss.mean() + 50*smooth_l1_loss


class BboxLoss(nn.Module):

    def __init__(self):
        super(BboxLoss, self).__init__()

    def forward(self, input, target, weight):
        loss = 0.0
        if isinstance(input['class'], dict):
            for k, v in input['class'].items():
                logits = {'class': v, 'offsets': input['offsets'][k]}
                loss += k*bbox_loss_level(logits, target, weight, k)
        else:
            loss = bbox_loss_level(input, target, weight)

        return loss


class HuberLoss(nn.Module):

    def __init__(self, delta=0.5):
        super(HuberLoss, self).__init__()
        self.deta = delta

    def forward(self, input, target, weight):
        deta = self.deta.cuda()
        if input.size() != target.size():
            input = input.permute(0, 2, 3, 1).squeeze()
        abs_diff = torch.abs(input - target)
        cond = abs_diff < delta
        loss = torch.where(cond, delta * abs_diff ** 2, abs_diff - delta)
        if weight is not None:
            loss = loss * weight.unsqueeze(1)
        return loss.mean()


def huber_loss(input, target, weight=None, delta=0.5, size_average=True):
    if input.size() != target.size():
        input = input.permute(0, 2, 3, 1).squeeze()
    abs_diff = torch.abs(input - target)
    cond = abs_diff < delta
    loss = torch.where(cond, delta * abs_diff ** 2, abs_diff - delta)
    if weight is not None:
        loss = loss * weight.unsqueeze(1)
    return loss.mean()


def bbox_loss_level(input, target, weight, stride=1):
    class_target = F.max_pool2d(target['class'], kernel_size=stride)
    offset_target = F.interpolate(target['offsets'], scale_factor=1/stride,
                                  mode='bilinear', align_corners=True)
    weight = F.max_pool2d(weight, kernel_size=stride)
    class_loss = focal_loss(input['class'], class_target, size_average=False)
    class_loss = (class_loss*weight.view(-1)).mean()
    bbox_loss = huber_loss(
        input['offsets'], offset_target, weight=weight)
    return class_loss + bbox_loss


def focal_loss(input, target, weights=None, sample_weight=None,
               gamma=1, size_average=True):
    c = input.size()[1]
    softmax_preds = F.softmax(input, dim=1).permute(
        0, 2, 3, 1).contiguous().view(-1, c)
    input, target = flatten_data(input, target)
    target = target.long()
    ce_loss = F.cross_entropy(input, target, weight=None,
                              ignore_index=255, reduction='none')
    target = target * (target != 255).long()
    softmax_preds = torch.gather(softmax_preds, 1, target.unsqueeze(1))
    focal_loss = ((1 - softmax_preds) ** gamma).squeeze() * ce_loss
    if size_average:
        return focal_loss.mean()
    else:
        return focal_loss


def flatten_data(input, target):
    n, c, h, w = input.size()
    input = input.squeeze()
    target = target.squeeze()
    if len(target.size()) == 2 and n == 1:
        ht, wt = target.size()
        input = input.transpose(0, 1).transpose(1, 2)
    elif len(target.size()) == 3 and n > 1:
        nt, ht, wt = target.size()
        input = input.transpose(1, 2).transpose(2, 3)
    else:
        raise ValueError('Check size of inputs and targets')

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt),
                              mode="bilinear", align_corners=True)
    input = input.contiguous().view(-1, c)
    target = target.view(-1)

    return input, target


def get_weight_mask(label):
    mask = torch.zeros_like(label).float()
    num_el = label.numel()
    beta = torch.sum((label == 0).float()) / num_el
    mask[label != 0] = beta
    mask[label == 0] = 1.0 - beta
    return mask


def make_one_hot(labels, num_classes=10):
    n, h, w = labels.size()
    one_hot = torch.zeros((n, num_classes, h, w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id, ...] = (labels == class_id+1)
    return one_hot.cuda()


# def get_mask_from_section(angles, section='horizontal'):
#     if section == 'horizontal':
#         return (get_mask_from_angle(angles, 0.0, 22.5) |
#                 get_mask_from_angle(angles, 157.5, 180)).float()
#     elif section == 'cnt_diag':
#         return get_mask_from_angle(angles, 22.5, 67.5).float()
#     elif section == 'lead_diag':
#         return get_mask_from_angle(angles, 112.5, 157.5).float()
#     elif section == 'vertical':
#         return get_mask_from_angle(angles, 67.5, 112.5).float()


# def get_mask_from_angle(angles, start, end):
#     return (angles >= start) & (angles < end)


# def get_normalized_responses(exp_preds, filter_dict,
#                              section='horizontal', eps=1e-6):

#     sum_boundary_responses = filter2D(exp_preds, filter_dict[section]) + eps
#     norm = torch.div(exp_preds, sum_boundary_responses)

#     return torch.clamp(norm, eps, 1)


# def get_nms_from_section(exp_preds, filter_dict, angles, section='horizontal'):
#     norm = get_normalized_responses(exp_preds, filter_dict,
#                                     section='horizontal',)
#     mask = get_mask_from_section(angles, section)
#     return (torch.log(norm) * mask).unsqueeze_(1)


# def get_all_nms(exp_preds, filter_dict, angles):
#     horiz_nms = get_nms_from_section(exp_preds, filter_dict,
#                                      angles, section='horizontal')
#     vert_nms = get_nms_from_section(exp_preds, filter_dict,
#                                     angles, section='cnt_diag')
#     lead_diag_nms = get_nms_from_section(exp_preds, filter_dict,
#                                          angles, section='lead_diag')
#     cnt_diag_nms = get_nms_from_section(exp_preds, filter_dict,
#                                         angles, section='vertical')

#     return torch.cat((horiz_nms, vert_nms, lead_diag_nms, cnt_diag_nms), 1)


# def get_filter_dict(r=2):
#     # can be 1D fiters?
#     filter_dict = {}
#     horiz = torch.zeros((2 * r - 1, 2 * r - 1))
#     horiz[r-1, :] = 1
#     filter_dict['horizontal'] = horiz.unsqueeze(0)
#     vert = torch.zeros((2 * r - 1, 2 * r - 1))
#     vert[:, r-1] = 1
#     filter_dict['vertical'] = vert.unsqueeze(0)
#     filter_dict['lead_diag'] = torch.eye(2*r-1).unsqueeze_(0)
#     filter_dict['cnt_diag'] = torch.flip(
#         torch.eye(2*r-1), dims=(0,)).unsqueeze(0)
#     return filter_dict


# def get_grads(target, eps=1e-6):
#     """
#     Calculate the direction of edges.
#     """
#     grads = grad2d(target)

#     angle = theta * 180 / np.pi
#     angle[angle < 0] += 180

#     return angle

# def weighted_binary_cross_entropy(input, target, weights=None):
#     # input, target = flatten_data(input, target)
#     n_batch = input.shape[0]
#     mean_bce = 0.0
#     mean_l1 = 0.0
#     for batch_id in range(n_batch):
#         label = target[batch_id].float()
#         mask = get_weight_mask(label)
#         prediction = input[batch_id].squeeze()
#         bce_loss = F.binary_cross_entropy_with_logits(
#             prediction, label, mask, size_average=True)
#         mean_bce += bce_loss
#         preds = torch.sigmoid(prediction)
#         l1_loss = huber_loss(preds, label, delta=0.3)
#         mean_l1 += l1_loss
#     mean_bce /= n_batch
#     mean_l1 /= n_batch
#     return 10*mean_bce + mean_l1


# def cross_entropy2d(input, target, weights=None, size_average=True):
#     input, target = flatten_data(input, target)
#     target = target.long()
#     loss = F.cross_entropy(input, target,
#                            weight=weights, ignore_index=255)

#     return loss
