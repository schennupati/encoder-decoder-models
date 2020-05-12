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


def cross_entropy2d(input, target, weights=None, size_average=True):
    input, target = flatten_data(input, target)
    target = target.long()
    loss = F.cross_entropy(input, target,
                           weight=weights, ignore_index=255)

    return loss


def duality_loss(input, target, weights=None, sample_weight=None,
                 size_average=True):
    argmax_preds = torch.argmax(input, dim=1)
    cnt_targets_mask = (target == 19).float()
    input, target = flatten_data(input, target)
    target = target.long()
    ce_loss = F.cross_entropy(input, target,
                              weight=weights, ignore_index=255)
    cnt_pred_mask = (argmax_preds == 19).float()
    l2_loss = F.mse_loss(cnt_pred_mask, cnt_targets_mask)
    # logging.info('ce_loss: {} l2_loss: {}'.format(ce_loss, l2_loss))
    return ce_loss + 50*l2_loss


def duality_focal_loss(input, target, weights=None, sample_weight=None,
                       gamma=1, size_average=True):
    n, c, h, w = input.size()
    argmax_preds = torch.argmax(input, dim=1)
    softmax_preds = F.softmax(input, dim=1).permute(
        0, 2, 3, 1).contiguous().view(-1, c)
    cnt_targets_mask = (target == 19).float()
    inst_targets_mask = (target == 20).float()
    input, target = flatten_data(input, target)
    target = target.long()
    ce_loss = F.cross_entropy(input, target, weight=None,
                              ignore_index=255, reduction='none')
    cnt_pred_mask = (argmax_preds == 19).float()
    inst_pred_mask = (argmax_preds == 20).float()
    l2_loss_seg_cnt = F.mse_loss(cnt_pred_mask, cnt_targets_mask)
    l2_loss_inst_cnt = F.mse_loss(inst_pred_mask, cnt_targets_mask)
    target = target * (target != 255).long()
    softmax_preds = torch.gather(softmax_preds, 1, target.unsqueeze(1))
    focal_loss = ((1 - softmax_preds)**gamma).squeeze() * ce_loss
    #focal_loss = focal_loss*(sample_weight.view(-1, 1).squeeze())
    # logging.info(
    #     'focal_loss: {}, ce_loss: {}, l2_loss: {}'.format(focal_loss, ce_loss.mean(), l2_loss))
    return focal_loss.mean() + 10*l2_loss_seg_cnt + 1e3*l2_loss_inst_cnt


def weighted_binary_cross_entropy(input, target, weights=None):
    # input, target = flatten_data(input, target)
    n_classes = input.shape[1]
    mean_loss = 0.0
    for class_id in range(n_classes):
        label = (target == class_id).float()
        mask = get_weight_mask(label)
        prediction = input[:, class_id, ...]
        loss = F.binary_cross_entropy_with_logits(prediction, label, mask)
        mean_loss += loss * weights[class_id] if weights is not None else loss
    return mean_loss


def get_weight_mask(label):
    num_el = label.numel()
    beta = torch.sum((label == 0).float()) / num_el
    label[label != 0] = beta
    label[label == 0] = 1 - beta
    return label


def weighted_binary_cross_entropy_with_nms(input, target, weights=None):
    mean_loss = weighted_binary_cross_entropy(input, target, weights=weights)
    nmsloss = StealNMSLoss()
    n_classes = input.shape[1]
    target = make_one_hot(target, n_classes)
    input = torch.softmax(input, dim=1)
    mean_ln = nmsloss(input, target)
    return mean_loss + mean_ln


def steal_nms_loss(input, target):

    angles = get_grads(target.float())

    # Create all possible direction NMS and mask them.
    exp_preds = torch.exp(input/self.tau)

    return - torch.mean(get_all_nms(exp_preds, filter_dict, angles))


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


def get_mask_from_section(angles, section='horizontal'):
    if section == 'horizontal':
        return (get_mask_from_angle(angles, 0.0, 22.5) |
                get_mask_from_angle(angles, 157.5, 180)).float()
    elif section == 'cnt_diag':
        return get_mask_from_angle(angles, 22.5, 67.5).float()
    elif section == 'lead_diag':
        return get_mask_from_angle(angles, 112.5, 157.5).float()
    elif section == 'vertical':
        return get_mask_from_angle(angles, 67.5, 112.5).float()


def get_mask_from_angle(angles, start, end):
    return (angles >= start) & (angles < end)


def get_normalized_responses(exp_preds, filter_dict,
                             section='horizontal', eps=1e-6):

    sum_boundary_responses = filter2D(exp_preds, filter_dict[section]) + eps
    norm = torch.div(exp_preds, sum_boundary_responses)

    return torch.clamp(norm, eps, 1)


def get_nms_from_section(exp_preds, filter_dict, angles, section='horizontal'):
    norm = get_normalized_responses(exp_preds, filter_dict,
                                    section='horizontal',)
    mask = get_mask_from_section(angles, section)
    return (torch.log(norm) * mask).unsqueeze_(1)


def get_all_nms(exp_preds, filter_dict, angles):
    horiz_nms = get_nms_from_section(exp_preds, filter_dict,
                                     angles, section='horizontal')
    vert_nms = get_nms_from_section(exp_preds, filter_dict,
                                    angles, section='cnt_diag')
    lead_diag_nms = get_nms_from_section(exp_preds, filter_dict,
                                         angles, section='lead_diag')
    cnt_diag_nms = get_nms_from_section(exp_preds, filter_dict,
                                        angles, section='vertical')

    return torch.cat((horiz_nms, vert_nms, lead_diag_nms, cnt_diag_nms), 1)


def get_filter_dict(r=2):
    # can be 1D fiters?
    filter_dict = {}
    horiz = torch.zeros((2 * r - 1, 2 * r - 1))
    horiz[r-1, :] = 1
    filter_dict['horizontal'] = horiz.unsqueeze(0)
    vert = torch.zeros((2 * r - 1, 2 * r - 1))
    vert[:, r-1] = 1
    filter_dict['vertical'] = vert.unsqueeze(0)
    filter_dict['lead_diag'] = torch.eye(2*r-1).unsqueeze_(0)
    filter_dict['cnt_diag'] = torch.flip(
        torch.eye(2*r-1), dims=(0,)).unsqueeze(0)
    return filter_dict


def get_grads(target, eps=1e-6):
    """
    Calculate the direction of edges.
    """
    grads = grad2d(target)

    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    return angle
