#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:55:03 2019

@author: sumche
"""
#Source: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from utils.data_utils import up_scale_tensors

def cross_entropy2d(input, target, weight=None, size_average=True):
    
    n, c, h, w = input.size()
    if len(target.size())==2 and n==1:
        ht, wt = target.size()
    elif len(target.size())==3 and n>1:
        nt, ht, wt = target.size()
    else:
        raise ValueError('Check size of inputs and targets')

    #Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=255)
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

def huber_loss(input, target, weight=None, size_average=True):
    if input.size() !=target.size():
        input = input.permute(0,2,3,1).squeeze()
    return F.smooth_l1_loss(input, target)

def mae_loss(input, target, weight=None, size_average=True):
    if input.size() !=target.size():
        input = input.permute(0,2,3,1).squeeze()
    return F.l1_loss(input, target)

def mse_loss(input, target, weight=None, size_average=True):
    if input.size() !=target.size():
        input = input.permute(0,2,3,1).squeeze()
    return F.mse_loss(input, target)

def instance_loss(input, target, weight=None, size_average=None):
    #target[target==0] = 0.1
    if input.size() !=target.size():
        input = input.permute(0,2,3,1).squeeze()
        target = target.squeeze()
    non_zeros = torch.nonzero(target.data).size(0)
    return F.mse_loss(input, target,reduction='mean')#/non_zeros

    
    

