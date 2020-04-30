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

def flatten_data(input, target):
    target = target.long()
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

    return input, target

def cross_entropy2d(input, target, weight=None, size_average=True):
    input, target = flatten_data(input, target)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=255)
    return loss

def weighted_binary_cross_entropy(input, target, weights=None):
    #input, target = flatten_data(input, target)
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[0] * (target * torch.log(input)) + \
               weights[1] * ((1 - target) * torch.log(1 - input))
    else:
        loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    
    return torch.neg(torch.mean(loss))

def make_one_hot(labels, num_classes=10):
    n,h,w = labels.size()
    one_hot = torch.zeros((n, num_classes,h, w), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id,...] = (labels==class_id+1)
    return one_hot.to(labels.get_device())

def BCELoss_ClassWeights(input, target, class_weights, reversed=False):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = class_weights[0]*target * torch.log(input) + class_weights[1]*(1 - target) * torch.log(1 - input)
    final_reduced_over_batch = -bce.sum()
    return final_reduced_over_batch

def dice_loss(input_soft: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~kornia.losses.DiceLoss` for details.
    """
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

    dice_score =   2. * intersection/(cardinality + eps)
    return torch.mean(-dice_score + 1.)

def weighted_multiclass_cross_entropy(input, target, weight=None, weights=None):
    if len(input.size()) == 4:
        n, c, h, w = input.size()
    else:
        c, h, w = input.size()
    if weight is None:
        weight = [1 for i in range(c)]
    input_soft = torch.sigmoid(input)
    target_onehot = make_one_hot(target)
    lc = weighted_multi_class_binary_cross_entropy(input_soft, target_onehot, weights=weights)
    #ld = dice_loss(input_soft, target_onehot)
    #l2 = F.l1_loss(input_soft, target_onehot, reduction='mean')

    return lc #+ ld + l2

def weighted_multiclass_cross_entropy_with_nms(input, target, weight=None, weights=None):
    nmsloss = StealNMSLoss()
    if len(input.size()) == 4:
        n, c, h, w = input.size()
    else:
        c, h, w = input.size()
    if weight is None:
        weight = [1 for i in range(c)]
    input_soft = torch.sigmoid(input)
    target_onehot = make_one_hot(target)
    lc = weighted_multi_class_binary_cross_entropy(input_soft, target_onehot, weights=weights)
    ln = nmsloss.__call__(target_onehot, input_soft)

    return lc + ln

def weighted_multi_class_binary_cross_entropy(input, target, weights):
    bce_loss = 0.0 
    n, c, h, w = input.shape
    assert input.shape == target.shape
    #import pdb; pdb.set_trace()
    
    for i in range(c):
        input_, target_ = input[:,i,...].view(n,h*w), target[:,i,...].view(n,h*w)
        bce_loss += BCELoss_ClassWeights(input_,target_,class_weights=weights,reversed=reversed)   
    return bce_loss

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

def instance_loss(input, target, weight=None, size_average=True):
    #target[target==0] = 0.1
    if input.size() !=target.size():
        input = input.permute(0,2,3,1).squeeze()
        target = target.squeeze()
    return F.l1_loss(input, target,reduction='mean')#/non_zeros
