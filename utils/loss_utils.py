#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:40:49 2019

@author: sumche
"""
import torch
import torch.nn as nn
#TODO:Implement MTL loss combinations
from utils.loss import (cross_entropy2d,bootstrapped_cross_entropy2d,
                        multi_scale_cross_entropy2d,huber_loss,mae_loss,
                        mse_loss,instance_loss)

loss_map = {
            'cross_entropy2d' : (cross_entropy2d),
            'multi_scale_cross_entropy2d' : (multi_scale_cross_entropy2d),
            'bootstrapped_cross_entropy2d': (bootstrapped_cross_entropy2d),
            'huber_loss': (huber_loss),
            'mae_loss' : (mae_loss),
            'mse_loss' : (mse_loss),
            'instance_loss': (instance_loss)
            }

def get_loss_fn(loss_type):
    return loss_map[loss_type] if loss_type in loss_map.keys() else None
    
def compute_task_loss(inputs,targets,weights,loss_type):
    loss_fn = get_loss_fn(loss_type)
    return loss_fn(inputs,targets,weights)

def compute_loss(predictions,targets,cfg,weights=None):
    out_loss = 0.0
    losses = {}
    
    for task in cfg.keys():
        prediction = predictions[task]
        target     = targets[task]
        weight = weights[task] if weights is not None else None
        loss_weight = cfg[task]['loss_weight']
        loss_type  = cfg[task]['loss']
        loss = compute_task_loss(prediction,target,weight,loss_type)
        
        losses[task] = loss
        out_loss += loss_weight*loss
        
    return losses,out_loss

class MultiTaskLoss(nn.Module):
    def __init__(self, cfg,weights=None, loss_fn='equal'):
        super(MultiTaskLoss, self).__init__()
        self.losses = {}
        self.cfg = cfg
        self.weights = weights
        self.tasks = list(self.cfg.keys())
        self.n = len(self.tasks)
        self.sigma = nn.Parameter(torch.ones(self.n))
        self.loss_fn = loss_fn

    def forward(self, predictions, targets):
        for task in self.tasks:
            prediction = predictions[task]
            target     = targets[task]
            weight = self.weights[task] if self.weights is not None else None
            loss_type  = self.cfg[task]['loss']
            self.losses[task] = compute_task_loss(prediction, 
                       target,weight,loss_type)
        l = [loss for loss in self.losses.values()]
        if self.loss_fn == 'log_uncertainity':
            l = self.log_uncertainity(l)
        elif self.loss_fn == 'exp_uncertainity':
            l = self.exp_uncertainity(l)
        elif self.loss_fn == 'equal':
            l = torch.Tensor(l)*nn.Parameter(torch.ones(self.n))
            l = l.sum()            
        return self.losses, l
    
    def log_uncertainity(self, l):
        l = 0.5 * (torch.Tensor(l) / self.sigma**2)
        l = l.sum() + torch.log(self.sigma.prod())
        return l
    
    def exp_uncertainity(self, l):
        eta = torch.log(self.sigma)
        l = (1.0/self.n) * (torch.Tensor(l)*torch.exp(-1.0*eta) + eta)
        l = l.sum()
        return l

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class loss_meters:
    def __init__(self,cfg):
        self.cfg = cfg    
        self.meters = self.get_loss_meters()
        
    def get_loss_meters(self):
        meters = {}
        for task in self.cfg.keys():
            meters[task] = averageMeter()
        return meters
    
    def update(self,losses):
        for task in self.cfg.keys():
            self.meters[task].update(losses[task])
    
    def reset(self):
        for task in self.cfg.keys():
            self.meters[task].reset()
    