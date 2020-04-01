#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:40:49 2019

@author: sumche
"""
import torch
import torch.nn as nn
import pdb
#TODO:Implement MTL loss combinations
from utils.loss import (cross_entropy2d,bootstrapped_cross_entropy2d,
                        multi_scale_cross_entropy2d,huber_loss,mae_loss,
                        mse_loss,instance_loss, weighted_binary_cross_entropy,
                        weighted_multiclass_cross_entropy, weighted_multiclass_cross_entropy_with_nms)

loss_map = {
            'cross_entropy2d' : (cross_entropy2d),
            'weighted_binary_cross_entropy' : (weighted_binary_cross_entropy),
            'weighted_multiclass_cross_entropy': (weighted_multiclass_cross_entropy),
            'weighted_multiclass_cross_entropy_with_nms': (weighted_multiclass_cross_entropy_with_nms),
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
    def __init__(self, cfg, weights=None, loss_type='fixed'):
        super(MultiTaskLoss, self).__init__()
        self.losses = {}
        self.cfg = cfg
        self.weights = weights
        self.tasks = list(self.cfg.keys())
        self.active_tasks = self.get_active_tasks()
        self.n = len(self.active_tasks)
        self.sigma = nn.Parameter(torch.ones(self.n))
        self.loss_type = loss_type
        self.l1 = nn.L1Loss(reduction='sum')
        self.l2 = nn.MSELoss(reduction='sum')
    
    def get_active_tasks(self):
        active_tasks = []
        for task in self.tasks:
            if self.cfg[task]['active']:
                active_tasks.append(task)
        return active_tasks
    
    def get_sigma(self, active_tasks):
        n = len(active_tasks)
        return nn.Parameter(torch.ones(n))


    def forward(self, predictions, targets):
        active_tasks = self.get_active_tasks()
        #sigma = self.get_sigma(active_tasks)
        for task in active_tasks:
            prediction = predictions[task]
            target = targets[task]
            weight = self.weights[task]
            loss_fn  = self.cfg[task]['loss']
            self.losses[task] = self.compute_task_loss(prediction, 
                                                      target, weight, loss_fn)

        device = self.losses[task].get_device()
        self.sigma.to(device)
        total_loss =  self.compute_total_loss()
        return self.losses, total_loss
    
    def compute_task_loss(self, prediction, target, weight, loss_fn):
        if loss_fn == 'cross_entropy2d':
            loss = cross_entropy2d(prediction, target.long(), weight=weight)
        elif loss_fn == 'weighted_binary_cross_entropy':
            weights = self.get_weights(target)
            loss = weighted_binary_cross_entropy(prediction, target.long(), weights=weights)
        elif loss_fn == 'weighted_multiclass_cross_entropy':
            weights = self.get_weights(target)
            loss = weighted_multiclass_cross_entropy(prediction, target.long(), weight=weight,weights=weights)
        elif loss_fn == 'weighted_multiclass_cross_entropy_with_nms':
            weights = self.get_weights(target)
            loss = weighted_multiclass_cross_entropy_with_nms(prediction, target.long(), weight=weight,weights=weights)
        elif loss_fn ==  'l1':
            non_zeros = torch.nonzero(target).size(0)
            if prediction.size() !=target.size():
                prediction = prediction.permute(0,2,3,1).squeeze()
            loss = self.l1(prediction, target)
            loss = loss/non_zeros if non_zeros >0 else torch.zeros_like(loss)
        elif loss_fn ==  'l2':
            non_zeros = torch.nonzero(target).size(0)
            if prediction.size() !=target.size():
                prediction = prediction.permute(0,2,3,1).squeeze()
            loss = self.l2(prediction, target)
            loss = loss/non_zeros if non_zeros >0 else torch.zeros_like(loss)
        return loss        
    
    def get_weights(self, target):
        non_zeros = torch.nonzero(target).size(0)
        if len(target.size())==2:
            ht, wt = target.size()
            nt = 1
        elif len(target.size())==3:
            nt, ht, wt = target.size()
        num_pixs  = nt*ht*wt
        beta = 1- non_zeros/num_pixs
        weights = torch.tensor([beta, 1- beta])
        device = target.get_device()
        return weights.to(device)

    def compute_total_loss(self):
        loss = 0.0
        if self.loss_type == 'fixed':
            for task in self.losses.keys():
                loss += self.losses[task]*self.cfg[task]['loss_weight']

        elif self.loss_type == 'uncertainty':
            for i, task in enumerate(self.losses.keys()):
                if self.cfg[task]['loss'] == 'cross_entropy2d' or \
                   self.cfg[task]['loss'] == 'weighted_binary_cross_entropy' or \
                   self.cfg[task]['loss'] == 'weighted_multiclass_cross_entropy':
                    loss += torch.exp(-self.sigma[i])*self.losses[task] + 0.5*self.sigma[i]
                elif self.cfg[task]['loss'] == 'l1' or self.cfg[task]['loss'] == 'l2':
                    loss += 0.5*(torch.exp(-self.sigma[i])*self.losses[task] + self.sigma[i])
        return loss

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
            if self.cfg[task]['active']:
                meters[task] = averageMeter()
        return meters
    
    def update(self,losses):
        for task in self.cfg.keys():
             if self.cfg[task]['active']:
                self.meters[task].update(losses[task])
    
    def reset(self):
        for task in self.cfg.keys():
             if self.cfg[task]['active']:
                self.meters[task].reset()
    