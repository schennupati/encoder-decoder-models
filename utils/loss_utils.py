#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:40:49 2019

@author: sumche
"""
from utils.data_utils import get_weights
import torch
import torch.nn as nn
import pdb
# TODO:Implement MTL loss combinations
from utils.loss import cross_entropy2d, huber_loss, mae_loss, mse_loss, \
    weighted_binary_cross_entropy, weighted_binary_cross_entropy_with_nms, \
    flatten_data, duality_focal_loss


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg, weights, loss_type='fixed'):
        super(MultiTaskLoss, self).__init__()
        self.losses = {}
        self.loss_fn = {}
        self.cfg = cfg
        self.weights = weights
        self.tasks = list(self.cfg.keys())
        self.active_tasks = self.get_active_tasks()
        self.make_loss_fn()
        self.n = len(self.active_tasks)
        if loss_type == 'fixed':
            self.sigma = {task: cfg[task]['loss_weight']
                          for task in self.active_tasks}
        elif loss_type == 'uncertainty':
            self.sigma = nn.Parameter(torch.ones(self.n)).cuda()
        else:
            raise ValueError('Unkown loss_type')
        self.loss_type = loss_type

    def get_active_tasks(self):
        active_tasks = []
        for task in self.tasks:
            if self.cfg[task]['active']:
                active_tasks.append(task)
        return active_tasks

    def make_loss_fn(self):
        for task in self.active_tasks:
            loss_fn = self.cfg[task]['loss']
            if loss_fn == 'cross_entropy2d':
                self.loss_fn[task] = (cross_entropy2d)
            elif loss_fn == 'weighted_binary_cross_entropy':
                self.loss_fn[task] = (weighted_binary_cross_entropy)
            elif loss_fn == 'weighted_binary_cross_entropy_with_nms':
                self.loss_fn[task] = (weighted_binary_cross_entropy)
            elif loss_fn == 'dualityfocalloss':
                self.loss_fn[task] = (duality_focal_loss)
            elif loss_fn == 'l1':
                self.loss_fn[task] = mae_loss
            elif loss_fn == 'l2':
                self.loss_fn[task] = mse_loss

    def forward(self, predictions, targets):
        active_tasks = self.get_active_tasks()
        # sigma = self.get_sigma(active_tasks)

        for task in active_tasks:
            weights = self.weights[task].cuda()
            if task != 'semantic_with_instance':
                self.losses[task] = \
                    self.loss_fn[task](predictions[task],
                                       targets[task], weights)
            else:
                self.losses[task] = \
                    self.loss_fn[task](predictions[task],
                                       targets[task], weights,
                                       targets['touching_boundaries'])

        total_loss = self.compute_total_loss()
        return self.losses, total_loss

    def compute_total_loss(self):
        loss = 0.0
        if self.loss_type == 'fixed':
            for task in self.losses.keys():
                loss_weight = torch.tensor(self.sigma[task]).cuda()
                task_loss = self.losses[task].cuda()
                loss += task_loss * loss_weight

        elif self.loss_type == 'uncertainty':
            sigma = (self.sigma).cuda()
            for i, task in enumerate(self.losses.keys()):
                task_loss = self.losses[task].cuda()
                if self.cfg[task]['loss'] \
                    in ['cross_entropy2d', 'weighted_binary_cross_entropy',
                        'weighted_multiclass_cross_entropy']:
                    loss += torch.exp(-sigma[i]) * task_loss + \
                        0.5*sigma[i]
                elif self.cfg[task]['loss'] in ['l1', 'l2']:
                    loss += 0.5 * \
                        torch.exp(-sigma[i]) * task_loss + sigma[i]
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
    def __init__(self, cfg):
        self.cfg = cfg
        self.meters = self.get_loss_meters()

    def get_loss_meters(self):
        meters = {}
        for task in self.cfg.keys():
            if self.cfg[task]['active']:
                meters[task] = averageMeter()
        return meters

    def update(self, losses):
        for task in self.cfg.keys():
            if self.cfg[task]['active']:
                self.meters[task].update(losses[task])

    def reset(self):
        for task in self.cfg.keys():
            if self.cfg[task]['active']:
                self.meters[task].reset()
