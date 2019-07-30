#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:40:49 2019

@author: sumche
"""
from utils.loss import (cross_entropy2d,bootstrapped_cross_entropy2d,
                        multi_scale_cross_entropy2d,huber_loss,mae_loss,mse_loss)

loss_map = {
            'cross_entropy2d' : (cross_entropy2d),
            'multi_scale_cross_entropy2d' : (multi_scale_cross_entropy2d),
            'bootstrapped_cross_entropy2d': (bootstrapped_cross_entropy2d),
            'huber_loss': (huber_loss),
            'mae_loss' : (mae_loss),
            'mse_loss' : (mse_loss)
            }

def get_loss_fn(loss_type):
    return loss_map[loss_type] if loss_type in loss_map.keys() else None
    
def compute_task_loss(inputs,targets,weights,loss_type):
    loss_fn = get_loss_fn(loss_type)
    return loss_fn(inputs,targets,weights)

def compute_loss(predictions,targets,cfg,device,weights=None):
    out_loss = 0.0
    losses = {}
    
    for task in cfg.keys():
        prediction = predictions[task]
        target     = targets[task].to(device)
        weight = weights[task] if weights is not None else None
        loss_type  = cfg[task]['loss']
        loss = compute_task_loss(prediction,target,weight,loss_type)
        
        losses[task] = loss
        out_loss += loss
        
    return losses,out_loss