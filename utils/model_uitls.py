#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:02:38 2019

@author: sumche
"""
import torch.nn as nn

def get_activation(activation,inplace=True):
    if activation == 'ELU':
        return nn.ELU(alpha=1.0, inplace=inplace)
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    elif activation == 'PReLU':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif activation == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif activation == 'Tanh':
        return nn.Tanh()