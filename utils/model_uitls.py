#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:02:38 2019

@author: sumche
"""
import torch.nn as nn


def get_activation(activation, inplace=True):
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


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)
