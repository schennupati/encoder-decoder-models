#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:22:59 2019

@author: sumche
"""
import torch
import os
import glob

def get_checkpoint(exp_name,base_dir):
    list_of_models = glob.glob(os.path.join(base_dir,'*.pkl'))
    if any(exp_name in model for model in list_of_models):
        checkpoint_name = max(list_of_models, key=os.path.getctime)
        return checkpoint_name, torch.load(checkpoint_name)
        