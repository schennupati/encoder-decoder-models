#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:10:15 2019

@author: sumche
"""
import torch

from datasets.cityscapes import Cityscapes
from .transforms import get_transforms


dataset_map = {'Cityscapes': (Cityscapes)}
 
def get_dataset(cfg,split='train',mode='fine',target_type='semantic',transforms=None):
     
     dataset_fn = dataset_map[cfg['dataset']]
     dataset = dataset_fn(root=cfg['root_path'],split=split,mode=mode,
                          target_type = target_type,
                          transform=transforms[split]['input'],
                          target_transform=transforms[split]['target'])
     return dataset
 
def get_dataloaders(cfg):
    
    data_loaders = {}
    target_type = []
    
    transforms = get_transforms(cfg['data']['transforms'])
    
    for target in cfg['tasks'].keys():
        target_type.append(target)
    
    if len(target_type) == 1:
        target_type = target_type[0]
        
    for split in ['train','val']:
        params = cfg['params'][split]
        dataset = get_dataset(cfg['data'],split,target_type=target_type,transforms=transforms)
        data_loaders[split] = torch.utils.data.DataLoader(dataset,
                                                          batch_size=params['batch_size'],
                                                          shuffle=params['shuffle'],
                                                          num_workers=params['n_workers'])
    return data_loaders
    
     
     
     