#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:06:24 2019

@author: sumche
"""
import torch
import numpy as np

from tqdm import tqdm

from utils.im_utils import labels

def transform_targets(targets,permute):
    return torch.squeeze((targets*255).permute(permute))
    
def convert_targets_semantic(targets,permute=(0,2,3,1),labels=labels):
    targets = transform_targets(targets,permute).numpy()
    new_targets = np.empty_like(targets)
    
    for label_id in np.unique(targets):
        train_id = labels[int(label_id)].trainId
        new_targets[np.where(targets==label_id)] = train_id
    
    return torch.tensor(new_targets)

def convert_targets_disparity(targets,permute=(0,2,3,1)):
    
    targets = torch.squeeze((targets).permute(permute)).numpy()
    mask = targets > 0
    dep_img = (.22*718)/(targets + (1.0 - mask))
    inv_dep = np.reciprocal(dep_img)
    min_inv_dep = np.min(inv_dep)
    max_inv_dep = np.max(inv_dep)
    
    normalized_dep = (inv_dep-min_inv_dep)/(max_inv_dep-min_inv_dep)
    #img_dep = np.repeat(np.expand_dims(normalized_dep[0],axis=-1),3,axis=-1)
    
    return torch.tensor(normalized_dep).type(torch.float32)

def convert_targets_instance(targets,permute=(0,2,3,1)):
    return torch.squeeze((targets).permute(permute))

def get_convert_fn(task):
    if task == 'semantic':
        return (convert_targets_semantic)
    elif task == 'disparity':
        return (convert_targets_disparity)
    elif task == 'instance':
        return (convert_targets_instance)
    else:
        return None

def convert_data_type(data,data_type):
    if data_type == 'double':
        return data.double()
    elif data_type == 'long':
        return data.long()
    elif data_type == 'float':
        return data.float()
    

def convert_targets(in_targets,cfg):
    #cfg['tasks']
    converted_tragets = {} 
    for i,task in enumerate(cfg.keys()):
        dat_type = cfg[task]['type']
        convert_fn = get_convert_fn(task)
        targets = in_targets[i] if isinstance(in_targets,list) else in_targets   
        converted_traget = convert_fn(targets) if convert_fn is not None else targets
        
        converted_tragets[task] = convert_data_type(converted_traget,dat_type)
            
    return converted_tragets

def convert_outputs(outputs,cfg):
    #cfg['tasks']
    converted_outputs = {}
    for i,task in enumerate(cfg.keys()):
        converted_outputs[task] = outputs[i]
    return converted_outputs

def get_class_weights_from_data(loader,num_classes):
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for data in tqdm(loader):
        _,labels = data
        for label_img in labels: 
            label_img = convert_targets(label_img, permute =(1,2,0))

            for trainId in range(num_classes):
                # count how many pixels in label_img which are of object class trainId:
                
                trainId_mask = np.equal(label_img, trainId)
                trainId_count = torch.sum(trainId_mask)

                # add to the total count:
                trainId_to_count[trainId] += trainId_count

    #compute the class weights according to the ENet paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    return class_weights

def cityscapes_semantic_weights(num_classes):
    if num_classes == 20 :
        class_weights = [2.955507538630981, 13.60952309186396, 5.56145316824849,
                         37.623098044056555, 35.219757095290035, 30.4509054117227,
                         46.155918742024745, 40.29336775103404, 7.1993048519013465,
                         31.964755676368643, 24.369833379633036, 26.667508196892037,
                         45.45602154799861, 9.738884687765038, 43.93387854348821,
                         43.46301980622594, 44.61855914531797, 47.50842372150186,
                         40.44117532401872, 12.772291423775606]
        
    elif num_classes == 19 :
        class_weights = [3.045383480249677, 12.862127312658735, 4.509888876996228, 
                         38.15694593009221, 35.25278401818165, 31.48260832348194, 
                         45.79224481584843, 39.69406346608758, 6.0639281852733715, 
                         32.16484408952653, 17.10923371690307, 31.5633201415795, 
                         47.33397232867321, 11.610673599796504, 44.60042610251128, 
                         45.23705196392834, 45.28288297518183, 48.14776939659858, 
                         41.924631833506794]
        
    elif num_classes == 34:
        return None #TODO: Compute weights
    else:
        raise ValueError('Invalid number of classes for Cityscapes dataset')
    
    return class_weights

def get_weights(cfg):
    dataset = cfg['data']['dataset']
    tasks = cfg['tasks']
    weights = {}
    for task in tasks.keys():
        if dataset == 'Cityscapes' and task == 'semantic':
            weight = cityscapes_semantic_weights(tasks[task]['out_channels'])
            weights[task] = torch.FloatTensor(weight).cuda()
        else:
            weights[task] = None
    
    return weights
            