#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:06:24 2019

@author: sumche
"""
import torch
import numpy as np
import yaml
from tqdm import tqdm
import pdb

from utils.im_utils import labels
from datasets.instance_to_clusters import convert_centroids, get_centers, to_rgb

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
    normalized_dep = []
    n = targets.size()[0] if len(targets.size())>2 else 1
    targets = torch.squeeze((targets).permute(permute)).numpy()
    targets[targets>0] = (targets[targets>0]-1)/256
    inv_dep = targets/(0.209313*2262.52)

    if n > 1:
        for i in range(n):    
            min_inv_dep = np.min(inv_dep[i])
            max_inv_dep = np.max(inv_dep[i])
            normalized_dep.append((inv_dep[i]-min_inv_dep)/(max_inv_dep-min_inv_dep))
    else:
        min_inv_dep = np.min(inv_dep)
        max_inv_dep = np.max(inv_dep)
        normalized_dep = (inv_dep-min_inv_dep)/(max_inv_dep-min_inv_dep)
        
    return torch.tensor(normalized_dep).type(torch.float32)

def convert_targets_instance_regression(targets,permute=(0,2,3,1)):

    targets = targets.permute(permute)
    #print('Normalized targets:',torch.unique(targets[0,:,:,0]*255.0))
    n, h, w, c = targets.size()
    stride = 1 #1024/h
    centroids = torch.zeros((n,h,w,c))
    for i in range(n):
        normalized_centroids = convert_centroids(torch.squeeze(targets[i]), 
                                                 op='denormalize')
        centroids[i] = convert_centroids(normalized_centroids, 
                                         op='up_scale',stride=stride)        
    #print('Denormalized targets:',torch.unique(centroids[0,:,:,0]))
    return centroids.permute(0,3,1,2)#torch.squeeze((targets).permute(permute))

def convert_targets_instance_probs(targets,permute=(0,2,3,1)):
    targets = torch.squeeze((targets).permute(permute))
    new_targets = torch.empty_like(targets)
    new_targets[torch.where(targets>=0.5)] = 1
    new_targets[torch.where(targets<0.5)] = 0
    return new_targets

def convert_targets_instance_heatmaps(targets,permute=(0,2,3,1)):
    targets = torch.squeeze((targets).permute(permute))
    return targets/targets.max()

def get_convert_fn(task):
    if task == 'semantic':
        return (convert_targets_semantic)
    elif task == 'disparity':
        return (convert_targets_disparity)
    elif task == 'instance_regression':
        return (convert_targets_instance_regression)
    elif task == 'instance_probs':
        return (convert_targets_instance_probs)
    elif task == 'instance_heatmaps':
        return (convert_targets_instance_heatmaps)
    else:
        return None

def convert_data_type(data,data_type):
    if data_type == 'double':
        return data.double()
    elif data_type == 'long':
        return data.long()
    elif data_type == 'float':
        return data.float()
    

def convert_targets(in_targets,cfg,device):
    #cfg['tasks']
    converted_tragets = {} 
    for i,task in enumerate(cfg.keys()):
        data_type = cfg[task]['type']
        convert_fn = get_convert_fn(task)
        targets = in_targets[i] if isinstance(in_targets,list) else in_targets   
        converted_traget = convert_fn(targets) if convert_fn is not None else targets
        
        converted_tragets[task] = convert_data_type(converted_traget,data_type).to(device)
            
    return converted_tragets

def convert_outputs(outputs,cfg):
    #cfg['tasks']
    converted_outputs = {}
    for i,task in enumerate(cfg.keys()):
        converted_outputs[task] = outputs[i]
    return converted_outputs

def post_process_outputs(outputs, cfg, targets):
    converted_outputs = {}
    for task in outputs.keys():
        if cfg[task]['postproc'] == 'argmax':
            converted_outputs[task] = torch.argmax(outputs[task],dim=1)
        elif cfg[task]['postproc'] == 'cluster_to_instance':
            converted_outputs[task] = cluster_to_instance(outputs[task])
        else:
            converted_outputs[task] = outputs[task]
    return converted_outputs

def cluster_to_instance(outputs):
    n,c,h,w = outputs.size()
    outputs = outputs.permute(0,2,3,1).cpu().numpy()
    predictions = np.zeros((n,h,w))
    for i in range(n):
        predictions[i,:,:] = get_centers(outputs[i,:,:,:])
    return predictions        

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

def get_weights(cfg,device):
    dataset = cfg['data']['dataset']
    tasks = cfg['tasks']
    weights = {}
    for task in tasks.keys():
        if dataset == 'Cityscapes' and task == 'semantic':
            weight = cityscapes_semantic_weights(tasks[task]['out_channels'])
            weights[task] = torch.FloatTensor(weight).to(device)
        else:
            weights[task] = None
    
    return weights

def up_scale_tensors(tensor):
    n,h,w,c = tensor.size()
    stride = 1024/h
    if len(tensor.shape)==4:
        tensor_x = tensor[:,:,:,1]*1024/stride
        tensor_y = tensor[:,:,:,0]*2048/stride
        
    elif len(tensor.shape)==3:
        tensor_x = tensor[:,:,:,1]*1024/stride
        tensor_y = tensor[:,:,:,0]*2048/stride
    
    return torch.stack((tensor_x,tensor_y),-1)

def regress_centers(Image):
    instances = np.unique(Image)
    instances = instances[instances > 1000]

    mask = np.zeros_like(Image)
    mask[np.where(Image > 1000)] = 1

    centroid_regression = np.zeros([Image.shape[0], Image.shape[1], 3])
    centroid_regression[:, :, 2] = mask

    for instance in instances:
        # step A - get a center (x,y) for each instance
        instance_pixels = np.where(Image == instance)
        y_c, x_c = np.mean(instance_pixels[0]), np.mean(instance_pixels[1])
        # step B - calculate dist_x, dist_y of each pixel of instance from its center
        y_dist = (-y_c + instance_pixels[0])
        x_dist = (-x_c + instance_pixels[1])
        for y, x, d_y, d_x in zip(instance_pixels[0], instance_pixels[1], y_dist, x_dist):
            centroid_regression[y, x, :2] = [d_y, d_x]  # remember - y is distance in rows, x in columns
    return centroid_regression

def get_cfg(config):
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    for task in list(cfg['tasks'].keys()):
        if not cfg['tasks'][task]['active']:
            del cfg['tasks'][task]
    return cfg
    
            