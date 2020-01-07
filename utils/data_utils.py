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
from utils.im_utils import labels, prob_labels
import matplotlib.pyplot as plt
import cv2

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

def convert_targets_instance(targets,permute=(0,2,3,1)):
        
    targets = torch.squeeze((targets).permute(permute)) 
    if len(targets.size()) > 2: 
        n,h,w = targets.size()
    elif len(targets.size()) == 2:
        h, w = targets.size()
        n = 1
        targets = targets.unsqueeze(0)
    vecs = torch.zeros((n,2,h,w))
    masks = torch.zeros((n,h,w))
    heatmaps = torch.zeros((n,h,w))
    contours = torch.zeros((n,h,w))
    for i in range(n):    
        reg, mask, heatmap, contour = compute_centroid_vector_torch(targets[i,:,:].float())
        #print((torch.min(reg), torch.max(reg)), (torch.min(heatmap), torch.max(heatmap)))
        vecs[i,:,:,:] = reg.float()
        masks[i,:,:] = mask.long()
        heatmaps[i,:,:] = heatmap.float()
        contours[i,:,:] = contour.long()
    
    converted_targets = {'instance_regression': vecs,
                         'instance_probs': masks,
                         'instance_heatmap':heatmaps,
                         'instance_contour': contours}    
    return converted_targets

def convert_targets_instance_contours(targets,permute=(0,2,3,1)):
        
    targets = torch.squeeze((targets).permute(permute)) 
    if len(targets.size()) > 2: 
        n,h,w = targets.size()
    elif len(targets.size()) == 2:
        h, w = targets.size()
        n = 1
        targets = targets.unsqueeze(0)
    contours = torch.zeros((n,h,w))
    
    for i in range(n):    
        cont = compute_instance_contours(targets[i,:,:].float())
        #print((torch.min(reg), torch.max(reg)), (torch.min(heatmap), torch.max(heatmap)))
        contours[i,:,:] = cont.long()
    
    converted_targets =  {'instance_contour': contours}
    return converted_targets

def get_convert_fn(task):
    if task == 'semantic':
        return (convert_targets_semantic)
    elif task == 'disparity':
        return (convert_targets_disparity)
    elif task == 'instance':
        return (convert_targets_instance) # Replace accordingly
    else:
        return None

def convert_data_type(data,data_type):
    if data_type == 'double':
        return data.double()
    elif data_type == 'long':
        return data.long()
    elif data_type == 'float':
        return data.float()
    

def convert_targets(in_targets,cfg,device=None):
    #cfg['tasks']
    converted_targets = {} 
    for i,task in enumerate(cfg.keys()):
        data_type = cfg[task]['type']
        convert_fn = get_convert_fn(task)
        targets = in_targets[i] if isinstance(in_targets,list) else in_targets
        if task != 'instance':
            converted_target = convert_fn(targets) if convert_fn is not None else targets
            converted_targets[task] = convert_data_type(converted_target,data_type).to(device)
        else:
            dict_targets = convert_fn(targets)
            for task in dict_targets.keys():
                dict_targets[task] = dict_targets[task].to(device)
            converted_targets.update(dict_targets)
            
    return converted_targets

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
        else:
            converted_outputs[task] = outputs[task]
    return converted_outputs   

def get_class_weights_from_data(loader,num_classes,cfg, task):
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for data in tqdm(loader):
        _,labels = data
        labels = convert_targets(labels,cfg['tasks'])
        for label_img in labels[task]: 
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

def cityscapes_contour_weights(num_classes):
    if num_classes == 11 :
        class_weights = [1.427197976828025, 47.66104006965641, 50.0977099173462,
                         44.04363870779025, 50.31372660864973, 50.31163764506638,
                         50.47265462912245, 50.471431327406826, 50.36620380700314,
                         50.32661428022733, 49.834611789928324]

    else:
        raise ValueError('Invalid number of classes for Cityscapes dataset')
    
    return class_weights

def get_weights(cfg,device):
    dataset = cfg['data']['dataset']
    tasks = cfg['model']['outputs']
    weights = {}
    for task in tasks.keys():
        if dataset == 'Cityscapes' and task == 'semantic':
            weight = cityscapes_semantic_weights(tasks[task]['out_channels'])
            weights[task] = torch.FloatTensor(weight).to(device)
        elif dataset == 'Cityscapes' and task == 'instance_contour':
            weight = cityscapes_contour_weights(tasks[task]['out_channels'])
            weights[task] = torch.FloatTensor(weight).to(device)
        else:
            weights[task] = None
    
    return weights

def get_2d_bbox_from_instance(xs, ys):
    vertex_1 = (torch.min(xs), torch.min(ys))
    vertex_2 = (torch.max(xs), torch.max(ys))
    return vertex_1, vertex_2

def get_instance_hw(xs, ys):
    vertex_1, vertex_2 = get_2d_bbox_from_instance(xs, ys)
    return (abs(vertex_1[0]-vertex_2[0]), abs(vertex_1[1]-vertex_2[1]))

def compute_centroid_vector_torch(instance_image):
    alpha = 10.0
    contours = np.zeros(instance_image.shape)
    contour_class_map = {i+24: i for i in range(10)}
    instance_image_tensor = torch.Tensor(instance_image)
    centroids_t = torch.zeros(instance_image.shape + (2,))
    w_h = torch.ones(instance_image.shape + (2,))
    for value in torch.unique(instance_image_tensor):
        xsys = torch.nonzero(instance_image_tensor == value)
        xs, ys = xsys[:, 0], xsys[:, 1]
        centroids_t[xs, ys] = torch.stack((torch.mean(xs.float()), torch.mean(ys.float())))
        if value > 1000:
            contour_class = contour_class_map[value.numpy()//1000]
            cont = np.array([xs.numpy(), ys.numpy()])
            for x in np.unique(cont[0,:]):
                idx = np.where(cont[0,:]==x)
                contours[x, np.min(cont[1,idx])] = contour_class+1
                contours[x, np.max(cont[1,idx])] = contour_class+1
            for y in np.unique(cont[1,:]):
                idx = np.where(cont[1,:]==y)
                contours[np.min(cont[0,idx]), y] = contour_class+1
                contours[np.max(cont[0,idx]), y] = contour_class+1
            w, h = get_instance_hw(xs, ys)
            if w!=0 and h!=0:
                w_h[xs, ys,0], w_h[xs, ys,1]  = w.float(), h.float()

    coordinates = torch.zeros(instance_image.shape + (2,))
    g1, g2 = torch.meshgrid(torch.arange(instance_image_tensor.size()[0]), torch.arange(instance_image_tensor.size()[1]))
    coordinates[:, :, 0] = g1
    coordinates[:, :, 1] = g2
    vecs = coordinates - centroids_t
    
    
    mask = instance_image_tensor >= 1000
    if len(mask.size()) > 1:
        mask = mask.int()
    elif mask is False:
        mask = np.zeros(instance_image.shape)
    else:
        mask = np.ones(instance_image.shape)
    vecs[:,:,0] = vecs[:,:,0]*mask
    vecs[:,:,1] = vecs[:,:,1]*mask
    heatmap_ = w_h - (torch.abs(vecs)*alpha)
    heatmap_ = np.clip(heatmap_, 0, torch.max(heatmap_))
    
    heatmap_[:,:,0] /= w_h[:,:,0]
    heatmap_[:,:,1] /= w_h[:,:,1]
    heatmap_t = heatmap_[:,:,0]*heatmap_[:,:,1]
    heatmap_t = heatmap_t*mask
    #kernel = np.ones((3,3), np.uint8)
    #contours = cv2.dilate(contours, kernel, iterations=1)
    #print((torch.min(vecs), torch.max(vecs)), (torch.min(heatmap_), torch.max(heatmap_)))
    return vecs.permute(2,0,1), mask, heatmap_t, torch.tensor(contours)

def compute_instance_contours(instance_image):
    contours = np.zeros(instance_image.shape)
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        if value>23:
            cont = np.array([xs, ys])
            for x in np.unique(cont[0,:]):
                idx = np.where(cont[0,:]==x)
                contours[x, np.min(cont[1,idx])] = 1
                contours[x, np.max(cont[1,idx])] = 1
            for y in np.unique(cont[1,:]):
                idx = np.where(cont[1,:]==y)
                contours[np.min(cont[0,idx]), y] = 1
                contours[np.max(cont[0,idx]), y] = 1
    kernel = np.ones((3,3), np.uint8)
    contours = cv2.dilate(contours, kernel, iterations=1)
    return torch.tensor(contours)

def get_cfg(config):
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    for task in list(cfg['tasks'].keys()):
        if not cfg['tasks'][task]['active']:
            del cfg['tasks'][task]
    return cfg
    
            