#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:25:54 2019

@author: sumche
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from clusters_to_instances import to_rgb
import matplotlib.pyplot as plt
path_to_annotations = '/home/sumche/datasets/Cityscapes/gtFine/val'


def get_total_files_count(path,ext='.png'):
    count = 0
    for root, dirs, names in os.walk(path, topdown=False):
        for name in names:
            if name.endswith(ext):
                count += 1 
    return count

annotations_count = get_total_files_count(path_to_annotations,'instanceIds.png')

def convert_centroids(centroids,op='normalize',stride=1,instance_prob=True):
    
    #converted_centroid_regression = torch.zeros_like(centroids) if tensor else np.zeros_like(centroids)
        
    if instance_prob:
        centroids[:,:,2] = centroids[:,:,2]

    if op == 'normalize':
        centroids[:,:,1] = normalize(centroids[:,:,1])
        centroids[:,:,0] = normalize(centroids[:,:,0])
    elif op == 'denormalize':
        centroids[:,:,1] = denormalize(centroids[:,:,1])
        centroids[:,:,0] = denormalize(centroids[:,:,0])
    elif op == 'down_scale':
        centroids[:,:,1] = down_scale(centroids[:,:,1],max_value=1024)
        centroids[:,:,0] = down_scale(centroids[:,:,0],max_value=2048)
    elif op == 'up_scale':
        centroids[:,:,1] = up_scale(centroids[:,:,1],max_value=1024,stride=stride)
        centroids[:,:,0] = up_scale(centroids[:,:,0],max_value=2048,stride=stride)
    else:
        raise ValueError('Unkown op to convert centroids')
    
    return centroids  

def normalize(array,max_value=1.0):
    return (array + max_value)/(2*max_value)
    
def denormalize(array,max_value=1.0):
    return max_value*(2*array-1)
    
def down_scale(array,max_value):
    return array/max_value

def up_scale(array,max_value,stride):
    return array*max_value/stride
    
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
        y_c, x_c = int(np.mean(instance_pixels[0])), int(np.mean(instance_pixels[1]))
        # step B - calculate dist_x, dist_y of each pixel of instance from its center
        y_dist = (-y_c + instance_pixels[0])
        x_dist = (-x_c + instance_pixels[1])
        for y, x, d_y, d_x in zip(instance_pixels[0], instance_pixels[1], y_dist, x_dist):
            centroid_regression[y, x, :2] = [d_y, d_x]  # remember - y is distance in rows, x in columns
            
            
    centroids = convert_centroids(centroid_regression,op='down_scale')               
    normalized_centroid_regression = convert_centroids(centroids,op='normalize')
    #plt.imshow(centroids[:,:,0])
    #plt.show()
    #plt.imshow(centroids[:,:,1])
    #plt.show()
    #plt.imshow(centroids[:,:,2])
    #plt.show()
    #denormalized_centroid_regression = convert_centroids(normalized_centroid_regression,op='denormalize')    
    return normalized_centroid_regression

def get_centers(centroids):
    dx_img, dy_img = centroids[:,:,1], centroids[:,:,0]
    centers = []
    instance_img = np.zeros_like(dx_img)
    for h, rows in enumerate(zip(dx_img, dy_img)):
        dx_row, dy_row = rows[0], rows[1]
        for w, deltas in enumerate(zip(dx_row, dy_row)):
            dx, dy = deltas[0], deltas[1]
            if dx!=0 and dy!=0:
                center = (int(w-dx),int(h-dy))
                if center not in centers:
                    centers.append(center)
                instance_img[h,w] = centers.index(center)
    return centers, instance_img

def convert_instance_to_clusters(path_to_annotations):
    for root, dirs, names in os.walk(path_to_annotations, topdown=False):
        for name in tqdm(names):
            if name.endswith("instanceIds.png") :
                identifier = name.split('.')[0]
                image = cv2.imread(os.path.join(root,name),-1)
                centroids = regress_centers(image)
                denormalized_centroids = convert_centroids(centroids,op='denormalize')               
                up_centroids= convert_centroids(denormalized_centroids,op='up_scale')
                centers, instance_img = get_centers(up_centroids)
                plt.imshow(to_rgb(instance_img))
                plt.show()
                '''
                for i,center in enumerate(centers):
                    print(i,center)
                    plt.imshow(instance_img==i)
                    plt.show()
                '''
                #clusters_img = calc_clusters_img(up_centroids)
                #plt.imshow(clusters_img)
                #plt.show()
                break
                #np.savez_compressed(os.path.join(root,identifier),centroids)



convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/train')
#convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/val')
