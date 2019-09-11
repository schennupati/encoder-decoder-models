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

path_to_annotations = '/home/sumche/datasets/Cityscapes/gtFine/val'


def get_total_files_count(path,ext='.png'):
    count = 0
    for root, dirs, names in os.walk(path, topdown=False):
        for name in names:
            if name.endswith(ext):
                count += 1 
    return count

annotations_count = get_total_files_count(path_to_annotations,'instanceIds.png')

def regress_centers(Image):
    instances = np.unique(Image)
    instances = instances[instances > 1000]

    mask = np.zeros_like(Image)
    mask[np.where(Image > 1000)] = 1

    centroid_regression = np.zeros([Image.shape[0], Image.shape[1], 3]).astype(np.int16)
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
    return centroid_regression

for root, dirs, names in os.walk(path_to_annotations, topdown=False):
    for name in tqdm(names):
        if name.endswith("instanceIds.png") :
            #os.remove(os.path.join(root,name))
            identifier = name.split('.')[0]
            image = cv2.imread(os.path.join(root,name),-1)

            centroids = regress_centers(image)
            np.savez_compressed(os.path.join(root,identifier),centroids)