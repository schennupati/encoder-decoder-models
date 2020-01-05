#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:25:54 2019

@author: sumche
"""
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.im_utils import cat_labels,prob_labels, decode_segmap
import torch
import torch.nn.functional as F

def compute_instance_contours(instance_image):
    contours = np.zeros(instance_image.shape)
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        if value>1000:
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


root = '/home/sumche/datasets/Cityscapes'
identifier = 'lindau_000037_000019'
img = 'leftImg8bit'
annot = 'gtFine'
split = 'val'
city = 'lindau'

img_tag = '_leftImg8bit.png'
seg_tag = '_gtFine_color.png'
instance_tag = '_gtFine_instanceIds.png'

def plot_images(identifier):

    inst_img = cv2.imread(os.path.join(root,annot,split,city,identifier+instance_tag),-1)
    contours = compute_contour(inst_img)
    
plot_images(identifier)
