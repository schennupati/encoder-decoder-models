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
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
path_to_annotations = '/home/sumche/datasets/Cityscapes/gtFine/val'

def get_total_files_count(path,ext='.png'):
    count = 0
    for root, dirs, names in os.walk(path, topdown=False):
        for name in names:
            if name.endswith(ext):
                count += 1 
    return count

annotations_count = get_total_files_count(path_to_annotations,'instanceIds.png')

def convert_centroids(centroids,op='normalize',stride=1):
    
    #converted_centroid_regression = torch.zeros_like(centroids) if tensor else np.zeros_like(centroids)
    if op == 'normalize':
        centroids[:,:,1] = normalize(centroids[:,:,1])
        centroids[:,:,0] = normalize(centroids[:,:,0])
    elif op == 'denormalize':
        centroids[:,:,1] = denormalize(centroids[:,:,1])
        centroids[:,:,0] = denormalize(centroids[:,:,0])

    elif op == 'down_scale':
        centroids[:,:,0] = down_scale(centroids[:,:,0],max_value=2048)
        centroids[:,:,1] = down_scale(centroids[:,:,1],max_value=1024)
    elif op == 'up_scale':
        centroids[:,:,0] = up_scale(centroids[:,:,0],max_value=2048,stride=stride)
        centroids[:,:,1] = up_scale(centroids[:,:,1],max_value=1024,stride=stride)
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

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
    return np.min(dist_2), np.argmin(dist_2)

def get_2d_bbox_from_instance(xs, ys):
    vertex_1 = (torch.min(xs), torch.min(ys))
    vertex_2 = (torch.max(xs), torch.max(ys))
    return vertex_1, vertex_2

def get_instance_hw(xs, ys):
    vertex_1, vertex_2 = get_2d_bbox_from_instance(xs, ys)
    return (abs(vertex_1[0]-vertex_2[0]), abs(vertex_1[1]-vertex_2[1]))

def get_2d_box_center(xs, ys):
    vertex_1, vertex_2 = get_2d_box_from_instance(xs, ys)
    return (int((vertex_1[0]+vertex_2[0])/2),int((vertex_1[1]+vertex_2[1])/2))

def save_plot(img, name):
    WIDTH = 100.0  # the number latex spits out
    FACTOR = 0.45  # the fraction of the width you'd like the figure to occupy
    figwidthpt  = WIDTH * FACTOR

    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

    figwidthin  = figwidthpt  # figure width in inches
    figheightin = figwidthin * golden_ratio   # figure height in inches
    fig_dims    = [figwidthin, figheightin] # fig dims as a list
    fig = plt.figure(figsize=fig_dims)
    plt.imshow(img)
    plt.axis('off')
    fig.savefig(name, bbox_inches='tight')

def get_color_inst(inst_seg):
    colour_inst = np.zeros((inst_seg.shape[0], inst_seg.shape[1], 3))
    colour_inst[:,:,:2] = inst_seg
    colour_inst = colour_inst - np.min(colour_inst)
    colour_inst = colour_inst / np.max(colour_inst)
    
    return colour_inst    

def compute_centroid_vectors(instance_image):
    alpha = 5.0
    centroids = np.zeros(instance_image.shape + (2,))
    w_h = np.ones(instance_image.shape + (2,))
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        if value>1000:
            w, h = get_instance_hw(xs, ys)
            if w!=0 and h!=0:
                w_h[xs, ys,0], w_h[xs, ys,1]  = w, h
        centroids[xs, ys] = np.array((np.mean(ys), np.mean(xs)))
    h, w = instance_image.shape[0], instance_image.shape[1]
    coordinates = np.zeros(instance_image.shape + (2,))
    g1, g2 = np.mgrid[range(h), range(w)]
    coordinates[:, :, 0] = g1
    coordinates[:, :, 1] = g2
    vecs = centroids - coordinates
    mask = np.ma.masked_where(instance_image >= 1000, instance_image)
    if len(mask.mask.shape) > 1:
        mask = np.asarray(mask.mask, dtype=np.uint8)
    elif mask.mask is False:
        mask = np.zeros(instance_image.shape, dtype=np.uint8)
    else:
        mask = np.ones(instance_image.shape, dtype=np.uint8)
    mask = np.stack((mask, mask))
    # We load the images as H x W x channel, but we need channel x H x W.
    # We don't need to transpose the mask as it has no channels.
    vecs = np.transpose(vecs, (2, 0, 1))
    vecs = vecs*mask
    vecs = np.transpose(vecs, (1, 2, 0))
    heatmap_ = w_h - np.abs(vecs)*alpha
    heatmap_ = np.clip(heatmap_, 0, np.max(heatmap_))
    heatmap_[:,:,0] /= w_h[:,:,0]
    heatmap_[:,:,1] /= w_h[:,:,1]
    heatmap_t = heatmap_[:,:,0]*heatmap_[:,:,1]
    heatmap_t = heatmap_t*mask[0]
    return vecs, mask, heatmap_t

def compute_centroid_vector_torch(instance_image):
    alpha = 5.0
    instance_image_tensor = torch.Tensor(instance_image.astype(np.int16))
    centroids_t = torch.zeros(instance_image.shape + (2,))
    w_h = torch.ones(instance_image.shape + (2,))
    for value in torch.unique(instance_image_tensor):
        xsys = torch.nonzero(instance_image_tensor == value)
        xs, ys = xsys[:, 0], xsys[:, 1]
        centroids_t[xs, ys] = torch.stack((torch.mean(xs.float()), torch.mean(ys.float())))
        if value > 1000:
            #pdb.set_trace()
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
    #print((torch.min(vecs), torch.max(vecs)), (torch.min(heatmap_), torch.max(heatmap_)))
    return vecs.permute(2,0,1).numpy(), mask.numpy(), heatmap_t.numpy()

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
    kernel = np.ones((9,9), np.uint8)  
    contours = cv2.dilate(contours, kernel, iterations=1)
    return contours

def get_color(num):
    return np.random.randint(0, 255, size=(3))

def to_rgb(bw_im):
    instances = np.unique(bw_im,axis=2)
    instances = instances[instances != 0]
    rgb_im = [np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int)]
    for instance in instances:
        color = get_color(instance)
        rgb_im[0][instance == bw_im] = color[0]
        rgb_im[1][instance == bw_im] = color[1]
        rgb_im[2][instance == bw_im] = color[2]
    return np.stack([rgb_im[0],rgb_im[1],rgb_im[2]],axis=-1)

labels ={}
classes = ['void','road','sidewalk','building','wall','fence',
           'pole','traffic light','traffic sign','vegetation',
           'terrain','sky','person','rider','car','truck','bus',
           'train','motorcycle','bicycle']
labels['void']          = [  0,  0,  0]
labels['road']          = [128, 64,128]
labels['sidewalk']      = [244, 35,232]
labels['building']      = [ 70, 70, 70]
labels['wall']          = [102,102,156]
labels['fence']         = [190,153,153]
labels['pole']          = [153,153,153]
labels['traffic light'] = [250,170, 30]
labels['traffic sign']  = [220,220,  0]
labels['vegetation']    = [107,142, 35]
labels['terrain']       = [152,251,152]
labels['sky']           = [ 70,130,180]
labels['person']        = [220, 20, 60]
labels['rider']         = [255,  0,  0]
labels['car']           = [  0,  0,142]
labels['truck']         = [  0,  0, 70]
labels['bus']           = [  0, 60,100]
labels['train']         = [  0, 80,100]
labels['motorcycle']    = [  0,  0,230]
labels['bicycle']       = [119, 11, 32]

def getSegmentationArr(img,classes=classes,labels=labels):
    
    nClasses = len(classes)
    seg_labels = np.zeros((img.shape[0],img.shape[1],nClasses))
    for c in range(nClasses):
        seg_labels[:,:,c] = (np.all(img==labels[classes[c]],axis=2))
    return seg_labels

def getSegImg(img,classes=classes,labels=labels):
    nClasses = len(classes)
    H = img.shape[0]
    W = img.shape[1]
    segImg = np.zeros((H,W,3))
    img = np.argmax(img,axis=2)
    for c in range(nClasses):
        color = labels[classes[c]]
        segImg[:,:,0] += ((img[:,:] == c )*( color[0] )).astype(np.uint8)
        segImg[:,:,1] += ((img[:,:] == c )*( color[1] )).astype(np.uint8)
        segImg[:,:,2] += ((img[:,:] == c )*( color[2] )).astype(np.uint8)
    return segImg.astype(np.uint8)
#convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/train')
#convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/val')

def to_rgb(bw_im):
    instances = np.unique(bw_im)
    instances = instances[instances != 0]
    rgb_im = [np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int)]
    for instance in instances:
        color = get_color(instance)
        rgb_im[0][instance == bw_im] = color[0]
        rgb_im[1][instance == bw_im] = color[1]
        rgb_im[2][instance == bw_im] = color[2]
    return np.stack([rgb_im[0],rgb_im[1],rgb_im[2]],axis=-1)

def get_clusters(inst_seg, mask, heatmap):
    inst_img = np.zeros_like(mask)
    h,w = mask.shape
    centroids = get_centroids(inst_seg, mask)
    heatmap = (heatmap*255.0).astype(np.uint8)
    _, labels = cv2.connectedComponents(heatmap)
    for i in range(h):
        for j in range(w):
            idx = tuple(centroids[:,i,j])
            inst_img[i,j] = labels[idx[0], idx[1]]
    return to_rgb(inst_img)

def get_centroids(inst_seg, mask):
    coordinates = np.zeros_like(inst_seg)
    #print(np.unique(inst_seg))
    g1, g2 = np.mgrid[range(inst_seg.shape[1]), range(inst_seg.shape[2])]
    coordinates[0, :, :] = g1
    coordinates[1, :, :] = g2
    centroids = coordinates - inst_seg
    centroids[0, :, :] = np.clip(centroids[0, :, :]*mask, 0,inst_seg.shape[1]-1)
    centroids[1, :, :] = np.clip(centroids[1, :, :]*mask, 0,inst_seg.shape[2]-1)
    return centroids.astype(np.int16)

root = '/home/sumche/datasets/Cityscapes'
identifier = 'lindau_000037_000019'
img = 'leftImg8bit'
annot = 'gtFine'
split = 'val'
city = 'lindau'

img_tag = '_leftImg8bit.png'
seg_tag = '_gtFine_color.png'
instance_tag = '_gtFine_instanceIds.png'

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def plot_images(identifier):
    
    raw_img = cv2.imread(os.path.join(root,img,split,city, identifier+img_tag),-1)
    b,g,r = cv2.split(raw_img)       # get b,g,r
    raw_img = cv2.merge([r,g,b])     # switch it to rgb
    #save_plot(raw_img, identifier +'_rgb_image.png')
    
    seg_img = cv2.imread(os.path.join(root,annot,split,city,identifier+seg_tag))
    b,g,r = cv2.split(seg_img)       # get b,g,r
    seg_img = cv2.merge([r,g,b])     # switch it to rgb
    seg_img = getSegmentationArr(seg_img)
    seg_img = getSegImg(seg_img)
    #save_plot(seg_img, identifier +'_seg_image.png')

    inst_img = cv2.imread(os.path.join(root,annot,split,city,identifier+instance_tag),-1)
    contours = compute_instance_contours(inst_img)
    #save_plot(contours, identifier +'_contours_image.png')
    
    vecs, mask, heatmap = compute_centroid_vector_torch(inst_img)
    #save_plot(get_color_inst(vecs), identifier +'_instance_offsets.png')
    save_plot(heatmap, identifier +'_instance_centroid.png')
    #save_plot(mask, identifier +'_instance_mask_image.png')
    
    inst_img = get_clusters(vecs, mask, heatmap)
    plt.imshow(inst_img)
    plt.show()
    diff = np.zeros_like(seg_img)
    for i in range(3):
        diff[:,:,i] = seg_img[:,:,i]*(1-contours)*mask[0]

    save_plot(diff, identifier +'_diff_image.png')
    
    ret, labels = cv2.connectedComponents(diff[:,:,2])
    instance_img = imshow_components(labels)
    #save_plot(instance_img, identifier +'_instance_seg.png')

    for i in range(3):
        seg_img[:,:,i] = seg_img[:,:,i]*(1-mask[0])

    pan_img = seg_img + instance_img
    #save_plot(pan_img, identifier +'_panoptic_seg.png')
#plot_images(identifier)
    