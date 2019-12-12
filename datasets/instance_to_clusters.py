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
plt.rcParams['image.cmap'] = 'jet'
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

def get_2d_box_from_instance(xs, ys):
    vertex_1 = (np.min(xs), np.min(ys))
    vertex_2 = (np.max(xs), np.max(ys))
    return vertex_1, vertex_2

def get_2d_box_center(xs, ys):
    vertex_1, vertex_2 = get_2d_box_from_instance(xs, ys)
    return (int((vertex_1[0]+vertex_2[0])/2),int((vertex_1[1]+vertex_2[1])/2))

def get_instance_hw(xs, ys):
    vertex_1, vertex_2 = get_2d_box_from_instance(xs, ys)
    return (abs(vertex_1[0]-vertex_2[0]), abs(vertex_1[1]-vertex_2[1]))

def compute_centroid_vectors(instance_image):
    alpha = 2.0
    centroids = np.zeros(instance_image.shape + (2,))
    heatmap = np.ones(instance_image.shape + (2,) )
    #img = np.zeros(instance_image.shape + (3,))
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        if value>1000:
            w, h = get_instance_hw(xs, ys)
            heatmap[xs, ys] = np.array(w,h)
            #alpha_h, alpha_w = int(h*alpha), int(w*alpha)
            #if alpha_h%2 == 0:
            #    alpha_h -= 1
            #if alpha_w%2 == 0:
            #    alpha_w -=1
            
        #    box_center = get_2d_box_center(ys, xs)
        #    pt1, pt2 = get_2d_box_from_instance(ys, xs)        
        #    img = cv2.rectangle(img, pt1, pt2, (255,255,255), 5)
        #    img = cv2.circle(img, center_coordinates, 3, (0,255,0), 5) 
        #    img = cv2.circle(img, box_center, 3, (255,0,0), 5) 
        centroids[xs, ys] = np.array((np.mean(ys), np.mean(xs)))
    h, w = instance_image.shape[0], instance_image.shape[1]
    coordinates = np.zeros(instance_image.shape + (2,))
    g1, g2 = np.mgrid[range(h), range(w)]
    coordinates[:, :, 1] = g1
    coordinates[:, :, 0] = g2
    vecs = centroids - coordinates
    heatmap_ = heatmap - np.abs(vecs)*alpha
    heatmap_ = np.clip(heatmap_, 0, np.max(heatmap_))
    heatmap_ = heatmap_/heatmap
    heatmap = heatmap_[:,:,0]*heatmap_[:,:,1]
    mask = np.ma.masked_where(instance_image >= 1000, instance_image)
    
    if len(mask.mask.shape) > 1:
        mask = np.asarray(mask.mask, dtype=np.uint8)
    elif mask.mask is False:
        mask = np.zeros(instance_image.shape, dtype=np.uint8)
    else:
        mask = np.ones(instance_image.shape, dtype=np.uint8)
    
    heatmap = heatmap*mask
    
    mask = np.stack((mask, mask))

    # We load the images as H x W x channel, but we need channel x H x W.
    # We don't need to transpose the mask as it has no channels.
    vecs = np.transpose(vecs, (2, 0, 1))
    vecs = vecs*mask
    vecs = np.transpose(vecs, (1, 2, 0))
    
    plt.figure()
    plt.subplot(311)
    plt.imshow(vecs[:,:,0])
    plt.title('x')
    plt.subplot(312)
    plt.imshow(vecs[:,:,1])
    plt.title('y')
    plt.subplot(313)
    plt.imshow(heatmap)
    plt.title('heatmap')
    plt.show()
    
    vecs = vecs - np.min(vecs)
    vecs = vecs/np.max(vecs)

    return vecs, mask

    
def convert_instance_to_clusters(path_to_annotations):
    for root, dirs, names in os.walk(path_to_annotations, topdown=False):
        for name in names:
            if name.endswith("instanceIds.png") :
                identifier = name.split('.')[0]
                if os.path.exists(os.path.join(root,identifier)+'.npz'):
                    os.remove(os.path.join(root,identifier)+'.npz')
                tag = '_'.join(identifier.split('_')[:-1])
                regression_tag = tag + '_instanceRegression'
                probability_tag = tag + '_instanceProbs'
                heatmap_tag = tag + '_instanceHeatmaps'
                image = cv2.imread(os.path.join(root, name),-1)
                vecs, mask = compute_centroid_vectors(image)
                #centroids, instance_prob, instance_heatmap = regress_centers(image)
                #denormalized = convert_centroids(centroids,op='denormalize')
                #centroids  = convert_centroids(denormalized,op='up_scale',stride=4)
                #vecs = vecs*mask
                #vecs = vecs - np.min(vecs)
                #vecs = vecs/np.max(vecs)
                '''
                plt.figure()
                plt.subplot(221)
                plt.imshow(vecs[0])
                plt.title('x')
                plt.subplot(222)
                plt.imshow(vecs[1])
                plt.title('y')
                plt.subplot(223)
                plt.imshow(mask[0])
                plt.title('probs')
                plt.subplot(224)
                plt.imshow(mask[1])
                plt.title('heatmaps')
                plt.show()
                break
                '''
                #np.savez_compressed(os.path.join(root,regression_tag),centroids)
                #np.savez_compressed(os.path.join(root,probability_tag),instance_prob)
                #np.savez_compressed(os.path.join(root,heatmap_tag),instance_heatmap)

def get_color(num):
    return np.random.randint(0, 255, size=(3))

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

annot_path = "/home/sumche/datasets/Cityscapes/gtFine/train/cologne/cologne_000000_000019_gtFine_instanceRegression.npz"

def _load_npz(path):
    im_array = np.load(path)['arr_0']
    #zeros = np.zeros(im_array.shape[:-1])
    #zeros = np.expand_dims(zeros, axis = -1)
    #im_array = np.concatenate((im_array, zeros), axis = -1)
    return im_array, Image.fromarray(np.uint8(im_array*255))
'''
a, centroids = _load_npz(annot_path)
to_tensor = transforms.ToTensor()
resize = transforms.Resize(size=512, interpolation=2)
to_pil = transforms.ToPILImage()
#centroids = to_tensor(resize(to_pil((a*255.0).astype(np.uint8))))
centroids = to_tensor(resize(centroids))

centroids = centroids.permute((1,2,0)).numpy()

centroids = convert_centroids(centroids, op='up_scale',stride=2)
centroids = convert_centroids(centroids, op='denormalize')
plt.figure()
plt.subplot(211)
plt.imshow(centroids[:,:,0])
plt.title('x')
#plt.subplot(212)
#plt.imshow(centroids[:,:,1])
#plt.title('y')
plt.show()
'''
convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/train')
#convert_instance_to_clusters('/home/sumche/datasets/Cityscapes/gtFine/val')
