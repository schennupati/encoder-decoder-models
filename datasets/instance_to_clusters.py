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
from tqdm import tqdm
from torchvision import transforms
#from clusters_to_instances import to_rgb
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
    

def compute_centroid_vectors(instance_image):
    centroids = np.zeros(instance_image.shape + (2,))
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        centroids[xs, ys] = np.array((np.mean(xs), np.mean(ys)))
        
    coordinates = np.zeros(instance_image.shape + (2,))
    g1, g2 = np.mgrid[range(instance_image.shape[0]), range(instance_image.shape[1])]
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
    #vecs = np.transpose(vecs, (2, 0, 1))
    #vecs = vecs*mask
    #vecs = np.transpose(vecs, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    mag_2d = np.linalg.norm(vecs,axis=-1)
    norm_vecs = np.zeros(mag_2d.shape + (2,))
    norm_vecs[:,:,0] = vecs[:,:,0]/mag_2d
    norm_vecs[:,:,1] = vecs[:,:,1]/mag_2d
    norm_vecs = norm_vecs*mask
    x_axis = unit_vector([0, 1])
    sign = np.sign(norm_vecs[:,:,0])
    angle = (np.arccos(np.clip((norm_vecs)*x_axis, -1.0, 1.0))/np.pi)[:,:,1]
    angle = (angle*sign + 1)*mask[:,:,0]
    angle /=2
    mag_2d = mag_2d*mask[:,:,0]
    print(np.unique(angle), np.unique(mag_2d))
    plt.figure()
    plt.subplot(211)
    plt.imshow(mag_2d)
    plt.title('x')
    plt.subplot(212)
    plt.imshow(angle)
    plt.title('y')
    plt.show()
    
    vecs = vecs - np.min(vecs)
    vecs = vecs/np.max(vecs)

    return vecs, mask

def regress_centers(Image):
    instances = np.unique(Image)
    instances = instances[instances > 1000]

    mask = np.zeros_like(Image)
    mask[np.where(Image > 1000)] = 1

    centroid_regression = np.zeros([Image.shape[0], Image.shape[1], 2])
    instance_prob = np.zeros([Image.shape[0], Image.shape[1], 1])
    instance_heatmap = np.zeros([Image.shape[0], Image.shape[1], 1])
    instance_prob = mask

    for instance in instances:
        # step A - get a center (x,y) for each instance
        instance_pixels = np.array(np.where(Image == instance))
        x_c, y_c = int(np.mean(instance_pixels[0])), int(np.mean(instance_pixels[1]))
        instance_heatmap[x_c, y_c] = 1
        # step B - calculate dist_x, dist_y of each pixel of instance from its center
        x_dist = (-x_c + instance_pixels[0])
        y_dist = (-y_c + instance_pixels[1])
        instance = np.array((instance_pixels[0,:]-x_c,instance_pixels[1,:]-y_c))
        x_axis = unit_vector([0, 1])
        mag_2d = np.linalg.norm(instance,axis=0)
        instance_norm = instance/mag_2d
        angle_2d = (np.arccos(np.clip(np.transpose(instance/mag_2d)*x_axis, -1.0, 1.0))/np.pi)[:,1]
        centroid_regression[instance_pixels] = mag_2d
        #for x, y, d_x, d_y in zip(instance_pixels[0], instance_pixels[1], x_dist, y_dist):
        #    x_axis = unit_vector([0, 1])
        #    instance = unit_vector([x-x_c, y-y_c])
        #    angle = (np.arccos(np.clip(np.dot(instance, x_axis), -1.0, 1.0)))/np.pi
        #    mag = np.sqrt((x-x_c)**2 + (y-y_c)**2)
        #    centroid_regression[x, y] = [d_y, d_x]
        #    centroid_regression[x, y] = [mag, angle]
      
    instance_heatmap = cv2.GaussianBlur(instance_heatmap,(9,9),0)
    instance_heatmap = instance_heatmap / instance_heatmap.max()
    #centroids = convert_centroids(centroid_regression,op='down_scale')             
    #normalized_centroid_regression = convert_centroids(centroids,op='normalize') 
    return centroid_regression, instance_prob, instance_heatmap

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_centers(centroids):
    dx_img, dy_img = centroids[:,:,1], centroids[:,:,0]
    centers = []
    instance_img = np.zeros_like(dx_img)
    im_h, im_w, _ = centroids.shape
    for h, rows in tqdm(enumerate(zip(dx_img, dy_img))):
        dx_row, dy_row = rows[0], rows[1]
        for w, deltas in enumerate(zip(dx_row, dy_row)):
            dx, dy = deltas[0], deltas[1]
            if abs(dx)>0.1 and abs(dy)>0.1:
                center = (int(w-dx),int(h-dy))
                if len(centers) !=0 :
                    closest_distance,closest_center = closest_node(center, centers)
                    if closest_distance > im_w/20 :
                        centers.append(center)
                    else: 
                        center = centers[closest_center]
                else:
                    centers.append(center)
                instance_img[h,w] = centers.index(center)
    return instance_img

def get_centers_vectorized(centroids):
    dx_img, dy_img = centroids[:,:,1], centroids[:,:,0]
    center_x, center_y = np.zeros_like(dx_img,dtype=int),np.zeros_like(dy_img,dtype=int)
    mask = (dx_img!=0).astype(int)*(dy_img!=0).astype(int)
    h,w = center_x.shape
    
    for i in range(w):
        center_x[:,i] = i
    for j in range(h):
        center_y[j,:] = j
        
    center_x = mask*(center_x - dx_img.astype(int))
    center_y = mask*(center_y - dy_img.astype(int))
    centers_new = np.where(((center_x * center_y) !=0),(center_x,center_y),0)
    instances = np.zeros_like(center_x)
    instance_id = 0
    for h in tqdm(np.unique(center_y)):
        for w in np.unique(center_x):
            instance_x = (centers_new[0,:,:]==w).astype(int)
            instance_y = (centers_new[1,:,:]==h).astype(int)
            if np.sum(instance_x*instance_y)!=0 and h*w !=0:
                instance_id +=1
                instances[np.where(instance_x*instance_y==1)] = instance_id
                
    return instances
    
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
                image = cv2.imread(os.path.join(root,name),-1)
                vecs, mask = compute_centroid_vectors(image)
                #centroids, instance_prob, instance_heatmap = regress_centers(image)
                #denormalized = convert_centroids(centroids,op='denormalize')
                #centroids  = convert_centroids(denormalized,op='up_scale',stride=4)
                vecs = vecs*mask
                vecs = vecs - np.min(vecs)
                vecs = vecs/np.max(vecs)
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
