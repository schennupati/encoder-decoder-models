#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 07:39:01 2019

@author: sumche
"""
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    Label('unlabeled',  0, 255, 'void', 0, False, True, (0,  0,  0)),
    Label('ego vehicle',  1, 255, 'void', 0, False, True, (0,  0,  0)),
    Label('rect border',  2, 255, 'void', 0, False, True, (0,  0,  0)),
    Label('out of roi',  3, 255, 'void', 0, False, True, (0,  0,  0)),
    Label('static',  4, 255, 'void', 0, False, True, (0,  0,  0)),
    Label('dynamic',  5, 255, 'void', 0, False, True, (111, 74,  0)),
    Label('ground',  6, 255, 'void', 0, False, True, (81,  0, 81)),
    Label('road',  7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk',  8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking',  9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guadrail', 14, 255, 'construction', 2, False, True, (180, 165, 18)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220,  0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255,  0,  0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0,  0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0,  0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0,  0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0,  0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', 34,  -1, 'vehicle', 7, False, True, (0, 0, 142)),
    Label('boundary', 35, 19, 'vehicle', 7, False, False, (255, 255, 0)),
    Label('t-boundary', 36, 20, 'vehicle', 7, False, False, (255, 0, 255))]

inst_labels = [
    Label('unlabeled', 0, 0, 'void', 0, False, True, (255, 255, 255)),
    Label('person', 24, 1, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 2, 'human', 6, True, False, (255,  1,  1)),
    Label('car', 26, 3, 'vehicle', 7, True, False, (1,  1, 142)),
    Label('truck', 27, 4, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 28, 5, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('train', 31, 6, 'vehicle', 7, True, False, (1, 80, 100)),
    Label('motorcycle', 32, 7, 'vehicle', 7, True, False, (1, 1, 230)),
    Label('bicycle', 33, 8, 'vehicle', 7, True, False, (119, 11, 32))]


cat_labels = [
    Label('road',  0, 0, 'flat', 1, False, False, (255, 255, 255)),
    Label('sidewalk',  1, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('building',  2, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall',  3, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence',  4, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('pole',  5, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('traffic light',  6, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign',  7, 7, 'object', 3, False, False, (220, 220,  0)),
    Label('vegetation',  8, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain',  9, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 10, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 11, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 12, 12, 'human', 6, True, False, (255,  0,  0)),
    Label('car', 13, 13, 'vehicle', 7, True, False, (0,  0, 142)),
    Label('truck', 14, 14, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 15, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('train', 16, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 17, 17, 'vehicle', 7, True, False, (0,  0, 230)),
    Label('bicycle', 18, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('boundary', 19, 19, 'vehicle', 7, False, False, (255, 255, 0)),
    Label('t-boundary', 36, 20, 'vehicle', 7, False, False, (255, 0, 255))]

prob_labels = [
    Label('stuff',  0, 0, 'flat', 1, False, False, (255, 255, 255)),
    Label('thing',  1, 1, 'flat', 1, True, False, (255,  0,  0))]


def get_trainId(id, labels=labels):
    for i, label in enumerate(labels):
        if label.trainId == id:
            return i


def decode_segmap(image, nc=21, labels=labels):

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        if l in image:
            idx = np.where(image == l)
            l = get_trainId(l, labels=labels)
            r[idx] = labels[l].color[0]
            g[idx] = labels[l].color[1]
            b[idx] = labels[l].color[2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
instance_trainId = {label.id: label.trainId for label in inst_labels}

# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# category to list of label objects
category2labels = {}
# Color to panoptic ID
panid2label = { l.color[0]+ 256*l.color[1]+ 256*256*l.color[2] : l for l in labels }


for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


def get_color_inst(inst_seg):
    colour_inst = np.zeros((inst_seg.shape[0], inst_seg.shape[1], 3))
    colour_inst[:, :, :2] = inst_seg
    colour_inst = colour_inst - np.min(colour_inst)
    colour_inst = colour_inst / np.max(colour_inst)

    return colour_inst


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
    return np.stack([rgb_im[0], rgb_im[1], rgb_im[2]], axis=-1)


def get_color(num):
    return np.random.randint(0, 255, size=(3))
