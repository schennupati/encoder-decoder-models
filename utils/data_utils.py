"""data utilities to support target conversion, postprocs etc."""

import copy
import torch
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import math
import torch.nn.functional as F
import torchvision

from utils.im_utils import labels, prob_labels, trainId2label, id2label, \
    decode_segmap, inst_labels, name2label, get_color_inst, instance_trainId
from utils.constants import TASKS, MODEL, OUTPUTS, TYPE, SEMANTIC, INSTANCE, \
    DISPARITY, PANOPTIC, ACTIVE, OUTPUTS_TO_TASK, INSTANCE_REGRESSION, \
    INSTANCE_PROBS, INSTANCE_HEATMAP, INSTANCE_CONTOUR, POSTPROCS, INPUTS, \
    INSTANCE_IMAGE, PANOPTIC, PANOPTIC_IMAGE, SEGMENT_INFO, DOUBLE, LONG, \
    FLOAT, POSTPROC, ARGMAX
from instance_to_clusters import get_clusters, imshow_components, to_rgb


class TargetGenerator():
    def __init__(self, cfg, useTrainId=False, postprocs=False,
                 scale_factor=1.0, min_pixel_ratio=256):
        self.tasks = list(get_active_tasks(cfg[MODEL][OUTPUTS]).keys())
        self.labels = {}
        self.postprocs = postprocs
        self.scale_factor = cfg[MODEL][OUTPUTS]['bounding_box']['scale_factor']
        self.min_pixel_ratio = min_pixel_ratio

    def prepare_targets(self, inst):
        if len(inst.shape) == 3:
            n, h, w = inst.shape
        else:
            n = 1
            h, w = inst.shape

        scaled_h = int(h * self.scale_factor)
        scaled_w = int(w * self.scale_factor)
        self.min_height = np.max([(scaled_h / self.min_pixel_ratio), 2])
        self.min_width = np.max([(scaled_w / self.min_pixel_ratio), 4])
        self.min_area = self.min_height * self.min_width

        self.instance_cnt = np.zeros_like(
            inst) if 'instance_contour' in self.tasks else None
        if 'semantic_with_instance' in self.tasks:
            self.semantic_cnt = np.zeros_like(inst)
            self.instance_cnt = np.zeros_like(inst)
            self.touching_cnt = np.zeros_like(inst)
        if ('panoptic' in self.tasks) or self.postprocs:
            self.panoptic_seg = np.zeros((inst.shape + (3,)), dtype=np.uint8)
            self.segmInfo = []
        else:
            self.panoptic_seg = None
            self.segmInfo = []
        if 'instance_regression' in self.tasks:
            if n != 1:
                self.instance_vecs = np.zeros((n, 2, h, w), dtype=np.int16)
                self.centroids = np.zeros((n, 2, h, w), dtype=np.int16)
                self.vec_loss_mask = np.ones((n, h, w), dtype=np.float32)
            else:
                self.instance_vecs = np.zeros((2, h, w), dtype=np.int16)
                self.centroids = np.zeros((2, h, w), dtype=np.int16)
                self.vec_loss_mask = np.ones((h, w), dtype=np.float32)
        else:
            self.instance_vecs = None
            self.centroids = None
        if 'bounding_box' in self.tasks:
            if n != 1:
                self.bbox_offsets = np.zeros((n, 4, scaled_h, scaled_w),
                                             dtype=np.int16)
                self.bbox_labels = np.zeros((n, scaled_h, scaled_w),
                                            dtype=np.int16)
                self.bboxes = {i: [] for i in range(n)}
                self.bbox_loss_mask = np.ones((n, scaled_h, scaled_w),
                                              dtype=np.float32)
            else:
                self.bbox_offsets = np.zeros((4, scaled_h, scaled_w),
                                             dtype=np.int16)
                self.bbox_labels = np.zeros((scaled_h, scaled_w),
                                            dtype=np.int16)
                self.bboxes = []
                self.bbox_loss_mask = np.ones((scaled_h, scaled_w),
                                              dtype=np.float32)
        else:
            self.bbox_offsets = None

    def return_targets(self):
        if 'semantic' in self.tasks:
            self.labels.update(
                {'semantic': torch.tensor(self.semantic_seg).cuda()})
        if 'semantic_with_instance' in self.tasks:
            self.labels.update(
                {'semantic_with_instance':
                 torch.tensor(self.semantic_seg).cuda(),
                 'touching_boundaries':
                 torch.tensor(self.touching_cnt).cuda()})
        if ('panoptic' in self.tasks) or self.postprocs:
            self.labels.update(
                {'panoptic': {'panoptic': self.panoptic_seg,
                              'segmInfo': self.segmInfo}})
        if 'instance_regression' in self.tasks:
            self.labels.update(
                {'instance_regression':
                 {'targets':
                  torch.tensor(self.instance_vecs).float().cuda(),
                  'loss_mask':
                  torch.tensor(self.vec_loss_mask).float().cuda()}})
        if 'bounding_box' in self.tasks:
            self.labels.update(
                {'bounding_box':
                 {'targets':
                  {'class': torch.tensor(self.bbox_labels).float().cuda(),
                   'offsets': torch.tensor(self.bbox_offsets).float().cuda()
                   },
                  'loss_mask':
                  torch.tensor(self.bbox_loss_mask).float().cuda(),
                  'bboxes': self.bboxes
                  }
                 })

        return self.labels

    def generate_targets(self, semantic, inst, useTrainId=False):
        # (TODO): generate instance vecs, area weight map, boundary weight map,
        # bbox offsets in support region.
        inst = inst.numpy() if torch.is_tensor(inst) else inst
        semantic = semantic.numpy() if torch.is_tensor(semantic) else semantic
        inst = inst.squeeze(1)
        self.semantic_seg = (semantic*255).squeeze(1)
        self.prepare_targets(inst)
        segmentIds = np.unique(inst)
        for segmentId in segmentIds:

            if segmentId < 1000:
                semanticId = segmentId
                isCrowd = 1
            else:
                semanticId = segmentId // 1000
                isCrowd = 0

            labelInfo = trainId2label[semanticId] \
                if useTrainId else id2label[semanticId]
            categoryId = labelInfo.trainId
            mask = inst == segmentId
            if labelInfo.ignoreInEval:
                continue

            if not labelInfo.hasInstances:
                isCrowd = 0

            if labelInfo.hasInstances and segmentId > 1000:
                self.instance_cnt = get_contours(self.instance_cnt,
                                                 inst, segmentId) \
                    if self.instance_cnt is not None else None
                (self.bbox_offsets, self.bbox_labels,
                 self.loss_mask, self.bboxes) =\
                    get_bbox_offsets(self.bbox_offsets, self.bbox_labels,
                                     self.bbox_loss_mask, self.bboxes,
                                     inst, segmentId, self.min_area,
                                     self.min_height, self.min_width) \
                    if self.bbox_offsets is not None else None
                self.centroids = get_centroids(self.centroids,
                                               self.vec_loss_mask,
                                               inst,
                                               segmentId, self.min_area) \
                    if self.centroids is not None else None

            if self.panoptic_seg is not None:
                self.panoptic_seg[mask] = [segmentId % 256, segmentId //
                                           256, segmentId // 256 // 256]
                self.segmInfo.append(get_segment_info(mask, segmentId,
                                                      categoryId, isCrowd))
        for value in np.unique(self.semantic_seg):
            self.semantic_seg[self.semantic_seg == value] = \
                id2label[value].trainId

        if 'semantic_with_instance' in self.tasks:
            self.semantic_seg, self.touching_cnt = \
                generate_semantic_with_instance(self.semantic_seg,
                                                self.touching_cnt,
                                                self.instance_cnt,
                                                self.semantic_cnt)
        if 'instance_regression' in self.tasks:
            self.instance_vecs = get_instance_vecs(self.centroids,
                                                   self.instance_vecs,
                                                   inst)

        # axis = plt.subplots(3, 3)[-1]
        # axis[0, 0].imshow(get_color_inst(self.instance_vecs[0, ...].squeeze()))
        # axis[0, 1].imshow(self.instance_vecs[0, 0, ...].squeeze())
        # axis[0, 2].imshow(self.instance_vecs[0, 1, ...].squeeze())
        # axis[1, 0].imshow(self.vec_loss_mask[0])
        # axis[1, 1].imshow(self.bbox_offsets[0, 0, ...])
        # axis[1, 2].imshow(self.bbox_offsets[0, 1, ...])
        # axis[2, 0].imshow(self.bbox_offsets[0, 2, ...])
        # axis[2, 1].imshow(self.bbox_offsets[0, 3, ...])
        # axis[2, 2].imshow(self.bbox_loss_mask[0])
        # plt.show()

        return self.return_targets()


def get_segment_info(mask, segmentId, categoryId, isCrowd):
    area = np.sum(mask.astype(np.uint8))
    # bbox computation for a segment
    bbox = get_bbox(mask)

    return {"id": int(segmentId),
            "category_id": int(categoryId),
            "area": int(area),
            "bbox": bbox,
            "iscrowd": isCrowd}


def get_instance_vecs(centroids_t, instance_vecs, inst):
    mask = (inst >= 1000).astype(np.uint8)
    h, w = instance_vecs.shape[-2], instance_vecs.shape[-1]
    g1, g2 = np.meshgrid(np.arange(w), np.arange(h))
    n = instance_vecs.shape[0] if len(instance_vecs.shape) == 4 else 1
    if n > 1:
        for i in range(n):
            instance_vecs[i, 0, ...] = g2
            instance_vecs[i, 1, ...] = g1
    else:
        instance_vecs[0, ...] = g2
        instance_vecs[1, ...] = g1
    instance_vecs[:, 0, ...] *= mask
    instance_vecs[:, 1, ...] *= mask
    instance_vecs -= centroids_t
    return instance_vecs


def get_centroids(centroids_t, vec_loss_mask, inst, segmentId, min_area):
    if len(inst.shape) == 3:
        n = inst.shape[0]
        for i in range(n):
            mask = (inst[i, ...] == segmentId)
            centroids_t[i, ...] = get_centroids_img(centroids_t[i, ...],
                                                    vec_loss_mask[i, ...],
                                                    mask, min_area)
        # print(np.unique(centroids_t[0, 0, ...]),
        #      np.unique(centroids_t[0, 1, ...]))
        return centroids_t
    else:
        mask = (inst == segmentId)
        return get_centroids_img(centroids_t, vec_loss_mask, mask, min_area)


def get_centroids_img(centroids_t, vec_loss_mask, mask, min_area):
    mask = mask.astype(np.uint8)
    area = np.sum(mask)
    # print(area)
    if area <= 0:
        return centroids_t
    xsys = np.nonzero(mask)
    xs, ys = xsys[0], xsys[1]
    # print(np.mean(xs), np.mean(ys))
    centroids_t[0, xs, ys] = np.mean(xs)
    centroids_t[1, xs, ys] = np.mean(ys)
    vec_loss_mask[xs, ys] = 1000/area
    # centroids_t[0, xs, ys] = int(bbox[0] + bbox[2]/2)
    # centroids_t[1, xs, ys] = int(bbox[1] + bbox[3]/2)

    return centroids_t


def get_contours(img, inst, segmentId):
    if len(inst.shape) == 3:
        n = inst.shape[0]
        for i in range(n):
            mask = (inst[i, ...] == segmentId)
            img[i, ...] = get_contour_img(img[i, ...].squeeze(), mask)
        return img
    else:
        mask = (inst == segmentId)
        return get_contour_img(img, mask)


def get_contour_img(img, mask):
    # kernel = np.ones((2, 2), np.uint8)
    mask = mask.numpy() if torch.is_tensor(mask) else mask
    mask = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, cnts, -1, 1, 2)  # .astype(np.uint8)
    # img =cv2.dilate(img, kernel, iterations=1)
    return img


def get_bbox_offsets(offsets, labels, loss_mask, bboxes, inst, segmentId,
                     min_area, min_h, min_w):
    h, w = offsets.shape[-2], offsets.shape[-1]
    if len(inst.shape) == 3:
        n = inst.shape[0]
        for i in range(n):
            resized_inst = cv2.resize(inst[i, ...], dsize=(w, h),
                                      interpolation=cv2.INTER_NEAREST)
            mask = (resized_inst == segmentId)
            offsets[i, ...], labels[i, ...], loss_mask[i, ...], bboxes[i] = \
                get_bbox_offset_img(offsets[i, ...], labels[i, ...],
                                    loss_mask[i, ...], bboxes[i],
                                    mask, segmentId, min_area, min_h, min_w)

        return offsets, labels, loss_mask, bboxes
    else:
        resized_inst = cv2.resize(inst, dsize=(w, h),
                                  interpolation=cv2.INTER_NEAREST)
        return get_bbox_offset_img(offsets, labels, loss_mask, bboxes, mask,
                                   segmentId, min_area, min_h, min_w)


def get_bbox_offset_img(offsets, labels, loss_mask, bbox_img, mask, segmentId,
                        min_area, min_h, min_w):
    mask = mask.astype(np.uint8)
    area = np.sum(mask)
    if area == 0:
        return offsets, labels, loss_mask, bbox_img
    bbox = get_bbox(mask)
    # if bbox[2] >= min_h and bbox[3] >= min_w:
    pt1, pt2 = get_corners(bbox)
    xmin, xmax, ymin, ymax = get_bbox_support_region(bbox)
    offsets[:, xmin:xmax, ymin:ymax] = \
        get_offsets_in_support_region(xmin, xmax, ymin, ymax, bbox)
    labels[xmin:xmax, ymin:ymax] = instance_trainId[segmentId//1000]
    loss_mask[xmin:xmax, ymin:ymax] = 1000 / area
    bbox_img.append([pt1[1], pt1[0], pt2[1], pt2[0]])

    return offsets, labels, loss_mask, bbox_img


def get_corners(bbox):
    left = int(bbox[0])
    top = int(bbox[1])
    width = int(bbox[2])
    height = int(bbox[3])
    pt1 = (left, top)
    pt2 = (int(left + width), int(top + height))

    return pt1, pt2


def get_bbox(mask):
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    hor_ = hor_idx[0]
    width = hor_idx[-1] - hor_ + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    vert_ = vert_idx[0]
    height = vert_idx[-1] - vert_ + 1

    return [int(hor_), int(vert_), int(width), int(height)]


def get_bbox_support_region(bbox, scale=0.25):
    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]

    center = (int(left + width/2), int(top + height/2))

    dx = (height * math.sqrt(scale)) / 2.0
    dy = (width * math.sqrt(scale)) / 2.0
    xmin = center[1] - dx
    ymin = center[0] - dy
    xmax = center[1] + dx + 1
    ymax = center[0] + dy + 1

    return int(xmin), int(xmax), int(ymin), int(ymax)


def get_offsets_in_support_region(xmin, xmax, ymin, ymax, bbox):

    xmin_offsets = np.expand_dims(np.arange(start=xmin, stop=xmax, step=1,
                                            dtype=np.float32) - bbox[1],
                                  axis=1)
    xmin_offsets = np.expand_dims(
        np.repeat(xmin_offsets, repeats=ymax - ymin, axis=1), axis=0)

    ymin_offsets = np.expand_dims(np.arange(start=ymin, stop=ymax, step=1,
                                            dtype=np.float32) - bbox[0],
                                  axis=0)
    ymin_offsets = np.expand_dims(
        np.repeat(ymin_offsets, repeats=xmax-xmin, axis=0), axis=0)

    xmax_offsets = np.expand_dims(np.arange(start=xmin, stop=xmax, step=1,
                                            dtype=np.float32) - (bbox[1] + bbox
                                                                 [3]), axis=1)
    xmax_offsets = np.expand_dims(
        np.abs(np.repeat(xmax_offsets, repeats=ymax - ymin, axis=1)), axis=0)

    ymax_offsets = np.expand_dims(np.arange(start=ymin, stop=ymax, step=1,
                                            dtype=np.float32) - (bbox[0] + bbox
                                                                 [2]), axis=0)
    ymax_offsets = np.expand_dims(
        np.abs(np.repeat(ymax_offsets, repeats=xmax-xmin, axis=0)), axis=0)

    offsets = np.concatenate([xmin_offsets, ymin_offsets,
                              xmax_offsets, ymax_offsets], axis=0)

    return offsets


def get_cfg(config):
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    for task in list(cfg[TASKS].keys()):
        if not cfg[TASKS][task][ACTIVE]:
            del cfg[TASKS][task]
    return cfg


def generate_semantic_with_instance(semantic_seg, touching_cnt,
                                    instance_cnt, semantic_cnt):
    for value in np.unique(semantic_seg):
        if id2label[value].hasInstances:
            semantic_cnt = get_contours(semantic_cnt, semantic_seg, value)

    # n = self.instance_cnt.shape[0]
    # kernel = np.ones((2, 2), np.uint8)
    # for i in range(n):
    #     img = self.instance_cnt[i, ...].astype(np.uint8)
    #     self.instance_cnt[i, ...] = cv2.dilate(img, kernel,
    #                                            iterations=1)

    # self.semantic_seg[self.instance_cnt != 0] = \
    #    name2label['boundary'].trainId
    touching_cnt = instance_cnt - semantic_cnt
    semantic_seg[touching_cnt != 0] = name2label['t-boundary'].trainId
    touching_cnt[touching_cnt != 0] = 10
    touching_cnt[touching_cnt == 0] = 1
    return semantic_seg, touching_cnt


def get_active_tasks(model_cfg):
    active_outputs_cfg = copy.deepcopy(model_cfg)
    for task in list(model_cfg.keys()):
        if not model_cfg[task][ACTIVE]:
            del active_outputs_cfg[task]

    return active_outputs_cfg


def map_outputs_to_task(output):
    return get_key(OUTPUTS_TO_TASK, output)


def get_key(_dict, val):
    for key, value in _dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def get_class_weights_from_data(loader, num_classes, cfg, task):
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for data in tqdm(loader):
        _, labels = data
        labels = get_labels(labels, cfg)
        for label_img in labels[task]:
            for trainId in range(num_classes):
                # count how many pixels in label_img which are of object class trainId:

                trainId_mask = np.equal(label_img, trainId)
                trainId_count = torch.sum(trainId_mask)

                # add to the total count:
                trainId_to_count[trainId] += trainId_count

    # compute the class weights according to the ENet paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/(trainId_prob + 1e-6)
        class_weights.append(trainId_weight)

    return class_weights


def cityscapes_semantic_weights(num_classes):
    if num_classes == 20:
        class_weights = [3.045383480249677, 12.862127312658735,
                         4.509888876996228, 38.15694593009221,
                         35.25278401818165, 31.48260832348194,
                         45.79224481584843, 39.69406346608758,
                         6.0639281852733715, 32.16484408952653,
                         17.10923371690307, 31.5633201415795,
                         47.33397232867321, 11.610673599796504,
                         44.60042610251128, 45.23705196392834,
                         45.28288297518183, 48.14776939659858,
                         41.924631833506794, 50]
    elif num_classes == 21:
        class_weights = [3.045383480249677, 12.862127312658735,
                         4.509888876996228, 38.15694593009221,
                         35.25278401818165, 31.48260832348194,
                         45.79224481584843, 39.69406346608758,
                         6.0639281852733715, 32.16484408952653,
                         17.10923371690307, 31.5633201415795,
                         47.33397232867321, 11.610673599796504,
                         44.60042610251128, 45.23705196392834,
                         45.28288297518183, 48.14776939659858,
                         41.924631833506794, 50, 60]
    elif num_classes == 19:
        class_weights = [3.045383480249677, 12.862127312658735,
                         4.509888876996228, 38.15694593009221,
                         35.25278401818165, 31.48260832348194,
                         45.79224481584843, 39.69406346608758,
                         6.0639281852733715, 32.16484408952653,
                         17.10923371690307, 31.5633201415795,
                         47.33397232867321, 11.610673599796504,
                         44.60042610251128, 45.23705196392834,
                         45.28288297518183, 48.14776939659858,
                         41.924631833506794]

    elif num_classes == 34:
        return None  # TODO: Compute weights
    else:
        raise ValueError('Invalid number of classes for Cityscapes dataset')

    return class_weights


def get_weights(cfg):
    dataset = cfg['data']['dataset']
    tasks = cfg['model']['outputs']
    weights = {}
    for task in tasks.keys():
        if dataset == 'Cityscapes' and task in ['semantic',
                                                'semantic_with_instance']:
            weight = cityscapes_semantic_weights(tasks[task]['out_channels'])
            weights[task] = torch.FloatTensor(weight)
        else:
            weights[task] = None

    return weights


def get_predictions(logits, cfg, targets):
    predictions = {}
    for task in logits.keys():
        if cfg[task][POSTPROC] == ARGMAX:
            predictions[task] = torch.argmax(logits[task], dim=1)
        elif cfg[task][POSTPROC] == 'bbox':
            predictions[task] = \
                get_bboxes_filtered(logits[task], targets[task],
                                    scale_factor=cfg[task]['scale_factor'],
                                    conf_thresh=cfg[task]['conf_thresh'],
                                    iou_thresh=cfg[task]['iou_thresh'])
        else:
            predictions[task] = logits[task]
    return predictions


def post_process_predictions(predictions, post_proc_cfg):
    outputs = {}
    for task in post_proc_cfg.keys():
        if post_proc_cfg[task][ACTIVE]:
            inputs_cfg = post_proc_cfg[task][INPUTS]
            inputs = get_active_tasks(inputs_cfg)
            predicted_tasks = set(predictions.keys())
            input_tasks = set(inputs.keys())

            if not input_tasks.issubset(predicted_tasks):
                raise ValueError("""Inputs to postproc are not a subset
                                  of network predictions. Inputs: {},
                                  prediction: {}""".format(input_tasks,
                                                           predicted_tasks))

            outputs[task] = generatePanopticFromPredictions(predictions,
                                                            inputs)
    return outputs


def get_bboxes_filtered(logits, targets, scale_factor=0.25,
                        conf_thresh=0.5, iou_thresh=0.5):
    confidence, offsets = logits['class'], logits['offsets']
    if isinstance(confidence, dict) and isinstance(offsets, dict):
        if len(offsets[1].size()) == 4:
            n, c, h, w = offsets[1].size()
        else:
            c, h, w = offsets[1].size()
            n = 1
        bboxes = {i: [] for i in range(n)}
        scores = {i: [] for i in range(n)}
        confidence_masked = {k: None for k in confidence.keys()}
        for k, v in confidence.items():
            confidence_masked[k] = get_confidence_mask(v, conf_thresh)

        for i in range(n):
            for k, v in confidence_masked.items():
                box, score = get_bbox_filtered(offsets[k][i, ...],
                                               v[i, ...], iou_thresh,
                                               h, w, k)
                bboxes[i].extend(box)
                scores[i].extend(score)

    else:
        confidence_masked = get_confidence_mask(confidence, conf_thresh)
        if len(offsets.size()) == 4:
            n, c, h, w = offsets.size()
        else:
            c, h, w = offsets.size()
            n = 1
        bboxes = {i: [] for i in range(n)}
        scores = {i: [] for i in range(n)}
        for i in range(n):
            bboxes[i], scores[i] = get_bbox_filtered(offsets[i, ...],
                                                     confidence_masked[i, ...], iou_thresh,
                                                     h, w)
    bboxes = get_nms_boxes(bboxes, scores, iou_thresh)
    return bboxes


def get_bbox_filtered(offset_img, confidence_img, iou_thresh=0.7, h=64, w=128,
                      stride=1):
    positions = torch.nonzero(confidence_img)*stride
    if positions.size()[0] == 0:
        return []
    with torch.no_grad():
        scores = confidence_img[positions[:, 0], positions[:, 1]]
        offsets = offset_img[:, positions[:, 0], positions[:, 1]]
        bbox_cords = torch.mul(offsets.T, torch.tensor([-1, -1, 1, 1]).cuda())
        bbox_cords *= stride
        bbox_cords += positions[:, [0, 1, 0, 1]]
        bbox_area = (bbox_cords[:, 2]-bbox_cords[:, 0]) * \
            (bbox_cords[:, 3]-bbox_cords[:, 1])
        keep_inds = torch.nonzero((bbox_area >= 5).float())
        filtered_boxes = bbox_cords[keep_inds].squeeze()
        scores = scores[keep_inds].squeeze()
        if filtered_boxes.size()[0] == 0:
            return []
        if len(filtered_boxes.size()) == 1:
            return [filtered_boxes]
        filtered_boxes[:, [0, 2]] = torch.clamp(
            filtered_boxes[:, [0, 2]], min=0, max=h)
        filtered_boxes[:, [1, 3]] = torch.clamp(
            filtered_boxes[:, [1, 3]], min=0, max=w)
        filtered_boxes = filtered_boxes.cpu()

    return filtered_boxes, scores


def get_nms_boxes(filtered_boxes, scores, iou_thresh):
    bbox = torchvision.ops.nms(
        filtered_boxes, scores.cpu(), iou_thresh)
    keep_boxes = filtered_boxes[bbox, :]
    return keep_boxes


def get_confidence_mask(confidence_img, conf_thresh=0.5):

    if len(confidence_img.size()) == 4:
        n, c, h, w = confidence_img.size()
        class_mask = torch.zeros((n, h, w))
        confidence_img = torch.softmax(confidence_img, dim=1)
        max_conf, class_ = confidence_img.max(1)
    else:
        c, h, w = confidence_img.size()
        class_mask = torch.zeros((h, w))
        confidence_img = torch.softmax(confidence_img, dim=0)
        max_conf, class_ = confidence_img.max(0)

    class_mask[class_ != 0] = 1
    max_conf[class_mask == 0] = 0
    max_conf[max_conf < conf_thresh] = 0
    return max_conf
