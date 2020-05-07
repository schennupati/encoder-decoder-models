"""data utilities to support target conversion, postprocs etc."""

import copy
import torch
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.im_utils import labels, prob_labels, trainId2label, id2label, \
    decode_segmap, inst_labels, name2label
from utils.constants import TASKS, MODEL, OUTPUTS, TYPE, SEMANTIC, INSTANCE, \
    DISPARITY, PANOPTIC, ACTIVE, OUTPUTS_TO_TASK, INSTANCE_REGRESSION, \
    INSTANCE_PROBS, INSTANCE_HEATMAP, INSTANCE_CONTOUR, POSTPROCS, INPUTS, \
    INSTANCE_IMAGE, PANOPTIC, PANOPTIC_IMAGE, SEGMENT_INFO, DOUBLE, LONG, \
    FLOAT, POSTPROC, ARGMAX
from instance_to_clusters import get_clusters, imshow_components, to_rgb


class TargetGenerator():
    def __init__(self, cfg, useTrainId=False, postprocs=False):
        self.tasks = list(get_active_tasks(cfg[MODEL][OUTPUTS]).keys())
        self.labels = {}
        self.postprocs = postprocs

    def prepare_targets(self, inst):
        self.instance_cnt = np.zeros_like(
            inst) if 'instance_contour' in self.tasks else None
        if 'semantic_with_instance' in self.tasks:
            self.semantic_cnt = np.zeros_like(inst)
            self.instance_cnt = np.zeros_like(inst)
        if ('panoptic' in self.tasks) or self.postprocs:
            self.panoptic_seg = np.zeros((inst.shape + (3,)), dtype=np.uint8)
            self.segmInfo = []
        else:
            self.panoptic_seg = None
            self.segmInfo = []
        if 'instance_regression' in self.tasks:
            self.instance_vec = np.zeros((inst.shape + (2,)), dtype=np.int16)

    def return_targets(self):
        if 'semantic' in self.tasks:
            self.labels.update(
                {'semantic': torch.tensor(self.semantic_seg).cuda()})
        if 'semantic_with_instance' in self.tasks:
            self.labels.update(
                {'semantic_with_instance':
                 torch.tensor(self.semantic_seg).cuda(),
                 'instance_contour':
                 torch.tensor(self.instance_cnt).cuda()})
        if ('panoptic' in self.tasks) or self.postprocs:
            self.labels.update(
                {'panoptic': self.panoptic_seg,
                 'segmInfo': self.segmInfo})
        if 'instance_regression' in self.tasks:
            self.labels.update(
                {'instance_regression':
                 torch.tensor(self.instance_vec).cuda(),
                 'instance_heatmap':
                 torch.tensor(self.instance_heatmap).cuda()})

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
            else:
                self.instance_cnt = get_contours(self.instance_cnt,
                                                 inst, segmentId) \
                    if self.instance_cnt is not None else None

            if self.panoptic_seg is not None:
                self.panoptic_seg[mask] = [segmentId % 256, segmentId //
                                           256, segmentId // 256 // 256]
                self.segmInfo.append(get_segment_info(mask, segmentId,
                                                      categoryId, isCrowd))
        if 'semantic_with_instance' in self.tasks:
            for value in np.unique(self.semantic_seg):
                self.semantic_cnt = get_contours(self.semantic_cnt,
                                                 self.semantic_seg,
                                                 value)
                self.semantic_seg[self.semantic_seg == value] = \
                    id2label[value].trainId

            n = self.instance_cnt.shape[0]
            kernel = np.ones((2, 2), np.uint8)
            for i in range(n):
                img = self.instance_cnt[i, ...].astype(np.uint8)
                self.instance_cnt[i, ...] = cv2.dilate(img, kernel,
                                                       iterations=1)

            self.semantic_seg[self.instance_cnt != 0] = \
                name2label['boundary'].trainId

        # fig, axis = plt.subplots(2, 2)
        # axis[0, 0].imshow(self.instance_cnt[0])
        # axis[0, 1].imshow(self.semantic_cnt[0])
        # axis[1, 0].imshow(decode_segmap(self.semantic_seg[0].astype(np.uint8)))
        # axis[1, 1].imshow(self.panoptic_seg[0])
        # plt.show()
        # #import pdb
        # # pdb.set_trace()

        return self.return_targets()


def get_contours(img, inst, segmentId):
    if len(img.shape) == 3:
        n = img.shape[0]
        contours = np.zeros_like(img)
        for i in range(n):
            mask = (inst[i, ...] == segmentId)
            contours[i, ...] = get_contour_img(img[i, ...].squeeze(), mask)
        return contours
    else:
        mask = (img == segmentId)
        return get_contour_img(img, mask)


def get_contour_img(img, mask):
    mask = mask.numpy() if torch.is_tensor(mask) else mask
    mask = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(mask,
                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(img, cnts, -1, 1, 1)


def get_segment_info(mask, segmentId, categoryId, isCrowd):
    area = np.sum(mask.astype(np.uint8))
    # bbox computation for a segment
    bbox = get_bbox(mask)

    return {"id": int(segmentId),
            "category_id": int(categoryId),
            "area": int(area),
            "bbox": bbox,
            "iscrowd": isCrowd}


def get_bbox(mask):
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    # bbox = top left (x,y) and (h,w)
    return [int(x), int(y), int(width), int(height)]


def get_cfg(config):
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    for task in list(cfg[TASKS].keys()):
        if not cfg[TASKS][task][ACTIVE]:
            del cfg[TASKS][task]
    return cfg


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
                         41.924631833506794, 60]

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
