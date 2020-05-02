"""data utilities to support target conversion, postprocs etc."""

import copy
import torch
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.im_utils import labels, prob_labels, trainId2label, id2label, \
    decode_segmap, inst_labels, instance_trainId
from utils.constants import TASKS, MODEL, OUTPUTS, TYPE, SEMANTIC, INSTANCE, \
    DISPARITY, PANOPTIC, ACTIVE, OUTPUTS_TO_TASK, INSTANCE_REGRESSION, \
    INSTANCE_PROBS, INSTANCE_HEATMAP, INSTANCE_CONTOUR, POSTPROCS, INPUTS, \
    INSTANCE_IMAGE, PANOPTIC, PANOPTIC_IMAGE, SEGMENT_INFO, DOUBLE, LONG, \
    FLOAT, POSTPROC, ARGMAX
from instance_to_clusters import get_clusters, imshow_components, to_rgb


def convert_targets_semantic(targets, permute=(0, 2, 3, 1), labels=labels):
    """Convert semantic segmentation targets.

    Arguments:
        targets {torch.tensor} -- [torch tensor containing targets]

    Keyword Arguments:
        permute {tuple} -- [swap axis] (default: {(0, 2, 3, 1)})
        labels {list} -- [list with named tuples containing semantic
        class information.] (default: {labels})

    Returns:
        [torch.tensor] -- [converted semantic segmentation targets]
    """
    targets = torch.squeeze((targets*255).permute(permute)).numpy()
    new_targets = np.empty_like(targets)
    for label_id in np.unique(targets):
        train_id = labels[int(label_id)].trainId
        new_targets[np.where(targets == label_id)] = train_id

    return torch.tensor(new_targets)


def prepare_targets(targets, permute=(0, 2, 3, 1)):
    """Prepare targets for conversion.

    Arguments:
        targets {torch.tensor} -- [torch tensor containing targets]

    Keyword Arguments:
        permute {tuple} -- [swap axis] (default: {(0, 2, 3, 1)})

    Returns:
        [tuple] -- [batch_size, dimensions(h,w), targets]
    """
    targets = torch.squeeze((targets).permute(permute))
    if len(targets.size()) > 2:
        n, h, w = targets.size()
    elif len(targets.size()) == 2:
        h, w = targets.size()
        n = 1
        targets = targets.unsqueeze(0)
    return n, h, w, targets


def convert_targets_disparity(targets, permute=(0, 2, 3, 1)):
    """Convert depth estimation targets.

    Arguments:
        targets {torch.tensor} -- [torch tensor containing targets]

    Keyword Arguments:
        permute {tuple} -- [swap axis] (default: {(0, 2, 3, 1)})

    Returns:
        [torch.tensor] -- [normalized depth map with values between (0,1)]
    """
    normalized_dep = []
    n = targets.size()[0] if len(targets.size()) > 2 else 1
    targets = torch.squeeze((targets).permute(permute)).numpy()
    targets[targets > 0] = (targets[targets > 0]-1)/256
    inv_dep = targets/(0.209313*2262.52)  # TODO: parse parameters as args.

    if n > 1:
        for i in range(n):
            min_inv_dep = np.min(inv_dep[i])
            max_inv_dep = np.max(inv_dep[i])
            normalized_dep.append(
                (inv_dep[i]-min_inv_dep)/(max_inv_dep-min_inv_dep))
    else:
        min_inv_dep = np.min(inv_dep)
        max_inv_dep = np.max(inv_dep)
        normalized_dep = (inv_dep-min_inv_dep)/(max_inv_dep-min_inv_dep)

    return torch.tensor(normalized_dep).type(torch.float32)


def convert_targets_instance(targets, permute=(0, 2, 3, 1),
                             contours_active=False,
                             regression_active=False):
    """Convert instance regression targets.

    Arguments:
        targets {torch.tensor} -- [torch tensor containing targets]

    Keyword Arguments:
        permute {tuple} -- [swap axis] (default: {(0, 2, 3, 1)})

    Returns:
        [dict] -- [Instance regression targets in with instance_image,
        centroids_regression, probabilites, heatmap, contours]
    """
    # TODO: Please clean this up.
    n, h, w, targets = prepare_targets(targets, permute)
    masks = torch.zeros((n, h, w))
    imgs = torch.zeros((n, h, w))
    if regression_active:
        vecs = torch.zeros((n, 2, h, w))
        heatmaps = torch.zeros((n, h, w))
    if contours_active:
        contours = torch.zeros((n, h, w))
    for i in range(n):
        target = targets[i, :, :].float()
        if contours_active and regression_active:
            img, reg, mask, heatmap, contour \
                = compute_instance_torch(target, True, True)
            imgs[i, :, :] = img.long()
            vecs[i, :, :, :] = reg.float()
            masks[i, :, :] = mask.long()
            heatmaps[i, :, :] = heatmap.float()
            contours[i, :, :] = contour.long()
        elif regression_active:
            img, reg, mask, heatmap \
                = compute_instance_torch(target, False, True)
            imgs[i, :, :] = img.long()
            vecs[i, :, :, :] = reg.float()
            masks[i, :, :] = mask.long()
            heatmaps[i, :, :] = heatmap.float()
        elif contours_active:
            img, mask, contour \
                = compute_instance_torch(target, True, False)
            imgs[i, :, :] = img.long()
            masks[i, :, :] = mask.long()
            contours[i, :, :] = contour.long()
    if regression_active and contours_active:
        converted_targets = {INSTANCE_IMAGE: imgs,
                             INSTANCE_REGRESSION: vecs,
                             INSTANCE_PROBS: masks,
                             INSTANCE_HEATMAP: heatmaps,
                             INSTANCE_CONTOUR: contours}
    elif contours_active:
        converted_targets = {INSTANCE_IMAGE: imgs,
                             INSTANCE_PROBS: masks,
                             INSTANCE_CONTOUR: contours}
    elif regression_active:
        converted_targets = {INSTANCE_IMAGE: imgs,
                             INSTANCE_REGRESSION: vecs,
                             INSTANCE_PROBS: masks,
                             INSTANCE_HEATMAP: heatmaps}
    return converted_targets


def convert_targets_panoptic(targets, permute=(0, 2, 3, 1)):
    n, h, w, targets = prepare_targets(targets, permute)
    panoptic = getPanopticEval(targets)

    converted_targets = {PANOPTIC: panoptic}

    return converted_targets


def get_labels_fn(task):
    if task == SEMANTIC:
        return (convert_targets_semantic)
    elif task == DISPARITY:
        return (convert_targets_disparity)
    elif task == INSTANCE:
        return (convert_targets_instance)
    else:
        return None


def convert_data_type(data, data_type):
    if data_type == DOUBLE:
        return data.double()
    elif data_type == LONG:
        return data.long()
    elif data_type == FLOAT:
        return data.float()


def get_labels(in_targets, cfg, device=None, get_postprocs=False):
    labels = {}
    active_outputs = get_active_tasks(cfg[MODEL][OUTPUTS])
    active_postprocs = get_active_tasks(cfg[POSTPROCS])
    for i, task in enumerate(cfg[TASKS].keys()):
        data_type = cfg[TASKS][task][TYPE]
        label_fn = get_labels_fn(task)
        targets = in_targets[i] if isinstance(in_targets, list) else in_targets
        if task in [SEMANTIC, DISPARITY]:
            if label_fn is not None:
                label = label_fn(targets)
            else:
                label = targets
            labels[task] = convert_data_type(label, data_type).to(device)
        elif task == INSTANCE:
            contours_active = (
                True if INSTANCE_CONTOUR in active_outputs else False)
            regression_active = (
                True if INSTANCE_REGRESSION in active_outputs else False)

            dict_targets = label_fn(targets, (0, 2, 3, 1),
                                    contours_active, regression_active)
            for task in dict_targets.keys():
                dict_targets[task] = dict_targets[task].to(device)

            panoptic_active = True if PANOPTIC in active_postprocs else False

            if panoptic_active and get_postprocs:
                dict_targets.update(convert_targets_panoptic(targets))

            labels.update(dict_targets)

    return labels


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


def generatePanopticFromPredictions(predictions, inputs):
    semantic = generateSemanticFromPredictions(predictions)
    instance = genrateInstanceFromPredictions(predictions, inputs)
    mask = generateMaskFromPredicitons(predictions)
    panoptic = semantic*(1-mask) + instance
    panoptic = getPanopticEval(panoptic, useTrainId=True)

    return panoptic


def genrateInstanceFromPredictions(predictions, inputs):
    semantic = generateSemanticFromPredictions(predictions)
    mask = generateMaskFromPredicitons(predictions)
    contours = generateMaskFromPredicitons(predictions)

    if INSTANCE_CONTOUR in inputs:
        instance = getInstanceFromContour(mask, semantic, contours)

    return instance


def generateSemanticFromPredictions(predictions):
    semantic = predictions.get(SEMANTIC, None)
    if semantic is not None:
        return semantic.detach().cpu().numpy()
    else:
        raise ValueError('{} not found in {}'.format(SEMANTIC,
                                                     predictions.keys()))


def generateContoursFromPredictions(predictions):
    contours = predictions.get(INSTANCE_CONTOUR, None)
    if contours is not None:
        return contours.detach().cpu().numpy()
    else:
        raise ValueError('{} not found in {}'.format(INSTANCE_CONTOUR,
                                                     predictions.keys()))


def generateMaskFromPredicitons(predictions):
    semantic = generateSemanticFromPredictions(predictions)
    #mask = predictions.get(INSTANCE_PROBS, None)
    # if mask is None:
    return (semantic >= 11).astype(np.uint8)
    # return mask.detach().cpu().numpy()


def getInstanceFromContour(mask, seg, contours):
    inst = np.zeros_like(seg)
    for i in range(inst.shape[0]):
        inst[i] = _get_instance_from_contour(mask[i], seg[i], contours[i])
    return inst


def _get_instance_from_contour(mask, seg, contours):
    inst = np.zeros_like(seg)
    for i in np.unique(contours):
        if i != 0:
            contour = (contours == i).astype(np.uint8)

            diff = seg*mask*contour
            _, labels = cv2.connectedComponents(diff.astype(np.uint8))
            inst += diff*1000 + labels
    return inst


def getPanopticEval(inst, useTrainId=False):
    panoptic_seg = np.zeros((inst.shape + (3,)), dtype=np.uint8)
    if torch.is_tensor(inst):
        inst = inst.numpy()
    segmentIds = np.unique(inst)
    segmInfo = []
    for segmentId in segmentIds:
        if segmentId < 1000:
            semanticId = segmentId
            isCrowd = 1
        else:
            semanticId = segmentId // 1000
            isCrowd = 0

        if useTrainId:
            labelInfo = trainId2label[semanticId]
            categoryId = labelInfo.id
        else:
            labelInfo = id2label[semanticId]
            categoryId = labelInfo.id
        if labelInfo.ignoreInEval:
            continue
        if not labelInfo.hasInstances:
            isCrowd = 0

        mask = inst == segmentId
        color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
        panoptic_seg[mask] = color

        area = np.sum(mask.astype(np.uint8))  # segment area computation

        # bbox computation for a segment
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        bbox = [int(x), int(y), int(width), int(height)]
        segmInfo.append({"id": int(segmentId),
                         "category_id": int(categoryId),
                         "area": int(area),
                         "bbox": bbox,
                         "iscrowd": isCrowd})
        panoptic = {PANOPTIC_IMAGE: panoptic_seg,
                    SEGMENT_INFO: segmInfo}

    return panoptic


def get_class_weights_from_data(loader, num_classes, cfg, task):
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for data in tqdm(loader):
        _, labels = data
        labels = convert_targets(labels, cfg['tasks'])
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
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    return class_weights


def cityscapes_semantic_weights(num_classes):
    if num_classes == 20:
        class_weights = [2.955507538630981, 13.60952309186396,
                         5.56145316824849, 37.623098044056555,
                         35.219757095290035, 30.4509054117227,
                         46.155918742024745, 40.29336775103404,
                         7.1993048519013465, 31.964755676368643,
                         24.369833379633036, 26.667508196892037,
                         45.45602154799861, 9.738884687765038,
                         43.93387854348821, 43.46301980622594,
                         44.61855914531797, 47.50842372150186,
                         40.44117532401872, 12.772291423775606]

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


def cityscapes_contour_weights(num_classes):
    if num_classes == 9:
        class_weights = [1.427197976828025, 47.66104006965641,
                         50.0977099173462, 44.04363870779025,
                         50.31372660864973, 50.31163764506638,
                         50.36620380700314, 50.32661428022733,
                         49.834611789928324]

    else:
        raise ValueError('Invalid number of classes for Cityscapes dataset')

    return class_weights


def get_weights(cfg, device):
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


def compute_instance_torch(instance_image, contours_active=False,
                           regression_active=False):
    instance_image_tensor = torch.Tensor(instance_image)
    mask = instance_image_tensor >= 2400
    if regression_active:
        alpha = 10.0
        centroids_t = torch.zeros(instance_image.shape + (2,))
        w_h = torch.ones(instance_image.shape + (2,))
    if contours_active:
        contours = np.zeros(instance_image.shape)

    for value in torch.unique(instance_image_tensor):
        if value >= 2400:
            class_id = int(value.numpy()//1000)
            xsys = torch.nonzero(instance_image_tensor == value)
            xs, ys = xsys[:, 0], xsys[:, 1]
            if regression_active:
                centroids_t[xs, ys] = torch.stack((torch.mean(xs.float()),
                                                   torch.mean(ys.float())))
                w, h = get_instance_hw(xs, ys)
                if w != 0 and h != 0:
                    w_h[xs, ys, 0], w_h[xs, ys, 1] = w.float(), h.float()

            if contours_active:
                cont_mask = np.zeros_like(instance_image)
                contour_class = instance_trainId.get(class_id, 0)
                cont_mask[np.where(instance_image == value)] = 255
                cont_img = np.zeros(instance_image.shape)
                _, thresh = cv2.threshold(cont_mask, 127, 255, 0)
                cnts, _ = cv2.findContours(thresh.astype(
                    np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cont_img = cv2.drawContours(cont_img, cnts, -1, 1, 1)
                contours[np.where(cont_img == 1.0)] = contour_class
                cont = np.array([xs.numpy(), ys.numpy()])
                for x in np.unique(cont[0, :]):
                    idx = np.where(cont[0, :] == x)
                    contours[x, np.min(cont[1, idx])] = contour_class
                    contours[x, np.max(cont[1, idx])] = contour_class
                for y in np.unique(cont[1, :]):
                    idx = np.where(cont[1, :] == y)
                    contours[np.min(cont[0, idx]), y] = contour_class
                    contours[np.max(cont[0, idx]), y] = contour_class

    if regression_active:
        coordinates = torch.zeros(instance_image.shape + (2,))
        g1, g2 = torch.meshgrid(torch.arange(instance_image_tensor.size()[
            0]), torch.arange(instance_image_tensor.size()[1]))
        coordinates[:, :, 0] = g1
        coordinates[:, :, 1] = g2
        vecs = coordinates - centroids_t

        if len(mask.size()) > 1:
            mask = mask.int()
        elif mask is False:
            mask = np.zeros(instance_image.shape)
        else:
            mask = np.ones(instance_image.shape)
        vecs[:, :, 0] = vecs[:, :, 0]*mask
        vecs[:, :, 1] = vecs[:, :, 1]*mask
        heatmap_ = w_h - (torch.abs(vecs)*alpha)
        heatmap_ = np.clip(heatmap_, 0, torch.max(heatmap_))

        heatmap_[:, :, 0] /= w_h[:, :, 0]
        heatmap_[:, :, 1] /= w_h[:, :, 1]
        heatmap_t = heatmap_[:, :, 0]*heatmap_[:, :, 1]
        heatmap_t = heatmap_t*mask

    if regression_active and contours_active:
        return (instance_image_tensor, vecs.permute(2, 0, 1),
                mask, heatmap_t, torch.tensor(contours))
    elif regression_active:
        return (instance_image_tensor, vecs.permute(2, 0, 1),
                mask, heatmap_t)
    elif contours_active:
        return (instance_image_tensor, mask, torch.tensor(contours))


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
