"""Metrics utilities for experiments."""
# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch
import copy

from utils.im_utils import labels, id2label
from utils.constants import SEGMENT_INFO, PANOPTIC_IMAGE, VOID


class semanticMetrics(object):
    def __init__(self, n_classes):

        self.n_classes = n_classes
        self.reset()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int)
                           + label_pred[mask], minlength=n_class ** 2
                           ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                     lp.flatten(),
                                                     self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) +
                               hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        cls_iou = dict(zip(range(self.n_classes), iou))

        mean = {"Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iou}

        return mean, cls_iou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class regressionAccruacy(object):
    def __init__(self):
        self.reset()
        self.reset()

    def accuracy(self, gt, pred, pct_close=1e-1):
        # data_x and data_y are numpy array-of-arrays matrices
        gt = torch.tensor(gt)
        pred = torch.tensor(pred)
        if gt.size() != pred.size():
            pred = pred.permute(0, 2, 3, 1).squeeze()
        n_items = pred.numel()
        n_correct = torch.sum((torch.abs(pred - gt) < pct_close))
        return (n_correct.item() * 100.0 / n_items)  # scalar

    def update(self, label_trues, label_preds, pct_close=0.025):
        self.acc_1 = self.accuracy(label_trues, label_preds, pct_close)
        self.acc_2 = self.accuracy(label_trues, label_preds, pct_close*1e-1)
        self.acc_3 = self.accuracy(label_trues, label_preds, pct_close*1e-2)

    def get_scores(self):
        acc_1 = self.acc_1
        acc_2 = self.acc_2
        acc_3 = self.acc_3
        return ({"Regression Acc delta_1": acc_1,
                 "Regression Acc delta_2": acc_2,
                 "Regression Acc delta_3": acc_3}, None)

    def reset(self):
        self.acc_1 = 0.0
        self.acc_2 = 0.0
        self.acc_3 = 0.0


class panopticMetrics(object):
    def __init__(self, labels):
        self.OFFSET = 256 * 256 * 256
        self.reset()

    def reset(self):
        self.metrics = {'iou': {label.id: 0 for label in labels},
                        'tp': {label.id: 0 for label in labels},
                        'fp': {label.id: 0 for label in labels},
                        'fn': {label.id: 0 for label in labels}}

    def rgb2id(self, color):
        if isinstance(color, np.ndarray) and len(color.shape) >= 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[..., 0] + \
                256 * color[..., 1] + \
                256 * 256 * color[..., 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def get_mean_metric(self, metric):
        metric_copy = []
        for _id, _ in enumerate(metric):
            label = id2label[_id]
            if not label.ignoreInEval:
                metric_copy.append(metric[_id])
        return np.nanmean(metric_copy)

    def update(self, label_true, label_preds):
        self.get_pan_labels_preds(label_true, label_preds)
        self.get_tp_iou()
        self.get_fn()
        self.get_fp()

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - PQ
            - RQ
            - SQ
            - mean_iu
        """
        # Calc Panoptic metrics (SQ, RQ, PQ)
        # (https://arxiv.org/pdf/1801.00868.pdf)

        iou = np.array(list(self.metrics['iou'].values()))
        tp = np.array(list(self.metrics['tp'].values()))
        fp = np.array(list(self.metrics['fp'].values()))
        fn = np.array(list(self.metrics['fn'].values()))

        sq = iou / tp
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq

        mean_metrics = {"Mean PQ": self.get_mean_metric(pq),
                        "Mean RQ": self.get_mean_metric(sq),
                        "Mean SQ": self.get_mean_metric(rq)}
        class_metrics = {'pq': pq,
                         'rq': rq,
                         'sq': sq}

        return mean_metrics, class_metrics

    def get_pan_labels_preds(self, label_true, label_preds):
        pan_label_true = label_true[PANOPTIC_IMAGE]
        segment_label_true = label_true[SEGMENT_INFO]
        pan_label_pred = label_preds[PANOPTIC_IMAGE]
        segment_label_pred = label_preds[SEGMENT_INFO]

        self.true_segms = {el['id']: el for el in segment_label_true}
        self.pred_segms = {el['id']: el for el in segment_label_pred}
        self.pred_labels_set = set(el['id'] for el in segment_label_pred)

        pan_label_true = self.rgb2id(pan_label_true)
        pan_label_pred = self.rgb2id(pan_label_pred)

        pan_true_pred = pan_label_true.astype(np.uint64)*self.OFFSET + \
            pan_label_pred.astype(np.uint64)
        self.pan_true_pred_map = {}
        labels, label_cnt = np.unique(pan_true_pred, return_counts=True)
        for label, intersection in zip(labels, label_cnt):
            true_id = label // self.OFFSET
            pred_id = label % self.OFFSET
            self.pan_true_pred_map[(true_id, pred_id)] = intersection

    def get_tp_iou(self):
        # count all matched pairs
        self.true_matched = set()
        self.pred_matched = set()
        for label_tuple, intersection in self.pan_true_pred_map.items():
            true_id, pred_id = label_tuple
            if true_id not in self.true_segms:
                continue
            if pred_id not in self.pred_segms:
                continue
            if self.true_segms[true_id]['iscrowd'] == 1:
                continue
            if self.true_segms[true_id]['category_id'] != \
                    self.pred_segms[pred_id]['category_id']:
                continue

            union = self.pred_segms[pred_id]['area'] + \
                self.true_segms[true_id]['area'] - \
                intersection - self.pan_true_pred_map.get((VOID, pred_id), 0)

            iou = intersection / union
            if iou > 0.5:
                true_cat_id = self.true_segms[true_id]['category_id']
                self.metrics['tp'][true_cat_id] += 1
                self.metrics['iou'][true_cat_id] += iou
                self.true_matched.add(true_id)
                self.pred_matched.add(pred_id)

    def get_fn(self):
        # count false negatives
        self.crowd_labels_dict = {}
        for true_id, true_seg_info in self.true_segms.items():
            true_cat_id = true_seg_info['category_id']
            if true_id in self.true_matched:
                continue
            # crowd segments are ignored
            if true_seg_info['iscrowd'] == 1:
                self.crowd_labels_dict[true_cat_id] = true_id
                continue
            self.metrics['fn'][true_cat_id] += 1

    def get_fp(self):
        # count false positives
        for pred_id, pred_seg_info in self.pred_segms.items():
            if pred_id in self.pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = self.pan_true_pred_map.get((VOID, pred_id), 0)
            # plus intersection with corresponding CROWD region if it exists
            pred_cat_id = pred_seg_info['category_id']
            if pred_cat_id in self.crowd_labels_dict:
                intersection += \
                    self.pan_true_pred_map.get((
                        self.crowd_labels_dict[pred_cat_id], pred_id), 0)
            # predicted segment is ignored if more than half of the segment
            # correspond to VOID and CROWD regions
            if intersection / pred_seg_info['area'] > 0.5:
                continue
            self.metrics['fp'][pred_cat_id] += 1


class metrics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics = self.get_metrics()

    def get_metrics(self):
        metrics = {}
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] == 'classification_metrics' and \
                    self.cfg[task]['active']:
                metrics[task] = semanticMetrics(self.cfg[task]['out_channels'])
            elif self.cfg[task]['metric'] == 'regression_metrics' and \
                    self.cfg[task]['active']:
                metrics[task] = regressionAccruacy()
            elif self.cfg[task]['metric'] == 'panoptic_metrics' and \
                    self.cfg[task]['active']:
                metrics[task] = panopticMetrics(labels)
            else:
                metrics[task] = None
        return metrics

    def update(self, label_trues, label_preds):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None' and self.cfg[task]['active']:
                gt = label_trues[task]
                pred = label_preds[task]
                if torch.is_tensor(gt):
                    gt = gt.data.cpu().numpy()
                if torch.is_tensor(pred):
                    pred = pred.data.cpu().numpy()
                self.metrics[task].update(gt, pred)

    def reset(self):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None' and self.cfg[task]['active']:
                self.metrics[task].reset()
