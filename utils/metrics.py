#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:41:09 2019

@author: sumche
"""
#Source: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch
from utils.im_utils import panid2label

class runningScore(object):
    def __init__(self, n_classes):

        self.n_classes = n_classes
        self.reset()

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
class regressionAccruacy(object):
    def __init__(self):
        self.reset()
        self.reset()
        
    def accuracy(self, gt, pred, pct_close = 1e-1 ):
        # data_x and data_y are numpy array-of-arrays matrices
        gt = torch.tensor(gt)
        pred = torch.tensor(pred)
        if gt.size() != pred.size():
            pred = pred.permute(0,2,3,1).squeeze()
        n_items = pred.numel()
        n_correct = torch.sum((torch.abs(pred- gt) < pct_close))
        return (n_correct.item() * 100.0 / n_items)  # scalar
    
    def update(self,label_trues, label_preds,pct_close=0.025):
        self.acc_1 = self.accuracy(label_trues, label_preds,pct_close)
        self.acc_2 = self.accuracy(label_trues, label_preds,pct_close*1e-1)
        self.acc_3 = self.accuracy(label_trues, label_preds,pct_close*1e-2)
        
    def get_scores(self):
        acc_1 = self.acc_1
        acc_2 = self.acc_2
        acc_3 = self.acc_3
        return ({"Regression Acc delta_1": acc_1,
                 "Regression Acc delta_2": acc_2,
                 "Regression Acc delta_3": acc_3},None)
        
    def reset(self):
        self.acc_1 = 0.0
        self.acc_2 = 0.0
        self.acc_3 = 0.0
        
class panopticMetrics(object):
    def __init__(self):
        self.OFFSET = 256*256*256
        self.reset()

    def convert_color_to_index(self, colors):
        if colors.shape[-1] != 3:
            np.transpose(colors, (0, 3, 1, 2))
        colors = colors[:, :, :, 0] + 256 * colors[:, :, :, 1] + 256 * 256 * colors[:, :, :, 2]
        batch_size = colors.shape[0]
        colors = colors.reshape(batch_size, -1)
        return colors

    def update(self, label_true, label_preds):
        #import pdb; pdb.set_trace()
        #print(label_true)
        #print(label_preds)
        label_true = self.convert_color_to_index(label_true)
        label_preds = self.convert_color_to_index(label_preds)
        combined_labels = label_true * self.OFFSET + label_preds
        comb_labels, comb_label_cnt = np.unique(combined_labels, return_counts=True)
        for label, cnts in zip(comb_labels, comb_label_cnt):
            true_label = int(label // self.OFFSET)
            pred_label = int(label % self.OFFSET)
            true_cat_id = panid2label[true_label].categoryId
            pred_cat_id = panid2label[pred_label].categoryId
            # TODO: Ignore 'Crowd' classes
            if true_label == pred_label:
                union = (label_preds == pred_label).sum() + (label_true == true_label).sum() - cnts
                iou = cnts / union
                if iou > 0.5:
                    self.metrics['iu'][true_cat_id] = self.metrics['iu'].get(true_cat_id, 0) + iou
                    self.metrics['tp'][true_cat_id] = self.metrics['tp'].get(true_cat_id, 0) + 1
            else:
                self.metrics['fn'][true_cat_id] = self.metrics['fn'].get(true_cat_id, 0) + 1
                self.metrics['fp'][pred_cat_id] = self.metrics['fp'].get(pred_cat_id, 0) + 1    
            
    def get_scores(self):
        """Returns accuracy score evaluation result.
            - PQ
            - RQ
            - SQ
            - mean_iu
        """
        # Calc Panoptic metrics (SQ, RQ, PQ) (https://arxiv.org/pdf/1801.00868.pdf)
        iu_nparray = np.array(self.metrics['iu'].values())
        tp_nparray = np.array(self.metrics['tp'].values())
        fp_nparray = np.array(self.metrics['fp'].values())
        fn_nparray = np.array(self.metrics['fn'].values())
        sq = iu_nparray / tp_nparray
        rq = tp_nparray / (tp_nparray + 0.5 * fp_nparray + 0.5 * fn_nparray)
        pq = sq * rq

        mean_iu = np.nanmean(iu_nparray)
        mean_sq = np.nanmean(sq)
        mean_rq = np.nanmean(rq)
        mean_pq = np.nanmean(pq)
        cls_metrics = {
            'classes': np.arange(self.n_classes),
            'iu': self.metrics['iu'],
            'pq': pq,
            'rq': rq,
            'sq': sq
        }

        return (
            {
                "Mean PQ": mean_pq,
                "Mean RQ": mean_rq,
                "Mean SQ": mean_sq,
                "Mean IU": mean_iu
            },
            cls_metrics,
        )

    def reset(self):
        self.metrics = {'iu': {}, 
                        'tp': {}, 
                        'fp': {}, 
                        'fn': {}}


class metrics:
    def __init__(self,cfg):
        self.cfg = cfg    
        self.metrics = self.get_metrics()
        
    def get_metrics(self):
        metrics = {}
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] == 'classification_metrics' and self.cfg[task]['active']:
                metrics[task] = runningScore(self.cfg[task]['out_channels'])
            elif self.cfg[task]['metric'] == 'regression_metrics' and self.cfg[task]['active']:
                metrics[task] = regressionAccruacy()
            elif self.cfg[task]['metric'] == 'panoptic_metrics' and self.cfg[task]['active']:
                # TODO: Send the number of panoptic classes from cfg
                metrics[task] = panopticMetrics()
            else:
                metrics[task] = None
        return metrics
    
    def update(self,label_trues, label_preds):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None' and self.cfg[task]['active']:
                if self.cfg[task]['metric'] == 'panoptic_metrics':
                    gt = label_trues['panoptic'].cpu().numpy()
                    pred = label_preds['panoptic'].cpu().numpy()
                else:
                    gt = label_trues[task].data.cpu().numpy()
                    pred = label_preds[task].data.cpu().numpy()
                self.metrics[task].update(gt, pred) 
    
    def reset(self):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None' and self.cfg[task]['active']:
                self.metrics[task].reset()