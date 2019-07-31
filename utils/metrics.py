#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:41:09 2019

@author: sumche
"""
#Source: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch

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
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
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
        
        
class metrics:
    def __init__(self,cfg):
        self.cfg = cfg    
        self.metrics = self.get_metrics()
        
    def get_metrics(self):
        metrics = {}
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] == 'classification_metrics':
                metrics[task] = runningScore(self.cfg[task]['out_channels'])
            elif self.cfg[task]['metric'] == 'regression_metrics':
                metrics[task] = regressionAccruacy()
            else:
                metrics[task] = None
        return metrics
    
    def update(self,label_trues, label_preds):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None':
                gt = label_trues[task].data.cpu().numpy()
                pred = label_preds[task].data.cpu().numpy()
                self.metrics[task].update(gt, pred) 
    
    def reset(self):
        for task in self.cfg.keys():
            if self.cfg[task]['metric'] != 'None':
                self.metrics[task].reset()