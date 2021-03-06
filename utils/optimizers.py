#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:23:02 2019

@author: sumche
"""
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import MultiStepLR

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

key2sched = {
    "MultiStepLR": MultiStepLR
}


def get_optimizer(cfg):
    if cfg["optimizer"] is None:
        return SGD

    else:
        opt_name = cfg["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError(
                "Optimizer {} not implemented".format(opt_name))

        return key2opt[opt_name]


def get_scheduler(cfg):
    if cfg["scheduler"] is None:
        return MultiStepLR

    else:
        sched_name = cfg["scheduler"]["name"]
        if sched_name not in key2sched:
            raise NotImplementedError(
                "Optimizer {} not implemented".format(sched_name))

        return key2sched[sched_name]
