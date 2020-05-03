#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:00:48 2019

@author: sumche
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from models import FCN, FPN, VGGNet, encoder_fn

decoder_map = {'fcn': (FCN), 'fpn': (FPN)}


def get_encoder_decoder(cfg, pretrained_backbone=True):

    encoder_name = cfg['model']['encoder']
    decoder_name = cfg['model']['decoder']
    tasks = cfg['tasks']

    encoder = encoder_fn[encoder_name](pretrained=pretrained_backbone,
                                       progress=True, multiscale=True)

    in_planes = list(encoder.in_planes_map.values())
    in_planes.reverse()
    out_planes = in_planes[1:3]

    model = Encoder_Decoder(encoder, decoder_map[decoder_name], cfg,
                            in_planes=in_planes, out_planes=out_planes)
    return model


def get_task_cls(in_channels, out_channels, kenrel_size=(1, 1)):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kenrel_size))


def get_activation(activation, inplace=True):
    if activation == 'ELU':
        return nn.ELU(alpha=1.0, inplace=inplace)
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    elif activation == 'PReLU':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif activation == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif activation == 'Tanh':
        return nn.Tanh()


class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, base_decoder_fn, cfg,
                 in_planes, out_planes):
        super().__init__()
        self.encoder = encoder
        self.class_decoder = base_decoder_fn(in_planes, out_planes)
        # self.reg_decoder = base_decoder_fn(in_planes, out_planes)
        self.task_cls = {}
        self.model = cfg['model']
        self.heads = cfg['model']['outputs']
        if self.heads['semantic']['active']:
            self.semantic = get_task_cls(out_planes[-1],
                                         self.heads['semantic']['out_channels'])
        if self.heads['instance_contour']['active']:
            num_classes = self.heads['instance_contour']['out_channels']
            self.ce_fusion = nn.Conv2d(4*num_classes, num_classes, groups=num_classes, kernel_size=1, stride=1, padding=0,
                                       bias=True)
        if self.heads['instance_regression']['active']:
            self.instance_regression = get_task_cls(out_planes[-1],
                                                    self.heads['instance_regression']['out_channels'])
        if self.heads['instance_heatmap']['active']:
            self.instance_heatmap = get_task_cls(out_planes[-1],
                                                 self.heads['instance_heatmap']['out_channels'])
        if self.heads['instance_probs']['active']:
            self.instance_probs = get_task_cls(out_planes[-1],
                                               self.heads['instance_probs']['out_channels'])

    def forward(self, x):
        outputs = {}
        x, intermediate_result = self.encoder(x)
        class_score, class_feats = self.class_decoder(intermediate_result)
        # reg_score, _ = self.reg_decoder(intermediate_result)
        if self.heads['semantic']['active']:
            out = self.semantic(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, scale_factor=4,
                                mode='bilinear', align_corners=True)
            outputs['semantic'] = out
        if self.heads['instance_contour']['active']:
            num_classes = self.heads['instance_contour']['out_channels']
            out = self._sliced_concat(
                class_feats[0], class_feats[1], class_feats[2], class_feats[3], num_classes)
            out = self.ce_fusion(out)
            out = F.interpolate(out, scale_factor=4,
                                mode='bilinear', align_corners=True)
            outputs['instance_contour'] = out
        if self.heads['instance_regression']['active']:
            out = self.instance_regression(reg_score)
            out = F.interpolate(out, scale_factor=4,
                                mode='bilinear', align_corners=True)
            outputs['instance_regression'] = out
        if self.heads['instance_heatmap']['active']:
            out = self.instance_heatmap(reg_score)
            out = F.interpolate(out, scale_factor=4,
                                mode='bilinear', align_corners=True)
            outputs['instance_heatmap'] = out
        if self.heads['instance_probs']['active']:
            out = self.instance_probs(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, scale_factor=4,
                                mode='bilinear', align_corners=True)
            outputs['instance_probs'] = out

        return outputs  # Size = (N, n_class, x.H/1, x.W/1)

    def _sliced_concat(self, res1, res2, res3, res5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(
            res1.size(0), out_dim, res1.size(2), res1.size(3))).cuda()
        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = res5[:, class_num, :, :]
            # It needs this trick for multibatch
            out_tensor[:, i + 1, :, :] = res1[:, 0, :, :]
            out_tensor[:, i + 2, :, :] = res2[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = res3[:, 0, :, :]
            class_num += 1
        return out_tensor

    def _normalize(self, out):
        out_tensor = torch.zeros_like(out)
        n = out.size(0)
        for i in range(0, n):
            out_tensor[i, :, :, :] = out[i, :, :, :] - \
                torch.min(out[i, :, :, :])
            if torch.max(out[i, :, :, :]) != 0:
                out_tensor[i, :, :, :] /= torch.max(out[i, :, :, :])
        return out_tensor
