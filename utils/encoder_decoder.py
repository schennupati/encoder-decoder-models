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

from models import FCN, build_fpn, EdgeDNN, DeepLabv3, encoder_fn

decoder_map = {'fcn': (FCN), 'fpn': (build_fpn), 'DeepLabv3': (DeepLabv3)}


def get_encoder_decoder(cfg, pretrained_backbone=True):

    encoder_name = cfg['model']['encoder']
    decoder_name = cfg['model']['decoder']
    tasks = cfg['tasks']
    if 'DeepLab' in decoder_name and 'resnet' in encoder_name:
        encoder = \
            encoder_fn[encoder_name](pretrained=pretrained_backbone,
                                     progress=True,
                                     replace_stride_with_dilation=[True,
                                                                   True,
                                                                   True])
        in_planes = list(encoder.in_planes_map.values())[-1]
        out_planes = [in_planes//8]

    else:
        encoder = encoder_fn[encoder_name](pretrained=pretrained_backbone,
                                           progress=True, multiscale=True)
        in_planes = list(encoder.in_planes_map.values())
        in_planes.reverse()
        if encoder_name in ['resnet18', ['resnet34']]:
            out_planes = in_planes[1:3]
        else:
            out_planes = [planes//8 for planes in in_planes[:2]]

    model = Encoder_Decoder(encoder, decoder_map[decoder_name], cfg,
                            in_planes=in_planes, out_planes=out_planes)
    return model


def get_task_cls(in_channels, out_channels, kenrel_size=(1, 1)):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kenrel_size))


class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, base_decoder_fn, cfg,
                 in_planes, out_planes):
        super().__init__()
        self.encoder = encoder
        self.heads = cfg['model']['outputs']
        self.class_decoder = base_decoder_fn(
            'symmetric', in_planes, out_planes)
        self.reg_decoder = base_decoder_fn('top_down', in_planes, out_planes)
        #self.bbox_decoder = base_decoder_fn(in_planes, out_planes)
        self.model = cfg['model']
        if self.heads['semantic']['active']:
            self.semantic = get_task_cls(out_planes[-1],
                                         self.heads['semantic']['out_channels'])
        if self.heads['semantic_with_instance']['active']:
            self.semantic_with_instance = get_task_cls(out_planes[-1],
                                                       self.heads['semantic_with_instance']['out_channels'])
        if self.heads['instance_contour']['active']:
            num_classes = self.heads['instance_contour']['out_channels']
            self.edge_decoder = EdgeDNN(in_planes, num_classes)
        if self.heads['instance_regression']['active']:
            self.instance_regression = get_task_cls(out_planes[-1],
                                                    self.heads['instance_regression']['out_channels'])
        if self.heads['bounding_box']['active']:
            self.bounding_box_class = get_task_cls(out_planes[-1],
                                                   self.heads['bounding_box']['out_channels'])
            self.bounding_box_offsets = get_task_cls(out_planes[-1], 4)

        if self.heads['instance_heatmap']['active']:
            self.instance_heatmap = get_task_cls(out_planes[-1],
                                                 self.heads['instance_heatmap']['out_channels'])
        if self.heads['instance_probs']['active']:
            self.instance_probs = get_task_cls(out_planes[-1],
                                               self.heads['instance_probs']['out_channels'])

    def forward(self, input):
        outputs = {}
        size = input.shape[-2:]
        _, intermediate_result = self.encoder(input)
        class_score, class_feats = self.class_decoder(intermediate_result)
        reg_score, reg_feats = self.reg_decoder(intermediate_result)
        #bbox_score, _ = self.bbox_decoder(intermediate_result)
        if self.heads['semantic']['active']:
            out = self.semantic(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, size=size,
                                mode='bilinear', align_corners=True)
            outputs['semantic'] = out
        if self.heads['semantic_with_instance']['active']:
            out = self.semantic_with_instance(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, size=size,
                                mode='bilinear', align_corners=True)
            outputs['semantic_with_instance'] = out
        if self.heads['instance_contour']['active']:
            outputs['instance_contour'] = self.edge_decoder(
                intermediate_result)
        if self.heads['instance_regression']['active']:
            out = self.instance_regression(class_score)
            out = F.interpolate(out, size=size,
                                mode='bilinear', align_corners=True)
            outputs['instance_regression'] = out
        if self.heads['bounding_box']['active']:
            scale_factor = 4*self.heads['bounding_box']['scale_factor']
            if self.heads['bounding_box']['multi-scale']:
                offsets_out = {2 ** i: None for i in range(len(reg_feats))}
                class_out = {2**i: None for i in range(len(reg_feats))}
                for i in range(len(reg_feats)):
                    offset_out = self.bounding_box_offsets(reg_feats[i])
                    offset_out = F.relu(offset_out, inplace=True)
                    offset_out = F.interpolate(offset_out,
                                               scale_factor=scale_factor,
                                               mode='bilinear', align_corners=True)

                    out = self.bounding_box_class(reg_feats[i])
                    out = F.relu(out, inplace=True)
                    out = F.interpolate(out, scale_factor=scale_factor,
                                        mode='bilinear', align_corners=True)
                    offsets_out[2**i] = offset_out
                    class_out[2**i] = out
                outputs['bounding_box'] = {
                    'class': class_out, 'offsets': offsets_out}
            else:
                offset_out = self.bounding_box_offsets(reg_score)
                offset_out = F.relu(offset_out, inplace=True)
                offset_out = F.interpolate(offset_out,
                                           scale_factor=scale_factor,
                                           mode='bilinear', align_corners=True)

                out = self.bounding_box_class(class_score)
                out = F.relu(out, inplace=True)
                out = F.interpolate(out, scale_factor=scale_factor,
                                    mode='bilinear', align_corners=True)
                outputs['bounding_box'] = {
                    'class': out, 'offsets': offsets_out}

        if self.heads['instance_heatmap']['active']:
            out = self.instance_heatmap(reg_score)
            out = F.interpolate(out, size=size,
                                mode='bilinear', align_corners=True)
            outputs['instance_heatmap'] = out
        if self.heads['instance_probs']['active']:
            out = self.instance_probs(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, size=size,
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
