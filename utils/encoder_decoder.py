#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:00:48 2019

@author: sumche
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.autograd import Variable

from models.fcn import FCN 
from models.fpn import FPN
from models.vgg import VGGNet
from models.efficientnet import efficientnet_b0, efficientnet_b1
from utils.model_uitls import get_activation

import pdb
__all__ = ['FCN']

inplanes_map = {
                'resnet':
                         {
                          'resnet18':[512,256,128],'resnet34':[512,256,128],
                          'resnet50':[2048,1024,512],'resnet101':[2048,1024,512],
                          'resnet152':[2048,1024,512],'resnext50_32x4d':[2048,1024,512],
                          'resnext101_32x8d':[2048,1024,512],'wide_resnet50_2':[2048,1024,512],
                          'wide_resnet101_2':[2048,1024,512]
                          },
                'vgg':
                      {
                       'vgg16':[512,512,256,128],'vgg19':[512,512,256,128]
                       },
                'efficientnet':
                      {
                       'efficientnet_b0':[320,112,40,24],'efficientnet_b1':[320,112,40,24]
                       }
                }

outplanes_map = {
                'resnet':
                         {
                          'fpn':[256,128], 'fcn':[512,256,128,64,32]
                          },
                'vgg':
                      {
                       'fpn':[256,128], 'fcn':[512,256,128,64,32]
                       },
                'efficientnet':
                      {
                       'fpn':[128,64], 'fcn':[320,112,40,40,32]
                       }
                }                    
                      
    
decoder_map = {'fcn': (FCN),'fpn':(FPN)}

def get_encoder_decoder(cfg, pretrained_backbone=True):
    
    encoder_name = cfg['model']['encoder']
    decoder_name = cfg['model']['decoder']
    tasks = cfg['tasks']
    
    if 'resnet' in encoder_name:
        if encoder_name in ['resnet18','resnet34'] and decoder_name=='fpn':
            raise ValueError('{} is not supported with fpn'.format(encoder_name))
        if decoder_name == 'fpn':
            encoder    = resnet_fpn_backbone(encoder_name,pretrained=True)
            inplanes   = [256,256,256,256]
            outplanes = outplanes_map['resnet'][decoder_name]
            decoder_fn = decoder_map['fpn']
            
        elif decoder_name == 'fcn':
            return_layers = {'layer2': 'out_1','layer3': 'out_2','layer4': 'out_final'}
            encoder = resnet.__dict__[encoder_name](pretrained=pretrained_backbone,
                                 replace_stride_with_dilation=[False, False, False])       
            encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
            inplanes = inplanes_map['resnet'][encoder_name]
            outplanes = outplanes_map['resnet'][decoder_name]
            decoder_fn = decoder_map['fcn']
            
    elif 'vgg' in encoder_name:
        encoder = VGGNet(model=encoder_name, requires_grad=True)
        inplanes = inplanes_map['vgg'][encoder_name]
        outplanes = outplanes_map['vgg'][decoder_name]
        if decoder_name =='fpn':
            decoder_fn = decoder_map['fpn']
        elif decoder_name =='fcn':
            decoder_fn = decoder_map['fcn']
    
    elif 'efficientnet' in encoder_name:
        if encoder_name == 'efficientnet_b0':
            encoder = efficientnet_b0(pretrained=True).features
            return_layers = {'3': 'out_1','5': 'out_2','11': 'out_3','16': 'out_final'}
            #for name,_ in encoder.named_children():
                #return_layers[name]=name
        elif encoder_name == 'efficientnet_b1':
            encoder = efficientnet_b1(pretrained=True).features
            #return_layers = {}
            return_layers = {'5': 'out_1','8': 'out_2','16': 'out_3','23': 'out_final'}
            #for name,_ in encoder.named_children():
                #return_layers[name]=name
        encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
        inplanes = inplanes_map['efficientnet'][encoder_name]
        outplanes = outplanes_map['efficientnet'][decoder_name]
        if decoder_name =='fpn':
            decoder_fn = decoder_map['fpn']
        elif decoder_name =='fcn':
            decoder_fn = decoder_map['fcn']
        
        
    ''' 
    decoders = nn.ModuleList()
    for task in tasks.keys():
        if tasks[task]['active']:
            task_cfg = tasks[task]
            decoder = decoder_fn(in_planes=inplanes, out_planes= outplanes,
                                 n_class=task_cfg["out_channels"],
                                 activation=task_cfg["activation"],
                                 activate_last=task_cfg["activate_last"])
            decoders.extend([decoder])
    '''
    model = Encoder_Decoder(encoder, decoder_fn, cfg, 
                            in_planes=inplanes, out_planes= outplanes)
    #pdb.set_trace()
    return model

def get_task_cls(in_channels, out_channels, kenrel_size = (1,1)):
    #pdb.set_trace()
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kenrel_size))
            
def get_intermediate_result(model, output):
    encoder= model['encoder']
    decoder= model['decoder']
    intermediate_result = OrderedDict()
    layers = [k for k,_ in output.items()]
    if 'resnet' in encoder and 'fpn' in decoder:
        layers = layers[:-1]
    elif 'efficientnet' in encoder and 'fcn' in decoder:
        layers = layers[1:]
    else: 
        layers = layers
    for layer in layers:
        intermediate_result[layer] = output[layer]
    return intermediate_result, layers

class _Encoder_Decoder(nn.Module):
    def __init__(self, encoder, base_decoder_fn, cfg, 
                 in_planes, out_planes):
        super().__init__()
        self.encoder = encoder
        self.class_decoder = base_decoder_fn(in_planes, out_planes)
        self.reg_decoder = base_decoder_fn(in_planes, out_planes)
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
        x = self.encoder(x)
        intermediate_result, layers = get_intermediate_result(self.model, x)
        class_score, class_feats = self.class_decoder(intermediate_result,layers)
        reg_score, _ = self.reg_decoder(intermediate_result,layers)
        if self.heads['semantic']['active']:
            out = self.semantic(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, scale_factor= 4, mode='bilinear', align_corners=True)
            outputs['semantic'] = out
        if self.heads['instance_contour']['active']:
            num_classes = self.heads['instance_contour']['out_channels']
            out = self._sliced_concat(class_feats[0], class_feats[1], class_feats[2], class_feats[3], num_classes)
            out = self.ce_fusion(out)
            out = F.interpolate(out, scale_factor= 4, mode='bilinear', align_corners=True)
            out = torch.sigmoid(out)
            outputs['instance_contour'] = out
        if self.heads['instance_regression']['active']:
            out = self.instance_regression(reg_score)
            out = F.interpolate(out, scale_factor= 4, mode='bilinear', align_corners=True)
            outputs['instance_regression'] = out
        if self.heads['instance_heatmap']['active']:
            out = self.instance_heatmap(reg_score)
            out = F.interpolate(out, scale_factor= 4, mode='bilinear', align_corners=True)
            outputs['instance_heatmap'] = out
        if self.heads['instance_probs']['active']:

            out = self.instance_probs(class_score)
            out = F.relu(out, inplace=True)
            out = F.interpolate(out, scale_factor= 4, mode='bilinear', align_corners=True)
            outputs['instance_probs'] = out 

        return outputs   # size=(N, n_class, x.H/1, x.W/1)
    
    def _sliced_concat(self, res1, res2, res3, res5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(res1.size(0), out_dim, res1.size(2), res1.size(3))).cuda()
        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = res5[:, class_num, :, :]
            out_tensor[:, i + 1, :, :] = res1[:, 0, :, :]  # it needs this trick for multibatch
            out_tensor[:, i + 2, :, :] = res2[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = res3[:, 0, :, :]

            class_num += 1

        return out_tensor

    def _normalize(self, out):
        out_tensor = torch.zeros_like(out)
        n = out.size(0)
        for i in range(0, n):
            out_tensor[i, :, :, :] = out[i, :, :, :] - torch.min(out[i, :, :, :])
            if torch.max(out[i, :, :, :]) != 0:
                out_tensor[i, :, :, :] /= torch.max(out[i, :, :, :])
        return out_tensor


class Encoder_Decoder(_Encoder_Decoder):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

