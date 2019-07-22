#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 08:32:02 2019

@author: sumche
"""

#TODO: Update code to use Semantic FPN ---> DONE 
#TODO: Update to use VGG, DenseNet etc  
#TODO: Update to use PSPNet, DeepLabV3 etc as decoders
#TODO: Implement multi-task feature.
import torch
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from models.vgg import VGGNet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.fcn import FCN, FPN_FCN

__all__ = ['FCN']

def get_encoder_decoder(encoder_name, decoder_name, num_classes, fpn=False, pretrained_backbone=True):
    
    if fpn and 'resnet' in encoder_name:
        encoder = resnet_fpn_backbone(encoder_name,pretrained=True)
        inplanes_map = {
            'resnet18':[256],'resnet34':[256],'resnet50':[256],
            'resnet101':[256],'resnet152':[256],'resnext50_32x4d':[256],
            'resnext101_32x8d':[256],'wide_resnet50_2':[256],'wide_resnet101_2':[256]
            }
    
        model_map = {
                'deeplab': (DeepLabHead, DeepLabV3),
                'fcn': (FPN_FCN)
                    }
    
    elif 'resnet' in encoder_name:    
        encoder = resnet.__dict__[encoder_name](
                pretrained=pretrained_backbone,
                replace_stride_with_dilation=[False, False, False])

        return_layers = {'layer2': 'out_1','layer3': 'out_2','layer4': 'out_final'}
        encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
        
        inplanes_map = {
            'resnet18':[512,256,128],
            'resnet34':[512,256,128],
            'resnet50':[2048,1024,512],
            'resnet101':[2048,1024,512],
            'resnet152':[2048,1024,512],
            'resnext50_32x4d':[2048,1024,512],
            'resnext101_32x8d':[2048,1024,512],
            'wide_resnet50_2':[2048,1024,512],
            'wide_resnet101_2':[2048,1024,512]
            }
    
        model_map = {
                'deeplab': (DeepLabHead, DeepLabV3),
                'fcn': (FCN)
                    }
    elif 'vgg' in encoder_name:
        encoder = VGGNet(model=encoder_name, requires_grad=True)
        inplanes_map = {
            'vgg16':[512,512,256]
            }
        model_map = {
                'deeplab': (DeepLabHead, DeepLabV3),
                'fcn': (FCN)
                    }
        
    out = encoder(torch.rand(1, 3, 512, 512))
    
    for _,layer in out.items():
        print(layer.size())

    inplanes = inplanes_map[encoder_name]
    base_model = model_map[decoder_name]
    
    model = base_model(encoder, inplanes, num_classes)
    
    return model

