#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:00:48 2019

@author: sumche
"""
#TODO: Update to use VGG, DenseNet etc  
#TODO: Update to use PSPNet, DeepLabV3 etc as decoders
#TODO: Implement multi-task feature.
import torch

from collections import OrderedDict

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.fcn import FCN 
from models.fpn import FPN
from models.vgg import VGGNet

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
                       }
                }
                      
    
decoder_map = {'fcn': (FCN),'fpn':(FPN)}

def get_encoder_decoder(cfg,device, pretrained_backbone=True):
    
    encoder_name = cfg['model']['encoder']
    decoder_name = cfg['model']['decoder']
    tasks = cfg['tasks']
    
    if 'resnet' in encoder_name:
        if encoder_name in ['resnet18','resnet34'] and decoder_name=='fpn':
            raise ValueError('{} is not supported with fpn'.format(encoder_name))
        if decoder_name == 'fpn':
            encoder    = resnet_fpn_backbone(encoder_name,pretrained=True)
            inplanes   = [256,256,256,256]
            decoder_fn = decoder_map['fpn']
            
        elif decoder_name == 'fcn':
            return_layers = {'layer2': 'out_1','layer3': 'out_2','layer4': 'out_final'}
            encoder = resnet.__dict__[encoder_name](pretrained=pretrained_backbone,
                                 replace_stride_with_dilation=[False, False, False])       
            encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
            inplanes = inplanes_map['resnet'][encoder_name]
            decoder_fn = decoder_map['fcn']
            
    elif 'vgg' in encoder_name:
        encoder = VGGNet(model=encoder_name, requires_grad=True)
        inplanes = inplanes_map['vgg'][encoder_name]
    
        if decoder_name =='fpn':
            decoder_fn = decoder_map['fpn']
        elif decoder_name =='fcn':
            decoder_fn = decoder_map['fcn']
        
    decoders = nn.ModuleList()
    
    for task in tasks.keys():
        decoder = decoder_fn(inplanes, tasks[task]["out_channels"])
        decoders.extend([decoder])
        
    
    if cfg['model']['cross_stitch']['flag']:
        model = Encoder_Decoder_Cross_Stitch(encoder, decoders, device,cfg['model'])
    else:
        model = Encoder_Decoder(encoder, decoders, cfg['model'])
    
    return model


class _Encoder_Decoder(nn.Module):

    def __init__(self, encoder, decoders, cfg):
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.cfg = cfg

    def forward(self, x):
        output = self.encoder(x)
        encoder= self.cfg['encoder']
        decoder= self.cfg['decoder']
        intermediate_result = OrderedDict()
        layers = [k for k,_ in output.items()]
        layers = layers[:-1] if 'resnet' in encoder and 'fpn' in decoder else layers
        for layer in layers:
            intermediate_result[layer] = output[layer]
        outputs = []

        for decoder in self.decoders:
            outputs.append(decoder(intermediate_result,layers))
        
        return outputs   # size=(N, n_class, x.H/1, x.W/1)

class _Encoder_Decoder_Cross_Stitch(nn.Module):

    def __init__(self, encoder, decoders,device, cfg):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.get_ops(decoders)
        self.fpn_cross = FPN_Cross_Stitch(self.op_names,self.ops,device)
        

    def forward(self, x):
        output = self.encoder(x)
        encoder= self.cfg['encoder']
        decoder= self.cfg['decoder']
        intermediate_result = OrderedDict()
        layers = [k for k,_ in output.items()]
        layers = layers[:-1] if 'resnet' in encoder and 'fpn' in decoder else layers
        for layer in layers:
            intermediate_result[layer] = output[layer]        
            
        return self.fpn_cross(intermediate_result, layers)  # size=(N, n_class, x.H/1, x.W/1)
    
    def get_ops(self,decoders):
        self.op_names = {}
        self.ops = {}
        for i,decoder in enumerate(decoders):
            self.op_names[i] = []
            self.ops[i] = nn.ModuleList()
            for op_name,op in decoder.named_children():
                self.op_names[i].append(op_name)
                self.ops[i].extend([op])
   
class _FPNModel_Cross_Stitch(nn.Module):

    def __init__(self, op_names,ops,device):
        super().__init__()
        self.op_names = op_names
        self.ops      = ops
        self.device   = device
        self.n_ops    = len(self.op_names[0])
        self.n_tasks  = len(self.op_names.keys())
        self.feats    = {}
        self.scores   = {}
        self.in_cfg   = [3,None,None,None,2,None,None,1,None,0,None]
        self.out_cfg  = [None,None,None,1,None,None,2,None,3,None,4]
        self.cross_cfg= [0,1,2,None,3,4,None,5,None,6,None]
        self.cross_stitch_layers = self.make_cross_stitch(7,self.n_tasks)

    def forward(self, intermediate_result, layers):
        
        for op in range(self.n_ops-1):
            self.feats[op] = {}
            for task in range(self.n_tasks):
                if self.in_cfg[op] is not None:
                    self.feats[op][task] = self.ops[task][op].to(self.device)(intermediate_result[self.in_cfg[op]])
                else:
                    self.feats[op][task] = self.ops[task][op].to(self.device)(self.feats[op-1][task])
                if op > 0 and self.out_cfg[op-1] is None:
                    self.feats[op-1][task] = None
            
            if self.cross_cfg[op] is not None:
                out_features = self.cross_stitch_layers[self.cross_cfg[op]](self.feats[op])#
                for i,out_feature in enumerate(out_features):
                    self.feats[op][i] = out_feature
                out_features = None
                
        for task in range(self.n_tasks):
            self.scores[task] = 0.0
            for i in range(len(self.out_cfg)):
                if self.out_cfg[i] is not None:                           
                    self.scores[task] += self.feats[i][task]
            self.scores[task] = self.ops[task][-1].to(self.device)(self.scores[task])
        return list(self.cross_stitch_layers.parameters()),self.scores # size=(N, n_class, x.H/1, x.W/1)
    
    def make_cross_stitch(self,n_ops,n_tasks):
        cross_stitch_layers = nn.ModuleList()
        for op in range(n_ops):
            cross_stitch_layers.extend([cross_stitch_layer(n_tasks)])
        
        return cross_stitch_layers
        

class _cross_stitch_layer(nn.Module):
    def __init__(self,n_tasks):
        super(_cross_stitch_layer,self).__init__()
        
        if n_tasks == 1:
            raise ValueError('Cross Stitch is applied on atleast 2 tasks')
        else:
            self.cross_stitch = nn.Parameter(torch.randn(n_tasks,n_tasks))#,device=device))
        self.cross_stitch.requiresGrad = True
        
        #self.cross_stitch_parameter = nn.ParameterList([self.cross_stitch])
    
    def forward(self,in_features):
        
        x_in = []
        
        for _,in_feature in in_features.items():
            n,h,w,c = in_feature.size()
            x_in.append(torch.unsqueeze(in_feature.view(n,-1,1),dim=-1))
        
        _in = torch.cat((x_in),dim=-1)
        _out = torch.matmul(_in,torch.sigmoid(self.cross_stitch))#_parameter[0])
        
        return _out.view(n,h,w,c,-1).permute(4,0,1,2,3)
    
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

class Encoder_Decoder_Cross_Stitch(_Encoder_Decoder_Cross_Stitch):
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

class cross_stitch_layer(_cross_stitch_layer):
    
    pass

class FPN_Cross_Stitch(_FPNModel_Cross_Stitch):
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