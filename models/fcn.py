from collections import OrderedDict
from torch import nn
import torch.nn.functional as F

#TODO: Update FCN to take FCN8,FCN16 or FCN32 based on inplanes
#TODO: Update FCN to take Cross-Stitch

class _FCNModel(nn.Module):

    def __init__(self, backbone, in_planes =[2048,1024,512], n_class =21):
        super().__init__()
        self.n_class = n_class
        self.backbone = backbone
        self.relu    = nn.ReLU(inplace=True)
        self.feat1   = nn.Sequential(nn.Conv2d(in_planes[0], 1024, 1), nn.ReLU(inplace=True))
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.feat2   = nn.Sequential(nn.Conv2d(in_planes[1], 512, 1), nn.ReLU(inplace=True))
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.feat3   = nn.Sequential(nn.Conv2d(in_planes[2], 256, 1), nn.ReLU(inplace=True))
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.backbone(x)
        intermediate_result = OrderedDict()
        layers = [k for k,_ in output.items()]
        for layer in layers:
            intermediate_result[layer] = output[layer]    # size=(N, 512, x.H/32, x.W/32)
        feat1 = self.feat1(intermediate_result[layers[-1]])
        score = self.bn1(self.relu(self.deconv1(feat1)))  # size=(N, 512, x.H/16, x.W/16)
        feat2 = self.feat2(intermediate_result[layers[-2]])
        score = score + feat2                             # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.feat3(intermediate_result[layers[-3]])
        score = score + feat3                             # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score   # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score   # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        #score = F.softmax(score,dim=1)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)




class FCN(_FCNModel):
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

class _FPN_FCNModel(nn.Module):

    def __init__(self, backbone, in_planes =[2048,1024,512], n_class =21):
        super().__init__()
        self.n_class = n_class
        self.backbone = backbone
        self.out_channels = 128
        self.upsample1_1  = self.upsample(in_planes[0],256)
        self.upsample1_2  = self.upsample(256,256)
        self.upsample1_3  = self.upsample(256,self.out_channels)
        self.bn1          = nn.BatchNorm2d(self.out_channels)
        self.upsample2_1  = self.upsample(in_planes[0],256)
        self.upsample2_2  = self.upsample(256,self.out_channels)
        self.bn2          = nn.BatchNorm2d(self.out_channels)
        self.upsample3_1  = self.upsample(in_planes[0],self.out_channels)
        self.bn3          = nn.BatchNorm2d(self.out_channels)
        self.upsample4    = nn.Sequential(nn.Conv2d(in_planes[0], self.out_channels, 3,padding=1), nn.ReLU(inplace=True))
        self.bn4          = nn.BatchNorm2d(self.out_channels)
        self.upsample5    = self.upsample(self.out_channels,self.out_channels,1,4,2)
        self.classifier   = nn.Conv2d(self.out_channels, self.n_class, kernel_size=1)

    def forward(self, x):
        output = self.backbone(x)
        intermediate_result = OrderedDict()
        layers = [k for k,_ in output.items()]
        for layer in layers:
            intermediate_result[layer] = output[layer]   
        # size=(N, 512, x.H/32, x.W/32)
        feat1 = self.upsample1_3(self.upsample1_2(self.upsample1_1(intermediate_result[layers[-2]])))
        feat1 = self.bn1(feat1)
        # size=(N, 512, x.H/16, x.W/16)
        feat2 = self.upsample2_2(self.upsample2_1(intermediate_result[layers[-3]]))
        feat2 = self.bn2(feat2)
        # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.upsample3_1(intermediate_result[layers[-4]])
        feat3 = self.bn3(feat3)
        # size=(N, 128, x.H/4, x.W/4)
        feat4 = self.upsample4(intermediate_result[layers[-5]])
        feat4 = self.bn4(feat4)
        score = feat1 + feat2 + feat3 + feat4
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.upsample5(score)
        score = self.classifier(score)
        #score = F.softmax(score,dim=1)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def upsample(self,in_channels,out_channels,kernel_size=3,factor=2,dilation=1):
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,padding=int(kernel_size/2)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(out_channels, out_channels, 
                                         kernel_size=3,stride=factor, 
                                         padding=1, dilation=dilation, output_padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
        
        
        




class FPN_FCN(_FPN_FCNModel):
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