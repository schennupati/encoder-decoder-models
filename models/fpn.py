from torch import nn

#TODO: Update FPN to take Cross-Stitch

class _FPNModel(nn.Module):

    def __init__(self, in_planes =[2048,1024,512], n_class =21):
        super().__init__()
        self.n_class = n_class
        self.out_channels = 128
        self.upsample1_1  = self.upsample(in_planes[0],256)
        self.upsample1_2  = self.upsample(256,256)
        self.upsample1_3  = self.upsample(256,self.out_channels)
        self.bn1          = nn.BatchNorm2d(self.out_channels)
        self.upsample2_1  = self.upsample(in_planes[1],256)
        self.upsample2_2  = self.upsample(256,self.out_channels)
        self.bn2          = nn.BatchNorm2d(self.out_channels)
        self.upsample3_1  = self.upsample(in_planes[2],self.out_channels)
        self.bn3          = nn.BatchNorm2d(self.out_channels)
        self.upsample4    = nn.Sequential(nn.Conv2d(in_planes[3], self.out_channels, 3,padding=1), nn.ReLU(inplace=True))
        self.bn4          = nn.BatchNorm2d(self.out_channels)
        self.upsample5    = self.upsample(self.out_channels,self.n_class,1,4,2)

    def forward(self, intermediate_result, layers):
        
        for layer in layers:
            print(intermediate_result[layer].size())
    
        # size=(N, 512, x.H/32, x.W/32)
        feat1 = self.upsample1_3(self.upsample1_2(self.upsample1_1(intermediate_result[layers[-1]])))
        feat1 = self.bn1(feat1)
        # size=(N, 512, x.H/16, x.W/16)
        feat2 = self.upsample2_2(self.upsample2_1(intermediate_result[layers[-2]]))
        feat2 = self.bn2(feat2)
        # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.upsample3_1(intermediate_result[layers[-3]])
        feat3 = self.bn3(feat3)
        # size=(N, 128, x.H/4, x.W/4)
        feat4 = self.upsample4(intermediate_result[layers[-4]])
        feat4 = self.bn4(feat4)
        score = feat1 + feat2 + feat3 + feat4
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.upsample5(score)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def upsample(self,in_channels,out_channels,kernel_size=3,factor=2,dilation=1,bn=True):
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,padding=int(kernel_size/2)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

class FPN(_FPNModel):
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