from torch import nn
from utils.model_uitls import get_activation
#TODO: Update FCN to take Cross-Stitch
import pdb
class _FCNModel(nn.Module):

    def __init__(self, in_planes =[2048,1024,512], out_planes = [512,256,128,64,32], 
                 n_class =19,activation='ReLU',activate_last=True):
        super().__init__()
        self. out_planes = out_planes
        self.activation = activation
        self.n_class = n_class
        self.activation = get_activation(activation,inplace=True)
        self.feat1   = nn.Sequential(nn.Conv2d(in_planes[0], out_planes[0], 1), nn.ReLU(inplace=True))
        self.deconv1 = nn.ConvTranspose2d(out_planes[0], out_planes[0], 
                                          kernel_size=3, stride=2, padding=1,
                                          dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(out_planes[0])
        self.feat2   = nn.Sequential(nn.Conv2d(in_planes[1], out_planes[0], 1), 
                                     nn.ReLU(inplace=True))
        self.deconv2 = nn.ConvTranspose2d(out_planes[0], out_planes[1], 
                                          kernel_size=3, stride=2, padding=1, 
                                          dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(out_planes[1])
        self.feat3   = nn.Sequential(nn.Conv2d(in_planes[2], out_planes[1], 1), 
                                     nn.ReLU(inplace=True))
        self.deconv3 = nn.ConvTranspose2d(out_planes[1], out_planes[2], 
                                          kernel_size=3, stride=2, padding=1, 
                                          dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(out_planes[2])
        self.deconv4 = nn.ConvTranspose2d(out_planes[2], out_planes[3],
                                          kernel_size=3, stride=2, padding=1, 
                                          dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(out_planes[3])
        self.deconv5 = nn.ConvTranspose2d(out_planes[3], out_planes[4], 
                                          kernel_size=3, stride=2, padding=1,
                                          dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(out_planes[4])
        self.classifier = nn.ConvTranspose2d(out_planes[4], n_class, kernel_size=1)
        
    def forward(self, intermediate_result, layers):
        
        #for layer in layers:
        # size=(N, 512, x.H/32, x.W/32)
        feat1 = self.feat1(intermediate_result[layers[-1]])
        score = self.bn1(self.activation(self.deconv1(feat1)))
        #print(intermediate_result[layers[-1]].size(), feat1.size(), score.size())
        # size=(N, 512, x.H/16, x.W/16)
        feat2 = self.feat2(intermediate_result[layers[-2]])
        score = score + feat2 
        #print(intermediate_result[layers[-2]].size(),feat2.size(), score.size())                    
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.activation(self.deconv2(score)))
        
        # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.feat3(intermediate_result[layers[-3]])
        #print(intermediate_result[layers[-3]].size(),feat3.size(), score.size())
        #pdb.set_trace()
        score = score + feat3
                             
        # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.activation(self.deconv3(score)))
        
        # size=(N, 128, x.H/4, x.W/4)
        #score = score
        
        # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.activation(self.deconv4(score)))
        
        # size=(N, 64, x.H/2, x.W/2)
        #score = score   
        
        # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.activation(self.deconv5(score)))
        score = self.classifier(score)
        # size=(N, n_class, x.H, x.W)
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