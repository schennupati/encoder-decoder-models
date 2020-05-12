import torch
from torch import nn
from torch.autograd import Variable


class EdgeDNN(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 256, 64], num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        in_planes.reverse()
        self.side1 = self.side_feature(in_planes=in_planes[0],
                                       num_classes=1, stride=2)
        self.side2 = self.side_feature(in_planes=in_planes[1],
                                       num_classes=1, stride=4)
        self.side3 = self.side_feature(in_planes=in_planes[2],
                                       num_classes=1, stride=8)
        # self.side4 = self.side_feature(in_planes=in_planes[3],
        #                               num_classes=1, stride=8)
        self.side5 = self.side_feature(in_planes=in_planes[4],
                                       num_classes=num_classes, stride=32)
        self.ce_fusion = nn.Sequential(nn.Conv2d(num_classes * 4, num_classes,
                                                 kernel_size=1,
                                                 groups=num_classes))

    def forward(self, intermediate_result):
        side1 = self.side1(intermediate_result[0])
        side2 = self.side2(intermediate_result[1])
        side3 = self.side3(intermediate_result[2])
        side5 = self.side5(intermediate_result[4])
        out = self._shared_concat(side1, side2, side3, side5, self.num_classes)
        out = self.ce_fusion(out)

        return out

    def side_feature(self, in_planes, num_classes=9, stride=1):
        layers = [nn.Conv2d(in_planes, num_classes, kernel_size=1)]
        if stride > 1:
            layers += [nn.Upsample(scale_factor=stride,
                                   mode='bilinear', align_corners=False)]
        return nn.Sequential(*layers)

    def _shared_concat(self, side1, side2, side3, side5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(
            side1.size(0), out_dim, side1.size(2), side1.size(3))).cuda()
        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = side5[:, class_num, :, :]
            # It needs this trick for multibatch
            out_tensor[:, i + 1, :, :] = side1[:, 0, :, :]
            out_tensor[:, i + 2, :, :] = side2[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = side3[:, 0, :, :]
            class_num += 1
        return out_tensor
