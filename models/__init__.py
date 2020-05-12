from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from models.efficientnet import efficientnet_b0, efficientnet_b1, \
    efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, \
    efficientnet_b6, efficientnet_b7

from models.fcn import FCN
from models.fpn import FPN
from models.hed import EdgeDNN
from models.steal import StealNMSLoss


encoder_fn = {'resnet18': (resnet18), 'resnet34': (resnet34),
              'resnet50': (resnet50), 'resnet101': (resnet101),
              'resnet152': (resnet152), 'resnext50_32x4d': (resnext50_32x4d),
              'resnext101_32x8d': (resnext101_32x8d),
              'wide_resnet50_2': (wide_resnet50_2),
              'wide_resnet101_2': (wide_resnet101_2),
              'efficientnet_b0': (efficientnet_b0),
              'efficientnet_b1': (efficientnet_b1),
              'efficientnet_b2': (efficientnet_b2),
              'efficientnet_b3': (efficientnet_b3),
              'efficientnet_b4': (efficientnet_b4),
              'efficientnet_b5': (efficientnet_b5),
              'efficientnet_b6': (efficientnet_b6),
              'efficientnet_b7': (efficientnet_b7)}
