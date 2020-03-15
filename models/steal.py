import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kornia import filters


class StealNMSLoss(nn.Module):
    def __init__(self, r=2):
        super(StealNMSLoss, self).__init__()
        self.grad2d = filters.SpatialGradient()
        self.nms_loss = 0
        self.r = r
    
    def get_grads(self, true_labels):
        """
        Calculate the direstion of edges.
        Ref: https://github.com/nv-tlabs/STEAL/blob/master/utils/edges_nms.m
        """
        eps = self.eps
        true_labels.unsqueeze_(0)
        true_labels.unsqueeze_(0)
        first_gradients = self.grad2d(true_labels)
        second_grad_x = self.grad2d(first_gradients[:, :, 0, :, :])
        second_grad_y = self.grad2d(first_gradients[:, :, 1, :, :])

        grad_yy = second_grad_y[:, :, 1, :, :]
        grad_xy = second_grad_y[:, :, 0, :, :]
        grad_xx = second_grad_x[:, :, 0, :, :]


        theta = torch.fmod(torch.atan((grad_yy * torch.sign(-grad_xy + eps)) / (grad_xx + eps)), np.pi)
        return theta

    def get_nms_per_batch(self, true_labels, predictions):
        """
        Computes the NMS loss for panoptic segmentation.
        
        Args:
            true_labels - True labels (HxW tensor)
            predictions - contour predictions (HxW tensor)
        """
        c,h,w = predictions.size()
        self.eps = torch.finfo(predictions.dtype).eps
        true_labels = F.one_hot(true_labels, num_classes=c)
        for i in range(c):
            preds = predictions[i]
            nms_tensor = torch.zeros_like(preds)
            masked_true = true_labels[:,:,i]
            theta = self.get_grads(masked_true.float())
            # Threshold theta to 4 regions
            thresh_theta = torch.fmod((theta * (5.0 / np.pi)).round() + 5, 5)
            n, c, h, w = list(thresh_theta.size())
            for x in range(self.r, h-self.r):
                for y in range(self.r, w-self.r):
                    dir = thresh_theta[:, :, x, y] % 4
                    #0 is horizontal
                    if dir == 0:
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / torch.sum(torch.exp(preds[x, y-self.r:y+self.r]))
        			#1 is NE SW
                    elif dir == 1: 
                        xx = [val for val in range(x-self.r, x+self.r)]
                        yy = [val for val in reversed(range(y-self.r, y+self.r))]
                        denom = torch.sum(torch.tensor([torch.exp(preds[_xx, _yy]) for _xx, _yy in zip(xx, yy)]))
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / denom
		        	 #2 is vertical
                    elif dir == 2:
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / torch.sum(torch.exp(preds[x-self.r:x+self.r, y]))
			         #3 is NW SE
                    elif dir == 3:
                        xx = [val for val in reversed(range(x-self.r, x+self.r))]
                        yy = [val for val in range(y-self.r, y+self.r)]
                        denom = torch.sum(torch.tensor([torch.exp(preds[_xx, _yy]) for _xx, _yy in zip(xx, yy)]))
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / denom
        
        self.nms_loss += torch.sum(nms_tensor)

    def __call__(self, pred_labels, true_labels):
        """
        NMS loss calculation.
        """
        for i in range(true_labels.size(0)):
            self.get_nms_per_batch(true_labels[i, :], pred_labels[i, :])
        
        return self.nms_loss
