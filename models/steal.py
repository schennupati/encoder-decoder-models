import torch.nn as nn
from kornia import filters


class StealNMSLoss(nn.module):
    def __init__(self, r=2):
        self.grad2d = filters.SpatialGradient()
        self.nms_loss = 0
        self.r = r
    
    def get_grads(self, true_labels):
        """
        Calculate the direstion of edges.
        Ref: https://github.com/nv-tlabs/STEAL/blob/master/utils/edges_nms.m
        """

        first_gradients = self.grad2d(true_labels)
        second_grad_x = self.grad2d(first_gradients[:, 0, :, :])
        second_grad_y = self.grad2d(first_gradients[:, 1, :, :])

        grad_yy = second_grad_y[:, 1, :, :]
        grad_xy = second_grad_y[:, 0, :, :]
        grad_xx = second_grad_x[:, 0, :, :]


        theta = torch.fmod(torch.atan(
                                     (grad_yy * torch.sign(-grad_xy + eps)) / (grad_xx + eps)),
                           np.pi)
        return theta

    def get_nms_per_batch(self, true_labels, preds):
        """
        Computes the NMS loss for panoptic segmentation.
        
        Args:
            true_labels - True labels (HxW tensor)
            preds - contour predictions (HxW tensor)
        """
        eps = torch.finfo(probs.dtype).eps
        labels = torch.unique(true)
        for label in labels:
            nms_tensor = tensor.zeros_like(preds)
            masked_true = true[true == label]
            theta = self.get_grads(masked_true)
            # Threshold theta to 4 regions
            thresh_theta = torch.fmod((theta * (5.0 / np.pi)).round() + 5, 5)
            h, w = list(thresh_theta.size())
            for x in range(self.r, h-self.r):
                for y in range(self.r, w-self.r):
                    dir = thresh_theta[x, y] % 4
                    if dir == 0: #0 is horizontal
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / torch.sum(torch.exp(pred[x, y-self.r:y+self.r]))
        			elif dir == 1: #1 is NE-SW
                        xx = [val for val in range(x-self.r, x+self.r)]
                        yy = [val for val in reversed(range(y-self.r, y+self.r))]
                        denom = torch.sum(torch.tensor([torch.exp(preds[_xx, _yy]) for _xx, _yy in zip(xx, yy)]))
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / denom
		        	elif dir == 2: #2 is vertical
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / torch.sum(torch.exp(pred[x-self.r:x+self.r, y]))
			        elif dir == 3: #3 is NW-SE
                        xx = [val for val in reversed(range(x-self.r, x+self.r))]
                        yy = [val for val in range(y-self.r, y+self.r)]
                        denom = torch.sum(torch.tensor([torch.exp(preds[_xx, _yy]) for _xx, _yy in zip(xx, yy)]))
                        nms_tensor[x, y] = torch.exp(preds[x, y]) / denom
        
        self.nms_loss += torch.sum(nms_tensor)

    def __call__(self, true_labels, pred_labels):
        """
        NMS loss calculation.
        """
        for i in range(true_labels.size(0)):
            self.get_nms_per_batch(true_labels[i, :], pred_labels[i, :])
        
        return self.nms_loss
