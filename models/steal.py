import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kornia import filters


class StealNMSLoss(nn.Module):
    def __init__(self, r=2, tau=0.1):
        super(StealNMSLoss, self).__init__()
        self.grad2d = filters.SpatialGradient()
        self.nms_loss = 0
        self.r = r
        self.tau = tau
        self.eps = 1e-7
        self.vert_k = torch.zeros((2*r-1, 2*r-1)).unsqueeze_(0)
        self.vert_k[:, :, r-1] = 1
        self.horiz_k = torch.zeros((2*r-1, 2*r-1)).unsqueeze_(0)
        self.horiz_k[:, r-1, :] = 1
        self.lead_diag_k = torch.eye(2*r-1).unsqueeze_(0)
        cnt_diag_k = np.array(np.fliplr(np.eye(2*r-1)))
        self.cnt_diag_k = torch.tensor(cnt_diag_k).unsqueeze_(0)
    
    def get_grads(self, true_labels):
        """
        Calculate the direstion of edges.
        Ref: https://github.com/nv-tlabs/STEAL/blob/master/utils/edges_nms.m
        """
        first_gradients = self.grad2d(true_labels)
        second_grad_x = self.grad2d(first_gradients[:, :, 0, :, :].squeeze_())
        second_grad_y = self.grad2d(first_gradients[:, :, 1, :, :].squeeze_())

        grad_yy = second_grad_y[:, :, 1, :, :].squeeze_()
        grad_xy = second_grad_y[:, :, 0, :, :].squeeze_()
        grad_xx = second_grad_x[:, :, 0, :, :].squeeze_()


        theta = torch.atan((grad_yy * torch.sign(-grad_xy + self.eps)) / (grad_xx + self.eps))
        angle = theta * 180 / np.pi
        angle[angle < 0] += 180
        
        return angle

    def __call__(self, true_labels, pred_labels):
        """
        NMS loss calculation.
        """        
        # Find edge directions
        grad_angles = self.get_grads(true_labels.float())
        # thresh_theta = torch.fmod((theta * (5.0 / np.pi)).round() + 5, 5)
        # print(torch.unique(thresh_theta, return_counts=True))
        true_edge_mask = (true_labels > 0).float()
        thresh_horiz = torch.mul(true_edge_mask, 
                                 (((grad_angles >= 0) & (grad_angles < 22.5)) | ((grad_angles >= 157.5) & (grad_angles < 180))).float())
        
        thresh_cnt_diag = torch.mul(true_edge_mask, 
                                    (((grad_angles >= 22.5) & (grad_angles < 67.5))).float())
        thresh_vert = torch.mul(true_edge_mask, 
                                (((grad_angles >= 67.5) & (grad_angles < 112.5))).float())
        thresh_lead_diag = torch.mul(true_edge_mask, 
                                     (((grad_angles > 112.5) & (grad_angles < 157.5))).float())
        # Create all possible direction NMS
        # print(true_labels.size())
        # print(torch.max(true_labels))
        # print(pred_labels.size())
        # print(torch.max(pred_labels))

        exp_preds = torch.exp(pred_labels.float() * self.tau)
        # print(torch.max(exp_preds))
        horiz_filter = filters.filter2D(exp_preds, self.horiz_k)
        vert_filter = filters.filter2D(exp_preds, self.vert_k)
        lead_diag_filter = filters.filter2D(exp_preds, self.lead_diag_k)
        cnt_diag_filter = filters.filter2D(exp_preds, self.cnt_diag_k)
        # print(f'4th filter: {torch.isnan(cnt_diag_filter)}')
        # print(horiz_filter.sum())
        # print(vert_filter.sum())
        # print(lead_diag_filter.sum())
        # print(cnt_diag_filter.sum())

        # Generate masked NMS
        horiz_nms = torch.clamp(torch.mul(thresh_horiz, 
                                          torch.div(exp_preds, horiz_filter+self.eps)).unsqueeze_(1),
                                self.eps, 1)
        # print(horiz_nms.sum())
        vert_nms = torch.clamp(torch.mul(thresh_vert,
                                         torch.div(exp_preds, vert_filter+self.eps)).unsqueeze_(1),
                               self.eps, 1)

        lead_diag_nms = torch.clamp(torch.mul(thresh_lead_diag,
                                              torch.div(exp_preds, lead_diag_filter+self.eps)).unsqueeze_(1),
                                    self.eps, 1)
        cnt_diag_nms = torch.clamp(torch.mul(thresh_cnt_diag,
                                             torch.div(exp_preds, cnt_diag_filter+self.eps)).unsqueeze_(1),
                                    self.eps, 1)

        all_nms = torch.clamp(torch.cat((horiz_nms, vert_nms, lead_diag_nms, cnt_diag_nms), 1), self.eps, 1)
        # print(torch.max(all_nms))
        # print(torch.min(all_nms))
        nms_loss = -torch.sum(torch.log(all_nms))
        # print(nms_loss)

        return nms_loss


class StealNMSLossOld(nn.Module):
    def __init__(self, r=2):
        super(StealNMSLossOld, self).__init__()
        self.grad2d = filters.SpatialGradient()
        self.nms_loss = 0
        self.r = r

    def get_grads(self, true_labels):
        """
        Calculate the direction of edges.
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
