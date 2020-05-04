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
        theta = torch.atan(
            (grad_yy * torch.sign(-grad_xy + self.eps)) / (grad_xx + self.eps))
        angle = theta * 180 / np.pi
        angle[angle < 0] += 180

        return angle

    def __call__(self, true_labels, pred_labels):
        """
        NMS loss calculation.
        """
        # Find edge directions
        grad_angles = self.get_grads(true_labels.float())

        true_edge_mask = (true_labels > 0).float()
        thresh_horiz = \
            torch.mul(true_edge_mask,
                      (((grad_angles >= 0) & (grad_angles < 22.5)) |
                       ((grad_angles >= 157.5) & (grad_angles < 180))).float())

        thresh_cnt_diag = \
            torch.mul(true_edge_mask,
                      (((grad_angles >= 22.5) & (grad_angles < 67.5))).float())

        thresh_vert = \
            torch.mul(true_edge_mask,
                      (((grad_angles >= 67.5) & (grad_angles < 112.5))).float())

        thresh_lead_diag = \
            torch.mul(true_edge_mask,
                      (((grad_angles > 112.5) & (grad_angles < 157.5))).float())

        # Create all possible direction NMS
        exp_preds = torch.exp(pred_labels.float() * self.tau)
        horiz_filter = filters.filter2D(exp_preds, self.horiz_k)
        vert_filter = filters.filter2D(exp_preds, self.vert_k)
        lead_diag_filter = filters.filter2D(exp_preds, self.lead_diag_k)
        cnt_diag_filter = filters.filter2D(exp_preds, self.cnt_diag_k)

        # Generate masked NMS
        horiz_nms = \
            torch.mul(thresh_horiz,
                      torch.log(torch.clamp(torch.div(exp_preds,
                                                      horiz_filter
                                                      + self.eps),
                                            self.eps, 1))).unsqueeze_(1)

        vert_nms = \
            torch.mul(thresh_vert,
                      torch.log(torch.clamp(torch.div(exp_preds,
                                                      vert_filter
                                                      + self.eps),
                                            self.eps, 1))).unsqueeze_(1)

        lead_diag_nms = \
            torch.mul(thresh_lead_diag,
                      torch.log(torch.clamp(torch.div(exp_preds,
                                                      lead_diag_filter
                                                      + self.eps),
                                            self.eps, 1))).unsqueeze_(1)
        cnt_diag_nms = \
            torch.mul(thresh_cnt_diag,
                      torch.log(torch.clamp(torch.div(exp_preds,
                                                      cnt_diag_filter
                                                      + self.eps),
                                            self.eps, 1))).unsqueeze_(1)

        all_nms = torch.cat(
            (horiz_nms, vert_nms, lead_diag_nms, cnt_diag_nms), 1)

        nms_loss = -torch.mean(torch.sum(all_nms, (1, 2, 3, 4)))

        return nms_loss
