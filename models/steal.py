import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kornia.filters import filter2D, SpatialGradient


class StealNMSLoss(nn.Module):
    def __init__(self, r=3, tau=0.1):
        super(StealNMSLoss, self).__init__()
        self.grad2d = SpatialGradient()
        self.nms_loss = 0
        self.r = r
        self.tau = tau
        self.eps = 1e-7
        self.filter_dict = get_filter_dict(r)

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

    def forward(self, pred_labels, true_labels):
        """
        NMS loss calculation.
        """
        # Find edge directions
        angles = self.get_grads(true_labels.float())

        # Create all possible direction NMS and mask them.
        exp_preds = torch.exp(pred_labels/self.tau)

        return - torch.mean(get_all_nms(exp_preds, self.filter_dict,
                                        angles))


def get_mask_from_section(angles, section='horizontal'):
    if section == 'horizontal':
        return (get_mask_from_angle(angles, 0.0, 22.5) |
                get_mask_from_angle(angles, 157.5, 180)).float()
    elif section == 'cnt_diag':
        return get_mask_from_angle(angles, 22.5, 67.5).float()
    elif section == 'lead_diag':
        return get_mask_from_angle(angles, 112.5, 157.5).float()
    elif section == 'vertical':
        return get_mask_from_angle(angles, 67.5, 112.5).float()


def get_mask_from_angle(angles, start, end):
    return (angles >= start) & (angles < end)


def get_normalized_responses(exp_preds, filter_dict,
                             section='horizontal', eps=1e-6):

    sum_boundary_responses = filter2D(exp_preds, filter_dict[section]) + eps
    norm = torch.div(exp_preds, sum_boundary_responses)

    return torch.clamp(norm, eps, 1)


def get_nms_from_section(exp_preds, filter_dict, angles, section='horizontal'):
    norm = get_normalized_responses(exp_preds, filter_dict,
                                    section='horizontal',)
    mask = get_mask_from_section(angles, section)
    return (torch.log(norm) * mask).unsqueeze_(1)


def get_all_nms(exp_preds, filter_dict, angles):
    horiz_nms = get_nms_from_section(exp_preds, filter_dict,
                                     angles, section='horizontal')
    vert_nms = get_nms_from_section(exp_preds, filter_dict,
                                    angles, section='cnt_diag')
    lead_diag_nms = get_nms_from_section(exp_preds, filter_dict,
                                         angles, section='lead_diag')
    cnt_diag_nms = get_nms_from_section(exp_preds, filter_dict,
                                        angles, section='vertical')

    return torch.cat((horiz_nms, vert_nms, lead_diag_nms, cnt_diag_nms), 1)


def get_filter_dict(r=2):
    # can be 1D fiters?
    filter_dict = {}
    horiz = torch.zeros((2 * r - 1, 2 * r - 1)).unsqueeze_(0)
    horiz[:, :, r-1] = 1
    filter_dict['horizontal'] = horiz
    vert = torch.zeros((2 * r - 1, 2 * r - 1)).unsqueeze_(0)
    vert[:, r-1, :] = 1
    filter_dict['vertical'] = vert
    filter_dict['cnt_diag'] = torch.eye(2*r-1).unsqueeze_(0)
    lead_diag = np.array(np.fliplr(np.eye(2*r-1)))
    filter_dict['lead_diag'] = torch.tensor(lead_diag).unsqueeze_(0)
    return filter_dict
