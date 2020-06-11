import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import pickle
import datetime


class extension_layer(nn.Module):
    def __init__(self):
        super(extension_layer, self).__init__()
        self.init_layers_weights()


    def forward(self, input, input_padded, roi_labels, box_info):
        if (len(input.shape)) == 2:
            input = input.unsqueeze(0)

        if (len(input_padded.shape)) == 2:
            input_padded = input_padded.unsqueeze(0)
        
        loss_list = [self.hand_contactstate_part(input_padded, roi_labels, box_info), \
                    self.hand_dxdymagnitude_part(input_padded, roi_labels, box_info),\
                    self.hand_handside_part(input, roi_labels, box_info)]

        return loss_list

    def init_layers_weights(self):

        self.hand_contact_state_layer = nn.Sequential(nn.Linear(2048, 32), \
            nn.ReLU(), \
            nn.Dropout(p=0.5),\
            nn.Linear(32, 5))
        self.hand_dydx_layer = torch.nn.Linear(2048, 3)
        self.hand_lr_layer = torch.nn.Linear(2048, 1)
        
        
        self.hand_contactstate_loss = nn.CrossEntropyLoss() 
        self.hand_dxdymagnitude_loss = nn.MSELoss()
        self.hand_handside_loss = nn.BCEWithLogitsLoss() 
        
        #
        self._init_weights()



    def hand_contactstate_part(self, input, roi_labels, box_info):
        contactstate_pred = self.hand_contact_state_layer(input)
        contactstate_loss = 0

        if self.training:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 0] # contactstate label
                index = roi_labels[i]==2 # if class is hand
                if index.sum() > 0:
                    contactstate_loss_sub = 0.1 * self.hand_contactstate_loss(contactstate_pred[i][index], gt_labels[index].long())
                
                    if not contactstate_loss:
                        contactstate_loss = contactstate_loss_sub
                    else:
                        contactstate_loss += contactstate_loss_sub

        return contactstate_pred, contactstate_loss



    def hand_dxdymagnitude_part(self, input, roi_labels, box_info):
        dxdymagnitude_pred = self.hand_dydx_layer(input)

        dxdymagnitude_pred_sub = 0.1 * F.normalize(dxdymagnitude_pred[:,:,1:], p=2, dim=2)

        dxdymagnitude_pred_norm = torch.cat([dxdymagnitude_pred[:,:,0].unsqueeze(-1), dxdymagnitude_pred_sub], dim=2)
        dxdymagnitude_loss = 0

        if self.training:
        # if 1:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 2:5] # [magnitude, dx, dy] label
                index = box_info[i,:,0] > 0 # in-contact
                
                if index.sum() > 0:
                    dxdymagnitude_loss_sub = 0.1 * self.hand_dxdymagnitude_loss(dxdymagnitude_pred_norm[i][index], gt_labels[index])
                    
                    if not dxdymagnitude_loss:
                        dxdymagnitude_loss = dxdymagnitude_loss_sub
                    else:
                        dxdymagnitude_loss += dxdymagnitude_loss_sub

        

        return dxdymagnitude_pred_norm, dxdymagnitude_loss



    def hand_handside_part(self, input, roi_labels, box_info):
        handside_pred = self.hand_lr_layer(input)
        handside_loss = 0

        if self.training:
            for i in range(input.size(0)):
                gt_labels = box_info[i, :, 1] # hand side label
                index = roi_labels[i]==2 # if class is hand
                if index.sum() > 0:
                    handside_loss_sub = 0.1 * self.hand_handside_loss(handside_pred[i][index], gt_labels[index].unsqueeze(-1))
                
                    if not handside_loss:
                        handside_loss = handside_loss_sub
                    else:
                        handside_loss += handside_loss_sub

        return handside_pred, handside_loss



    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) 
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.hand_contact_state_layer[0], 0, 0.01)
        normal_init(self.hand_contact_state_layer[3], 0, 0.01)
        normal_init(self.hand_dydx_layer, 0, 0.01)
        normal_init(self.hand_lr_layer, 0, 0.01)
