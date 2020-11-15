import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from model.utils.config import cfg
import cv2
import pdb
import random
from PIL import Image, ImageDraw, ImageFont
from model.utils.viz_hand_obj import *

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        # print(f'hand score = {score}')
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            if class_name == 'hand':
                cv2.putText(im, '%s: %.3f lr %.1f s %.1f' % (class_name, score, lr, state), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
            else:
                cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
    return im

def vis_detections_filtered_objects(im, obj_dets, hand_dets, thresh=0.8):
    """Visual debugging of detections."""
    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        for i in range(np.minimum(10, obj_dets.shape[0])):
            bbox = tuple(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            if score > thresh and i in img_obj_id:
                cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                cv2.putText(im, '%s: %.3f' % ('object', score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
        for i in range(np.minimum(10, hand_dets.shape[0])):
            bbox = tuple(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if score > thresh:
                cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                cv2.putText(im, '%s: %.3f lr %.1f s %.1f' % ('hand', score, lr, state), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
                if state > 0:
                    obj_cc, hand_cc =  calculate_center(obj_dets[img_obj_id[i],:4]), calculate_center(bbox)
                    cv2.line(im, (int(obj_cc[0]), int(obj_cc[1])), (int(hand_cc[0]), int(hand_cc[1])), (0, 204, 0))
    elif hand_dets is not None:
        im = vis_detections(im, 'hand', hand_dets, thresh)
    return im



def vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01, font_path='lib/model/utils/times_b.ttf'):

    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size 

    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            if score > thresh_obj and i in img_obj_id:
                # viz obj by PIL
                image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height, font)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)

                if state > 0: # in contact hand

                    obj_cc, hand_cc =  calculate_center(obj_dets[img_obj_id[i],:4]), calculate_center(bbox)
                    # viz line by PIL
                    if lr == 0:
                        side_idx = 0
                    elif lr == 1:
                        side_idx = 1
                    draw_line_point(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))


        

    elif hand_dets is not None:
        image = vis_detections_PIL(im, 'hand', hand_dets, thresh_hand, font_path)
        
    return image

def vis_detections_PIL(im, class_name, dets, thresh=0.8, font_path='lib/model/utils/times_b.ttf'):
    """Visual debugging of detections."""
    
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size
    
    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)
            
    return image



def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0:
            img_obj_id.append(-1)
            continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
    return img_obj_id

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)

    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta
