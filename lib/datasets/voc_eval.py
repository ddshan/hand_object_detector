# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os, sys, pdb, math
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#sys.path.append('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/lib/model/utils')
# from lib.datasets.viz_hand_obj_debug import *

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]

    obj_struct['handstate'] = 0 if obj.find('contactstate').text is None else int(obj.find('contactstate').text)
    obj_struct['leftright'] = 0 if obj.find('handside').text is None else int(obj.find('handside').text)


    obj_struct['objxmin'] = None if obj.find('objxmin').text in [ None, 'None'] else float(obj.find('objxmin').text)
    obj_struct['objymin'] = None if obj.find('objymin').text in [ None, 'None'] else float(obj.find('objymin').text)
    obj_struct['objxmax'] = None if obj.find('objxmax').text in [ None, 'None'] else float(obj.find('objxmax').text)
    obj_struct['objymax'] = None if obj.find('objymax').text in [ None, 'None'] else float(obj.find('objymax').text)

    if obj_struct['objxmin'] is not None and obj_struct['objymin'] is not None and obj_struct['objxmax'] is not None and obj_struct['objymax'] is not None:
      obj_struct['objectbbox'] = [obj_struct['objxmin'], obj_struct['objymin'], obj_struct['objxmax'], obj_struct['objymax']]
    else:
      obj_struct['objectbbox'] = None



    objects.append(obj_struct) 

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap




'''
@description: raw evaluation for fasterrcnn
'''
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  print(f'\n\n thd = {ovthresh}\n\n')

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'].lower() == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:2+4]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)


  return rec, prec, ap




'''
@description: eval hands
@compare: hand_bbox, object_bbox, state, side
TODO:
(1) prepare gt and det of hand --> (image_path, score, handbbox, state, side, objectbbox)
'''
def voc_eval_hand(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             constraint=''
             ):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  [constraint]：[handstate, handside, objectbbox]
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # ------------------------------------------
  # cachefile = test.txt_annots.pkl
  # imagesetfile = test.txt
  # annopath.format(imagename): filename in Annotations, eg. xxxx.xml
  # detpath = comp4_det_test_{classname}.txt: path, score, bbox, state. vector, side, xxx


  print(f'\n\n*** current overlap thd = {ovthresh}')
  print(f'*** current constraint = {constraint}')
  assert constraint in ['', 'handstate', 'handside', 'objectbbox', 'all']

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile) # cachefile = test.txt_annots.pkl
  # read list of images
  with open(imagesetfile, 'r') as f: 
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'].lower() == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    handstate = np.array([x['handstate'] for x in R]).astype(np.int)
    leftright = np.array([x['leftright'] for x in R]).astype(np.int)
    objectbbox = np.array([x['objectbbox'] for x in R])
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'handstate': handstate,
                             'leftright':leftright,
                             'objectbbox':objectbbox,
                             'det': det}


  # ======== hand all det ======== #
  BB_det_object, image_ids_object, detfile_object = extract_BB(detpath, extract_class='targetobject')
  BB_det_hand, image_ids_hand, detfile_hand = extract_BB(detpath, extract_class='hand')
  
  ho_dict = make_hand_object_dict(BB_det_object, BB_det_hand, image_ids_object, image_ids_hand)
  hand_det_res = gen_det_result(ho_dict) # [image_path, score, handbbox, state, vector, side, objectbbox, objectbbox_score]

  # print(f'det len: obj-bbox={len(BB_det_object)}, obj_image={len(image_ids_object)}, {detfile_object}')
  # print(f'det len: hand-bbox={len(BB_det_hand)}, hand_image={len(image_ids_hand)}, {detfile_hand}')
  # print('\n\n\n\n')
  # pdb.set_trace() 
  # for key, val in ho_dict.items():
  #   print(key, val, '\n\n\n')
  # ============================= #

  image_ids = [x[0] for x in hand_det_res]
  confidence = np.array([x[1] for x in hand_det_res])
  BB_det = np.array([[float(z) for z in x[2]] for x in hand_det_res])
  handstate_det = np.array([int(x[3]) for x in hand_det_res]) # get handstate
  leftright_det = np.array([int(x[5]) for x in hand_det_res]) # get leftright
  objectbbox_det = [ x[6] for x in hand_det_res]
  objectbbox_score_det = [ x[7] for x in hand_det_res]
  

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB_det.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    # ======== det ======== #
    image_ids = [image_ids[x] for x in sorted_ind]
    confidence_det = [confidence[x] for x in sorted_ind]
    BB_det = BB_det[sorted_ind, :]
    handstate_det = handstate_det[sorted_ind]
    leftright_det = leftright_det[sorted_ind]
    objectbbox_det = [objectbbox_det[x] for x in sorted_ind] #objectbbox_det[sorted_ind, :]
    objectbbox_score_det = [objectbbox_score_det[x] for x in sorted_ind] #objectbbox_det[sorted_ind, :]
    # ============================= #
    

    # go down dets and mark TPs and FPs
    for d in range(nd):

      # det
      image_id_det =  image_ids[d]
      score_det = confidence_det[d]
      bb_det = BB_det[d, :].astype(float)
      hstate_det = handstate_det[d].astype(int)
      hside_det = leftright_det[d].astype(int)
      objbbox_det = objectbbox_det[d]#.astype(float)
      objbbox_score_det = objectbbox_score_det[d]
      #print(f'debug hand-obj: {bb_det} {objbbox_det}')

      # gt
      ovmax = -np.inf
      R = class_recs[image_ids[d]]
      BBGT = R['bbox'].astype(float)
      hstate_GT = R['handstate'].astype(int)
      hside_GT = R['leftright'].astype(int)
      objbbox_GT = R['objectbbox']#.astype(float)

      

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb_det[0])
        iymin = np.maximum(BBGT[:, 1], bb_det[1])
        ixmax = np.minimum(BBGT[:, 2], bb_det[2])
        iymax = np.minimum(BBGT[:, 3], bb_det[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb_det[2] - bb_det[0] + 1.) * (bb_det[3] - bb_det[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)


        # plot
        if 0:
          det_info = [bb_det, hstate_det, hside_det, objbbox_det, score_det, objbbox_score_det]
          gt_info = [BBGT[jmax], hstate_GT[jmax], hside_GT[jmax], objbbox_GT[jmax]]
          debug_det_gt(image_ids[d], det_info, gt_info, d)


      if constraint == '':
        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax]: # add diff constraints here for diff eval
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.

      

      elif constraint == 'handstate':
        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax] and hstate_GT[jmax] == hstate_det: # add diff constraints here for diff eval
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.


          
      elif constraint == 'handside':
        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax] and hside_GT[jmax] == hside_det: # add diff constraints here for diff eval
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.

          

      elif constraint == 'objectbbox':
        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax] and val_objectbbox(objbbox_GT[jmax], objbbox_det, image_ids[d]): # add diff constraints here for diff eval
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.
          
      
      elif constraint == 'all':
        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax] and hstate_GT[jmax] == hstate_det and hside_GT[jmax] == hside_det and val_objectbbox(objbbox_GT[jmax], objbbox_det, image_ids[d]): # add diff constraints here for diff eval
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.



  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)


  return rec, prec, ap



# ======== debug ======== #
def debug_det_gt(image_name, det_info, gt_info, d):
  
  os.makedirs('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/images/debug', exist_ok=True)
  # det_info = [bb_det, hstate_det, hside_det, objbbox_det, score_det， objbbox_score_det]
  # gt_info = [BBGT[jmax], hstate_GT[jmax], hside_GT[jmax], objbbox_GT[jmax]]

  genre, vid_folder = image_name.split('_', 1)[0], image_name.split('_', 1)[1][:13]
  genre_name = f'{genre}_videos'
  image_path = os.path.join('/y/jiaqig/hand_cache', genre_name, vid_folder, image_name+'.jpg')
  image = Image.open(image_path).convert("RGBA")


  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/lib/model/utils/times_b.ttf', size=20)
  width, height = image.size 

  # ======== plot det ======== #
  
  hand_bbox_det = list(det_info[0])
  hand_bbox_det = list(int(np.round(x)) for x in hand_bbox_det)
  image = draw_hand_mask(image, draw, 0, hand_bbox_det, det_info[4], det_info[2], det_info[1], width, height, font)

  if det_info[3] is not None:
    object_bbox_det = list(det_info[3])
    object_bbox_det = list(int(np.round(x)) for x in object_bbox_det)
    image = draw_obj_mask(image, draw, 0, object_bbox_det, det_info[5], width, height, font)

    if det_info[1] > 0 : # in contact hand

      obj_cc, hand_cc =  calculate_center_PIL(hand_bbox_det), calculate_center_PIL(object_bbox_det)
      draw_line_point(draw, 0, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))


  # ======== plot gt ======== #

  hand_bbox_gt = list(gt_info[0])
  hand_bbox_gt = list(int(np.round(x)) for x in hand_bbox_gt)
  image = draw_hand_mask(image, draw, 1, hand_bbox_gt, 1.0, gt_info[2], gt_info[1], width, height, font)

  if gt_info[3] is not None:
    object_bbox_gt = list(gt_info[3])
    object_bbox_gt = list(int(np.round(x)) for x in object_bbox_gt)
    image = draw_obj_mask(image, draw, 1, object_bbox_gt, 1.0, width, height, font)
  
    if gt_info[1] > 0: # in contact hand

        obj_cc, hand_cc =  calculate_center_PIL(hand_bbox_gt), calculate_center_PIL(object_bbox_gt)
        draw_line_point(draw, 1, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))

  
  # ======== save ======== #
  
  save_name = image_name + f'_draw_{d:04d}.png'
  image.save(os.path.join('/y/dandans/Hand_Object_Detection/faster-rcnn.pytorch/images/debug', save_name))






# ======== auxiluary functions ======== #
def val_objectbbox(objbbox_GT, objbbox_det, imagepath, threshold=0.5):
  if objbbox_GT is None and objbbox_det is None:
    #print('None - None')
    return True
  elif objbbox_GT is not None and objbbox_det is not None:
    if get_iou(objbbox_GT, objbbox_det) > threshold:
      #print('Yes', get_iou(objbbox_GT, objbbox_det), objbbox_GT, objbbox_det, imagepath)
      return True
    #else:
      #print('No', get_iou(objbbox_GT, objbbox_det), objbbox_GT, objbbox_det, imagepath)
      
  else:
    #print(f'None - Float')
    False
    
  

def get_iou(bb1, bb2):


    assert(bb1[0] <= bb1[2] and bb1[1] <= bb1[3] and bb2[0] <= bb2[2] and bb2[1] <= bb2[3]), print(bb1, bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def extract_BB(detpath, extract_class):
  '''
  @description
  ---> hand：
  image_ids item = image_path
  BB item =[score(0), bbox(1:1+4), state(5), vector(6:6+3), side(9)]
  --> object:
  image_ids item = image_path
  BB item = [score(0), bbox(1,1+4)]
  '''
  # read dets
  detfile = detpath.format(extract_class)
  with open(detfile, 'r') as f:
    lines = f.readlines()
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  BB = np.array([[float(z) for z in x[1:]] for x in splitlines])

  #print(f'in-function, det len: {extract_class}-bbox={len(BB)}, {extract_class}_image={len(image_ids)}, {detfile}')
  return BB, image_ids, detfile

def make_hand_object_dict(BB_o, BB_h, image_o, image_h):
  ho_dict = {}
  for bb_h, id_h in zip(BB_h, image_h):
    if id_h in ho_dict:
      ho_dict[id_h]['hands'].append(bb_h)
    else:
      ho_dict[id_h] = {'hands': [bb_h], 'objects': []}

  for bb_o, id_o in zip(BB_o, image_o):
    if id_o in ho_dict:
      ho_dict[id_o]['objects'].append(bb_o)
    else:
      ho_dict[id_o] = {'hands': [], 'objects': [bb_o]}
  return ho_dict

def calculate_center(bb):
  return [(bb[1] + bb[3])/2, (bb[2] + bb[4])/2]


'''
@description: 
[image_path, score, handbbox, state, vector, side, objectbbox]
'''
def gen_det_result(ho_dict):

  # take all results
  hand_det_res = []

  for key, info in ho_dict.items():
    object_cc_list = []
    object_bb_list = []
    object_score_list = []

    for j, object_info in enumerate(info['objects']):
      object_bbox = [object_info[1], object_info[2], object_info[3], object_info[4]]
      object_cc_list.append(calculate_center(object_info)) # is it wrong???
      object_bb_list.append(object_bbox)
      object_score_list.append(float(object_info[0]))
    object_cc_list = np.array(object_cc_list)

    for i, hand_info in enumerate(info['hands']):
      hand_path = key
      hand_score = hand_info[0]
      hand_bbox = hand_info[1:5]
      hand_state = hand_info[5]
      hand_vector = hand_info[6:9]
      hand_side = hand_info[9] 
      
      if hand_state <= 0 or len(object_cc_list) == 0 :
        to_add = [hand_path, hand_score, hand_bbox, hand_state, hand_vector, hand_side, None, None]
        hand_det_res.append(to_add)
      else:
        hand_cc = np.array(calculate_center(hand_info))
        point_cc = np.array([(hand_cc[0]+hand_info[6]*10000*hand_info[7]), (hand_cc[1]+hand_info[6]*10000*hand_info[8])])
        dist = np.sum( (object_cc_list - point_cc)**2 , axis=1)

        dist_min = np.argmin(dist)
        # get object bbox
        target_object_score = object_score_list[dist_min]
        #
        target_object_bbox = object_bb_list[dist_min]
        to_add = [hand_path, hand_score, hand_bbox, hand_state, hand_vector, hand_side, target_object_bbox, target_object_score]
        hand_det_res.append(to_add)
        
        
  return hand_det_res