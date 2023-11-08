import os
from typing import Dict
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import config
import time
import numpy as np
import logging
import math

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SSD300, SSD300_3Way, MultiBoxLoss, SceneLoss
from sklearn.cluster import KMeans
from datasets2 import KAISTPed
from train_utils import *
from utils import utils
from utils.transforms import FusionDeadZone
from utils.evaluation_script import evaluate
from vis import visualize

def detect(detections, len_anno):
    input_size = config.test.input_size
    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)
    results = dict()
    boxes = list()
    labels = list()

    det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

    for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, range(len_anno)):
        boxes_np = boxes_t.cpu().detach().numpy().reshape(-1, 4)
        scores_np = scores_t.cpu().detach().numpy().mean(axis=1).reshape(-1, 1)

        xyxy_np = boxes_np * xyxy_scaler_np
        xywh_np = xyxy_np
        xywh_np[:, 2] -= xywh_np[:, 0]
        xywh_np[:, 3] -= xywh_np[:, 1]
        
        results[image_id + 1] = np.hstack([xywh_np, scores_np])
    
    temp_box = defaultdict(list)

    length = len(results.keys())
    
    for i in range(length):
        for key, value in results.items():
            if key == i+1:
                for line in results[key]:
                    x, y, w, h, score = line
                    if float(score) >= 0.5:
                        temp_box[key-1].append([float(x), float(y), float(w), float(h)])

    for num in range(len_anno):
        temp_boxes = temp_box[num]
        vis_boxes = np.array(temp_boxes, dtype=np.float)
        lwir_boxes  = np.array(temp_boxes, dtype=np.float)

        boxes_vis = [[0, 0, 0, 0, -1]]
        boxes_lwir = [[0, 0, 0, 0, -1]]

        for i in range(len(vis_boxes)):
            bndbox = [int(i) for i in vis_boxes[i][0:4]]
            bndbox[2] = min( bndbox[2] + bndbox[0], width )
            bndbox[3] = min( bndbox[3] + bndbox[1], height )
            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            bndbox.append(1)
            boxes_vis += [bndbox]

        for i in range(len(lwir_boxes)) :
            bndbox = [int(i) for i in lwir_boxes[i][0:4]]
            bndbox[2] = min( bndbox[2] + bndbox[0], width )
            bndbox[3] = min( bndbox[3] + bndbox[1], height )
            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            bndbox.append(1)
            boxes_lwir += [bndbox]

        boxes_vis = np.array(boxes_vis, dtype=np.float64)
        boxes_lwir = np.array(boxes_lwir, dtype=np.float64)

        #boxes_vis to Tensor
        boxes_vis = torch.tensor(boxes_vis)
        boxes_lwir = torch.tensor(boxes_lwir)

        if len(boxes_vis.shape) != 1 :
            boxes_vis[1:,4] = 3
        if len(boxes_lwir.shape) != 1 :
            boxes_lwir[1:,4] = 3

        boxes_p = torch.cat((boxes_vis,boxes_lwir), dim=0)
        boxes_p = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes_p.numpy()]))))   

        boxes.append(boxes_p[:,0:4])
        labels.append(boxes_p[:,4])

    return temp_box, boxes, labels


def softTeaching_every_batch(s_model: SSD300_3Way,
                             t_model: SSD300_3Way,
                             L_dataloader: torch.utils.data.DataLoader,
                             U_dataloader: torch.utils.data.DataLoader,
                             criterion: MultiBoxLoss,
                             scene_criterion: SceneLoss,
                             optimizer: torch.optim.Optimizer,
                             logger: logging.Logger,
                             tau: float,
                             **kwargs: Dict) -> float:
    """Train the student model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    criterion: MultiBoxLoss
        Compute multibox loss for single-shot detection
    optimizer: torch.optim.Optimizer
        Pytorch optimizer(e.g. SGD, Adam, etc)
    logger: logging.Logger
        Logger instance
    kwargs: Dict
        Other parameters to control grid_clip, print_freq

    Returns
    -------
    float
        A single scalar value for averaged loss
    """

    device = next(s_model.parameters()).device
    s_model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum
    # losses_loc = utils.AverageMeter()  # loss_loc
    # losses_cls = utils.AverageMeter()  # loss_cls

    start = time.time()

    # Batches
    for (batch_idx, (wL_image_vis, wL_image_lwir, sL_image_vis, sL_image_lwir, L_boxes, L_labels, L_is_anno)), \
        (        _, (wU_image_vis, wU_image_lwir, sU_image_vis, sU_image_lwir, U_boxes, U_labels, U_is_anno)) \
            in zip(enumerate(L_dataloader), enumerate(U_dataloader)):
        
        data_time.update(time.time() - start)

        # 두 DataLoader에서 가져온 텐서들을 합칩니다.
        s_image_vis = torch.cat((sL_image_vis, sU_image_vis), 0).to(device)
        s_image_lwir = torch.cat((sL_image_lwir, sU_image_lwir), 0).to(device)

        with torch.no_grad():
            w_image_vis = torch.cat((wL_image_vis, wU_image_vis), 0).to(device)
            w_image_lwir = torch.cat((wL_image_lwir, wU_image_lwir), 0).to(device)

            # inference
            fusion_teacher_locs, fusion_teacher_scores, \
            vis_teacher_locs, vis_teacher_scores, \
            lwir_teacher_locs, lwir_teacher_scores, \
            fusion_teacher_features, vis_teacher_features, lwir_teacher_features = t_model(w_image_vis, w_image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

            # get unlabeled inference results
            fusion_teacher_locs, fusion_teacher_scores = fusion_teacher_locs[len(L_boxes):], fusion_teacher_scores[len(L_boxes):]
            # vis_teacher_locs, vis_teacher_scores = vis_teacher_locs[len(L_boxes):], vis_teacher_scores[len(L_boxes):]
            # lwir_teacher_locs, lwir_teacher_scores = lwir_teacher_locs[len(L_boxes):], lwir_teacher_scores[len(L_boxes):]

            # Detect objects in SSD output
            fusion_teacher_inference = t_model.module.detect_objects_cuda(fusion_teacher_locs, fusion_teacher_scores, min_score=0.1, max_overlap=0.425, top_k=200)
            # vis_teacher_inference = t_model.module.detect_objects_cuda(vis_teacher_locs, vis_teacher_scores, min_score=0.1, max_overlap=0.425, top_k=200)
            # lwir_teacher_inference = t_model.module.detect_objects_cuda(lwir_teacher_locs, lwir_teacher_scores, min_score=0.1, max_overlap=0.425, top_k=200)
            
            # convert coordination as xywh
            fusion_id_box_dict, fusion_pseudo_boxes, fusion_pseudo_labels = detect(fusion_teacher_inference, len(U_is_anno))
            # vis_id_box_dict, vis_pseudo_boxes, vis_pseudo_labels = detect(vis_teacher_inference, len(U_is_anno))
            # lwir_id_box_dict, lwir_pseudo_boxes, lwir_pseudo_labels = detect(lwir_teacher_inference, len(U_is_anno))

        # calc GAP weight
        GAP_weight = calc_weight_by_GAPVector_distance(fusion_teacher_features, L_boxes, fusion_id_box_dict, len(L_boxes), config.test.input_size)
        # fusion_GAP_weights, _ = calc_weight_by_GAPVector_distance_3way(fusion_teacher_features, L_boxes, fusion_id_box_dict, len(L_boxes), config.test.input_size)
        # vis_GAP_weights, is_non_zero_vis = calc_weight_by_GAPVector_distance_3way(vis_teacher_features, L_boxes, fusion_id_box_dict, len(L_boxes), config.test.input_size)
        # lwir_GAP_weights, is_non_zero_lwir = calc_weight_by_GAPVector_distance_3way(lwir_teacher_features, L_boxes, fusion_id_box_dict, len(L_boxes), config.test.input_size)
        
        # vis_GAP_weights = vis_GAP_weights if is_non_zero_vis else fusion_GAP_weights
        # lwir_GAP_weights = lwir_GAP_weights if is_non_zero_lwir else fusion_GAP_weights

        # fusion_GAP_weights = fusion_GAP_weights.to(device)
        # vis_GAP_weights = vis_GAP_weights.to(device)
        # lwir_GAP_weights = lwir_GAP_weights.to(device)

        # # GAPs = [fusion_GAP_weight, vis_GAP_weight, lwir_GAP_weight]
        # # max_GAP_index = GAPs.index(max(GAPs))
        # # if max_GAP_index == 0:
        # #     pseudo_boxes, pseudo_labels = fusion_pseudo_boxes, fusion_pseudo_labels
        # #     GAP_weight = fusion_GAP_weight
        # # elif max_GAP_index == 1:
        # #     pseudo_boxes, pseudo_labels = vis_pseudo_boxes, vis_pseudo_labels
        # #     GAP_weight = vis_GAP_weight
        # # elif max_GAP_index == 2:
        # #     pseudo_boxes, pseudo_labels = lwir_pseudo_boxes, lwir_pseudo_labels
        # #     GAP_weight = lwir_GAP_weight
        
        # # GAPs = [fusion_GAP_weight, vis_GAP_weight, lwir_GAP_weight]
        # # GAPs = [GAP if GAP > 0 else 1000 for GAP in GAPs]
        # # min_GAP_index = GAPs.index(min(GAPs))

        # Forward prop.
        fusion_student_locs, fusion_student_scores, \
        vis_student_locs, vis_student_scores, \
        lwir_student_locs, lwir_student_scores, \
        fusion_student_features, vis_student_features, lwir_student_features = s_model(s_image_vis, s_image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        # split results as sup and unsup
        fusion_sup_locs, fusion_un_locs = fusion_student_locs[:len(L_boxes)], fusion_student_locs[len(L_boxes):]
        vis_sup_locs, vis_un_locs = vis_student_locs[:len(L_boxes)], vis_student_locs[len(L_boxes):]
        lwir_sup_locs, lwir_un_locs = lwir_student_locs[:len(L_boxes)], lwir_student_locs[len(L_boxes):]
        fusion_sup_scores, fusion_un_scores = fusion_student_scores[:len(L_boxes)], fusion_student_scores[len(L_boxes):]
        vis_sup_scores, vis_un_scores = vis_student_scores[:len(L_boxes)], vis_student_scores[len(L_boxes):]
        lwir_sup_scores, lwir_un_scores = lwir_student_scores[:len(L_boxes)], lwir_student_scores[len(L_boxes):]

        # create backward tensors
        fusion_sup_predicted_locs = torch.FloatTensor([]).to(device)
        fusion_sup_predicted_scores = torch.FloatTensor([]).to(device)
        vis_sup_predicted_locs = torch.FloatTensor([]).to(device)
        vis_sup_predicted_scores = torch.FloatTensor([]).to(device)
        lwir_sup_predicted_locs = torch.FloatTensor([]).to(device)
        lwir_sup_predicted_scores = torch.FloatTensor([]).to(device)
        fusion_un_predicted_locs = torch.FloatTensor([]).to(device)
        fusion_un_predicted_scores = torch.FloatTensor([]).to(device)
        vis_un_predicted_locs = torch.FloatTensor([]).to(device)
        vis_un_predicted_scores = torch.FloatTensor([]).to(device)
        lwir_un_predicted_locs = torch.FloatTensor([]).to(device)
        lwir_un_predicted_scores = torch.FloatTensor([]).to(device)
        sup_boxes, un_boxes, sup_labels, un_labels = list(), list(), list(), list()
        # un_fusion_boxes, un_vis_boxes, un_lwir_boxes, un_fusion_labels, un_vis_labels, un_lwir_labels = list(), list(), list(), list(), list(), list()

        for floc, fscore, vloc, vscore, lloc, lscore, box, label in zip(fusion_sup_locs, fusion_sup_scores, vis_sup_locs, vis_sup_scores, 
                                                                        lwir_sup_locs, lwir_sup_scores, L_boxes, L_labels):
            fusion_sup_predicted_locs = torch.cat([fusion_sup_predicted_locs, floc.unsqueeze(0).to(device)], dim=0)
            fusion_sup_predicted_scores = torch.cat([fusion_sup_predicted_scores, fscore.unsqueeze(0).to(device)], dim=0)
            vis_sup_predicted_locs = torch.cat([vis_sup_predicted_locs, vloc.unsqueeze(0).to(device)], dim=0)
            vis_sup_predicted_scores = torch.cat([vis_sup_predicted_scores, vscore.unsqueeze(0).to(device)], dim=0)
            lwir_sup_predicted_locs = torch.cat([lwir_sup_predicted_locs, lloc.unsqueeze(0).to(device)], dim=0)
            lwir_sup_predicted_scores = torch.cat([lwir_sup_predicted_scores, lscore.unsqueeze(0).to(device)], dim=0)
            sup_boxes.append(box.to(device))
            sup_labels.append(label.to(device))
        for floc, fscore, vloc, vscore, lloc, lscore, box, label in zip(fusion_un_locs, fusion_un_scores, vis_un_locs, vis_un_scores, \
                                                                        lwir_un_locs, lwir_un_scores, fusion_pseudo_boxes, fusion_pseudo_labels):
            fusion_un_predicted_locs = torch.cat([fusion_un_predicted_locs, floc.unsqueeze(0).to(device)], dim=0)
            fusion_un_predicted_scores = torch.cat([fusion_un_predicted_scores, fscore.unsqueeze(0).to(device)], dim=0)
            vis_un_predicted_locs = torch.cat([vis_un_predicted_locs, vloc.unsqueeze(0).to(device)], dim=0)
            vis_un_predicted_scores = torch.cat([vis_un_predicted_scores, vscore.unsqueeze(0).to(device)], dim=0)
            lwir_un_predicted_locs = torch.cat([lwir_un_predicted_locs, lloc.unsqueeze(0).to(device)], dim=0)
            lwir_un_predicted_scores = torch.cat([lwir_un_predicted_scores, lscore.unsqueeze(0).to(device)], dim=0)
            un_boxes.append(box.to(device))
            un_labels.append(label.to(device))

        # calc sup loss and un loss
        fusion_sup_loss, fusion_sup_cls_loss, fusion_sup_loc_loss, fusion_sup_n_positives = criterion(fusion_sup_predicted_locs, fusion_sup_predicted_scores, sup_boxes, sup_labels) 
        # vis_sup_loss, vis_sup_cls_loss, vis_sup_loc_loss, vis_sup_n_positives = criterion(vis_sup_predicted_locs, vis_sup_predicted_scores, sup_boxes, sup_labels) 
        # lwir_sup_loss, lwir_sup_cls_loss, lwir_sup_loc_loss, lwir_sup_n_positives = criterion(lwir_sup_predicted_locs, lwir_sup_predicted_scores, sup_boxes, sup_labels) 
        # fusion_un_loss, fusion_un_cls_loss, fusion_un_loc_loss, fusion_un_n_positives = criterion(fusion_un_predicted_locs, fusion_un_predicted_scores, un_boxes, un_labels, fusion_GAP_weights)
        # vis_un_loss, vis_un_cls_loss, vis_un_loc_loss, vis_un_n_positives = criterion(vis_un_predicted_locs, vis_un_predicted_scores, un_boxes, un_labels, vis_GAP_weights)
        # lwir_un_loss, lwir_un_cls_loss, lwir_un_loc_loss, lwir_un_n_positives = criterion(lwir_un_predicted_locs, lwir_un_predicted_scores, un_boxes, un_labels, lwir_GAP_weights)
        # fusion_un_loss, fusion_un_cls_loss, fusion_un_loc_loss, fusion_un_n_positives = criterion(fusion_un_predicted_locs, fusion_un_predicted_scores, un_fusion_boxes, un_fusion_labels)
        # vis_un_loss, vis_un_cls_loss, vis_un_loc_loss, vis_un_n_positives = criterion(vis_un_predicted_locs, vis_un_predicted_scores, un_vis_boxes, un_vis_labels)
        # lwir_un_loss, lwir_un_cls_loss, lwir_un_loc_loss, lwir_un_n_positives = criterion(lwir_un_predicted_locs, lwir_un_predicted_scores, un_lwir_boxes, un_lwir_labels)
        fusion_un_loss, fusion_un_cls_loss, fusion_un_loc_loss, fusion_un_n_positives = criterion(fusion_un_predicted_locs, fusion_un_predicted_scores, un_boxes, un_labels)
        vis_un_loss, vis_un_cls_loss, vis_un_loc_loss, vis_un_n_positives = criterion(vis_un_predicted_locs, vis_un_predicted_scores, un_boxes, un_labels)
        lwir_un_loss, lwir_un_cls_loss, lwir_un_loc_loss, lwir_un_n_positives = criterion(lwir_un_predicted_locs, lwir_un_predicted_scores, un_boxes, un_labels)

        # fusion_teacher_relation_matrix = create_relation_matrix(fusion_teacher_features, len(L_boxes), device)
        # vis_teacher_relation_matrix = create_relation_matrix(vis_teacher_features, len(L_boxes), device)
        # lwir_teacher_relation_matrix = create_relation_matrix(lwir_teacher_features, len(L_boxes), device)
        fusion_student_relation_matrix = create_relation_matrix(fusion_student_features, len(L_boxes), device)
        vis_student_relation_matrix = create_relation_matrix(vis_student_features, len(L_boxes), device)
        lwir_student_relation_matrix = create_relation_matrix(lwir_student_features, len(L_boxes), device)
        
        # 1. teacher와 student의 feature 비교
        # fusion_un_KL_loss = scene_criterion(fusion_teacher_relation_matrix, fusion_student_relation_matrix)
        # vis_un_KL_loss = scene_criterion(vis_teacher_relation_matrix, vis_student_relation_matrix)
        # lwir_un_KL_loss = scene_criterion(lwir_teacher_relation_matrix, lwir_student_relation_matrix)
        # loss = fusion_sup_loss + GAP_weight * (fusion_un_loss + vis_un_loss + lwir_un_loss) + fusion_un_KL_loss + vis_un_KL_loss + lwir_un_KL_loss
        # 2. student의 feature 내에서 비교
        vis_un_KL_loss = scene_criterion(fusion_student_relation_matrix, vis_student_relation_matrix)
        lwir_un_KL_loss = scene_criterion(fusion_student_relation_matrix, lwir_student_relation_matrix)
        loss = fusion_sup_loss + GAP_weight * (fusion_un_loss + vis_un_loss + lwir_un_loss + vis_un_KL_loss + lwir_un_KL_loss)

        # Now compute the losses
        # un_vis_weight, un_lwir_weight = calc_unvis_unlwir_weight_by_loss(vis_un_loss, lwir_un_loss)
        
        # calc total loss
        # loss = fusion_sup_loss + vis_sup_loss + lwir_sup_loss + GAP_weight * (fusion_un_loss + un_vis_weight * vis_un_loss + un_lwir_weight * lwir_un_loss)
        # loss = fusion_sup_loss + vis_sup_loss + lwir_sup_loss + GAP_weight * (fusion_un_loss + vis_un_loss + lwir_un_loss)
        # loss = fusion_sup_loss + vis_sup_loss + lwir_sup_loss + (GAP_weight * fusion_un_loss)  + vis_un_loss + lwir_un_loss
        # loss = fusion_sup_loss + (GAP_weight * fusion_un_loss)  + vis_un_loss + lwir_un_loss
        # loss = fusion_sup_loss + GAP_weight * (fusion_un_loss + vis_un_loss + lwir_un_loss)
        # loss = fusion_sup_loss + (fusion_GAP_weight * fusion_un_loss + vis_GAP_weight * vis_un_loss + lwir_GAP_weight * lwir_un_loss)
        # loss = fusion_sup_loss + (fusion_un_loss + vis_un_loss + lwir_un_loss)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if kwargs.get('grad_clip', None):
            utils.clip_gradient(optimizer, kwargs['grad_clip'])

        # Update model
        optimizer.step()

        losses_sum.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # soft_update(t_model, s_model, tau)

        # Print status
        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'num of Positive '#{vis_sup_n_positives} {lwir_sup_n_positives} '
                        '{fusion_sup_n_positives} {vis_un_n_positives} '
                        '{lwir_un_n_positives} {fusion_un_n_positives}\t'.format(
                                                              batch_idx, 
                                                              len(U_dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                            #   vis_sup_n_positives=vis_sup_n_positives,
                                                            #   lwir_sup_n_positives=lwir_sup_n_positives,
                                                              fusion_sup_n_positives=fusion_sup_n_positives,
                                                              vis_un_n_positives=vis_un_n_positives,
                                                              lwir_un_n_positives=lwir_un_n_positives,
                                                              fusion_un_n_positives=fusion_un_n_positives))
    
    return losses_sum.avg


def train_epoch(model: SSD300,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                **kwargs: Dict) -> float:
    """Train the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    criterion: MultiBoxLoss
        Compute multibox loss for single-shot detection
    optimizer: torch.optim.Optimizer
        Pytorch optimizer(e.g. SGD, Adam, etc)
    logger: logging.Logger
        Logger instance
    kwargs: Dict
        Other parameters to control grid_clip, print_freq

    Returns
    -------
    float
        A single scalar value for averaged loss
    """

    device = next(model.parameters()).device
    model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum
    # losses_loc = utils.AverageMeter()  # loss_loc
    # losses_cls = utils.AverageMeter()  # loss_cls

    start = time.time()

    # Batches
    for batch_idx, (image_vis, image_lwir, vis_box, lwir_box, vis_labels, lwir_labels, _, is_anno) in enumerate(dataloader):
        data_time.update(time.time() - start)

        # Move to default device
        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        sup_vis_box = list()
        sup_lwir_box = list()
        sup_vis_labels = list()
        sup_lwir_labels = list()
        sup_predicted_locs = torch.FloatTensor([]).to(device)
        sup_predicted_scores = torch.FloatTensor([]).to(device)

        un_vis_box = list()
        un_lwir_box = list()
        un_vis_labels = list()
        un_lwir_labels = list()
        un_predicted_scores = torch.FloatTensor([]).to(device)
        un_predicted_locs = torch.FloatTensor([]).to(device)

        for anno, vb, lb, vl, ll, pl, ps in zip(is_anno, vis_box, lwir_box, vis_labels, lwir_labels, predicted_locs, predicted_scores):
            if anno:
                sup_vis_box.append(vb.to(device))
                sup_lwir_box.append(lb.to(device))
                sup_vis_labels.append(vl.to(device))
                sup_lwir_labels.append(ll.to(device))
                sup_predicted_locs = torch.cat([sup_predicted_locs, pl.unsqueeze(0).to(device)], dim=0)
                sup_predicted_scores = torch.cat([sup_predicted_scores, ps.unsqueeze(0).to(device)], dim=0)
            else:
                un_vis_box.append(vb.to(device))
                un_lwir_box.append(lb.to(device))
                un_vis_labels.append(vl.to(device))
                un_lwir_labels.append(ll.to(device))
                un_predicted_locs = torch.cat([un_predicted_locs, pl.unsqueeze(0).to(device)], dim=0)
                un_predicted_scores = torch.cat([un_predicted_scores, ps.unsqueeze(0).to(device)], dim=0)

        sup_loss, un_loss = torch.zeros(1)[0].to(device), torch.zeros(1)[0].to(device)
        sup_vis_n_positives, sup_lwir_n_positives, un_vis_n_positives, un_lwir_n_positives = 0, 0, 0, 0

        if len(sup_vis_box) > 0:
            sup_vis_loss, sup_vis_cls_loss, sup_vis_loc_loss, sup_vis_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_vis_box, sup_vis_labels) 
            sup_lwir_loss, sup_lwir_cls_loss, sup_lwir_loc_loss, sup_lwir_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_lwir_box, sup_lwir_labels)
            sup_loss = sup_vis_loss + sup_lwir_loss
        if len(un_vis_box) > 0:
            un_vis_loss, un_vis_cls_loss, un_vis_loc_loss, un_vis_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_vis_box, un_vis_labels)
            un_lwir_loss, un_lwir_cls_loss, un_lwir_loc_loss, un_lwir_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_lwir_box, un_lwir_labels)
            un_loss = un_vis_loss + un_lwir_loss + F.mse_loss(un_vis_cls_loss.float(), un_lwir_cls_loss.float()) + F.mse_loss(un_vis_loc_loss.float(), un_lwir_loc_loss.float())

        loss = sup_loss + un_loss

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # TODO(sohwang): Do we need this?
        #if np.isnan(loss.item()):
            #loss, cls_loss, loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Clip gradients, if necessary
        if kwargs.get('grad_clip', None):
            utils.clip_gradient(optimizer, kwargs['grad_clip'])

        # Update model
        optimizer.step()

        losses_sum.update(loss.item())
        # losses_loc.update(loc_loss.sum().item())
        # losses_cls.update(cls_loss.sum().item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\n'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}),\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        # 'sup_vis_loss {sup_vis_loss},\t'
                        # 'sup_vis_cls_loss {sup_vis_cls_loss},\t'
                        # 'sup_vis_loc_loss {sup_vis_loc_loss}\n'
                        # 'sup_lwir_loss {sup_lwir_loss},\t'
                        # 'sup_lwir_cls_loss {sup_lwir_cls_loss},\t'
                        # 'sup_lwir_loc_loss {sup_lwir_loc_loss}\n'
                        # 'un_vis_loss {un_vis_loss},\t'
                        # 'un_vis_cls_loss {un_vis_cls_loss},\t'
                        # 'un_vis_loc_loss {un_vis_loc_loss}\n'
                        # 'un_lwir_loss {un_lwir_loss},\t'
                        # 'un_lwir_cls_loss {un_lwir_cls_loss},\t'
                        # 'un_lwir_loc_loss {un_lwir_loc_loss}\n'
                        # 'is_anno {is_anno}\n'
                        'num of Positive {sup_vis_n_positives} {sup_lwir_n_positives} {un_vis_n_positives} {un_lwir_n_positives}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                            #   sup_vis_loss=sup_vis_loss,
                                                            #   sup_vis_cls_loss=sup_vis_cls_loss,
                                                            #   sup_vis_loc_loss=sup_vis_loc_loss,
                                                            #   sup_lwir_loss=sup_lwir_loss,
                                                            #   sup_lwir_cls_loss=sup_lwir_cls_loss,
                                                            #   sup_lwir_loc_loss=sup_lwir_loc_loss,
                                                            #   un_vis_loss=un_vis_loss,
                                                            #   un_vis_cls_loss=un_vis_cls_loss,
                                                            #   un_vis_loc_loss=un_vis_loc_loss,
                                                            #   un_lwir_loss=un_lwir_loss,
                                                            #   un_lwir_cls_loss=un_lwir_cls_loss,
                                                            #   un_lwir_loc_loss=un_lwir_loc_loss,
                                                            #   is_anno=is_anno,
                                                              sup_vis_n_positives=sup_vis_n_positives,
                                                              sup_lwir_n_positives=sup_lwir_n_positives,
                                                              un_vis_n_positives=un_vis_n_positives,
                                                              un_lwir_n_positives=un_lwir_n_positives))

    return losses_sum.avg


def val_epoch(model: SSD300_3Way, dataloader: DataLoader, dataset_type: str, input_size: Tuple, min_score: float = 0.1) -> Dict:
    """Validate the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    input_size: Tuple
        A tuple of (height, width) for input image to restore bounding box from the raw prediction
    min_score: float
        Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    model.eval()

    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    device = next(model.parameters()).device
    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')):
            if dataset_type == "KAISTPed":
                image_vis, image_lwir, boxes, labels, indices = blob
            else:
                image_vis, image_lwir, vis_boxes, lwir_boxes, vis_labels, lwir_labels, indices, _ = blob

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores, _, _, _, _, _, _, _ = model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

            # Detect objects in SSD output
            detections = model.module.detect_objects_cuda(predicted_locs, predicted_scores,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)

            det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

            for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                # TODO(sohwang): check if labels are required
                # labels_np = labels_t.cpu().numpy().reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]
                
                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])

    return results


def run_inference(model_path: str, fdz_case: str) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset with FDZ setting.

    Parameters
    ----------
    model_path: str
        Full path of pytorch model
    fdz_case: str
        Fusion dead zone case defined in utils/transforms.py:FusionDeadZone

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)

    input_size = config.test.input_size

    # Load dataloader for Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    config.test.img_transform.add(FDZ)

    args = config.args
    batch_size = config.test.batch_size * torch.cuda.device_count()
    test_dataset = KAISTPed(args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)

    results = val_epoch(model, test_loader, input_size)
    return results


def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """

    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')


if __name__ == '__main__':

    FDZ_list = FusionDeadZone._FDZ_list

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--FDZ', default='original', type=str, choices=FDZ_list,
                        help='Setting for the "Fusion Dead Zone" experiment. e.g. {}'.format(', '.join(FDZ_list)))
    parser.add_argument('--model-path', required=True, type=str,
                        help='Pretrained model for evaluation.')
    parser.add_argument('--result-dir', type=str, default='../result',
                        help='Save result directory')
    parser.add_argument('--vis', action='store_true', 
                        help='Visualizing the results')
    arguments = parser.parse_args()

    print(arguments)

    fdz_case = arguments.FDZ.lower()
    model_path = Path(arguments.model_path).stem.replace('.', '_')

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True)
    result_filename = opj(arguments.result_dir,  f'{fdz_case}_{model_path}_TEST_det')

    # Run inference
    results = run_inference(arguments.model_path, fdz_case)

    # Save results
    save_results(results, result_filename)

    # Eval results
    phase = "Multispectral"
    evaluate(config.PATH.JSON_GT_FILE, result_filename + '.txt', phase) 
    
    # Visualizing
    if arguments.vis:
        vis_dir = opj(arguments.result_dir, 'vis', model_path, fdz_case)
        os.makedirs(vis_dir, exist_ok=True)
        visualize(result_filename + '.txt', vis_dir, fdz_case)
