from datetime import datetime
from typing import Dict
import config
import logging
import numpy as np
import os
import time

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets_batch import KAISTPed, KAISTPedWS
from inference import val_epoch, save_results
from model import SSD300, MultiBoxLoss
from train_utils import *
from utils import utils
from utils.evaluation_script import evaluate
from easydict import EasyDict as edict
from collections import defaultdict

torch.backends.cudnn.benchmark = False

# random seed fix 
utils.set_seed(seed=9)

def main():
    """Train and validate a model"""

    # TODO(sohwang): why do we need these global variables?
    # global epochs_since_improvement, start_epoch, label_map, best_loss, epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config.args
    train_conf = config.train
    checkpoint = train_conf.checkpoint
    start_epoch = train_conf.start_epoch
    epochs = train_conf.epochs
    phase = "Multispectral"

    # Initialize student model and load teacher checkpoint
    s_model, s_optimizer, s_optim_scheduler, \
    t_model, t_optimizer, t_optim_scheduler = load_SoftTeacher(config)

    # Move to default device
    s_model = s_model.to(device)
    s_model = nn.DataParallel(s_model)
    t_model = t_model.to(device)
    t_model = nn.DataParallel(t_model)
    
    criterion = MultiBoxLoss(priors_cxcy=s_model.module.priors_cxcy).to(device)


    #weak_aug_dataset, weak_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="weak", condition="train", propname = propname)
    #strong_aug_dataset, strong_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="strong", condition="train", propname = propname)
    #test_dataset, test_loader = create_dataloader(config, KAISTPed, condition="test")
    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    
    

    # TODO(sohwang): should config.exp_name be updated from command line argument?
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

    # Make logger
    logger = utils.make_logger(args)

    # create dataloader
    #weak_aug_dataset, weak_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="weak", condition="train", propname = )
    #strong_aug_dataset, strong_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="strong", condition="train", propname = )
    #test_dataset, test_loader = create_dataloader(config, KAISTPed, condition="test")

    

    # Epochs
    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(start_epoch, epochs):
        propname = os.path.join(jobs_dir, f'props_{epoch:3d}.txt')
        # create dataloader
        weak_aug_dataset, weak_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="weak", condition="train", propname = propname)
        strong_aug_dataset, strong_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="strong", condition="train", propname = propname)
        test_dataset, test_loader = create_dataloader(config, KAISTPed, condition="test")

        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        #t_infer_result = val_epoch(t_model, weak_aug_loader, 
        #                           "KAISTPedWS",
        #                           config.test.input_size, 
        #                           min_score=0.1)
        #result_filename = os.path.join(jobs_dir, f'teacher_inferece_Epoch{epoch:3d}.txt')
        #save_results(t_infer_result, result_filename)

        not_anno_names = "./imageSets/Unlabeled_90.txt"
        #converter(train_conf.teacher_img_set, result_filename, convert_name)

        #global propname
        #propname = os.path.join(jobs_dir, f'props_{epoch:3d}.txt')

        strong_aug_dataset.load_teacher_inference(not_anno_names)
        s_train_loss = train_epoch(s_model=s_model,
                                   t_model = t_model,
                                 dataloader=strong_aug_loader,
                                 criterion=criterion,
                                 optimizer=s_optimizer,
                                 logger=logger,
                                 #teachingValue=t_infer_result,
                                 **kwargs)

        s_optim_scheduler.step()

        # Save checkpoint
        utils.save_checkpoint(epoch, s_model.module, s_optimizer, s_train_loss, jobs_dir)

        if epoch >= 0:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')

            # High min_score setting is important to guarantee reasonable number of detections
            # Otherwise, you might see OOM in validation phase at early training epoch
            results = val_epoch(s_model, test_loader, "KAISTPed", config.test.input_size, min_score=0.1)

            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase) 


def train_epoch(s_model: SSD300,
                t_model : SSD300,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                #teachingValue: list,
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

    device = next(s_model.parameters()).device
    s_model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum
    # losses_loc = utils.AverageMeter()  # loss_loc
    # losses_cls = utils.AverageMeter()  # loss_cls

    start = time.time()

    # Batches
    for batch_idx, (image_vis, image_lwir, vis_box, lwir_box, vis_labels, lwir_labels, _, is_anno) in enumerate(dataloader):
        data_time.update(time.time() - start)

        #print(f"\nanno : {is_anno}\n")
        #t_model.eval()

        # Move to default device
        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)

        input_size = config.test.input_size
        height, width = input_size
        min_score = 0.1
        xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)
        results = dict()
        predicted_locs, predicted_scores = t_model(image_vis, image_lwir)

        anno_index = list()
        for index, anno in enumerate(is_anno):
            if not anno:
                anno_index.append(index)

        #print(anno_index)

        predicted_locs_non, predicted_scores_non = predicted_locs[anno_index], predicted_scores[anno_index]

        # Detect objects in SSD output
        detections = t_model.module.detect_objects(predicted_locs_non, predicted_scores_non,
                                                min_score=min_score, max_overlap=0.425, top_k=200)

        det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

        indices = [i for i in range(len(anno_index))]

        #print(indices)

        #print(det_boxes_batch, det_labels_batch, det_scores_batch, indices)

        for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
            boxes_np = boxes_t.cpu().detach().numpy().reshape(-1, 4)
            scores_np = scores_t.cpu().detach().numpy().mean(axis=1).reshape(-1, 1)

            # TODO(sohwang): check if labels are required
            # labels_np = labels_t.cpu().numpy().reshape(-1, 1)

            xyxy_np = boxes_np * xyxy_scaler_np
            xywh_np = xyxy_np
            xywh_np[:, 2] -= xywh_np[:, 0]
            xywh_np[:, 3] -= xywh_np[:, 1]
            
            results[image_id + 1] = np.hstack([xywh_np, scores_np])
        
        temp_box = defaultdict(list)

        length = len(results.keys())
        #print(f"\n\nresults boxes : {results}\n\n")

        for i in range(length):
            for key, value in results.items():
                if key == i+1:
                    #print(key)
                    for line in results[key]:
                        #print(f"key : {key}, line : {line}")
                        x, y, w, h, score = line
                        if float(score) >= 0.5:
                            temp_box[key-1].append([float(x), float(y), float(w), float(h)])
                    #print(f"\n\nkey : {key}, value : {value}\n\n")

        for num, j in enumerate(anno_index):
            #print(f"j : {j}")
            boxes = temp_box[num]
            vis_boxes = np.array(boxes, dtype=np.float)
            lwir_boxes  = np.array(boxes, dtype=np.float)

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
                ##print(f"lwir : {lwir_boxes}\n")
                name = lwir_boxes[i][0]
                bndbox = [int(i) for i in lwir_boxes[i][0:4]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float64)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float64)
            if len(boxes_vis.shape) != 1 :
                    boxes_vis[1:,4] = 3
            if len(boxes_lwir.shape) != 1 :
                boxes_lwir[1:,4] = 3

            #print(f"boxes_vis : {boxes_vis}, boxes_lwir : {boxes_lwir}")

            label_vis = boxes_vis[:,4]
            label_lwir = boxes_lwir[:,4]
            boxes_vis = boxes_vis[:,0:4]
            boxes_lwir = boxes_lwir[:,0:4]

            boxes_vis = torch.FloatTensor(boxes_vis).to(device)
            boxes_lwir = torch.FloatTensor(boxes_lwir).to(device)

            #print(f"\n\nvis_labels : {label_vis}\n\n", f"\n\nlwir_labels : {label_lwir}\n\n", f"\n\nvis_boxes : {boxes_vis}\n\n", f"\n\nlwir_boxes : {boxes_lwir}\n\n") 
            vis_box[j] = boxes_vis
            lwir_box[j] = boxes_lwir
            vis_labels[j] = label_vis
            lwir_labels[j] = label_lwir

            #print(f"j : {j}, vis_labels[i] : {vis_labels[j]}")

#                if vis_labels[i] == None:
 #                   vis_labels[i] = torch.FloatTensor([-1]).to(device)
        

        #print(f"\n\nvis_box : {vis_box}\n\n")
        #print(f"\n\nlwir_box : {lwir_box}\n\n")
        """
        for num, i in enumerate(anno_index):
            for f in results[num]:
                for line in f:
                    x, y, w, h, score = line.strip().split(",")
                    if float(score) >= 0.5:
                        vis_box[i].append([float(x), float(y), float(w), float(h)])
                        lwir_box[i].append([float(x), float(y), float(w), float(h)])
        """

        # Forward prop.
        predicted_locs, predicted_scores = s_model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

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
                vl = torch.tensor(vl).to(device)
                ll = torch.tensor(ll).to(device)
                un_vis_box.append(vb.to(device))
                un_lwir_box.append(lb.to(device))
                un_vis_labels.append(vl.to(device))
                un_lwir_labels.append(ll.to(device))
                un_predicted_locs = torch.cat([un_predicted_locs, pl.unsqueeze(0).to(device)], dim=0)
                un_predicted_scores = torch.cat([un_predicted_scores, ps.unsqueeze(0).to(device)], dim=0)
        

        sup_loss, un_loss = torch.zeros(1)[0].to(device), torch.zeros(1)[0].to(device)
        sup_vis_n_positives, sup_lwir_n_positives, un_vis_n_positives, un_lwir_n_positives = 0, 0, 0, 0

        if len(sup_vis_box) > 0 or len(sup_lwir_box) > 0:
            #print(f"sup_vis_labels : {sup_vis_labels}")
            #print(f"sup_lwir_labels : {sup_lwir_labels}")
            sup_vis_loss, sup_vis_cls_loss, sup_vis_loc_loss, sup_vis_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_vis_box, sup_vis_labels) 
            sup_lwir_loss, sup_lwir_cls_loss, sup_lwir_loc_loss, sup_lwir_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_lwir_box, sup_lwir_labels)
            sup_loss = sup_vis_loss + sup_lwir_loss
        if len(un_vis_box) > 0 or len(un_lwir_box) > 0:
            #print("un_vis_box is not None")
            #print(len(un_vis_box), len(un_lwir_box))
            #print(f"un_vis_labels : {un_vis_labels}")
            #print(f"un_lwir_labels : {un_lwir_labels}")
            un_vis_loss, un_vis_cls_loss, un_vis_loc_loss, un_vis_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_vis_box, un_vis_labels)
            un_lwir_loss, un_lwir_cls_loss, un_lwir_loc_loss, un_lwir_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_lwir_box, un_lwir_labels)
            un_loss = un_vis_loss + un_lwir_loss
            #+ F.mse_loss(un_vis_cls_loss.float(), un_lwir_cls_loss.float()) + F.mse_loss(un_vis_loc_loss.float(), un_lwir_loc_loss.float())

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

        soft_update(t_model, s_model, config.args.tau)

        # Print status
        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        #'sup_vis_cls_loss {sup_vis_cls_loss}\t'
                        #'sup_vis_loc_loss {sup_vis_loc_loss}\t'
                        #'sup_lwir_cls_loss {sup_lwir_cls_loss}\t'
                        #'sup_lwir_loc_loss {sup_lwir_loc_loss}\t'
                        #'un_vis_cls_loss {un_vis_cls_loss}\t'
                        #'un_vis_loc_loss {un_vis_loc_loss}\t'
                        #'un_lwir_cls_loss {un_lwir_cls_loss}\t'
                        #'un_lwir_loc_loss {un_lwir_loc_loss}\t'
                        #'cls mse loss {cls_mse_loss}\t'
                        #'loc mse loss {loc_mse_loss}\t'
                        #'is_anno {is_anno}\t'
                        'num of Positive {sup_vis_n_positives} {sup_lwir_n_positives} {un_vis_n_positives} {un_lwir_n_positives}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                              #sup_vis_cls_loss=sup_vis_cls_loss,
                                                              #sup_vis_loc_loss=sup_vis_loc_loss,
                                                              #sup_lwir_cls_loss=sup_lwir_cls_loss,
                                                              #sup_lwir_loc_loss=sup_lwir_loc_loss,
                                                              #un_vis_cls_loss=un_vis_cls_loss,
                                                              #un_vis_loc_loss=un_vis_loc_loss,
                                                              #un_lwir_cls_loss=un_lwir_cls_loss,
                                                              #un_lwir_loc_loss=un_lwir_loc_loss,
                                                              #cls_mse_loss=F.mse_loss(un_vis_cls_loss.float(), un_lwir_cls_loss.float()),
                                                              #loc_mse_loss=F.mse_loss(un_vis_loc_loss.float(), un_lwir_loc_loss.float()),
                                                              #is_anno=is_anno,
                                                              sup_vis_n_positives=sup_vis_n_positives,
                                                              sup_lwir_n_positives=sup_lwir_n_positives,
                                                              un_vis_n_positives=un_vis_n_positives,
                                                              un_lwir_n_positives=un_lwir_n_positives))
    
    

    return losses_sum.avg


if __name__ == '__main__':
    main()
