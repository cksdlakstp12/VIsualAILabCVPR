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

from datasets import KAISTPed, KAISTPedWS
from inference import val_epoch, save_results
from model import SSD300, MultiBoxLoss
from train_utils import *
from utils import utils
from utils.evaluation_script import evaluate

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

    # create dataloader
    weak_aug_dataset, weak_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="weak", condition="train")
    strong_aug_dataset, strong_aug_loader = create_dataloader(config, KAISTPedWS, aug_mode="strong", condition="train")
    test_loader = create_dataloader(config, KAISTPed, condition="test")

    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    
    # TODO(sohwang): should config.exp_name be updated from command line argument?
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

    # Make logger
    logger = utils.make_logger(args)

    # Epochs
    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        t_infer_result = val_epoch(t_model, weak_aug_loader, 
                                   config.test.input_size, 
                                   min_score=0.1)
        result_filename = os.path.join(jobs_dir, f'teacher_inferece_Epoch{epoch:3d}.txt')
        save_results(t_infer_result, result_filename)
        converter(args.txt_path, result_filename, args.cnvt_path)

        strong_aug_dataset.load_teacher_inference()
        s_train_loss = train_epoch(model=s_model,
                                 dataloader=strong_aug_loader,
                                 criterion=criterion,
                                 optimizer=s_optimizer,
                                 logger=logger,
                                 teachingValue=t_infer_result,
                                 **kwargs)

        s_optim_scheduler.step()

        # Save checkpoint
        utils.save_checkpoint(epoch, s_model.module, s_optimizer, s_train_loss, jobs_dir)
        
        soft_update(t_model, s_model, args.tau)

        if epoch >= 0:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')

            # High min_score setting is important to guarantee reasonable number of detections
            # Otherwise, you might see OOM in validation phase at early training epoch
            results = val_epoch(s_model, test_loader, config.test.input_size, min_score=0.1)

            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase) 


def train_epoch(model: SSD300,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                teachingValue: list,
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
        print(sup_loss)
        print(un_loss)
        if len(sup_vis_box) > 0:
            sup_vis_loss, sup_vis_cls_loss, sup_vis_loc_loss, sup_vis_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_vis_box, sup_vis_labels) 
            sup_lwir_loss, sup_lwir_cls_loss, sup_lwir_loc_loss, sup_lwir_n_positives = criterion(sup_predicted_locs, sup_predicted_scores, sup_lwir_box, sup_lwir_labels)

            sup_loss = sup_vis_loss + sup_lwir_loss
            print(sup_loss)

        if len(un_vis_box) > 0:
            un_vis_loss, un_vis_cls_loss, un_vis_loc_loss, un_vis_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_vis_box, un_vis_labels)
            un_lwir_loss, un_lwir_cls_loss, un_lwir_loc_loss, un_lwir_n_positives = criterion(un_predicted_locs, un_predicted_scores, un_lwir_box, un_lwir_labels)

            un_loss = un_vis_loss + un_lwir_loss + F.mse_loss(un_vis_cls_loss, un_lwir_cls_loss) + F.mse_loss(un_vis_loc_loss, un_lwir_loc_loss)
            print(un_loss)

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
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'num of Positive {sup_vis_n_positives} {sup_lwir_n_positives} {un_vis_n_positives} {un_lwir_n_positives}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                              sup_vis_n_positives=sup_vis_n_positives,
                                                              sup_lwir_n_positives=sup_lwir_n_positives,
                                                              un_vis_n_positives=un_vis_n_positives,
                                                              un_lwir_n_positives=un_lwir_n_positives))

    return losses_sum.avg


if __name__ == '__main__':
    main()
