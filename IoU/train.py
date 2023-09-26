from datetime import datetime
import config
import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from datasets_2 import KAISTPed, KAISTPedWSEpoch, KAISTPedWSIter
from run_epoch2 import val_epoch, save_results, train_epoch, softTeaching_every_iter
from model import MultiBoxLoss
from train_utils import *
from utils import utils
from utils.evaluation_script import evaluate

torch.backends.cudnn.benchmark = False

# random seed fix 
utils.set_seed(seed=42)

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


    
    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    
    # TODO(sohwang): should config.exp_name be updated from command line argument?
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

       # Initialize student model and load teacher checkpoint
    if config.soft_teacher.student_checkpoint == config.soft_teacher.teacher_checkpoint:
        s_model, s_optimizer, s_optim_scheduler, start_epoch, s_train_loss,\
        t_model, t_optimizer, t_optim_scheduler, _, _ = load_SoftTeacher(config)
        start_epoch = 0
    else:
        s_model, s_optimizer, s_optim_scheduler,start_epoch, s_train_loss,\
        t_model, t_optimizer, t_optim_scheduler, _, _ = load_SoftTeacher(config)
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        s_train_loss = checkpoint['loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, s_train_loss))
        s_model = checkpoint['model']
        s_optimizer = checkpoint['optimizer']
        s_optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(s_optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)

    # Move to default device
    s_model = s_model.to(device)
    s_model = nn.DataParallel(s_model)
    t_model = t_model.to(device)
    t_model = nn.DataParallel(t_model)

    criterion = MultiBoxLoss(priors_cxcy=s_model.module.priors_cxcy).to(device)

    # create dataloader
    weak_aug_dataset, weak_aug_loader = create_dataloader(config, KAISTPedWSEpoch, aug_mode="weak", condition="train")
    strong_aug_dataset, strong_aug_loader = create_dataloader(config, KAISTPedWSEpoch, aug_mode="strong", condition="train")
    test_dataset, test_loader = create_dataloader(config, KAISTPed, condition="test")
    if train_conf.soft_update_mode == "iter":
        strong_aug_dataset, L_aug_loader, U_aug_loader = create_dataloader(config, KAISTPedWSIter, sample_mode = "two", condition="train")

    # EMA Scheduler
    ema_scheduler = EMAScheduler(config)

    # Make logger
    logger = utils.make_logger(args)

    #initialize min_MR_all
    min_MR_all = float('inf') 

    # Epochs
    cnt = 0

    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(start_epoch, epochs):
        
        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        s_train_loss = None
        if train_conf.soft_update_mode == "epoch":
            weak_aug_dataset.set_propFilePath_by_epoch(epoch)
            strong_aug_dataset.set_propFilePath_by_epoch(epoch)
            t_infer_result = val_epoch(t_model, weak_aug_loader, 
                                        "KAISTPedWSEpoch",
                                        config.test.input_size, 
                                        min_score=0.1)
            
            strong_aug_dataset.parse_teacher_inference(t_infer_result)
            strong_aug_dataset.load_propFile()
            s_train_loss = train_epoch(model=s_model,
                                        dataloader=strong_aug_loader,
                                        criterion=criterion,
                                        optimizer=s_optimizer,
                                        logger=logger,
                                        teachingValue=t_infer_result,
                                        **kwargs)
            s_optim_scheduler.step()
            
            soft_update(t_model, s_model, ema_scheduler.get_tau(epoch))
        
        elif train_conf.soft_update_mode == "iter":
            
            s_train_loss = softTeaching_every_iter(s_model=s_model,
                                    t_model=t_model,
                                    L_dataloader=L_aug_loader,
                                    U_dataloader=U_aug_loader,
                                    criterion=criterion,
                                    optimizer=s_optimizer,
                                    logger=logger,
                                    epoch=epoch,
                                    tau=ema_scheduler.get_tau(epoch),
                                    **kwargs)
            s_optim_scheduler.step()
        
        else:
            raise Exception("You should choise train mode between batch or iter")

        # Save checkpoint
        assert s_train_loss is not None, "s_train_loss should not be None"
        utils.save_checkpoint(epoch, s_model.module, s_optimizer, s_train_loss, jobs_dir)
    
        if epoch >= 0:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')

            # High min_score setting is important to guarantee reasonable number of detections
            # Otherwise, you might see OOM in validation phase at early training epoch
            results = val_epoch(s_model, test_loader, "KAISTPed", config.test.input_size, min_score=0.1)

            save_results(results, result_filename)
            
            eval_result = evaluate(config.PATH.JSON_GT_FILE, result_filename, phase)
            
            """
            MR_all = eval_result['all'].summarize(0)

            if MR_all < min_MR_all:
                min_MR_all = MR_all
                # 최소 MR_all을 가진 모델의 가중치 저장
                torch.save(s_model.state_dict(), 'best_model_weights.pth')
                cnt = 0
            else:
                cnt +=1
            

        if epoch >= 10 and cnt == 3:
                cnt = 0
                s_model.load_state_dict(torch.load('best_model_weights.pth'))
            """

if __name__ == '__main__':
    main()
