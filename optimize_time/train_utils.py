from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np

from model import SSD300

def initialize_state(n_classes, train_conf):
    model = SSD300(n_classes=n_classes)
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state(args.n_classes, train_conf)
    else:
        model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint(train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state(config, student_checkpoint)
    teacher_state = load_state(config, teacher_checkpoint)

    return *student_state, *teacher_state

def create_dataloader(config, dataset_class, sample_mode = None, **kwargs):
    if kwargs["condition"] == "train":
        if sample_mode == "two":
            sample = "Labeled"
            dataset = dataset_class(config.args, sample = sample,**kwargs)
            L_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            sample = "Unlabeled"
            dataset = dataset_class(config.args, sample = sample, **kwargs)
            U_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            return dataset, L_loader, U_loader
        else: 
            dataset = dataset_class(config.args, **kwargs)
            loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
    else:
        dataset = dataset_class(config.args, **kwargs)
        test_batch_size = config.args["test"].eval_batch_size * torch.cuda.device_count()
        loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    return dataset, loader

def converter(originpath, changepath, wantname):
    # Loading the 90percents.txt file and creating a dictionary where keys are the index
    with open("./imageSets/" + originpath, 'r') as f:
        data_90 = {idx+1: line.strip() for idx, line in enumerate(f)}

    # Loading the test2.txt file
    with open(changepath, 'r') as f:
        data_test2 = f.readlines()

    # Replacing the first number of each line in test2.txt with corresponding line in 90percents.txt
    data_test2_new = []
    for line in data_test2:
        items = line.split(',')
        index = int(items[0])
        items[0] = data_90[index]
        data_test2_new.append(','.join(items))

    # Writing the new data into a new file
    with open(wantname, 'w') as f:
        for line in data_test2_new:
            f.write(line)

def soft_update(teacher_model, student_model, tau):
    """
    Soft update model parameters.
    θ_teacher = τ*θ_student + (1 - τ)*θ_teacher

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    :param tau: interpolation parameter (0.001 in your case)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(tau*student_param.data + (1.0-tau)*teacher_param.data)

class EMAScheduler():
    def __init__(self, config):
        self.use_scheduler = config.ema.use_scheduler
        self.start_tau = config.ema.tau
        self.scheduling_start_epoch = config.ema.scheduling_start_epoch
        self.max_tau = config.ema.max_tau
        self.min_tau = config.ema.min_tau
        self.last_tau = config.ema.tau

    @staticmethod
    def calc_tau(epoch, tau):
        tau = tau
        return tau
    
    def get_tau(self, epoch):
        if not self.use_scheduler:
            return self.start_tau
        else:
            new_tau = EMAScheduler.calc_tau(epoch, self.last_tau)
            return new_tau
            """
            if new_tau > self.max_tau: 
                return self.max_tau
            elif new_tau < self.min_tau: 
                return self.min_tau
            else:
                self.last_tau = new_tau 
                return new_tau
            """
