from torch.utils.data import DataLoader
import torch
import torch.nn as nn

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
    return model, optimizer, optim_scheduler

def load_state_from_checkpoint(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler

def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler = initialize_state(args.n_classes, train_conf)
    else:
        model, optimizer, optim_scheduler = load_state_from_checkpoint(train_conf, checkpoint)

    return (model, optimizer, optim_scheduler)

def load_SoftTeacher(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state(config, student_checkpoint)
    teacher_state = load_state(config, teacher_checkpoint)

    return *student_state, *teacher_state

def create_dataloader(config, dataset_class, **kwargs):
    dataset = dataset_class(config.args, **kwargs)
    if kwargs["condition"] == "test":
        test_batch_size = config.args["test"].eval_batch_size * torch.cuda.device_count()
        loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    else:
        loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    return loader
