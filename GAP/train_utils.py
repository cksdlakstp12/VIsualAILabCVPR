from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle
from joblib import load

from model import SSD300, SSD300ReturnFeatures

def initialize_state(n_classes, train_conf):
    if train_conf.return_feature:
        model = SSD300ReturnFeatures(n_classes=n_classes)
    else:
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
    optim_scheduler = None
    if optimizer is not None:
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

def load_KMeans_model(model_path):
    loaded_kmeans = load(model_path)
    return loaded_kmeans

def load_mean_std_dict(dict_path):
    with open(dict_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def translate_coordinate(box, feature_w, feature_h, ori_w, ori_h):
    x, y, w, h = box
    new_x = int(x / ori_w * feature_w)
    new_y = int(y / ori_h * feature_h)
    new_w = int(w / ori_w * feature_w)
    new_h = int(h / ori_h * feature_h)
    return new_x, new_y, new_w, new_h

def split_features(features, len_L):
    L_features = list()
    U_features = list()
    for feature in features:
        L_feature = feature[:len_L]
        U_feature = feature[len_L:]
        L_features.append(L_feature)
        U_features.append(U_feature)
    return L_features, U_features

def calc_weight_by_GAPVector_distance(features, GT, PL, len_L, input_size, device):
        ori_h, ori_w = input_size
        with torch.no_grad():
            L_features, U_features = split_features(features, len_L)
            per_image_mean_gaps_GT = torch.empty((0, 512)).to(device)
            for idx, boxes in enumerate(GT):
                mean_gaps = torch.empty((0, 512)).to(device)
                for box in boxes:
                    if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0: continue
                    gaps = torch.empty((0, 512)).to(device)
                    for features in L_features:
                        feature = features[idx]
                        _, feature_h, feature_w = feature.size()
                        print(_, feature_h, feature_w)
                        print(box)
                        x, y, w, h = translate_coordinate(box, feature_w, feature_h, ori_w, ori_h)
                        print(x, y, w, h)
                        obj = feature[:, y:y+h, x:x+w]
                        gap_obj = F.avg_pool2d(obj.unsqueeze(0), kernel_size=obj.size()[1:]).squeeze(0)
                        gaps = torch.cat((gaps, gap_obj), dim=0)

                    mean_gap = torch.mean(gaps, dim=0)
                    mean_gaps = torch.cat((mean_gaps, mean_gap), dim=0)

                per_image_mean_gap = torch.mean(mean_gaps, dim=0)
                per_image_mean_gaps_GT = torch.cat((per_image_mean_gaps_GT, per_image_mean_gap), dim=0)

            per_image_mean_gaps_PL = torch.empty((0, 512)).to(device)
            for idx, boxes in PL.items():
                for box in boxes:
                    gaps = torch.empty((0, 512)).to(device)
                    for features in U_features:
                        feature = features[idx]
                        _, feature_h, feature_w = feature.size()
                        x, y, w, h = translate_coordinate(box, feature_w, feature_h, ori_w, ori_h)
                        obj = feature[:, y:y+h, x:x+w]
                        gap_obj = F.avg_pool2d(obj.unsqueeze(0), kernel_size=obj.size()[1:]).squeeze(0)
                        gaps = torch.cat((gaps, gap_obj), dim=0)

                    mean_gap = torch.mean(gaps, dim=0)
                    mean_gaps = torch.cat((mean_gaps, mean_gap), dim=0)

                per_image_mean_gap = torch.mean(mean_gaps, dim=0)
                per_image_mean_gaps_PL = torch.cat((per_image_mean_gaps_PL, per_image_mean_gap), dim=0)
            
            mse = torch.sqrt(torch.sum((
                per_image_mean_gaps_GT - per_image_mean_gaps_PL
            ) ** 2, dim=1))
            mse_norm = torch.mean(mse).item()
            # weight = mse_norm
            weight = np.exp(-mse_norm)
            return weight
