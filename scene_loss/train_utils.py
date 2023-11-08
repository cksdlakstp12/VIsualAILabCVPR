from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle
from collections import defaultdict

from model import SSD300, SSD300_3Way

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
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_with_new_optim(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    
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
    
    return model, optimizer, optim_scheduler, start_epoch, None
    
def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state(args.n_classes, train_conf)
    else:
        if "teacher_weights.pth.tar071" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_with_new_optim(train_conf, checkpoint)
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

def initialize_state_3way(n_classes, train_conf):
    model = SSD300_3Way(n_classes=n_classes)

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

def load_state_from_checkpoint_3way(n_classes, train_conf, checkpoint):
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

def load_state_from_checkpoint_3way_with_new_optim(n_classes, train_conf, checkpoint):
    init_model, optimizer, optim_scheduler, train_conf.start_epoch, _ = initialize_state_3way(n_classes, train_conf)

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    pret_model = checkpoint['model']
    
    for pret_name, pret_param in pret_model.named_parameters():
        if "pred_convs" in pret_name:
            for name_pred in ["pred_convs", "pred_convs_vis", "pred_convs_lwir"]:
                seach_name = pret_name.replace("pred_convs", name_pred)
                init_param = dict(init_model.named_parameters())[seach_name]
                init_param.data.copy_(pret_param.data)
        elif pret_name in dict(pret_model.named_parameters()):
            init_param = dict(init_model.named_parameters())[pret_name]
            init_param.data.copy_(pret_param.data)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in init_model.named_parameters():
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

    return init_model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_3way(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state_3way(args.n_classes, train_conf)
    else:
        if "teacher_weights.pth.tar071" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way_with_new_optim(args.n_classes, train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way(args.n_classes, train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher_3way(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state_3way(config, student_checkpoint)
    teacher_state = load_state_3way(config, teacher_checkpoint)

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

def copy_student_to_teacher(teacher_model, student_model):
    """
    Copy student model to teacher model.
    θ_teacher = θ_student

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)

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

def translate_coordinate(box, feature_w, feature_h, ori_w, ori_h, is_GT):
    x, y, w, h = box
    new_x = int(x * feature_w) if is_GT else int(x / ori_w * feature_w)
    new_y = int(y * feature_h) if is_GT else int(y / ori_h * feature_h)
    new_w = int(w * feature_w) if is_GT else int(w / ori_w * feature_w)
    new_h = int(h * feature_h) if is_GT else int(h / ori_h * feature_h)
    return new_x, new_y, new_w, new_h

def split_features(features, len_L):
    L_features = [feature[:len_L] for feature in features]
    U_features = [feature[len_L:] for feature in features]
    return L_features, U_features

def compute_gap_from_features(features, box, ori_w, ori_h, idx, is_GT):
    gaps = []
    for feature in features:
        feature = feature[idx]
        _, feature_h, feature_w = feature.size()
        x, y, w, h = translate_coordinate(box, feature_w, feature_h, ori_w, ori_h, is_GT)
        obj = feature[:, y:y+h, x:x+w]
        if obj.size(1) > 0 and obj.size(2) > 0:
            gap_obj = F.avg_pool2d(obj.unsqueeze(0), kernel_size=obj.size()[1:]).squeeze()
            gaps.append(gap_obj)
    if not gaps:  # Check if gaps is empty
        return None
    return torch.mean(torch.stack(gaps), dim=0)

def calc_weight_by_GAPVector_distance(features, GT, PL, len_L, input_size):
    ori_h, ori_w = input_size
    with torch.no_grad():
        L_features, U_features = split_features(features, len_L)

        per_image_mean_gaps_GT = []
        for idx, boxes in enumerate(GT):
            # if len(boxes) <= 1: continue
            mean_gaps = [gap for gap in (compute_gap_from_features(L_features, box, ori_w, ori_h, idx, True) for box in boxes[1:]) if gap is not None]
            if mean_gaps:  # Check if mean_gaps is not empty
                per_image_mean_gaps_GT.append(torch.mean(torch.stack(mean_gaps), dim=0))

        per_image_mean_gaps_PL = []
        for idx, boxes in PL.items():
            mean_gaps = [gap for gap in (compute_gap_from_features(U_features, box, ori_w, ori_h, idx, False) for box in boxes) if gap is not None]
            if mean_gaps:  # Check if mean_gaps is not empty
                per_image_mean_gaps_PL.append(torch.mean(torch.stack(mean_gaps), dim=0))

        if per_image_mean_gaps_GT and per_image_mean_gaps_PL:  # Check if both lists are not empty
            per_image_mean_gaps_GT = torch.mean(torch.stack(per_image_mean_gaps_GT), dim=0)
            per_image_mean_gaps_PL = torch.mean(torch.stack(per_image_mean_gaps_PL), dim=0)    
            
            mse = torch.sqrt(torch.sum((per_image_mean_gaps_GT - per_image_mean_gaps_PL) ** 2, dim=0))
            weight = np.log(1 + mse.item()) 
            weight = np.mean(weight)
            return weight
            # mse_norm = torch.mean(mse).item()
            # weight = mse_norm
            # weight = np.exp(-mse_norm)
            # weight = np.exp(-mse_norm)
            # weight = np.log(1 + mse_norm + 1e-7)
            # weight = np.exp(mse_norm)
            # weight = 1 / mse_norm
        else:
            # Handle the case where one or both of the lists are empty
            # You can return a default weight or handle it in another way depending on your requirements
            return 0.0  # Default weight

def calc_weight_by_GAPVector_distance_3way(features, GT, PL, len_L, input_size):
    ori_h, ori_w = input_size
    with torch.no_grad():
        L_features, U_features = split_features(features, len_L)

        per_image_mean_gaps_GT = []
        for idx, boxes in enumerate(GT):
            # if len(boxes) <= 1: continue
            mean_gaps = [gap for gap in (compute_gap_from_features(L_features, box, ori_w, ori_h, idx, True) for box in boxes[1:]) if gap is not None]
            if mean_gaps:  # Check if mean_gaps is not empty
                per_image_mean_gaps_GT.append(torch.mean(torch.stack(mean_gaps), dim=0))
            else:
                per_image_mean_gaps_GT.append(None)

        per_image_mean_gaps_PL = []
        for idx, boxes in PL.items():
            mean_gaps = [gap for gap in (compute_gap_from_features(U_features, box, ori_w, ori_h, idx, False) for box in boxes) if gap is not None]
            if mean_gaps:  # Check if mean_gaps is not empty
                per_image_mean_gaps_PL.append(torch.mean(torch.stack(mean_gaps), dim=0))
            else:
                per_image_mean_gaps_PL.append(None)

        weights = torch.FloatTensor([])
        zero_cnt = 0
        for GT_gap, PL_gap in zip(per_image_mean_gaps_GT, per_image_mean_gaps_PL):
            if GT_gap is None or PL_gap is None:
                weights = torch.cat([weights, torch.zeros(1, dtype=torch.float64)])
                zero_cnt += 1
            else:
                mse = torch.sqrt(torch.sum((GT_gap - PL_gap) ** 2, dim=0))
                weight = torch.log(1 + mse + 1e-7).to("cpu") # log 사용시 min로 변경
                weights = torch.cat([weights, weight.unsqueeze(0)])
    
        assert weights.size(0) == len(GT), f"weights 텐서의 크기가 배치 크기와 일치해야 합니다. 현재 크기: {weights.size(0)}, 배치 크기: {batch_size}"

        return weights, False if zero_cnt == len_L else True

        # if per_image_mean_gaps_GT and per_image_mean_gaps_PL:  # Check if both lists are not empty
        #     per_image_mean_gaps_GT = torch.mean(torch.stack(per_image_mean_gaps_GT), dim=0)
        #     per_image_mean_gaps_PL = torch.mean(torch.stack(per_image_mean_gaps_PL), dim=0)    
            
        #     sq = (per_image_mean_gaps_GT - per_image_mean_gaps_PL) ** 2
        #     print("sq : ", sq.size())
        #     s = torch.sum(sq, dim=0)
        #     print("s : ", s)
        #     mse = torch.sqrt(s)
        #     # mse = torch.sqrt(torch.sum((per_image_mean_gaps_GT - per_image_mean_gaps_PL) ** 2, dim=0))
        #     print("mse : ", mse)
        #     weight = torch.log(1 + mse + 1e-7) # log 사용시 min로 변경
        #     print("weight : ", weight)
        #     # weight = torch.FloatTensor(weight)
        #     # weight = np.mean(weight)
        #     # mse_norm = torch.mean(mse).item()
        #     # weight = mse_norm
        #     # weight = np.exp(-mse_norm)
        #     # weight = np.log(1 + mse_norm + 1e-7)
        #     # weight = np.exp(mse_norm)
        #     # weight = 1 / mse_norm
        #     return weight, True
        #     # return weight * 100
        # else:
        #     # Handle the case where one or both of the lists are empty
        #     # You can return a default weight or handle it in another way depending on your requirements
        #     return torch.FloatTensor([0.0] * len_L), False  # Default weight
    
def calc_unvis_unlwir_weight_by_loss(vis_un_loss, lwir_un_loss):
    with torch.no_grad():
        un_vis_loss_value = vis_un_loss.item()
        un_lwir_loss_value = lwir_un_loss.item()
        un_sum_loss = un_vis_loss_value + un_lwir_loss_value
        if un_sum_loss == 0:
            return 0, 0
        un_vis_weight = un_vis_loss_value / un_sum_loss
        un_lwir_weight = un_lwir_loss_value / un_sum_loss
        return un_vis_weight, un_lwir_weight

def create_relation_matrix(features, len_L, device):
    with torch.no_grad():
        _, U_features = split_features(features, len_L)
        
        relation_matrixes = torch.FloatTensor([])
        for feature in U_features:
            feature_flat = feature.view(len_L, -1)

            relation_matrix = torch.zeros(len_L, len_L)

            for i in range(len_L):
                for j in range(len_L):
                    relation_matrix[i, j] = F.cosine_similarity(feature_flat[i].unsqueeze(0),
                                                                feature_flat[j].unsqueeze(0),
                                                                dim=1)
            relation_matrixes = torch.cat((relation_matrixes, relation_matrix.unsqueeze(dim=0)), dim=0)
        return relation_matrixes.to(device)
