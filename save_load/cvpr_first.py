from datetime import datetime
import config_case1 as config
import logging
import time

from inference_cvpr import val_epoch, save_results, run_inference
from model import SSD300, MultiBoxLoss
from utils import utils

import os
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
#import config_cvpr as config
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_case1 import KAISTPed
from datasets_cvpr import KAISTPed as cvpr_KAISTPed
from utils.transforms import FusionDeadZone
from utils.evaluation_script import evaluate
from vis import visualize

from model import SSD300
from train_eval_cvpr import main

import pandas as pd

def converter(originpath, changepath, wantname):
    # Loading the 90percents.txt file and creating a dictionary where keys are the index
    with open(f'{originpath}', 'r') as f:
        data_90 = {idx+1: line.strip() for idx, line in enumerate(f)}

    # Loading the test2.txt file
    with open(f'{changepath}', 'r') as f:
        data_test2 = f.readlines()

    # Replacing the first number of each line in test2.txt with corresponding line in 90percents.txt
    data_test2_new = []
    for line in data_test2:
        items = line.split(',')
        index = int(items[0])
        items[0] = data_90[index]
        data_test2_new.append(','.join(items))

    # Writing the new data into a new file
    with open(f'{wantname}.txt', 'w') as f:
        for line in data_test2_new:
            f.write(line)

torch.backends.cudnn.benchmark = False

# random seed fix 
utils.set_seed(seed=9)

epoch = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shell 파일에서 사용한 인수와 동일한 인수를 전달하여 함수 호출

#args = config.args

model_path = "jobs3/2023-08-08_02h19m00s_/checkpoint_ssd300.pth.tar013"
fdz_case = "original"

for i in range(1,epoch+1):
    #results = run_inference(FDZ=fdz_case, model_path=model_path, result_dir="../cvpr", vis=False)
    #esult_filename = f"../cvpr/test5.txt"
    
    # Save results
    #save_results(results, result_filename)

    #converter("../cvpr/90percents.txt","../cvpr/test5.txt", "../cvpr/new_annotations2")

    #train 90pencents
    main()

    #phase = "Multispectral"
    #evaluate(config.PATH.JSON_GT_FILE, result_filename + '.txt', phase)
    
    # config.py와 model.py에서 90perecnts.txt를 라벨링되지 않은 데이터로 불러오고 checkpoint_ssd300.pth.tar071의 가중치를  pretrained 모델로 불러와서 학습 시작
    # inference.py를 사용하여 라벨링되지 않은 데이터를  data labeling
    # 라벨링 된 데이터를 train_eval_case1.py에서 학습