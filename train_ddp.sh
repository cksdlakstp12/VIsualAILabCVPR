#!/bin/bash

#SBATCH --job-name pedestrian
#SBATCH --error=error/train_error-%A-%x.txt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time=72:00:00
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

pip install easydict
pip install Pillow==8.3.0

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train.py

exit 0
