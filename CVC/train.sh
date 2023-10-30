#!/bin/bash

#SBATCH --job-name CVC_10p_baseline
#SBATCH --error=error/train_error-%A-%x.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition batch_ce_ugrad
#SBATCH -o slurm/logs/slurm-%A-%x.out

pip install easydict
pip install Pillow==8.3.0

cd src

CUDA_VISIBLE_DEVICES=0 python train_eval.py

exit 0
