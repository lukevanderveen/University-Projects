#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3
#SBATCH --mem=80G
#SBATCH --time=11:59:59
#SBATCH --cpus-per-task=4

source /etc/profile
module add cuda/12.0
module add anaconda3/2023.09

source activate /storage/hpc/00/zhangz65/acaconda3/envs/satlas
python training.py
