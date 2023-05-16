#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1

# module load use.own
# module load python/3.7.4
# module load cudnn/7.6.5-cuda-10.2
# module load cuda/10.2

#python test.py
# copy it back
# python3 run_list35.py 

python3 run_list36.py 

