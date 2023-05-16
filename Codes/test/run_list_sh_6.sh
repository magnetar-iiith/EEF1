#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1

#python test.py
# copy it back
python3 run_list40.py 


