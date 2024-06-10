#!/bin/sh
#BSUB -J train_gen_trial
#BSUB -o train_gen_trial%J.out
#BSUB -e train_gen_trial%J.err
### BSUB -q gpuh100
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate fresh-env

python3 train_gen_trial.py
