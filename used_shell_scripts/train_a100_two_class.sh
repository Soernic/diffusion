#!/bin/sh
#BSUB -J train_gen_trial_a100
#BSUB -o train_gen_trial_a100%J.out
#BSUB -e train_gen_trial_a100%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate fresh-env

python3 train_gen_trial.py
