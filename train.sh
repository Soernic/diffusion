#!/bin/sh
#BSUB -J train_gen
#BSUB -o train_gen_%J.out
#BSUB -e train_gen%J.err
### BSUB -q gpuv100
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=5G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate dpm-pc-gen

python matmul_torch.py
