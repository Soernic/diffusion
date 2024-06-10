#!/bin/sh
#BSUB -J pointnet
#BSUB -o train_gen_%J.out
#BSUB -e pointnet%J.err
### BSUB -q gpuv100
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 15
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate dpm-pc-gen

python z_ClassifierGPT.py
