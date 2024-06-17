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
#BSUB -W 06:00
#BSUB -u s224188@dtu.dk
#BSUB -B
#BSUB -N 
### end of BSUB options

source /zhome/62/3/187432/anaconda3/etc/profile.d/conda.sh

conda activate base

python3 train_gen.py --max_iters=1000000 --load_ckpt=True --ckpt='./logs_gen/GEN_2024_06_14__01_11_37/ckpt_0.000000_870000.pt'
