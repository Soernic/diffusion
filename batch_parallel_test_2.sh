#!/bin/sh
#BSUB -J para_test_v100
#BSUB -o para_test_v100%J.out
#BSUB -e para_test_v100%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=1G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate fresh-env

python3 train_gen_para.py --tag para_test --parallel True --max_iters 30000
