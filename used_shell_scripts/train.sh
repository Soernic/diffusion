#!/bin/sh
#BSUB -J class_2_max
#BSUB -o class_2_max%J.out
#BSUB -e class_2_max%J.err
### BSUB -q gpuh100
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -u s224183@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate fresh-env

python3 test_gen_s_values.py
