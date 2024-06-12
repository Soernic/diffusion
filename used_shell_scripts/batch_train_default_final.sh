#!/bin/sh
#BSUB -J default_24_hours
#BSUB -o default_24_hours%J.out
#BSUB -e default_24_hours%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=1G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s224169@dtu.dk
#BSUB -B
#BSUB -N
### end of BSUB options

source $HOME/miniconda3/bin/activate fresh-env

python3 train_gen.py --tag default_long --parallel True --max_iters 1000000
