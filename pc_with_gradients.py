
import argparse
import math
import os
import time

import torch
from tqdm.auto import tqdm
from typing import List # just type better type hints 
from pdb import set_trace # for debugging

from evaluation import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.vae_flow import *
from models.vae_gaussian import *
from models.classifier import *
from utils.data import *
from utils.dataset import *
from utils.misc import *

# For animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from pc_sampler import *


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_sup_1M.pt')
    parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/classifier_all_14k.pt')
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
    parser.add_argument('--ret_traj', type=eval, default=True, choices=[True, False])
    parser.add_argument('--num_steps', type=int, default=200)
    args = parser.parse_args()



    t = args.num_steps
    classifier = Classifier(args)
    diffusion = Diffusion(args)
    labels = []

    desired_class = 0       # This is airplane
    s = 10                   # This is gradient scale

    for idx in tqdm(range(100)):
        pc_batch = diffusion.sample(
            classifier=classifier,
            desired_class=desired_class,
            s=s
        )
        
        label = pc_batch[0].classify(classifier)
        labels.append(label)

        # label = pc_batch.batch_list[0].classify(classifier)
        # pc_batch.animate(f'gradient_{idx}_{label}')

    print(f'Ratio of airplanes vs. baseline ~38%: {(labels.count('airplane') / len(labels) * 100):.1f}')
    