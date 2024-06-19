
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
from pc_sampler import *

# For animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/airplane_chair_200k.pt')
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
args = parser.parse_args()


if __name__ == '__main__':

    print('generating gifs')
    # Initialize classifier and diffusion model with arguments
    classifier = Classifier(args)
    diffusion = Diffusion(args)
    #ckpt = torch.load(args.ckpt)
    #it = ckpt['others']['scheduler']['_step_count'] - 1
    # Get a PointCloudBatch object containing the trajectory of the diffusion process
    pc_batch = diffusion.sample()
    pc = pc_batch.batch_list[0]
    # Classify the final image (which is idx 0 in the batch)
    label = pc.classify(classifier)
    print(pc)
    # Animate the process.
    # TODO: MODIFY NAME HERE IF YOU WANT IT DIFFERENT. 
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the initial point cloud

    # Create a scatter plot
    scatter = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 0], cmap='Blues', s=30, edgecolor='none', alpha=0.7)

    # Set up the axes limits
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.savefig(f'{1}_{label}.png')