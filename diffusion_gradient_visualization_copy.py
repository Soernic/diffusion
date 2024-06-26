import argparse
import math
import os
import time

import torch
from tqdm.auto import tqdm
from typing import List
from pdb import set_trace

from evaluation import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.vae_flow import *
from models.vae_gaussian import *
from models.classifier import *
from utils.data import *
from utils.dataset import *
from utils.misc import *
from pc_with_gradients import *
from plots.plots import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

class GradientPointCloudBatch:
    def __init__(self):
        self.batch_list = list()

    def append(self, pc: GradientPointCloud):
        self.batch_list.append(pc)
    
    @staticmethod
    def update(frame, point_clouds, scatter, ax):
        point_cloud = point_clouds[frame]
        scatter._offsets3d = (point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

        ax.view_init(elev=30, azim=45 + 0.1 * frame)

        return scatter,

    def animate(self, name):
        angle = np.pi / 2  # 90 degrees
        point_clouds = [self.rotate(pc.pc, 'x', angle) for pc in self.batch_list]

        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax = fig.add_subplot(111, projection='3d')

        initial_pc = point_clouds[0]

        scatter = ax.scatter(initial_pc[:, 0], initial_pc[:, 1], initial_pc[:, 2],
                             c=initial_pc[:, 0], cmap='Blues', s=50, edgecolor='none', alpha=0.7)

        # Set up the axes limits
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        ax.set_axis_off()

        ani = FuncAnimation(fig, self.update, frames=len(point_clouds), fargs=(point_clouds, scatter, ax), interval=100)

        ani.save(f'{name}.gif', writer='pillow', fps=60)

    def rotate(self, pc, axis, angle):
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return np.dot(pc, rotation_matrix.T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--ret_traj', type=eval, default=False, choices=[True, False])
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])

    parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_base_800k.pt')
    parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/cl_2_max_100.pt')
    parser.add_argument('--categories_classifier', type=str_list, default=['airplane', 'chair'])
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--desired_class', type=int, default=0)

    parser.add_argument('--ckpt_second_classifier', type=str, default=None)
    parser.add_argument('--second_categories_classifier', type=str_list, default=['airplane', 'chair'])

    parser.add_argument('--save_name', type=str, default='testing')

    args = parser.parse_args()
    seed_all(args.seed)

    args.batch_size = 1
    y = torch.ones(args.batch_size, dtype=torch.long).to(args.device) * args.desired_class

    gradient_batch = GradientPointCloudBatch()

    gradient_scales = np.arange(0, 2000, step=3)
    for s in tqdm(gradient_scales, 's-values'):
        args.gradient_scale = s
        seed_all(args.seed)     
        models = set_up_classifier_diffusion(args)
        seed_all(args.seed)
        batch = generate_batch(args, models, y, q=True)
        gradient_batch.append(batch[0])

    gradient_batch.animate(args.save_name)
