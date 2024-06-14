
import argparse
import math
import os
import time
import numpy as np

# import torch
# from tqdm.auto import tqdm
# from typing import List # just type better type hints 
# from pdb import set_trace # for debugging

# from evaluation import *
# from models.flow import add_spectral_norm, spectral_norm_power_iteration
# from models.vae_flow import *
# from models.vae_gaussian import *
# from models.classifier import *
# from utils.data import *
# from utils.dataset import *
# from utils.misc import *


def load_point_cloud(file_path):
    point_cloud = np.load(file_path)
    return point_cloud

pc = load_point_cloud('./pcs/airplane_1_0.npy')
print(pc)
print(np.mean(pc, axis=2))
print(np.var(pc, axis=2))
print(pc.shape)
