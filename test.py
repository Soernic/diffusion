import os
import argparse
import torch
# import torch.utils.tensorboard
# from torch.nn.utils import clip_grad_norm_
# from tqdm.auto import tqdm

# from utils.dataset import *
# from utils.misc import *
# from utils.data import *
# from utils.transform import *
# from evaluation import EMD_CD
# # from pointnet.z_PointNetCls import *
# from models.classifier import *
# from models.autoencoder import *
# # Arguments
from pdb import set_trace


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





beta_1 = 1e-4
beta_T = 0.02

# Get betas and pad the first step
betas = torch.linspace(beta_1, beta_T, steps=100)
betas = torch.cat([torch.zeros([1]), betas], dim=0)

# Get alphas from betas
alphas = 1 - betas

# Can't do cumprod since 0 in first step, but we can do this
log_alphas = torch.log(alphas)
for i in range(1, log_alphas.size(0)):
    log_alphas[i] += log_alphas[i - 1]
alpha_bars = log_alphas.exp()
set_trace()

