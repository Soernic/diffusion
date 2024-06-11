import argparse
import math
import os
import time

import torch
from tqdm.auto import tqdm
from typing import List

from evaluation import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.vae_flow import *
from models.vae_gaussian import *
from models.classifier import *
from utils.data import *
from utils.dataset import *
from utils.misc import *





def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=4)

# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
# ckpt = torch.load(args.ckpt)
# Checkpoint
ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))

seed_all(args.seed)


# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=args.normalize,
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)


# Model
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])



gen_pcs = []
for i in tqdm(range(1), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)


pc = gen_pcs[0]
print(pc.shape)
pc = pc.unsqueeze(0).transpose(1, 2)
print(pc.shape)


class Classifier:
    def __init__(self, path: str, classes: List[str] = ['airplane', 'chair'], device='cpu'):
        self.classes = classes
        self.device = device
        self.model = self.get_classifier(path)
        self.mask = {i: self.classes[i] for i in range(len(self.classes))} # translator for labels 0, 1, ... to actual labels airplane, chair, ...

    def get_classifier(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model = PointNet(k=len(self.classes), feature_transform=True).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()  # Set the model to evaluation mode
        return model

    def predict(self, x):
        # Ensure x is a tensor and move it to the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        
        # Add a batch dimension if x doesn't have one
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # TODO: enable gradients later
        with torch.no_grad():  # Disable gradient calculation for prediction
            outputs = self.model(x)
        return outputs
    
    def __repr__(self):
        return f'Classifier for classes {self.classes}'


classifier_path = 'logs_pointnet/pointnet_classifier_two/classifier.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = Classifier(classifier_path, device=device)
print(classifier)

# Example input data, ensure this is in the correct shape
# Assuming pc is already a torch tensor with the correct shape [batch_size, num_features, num_points]
# pc = torch.randn(1, 3, 2048)  # Example tensor, adjust shape according to your data

outputs, _ = classifier.predict(pc)
_, predicted = torch.max(outputs.data, 1)
print(classifier.mask[predicted[0].item()])


# def visualize_point_cloud(point_cloud):
#     # Convert tensor to numpy array
#     point_cloud_np = point_cloud.numpy()

#     # Create an Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()

#     # Assign points
#     pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

#     # Visualize
#     o3d.visualization.draw_geometries([pcd])

# print(gen_pcs)

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def visualize_point_cloud_matplotlib(point_cloud):
#     point_cloud_np = point_cloud.numpy()
    
#     x = point_cloud_np[:, 0]
#     y = point_cloud_np[:, 1]
#     z = point_cloud_np[:, 2]
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z, c=z, cmap='cool', marker='o')
    
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
    
#     plt.show()

# # Example usage
# for pc in gen_pcs:
#     visualize_point_cloud_matplotlib(pc)



# visualize_point_cloud(gen_pcs[0])  # Visualize the first generated point cloud