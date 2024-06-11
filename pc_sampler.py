
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



def normalize_point_clouds(pcs, mode, logger=None):
    if mode is None:
        # logger.info('Will not normalize point clouds.')
        return pcs
    # logger.info('Normalization mode: %s' % mode)
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
parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/airplane_chair_200k.pt')
parser.add_argument('--ckpt_classifier', type=str, default='logs_pointnet/pointnet_classifier_two/classifier.pt')
parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
args = parser.parse_args()



class VariantDiffusionPoint(DiffusionPoint):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__(net, var_sched)
        
    def sample(
            self, 
            num_points, 
            context, 
            point_dim=3, 
            flexibility=0.0, 
            ret_traj=False,
            classifier=None,
            desired_class=None,
            s=1
            ):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]
        

class VariantFlowVAE(FlowVAE):
    pass    





class Classifier:
    def __init__(self, args):
        
        self.classes = args.categories
        self.device = args.device
        self.model = self.get_classifier(args.ckpt_classifier)
        self.mask = {i: self.classes[i] for i in range(len(self.classes))} # translator for labels 0, 1, ... to actual labels airplane, chair, ...


    def get_classifier(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model = PointNet(k=len(self.classes), feature_transform=True).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        # TODO: Remove this later
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
    


class Diffusion:
    def __init__(self, args):
        self.args = args
        self.path = args.ckpt
        self.batch_size = args.batch_size
        self.num_batches = args.num_batches
        self.model = self.load_model()

    
    def load_model(self):
        self.ckpt = torch.load(self.path, map_location=torch.device(self.args.device))
        # TODO: exchange for VariantFlowVAE()
        model = FlowVAE(self.ckpt['args']).to(self.args.device)
        model.load_state_dict(self.ckpt['state_dict']) 
        return model
       

    def sample(self) -> list:
        self.pcs = []
        for i in tqdm(range(self.num_batches), 'Sampling...'):
            with torch.no_grad():
                z = torch.randn([self.batch_size, self.ckpt['args'].latent_dim]).to(self.args.device)
                x = self.model.sample(z, self.args.sample_num_points, flexibility=self.ckpt['args'].flexibility)
                self.pcs.append(x.detach().cpu())
        self.pcs = torch.cat(self.pcs, dim=0)  # [:len(test_dset)] we don't need this right?
        # print(self.pcs[0])
        
        if args.normalize is not None:
            # TODO: move normalize to be part of this class
            self.pcs = normalize_point_clouds(self.pcs, mode=args.normalize)

        # Convert it to a shape that the classifier understands
        # self.pcs = self.pcs.permute(0, 2, 1)
        # print(f'Shape inside function: {self.pcs.shape}')

        # Now, conver them to pointcloud objects so they are formatted right
        self.pcs = [PointCloud(pc) for pc in self.pcs]
        return self.pcs
    

class PointCloud:
    def __init__(self, pc):
        self.pc = pc
        self.format_for_classifier()
        # print(f'Shape after formatting in PC object: {self.pc.shape}')

    def format_for_classifier(self):
        self.pc = self.pc.transpose(0, 1).unsqueeze(0) # switching around first and second dim and then adding one in front

    def classify(self, classifier: Classifier):
        # TODO: implement classification code
        outputs, _ = classifier.predict(self.pc)
        _, predicted = torch.max(outputs.data, 1)
        predicted = classifier.mask[predicted[0].item()]
        return predicted
        


def main():
    pass


if __name__ == '__main__':
    classifier = Classifier(args)
    diffusion = Diffusion(args)
    pcs = diffusion.sample()

    labels = list()
    for pc in pcs:
        labels.append(pc.classify(classifier))

    print(f'Number of airplanes: {labels.count('airplane')}')
    print(f'Number of chairs: {labels.count('chair')}')
        