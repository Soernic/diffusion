
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
    for i in range(pcs.size(0)):
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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
args = parser.parse_args()



class VariantDiffusionPoint(DiffusionPoint):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__(net, var_sched)

    def modify_e_theta(self, e_theta, x_t, classifier, desired_class, s, alpha_bar):
        x_t.requires_grad = True
        x_t = x_t.permute(0, 2, 1)
        e_theta = e_theta.permute(0, 2, 1)
        classifier_output = classifier.predict_variant(x_t, guidance=True)
        desired_output = classifier_output[:, desired_class]
        classifier_grad = torch.autograd.grad(outputs=desired_output.sum(), inputs=x_t, retain_graph=True)[0]
        classifier_grad_scaled = s * classifier_grad
        temp = e_theta - torch.sqrt(1 - alpha_bar) * classifier_grad_scaled
        temp = temp.permute(0, 2, 1)
        return temp
    
    def get_mu_addition(self, x_t, sigma, classifier, desired_class, s):
        x_t.requires_grad = True
        classifier_output = classifier.predict_variant(x_t.permute(0, 2, 1), guidance=True)
        desired_output = classifier_output[:, desired_class]
        classifier_grad = torch.autograd.grad(outputs=desired_output.sum(), inputs=x_t, retain_graph=True)[0]
        mu_addition = s * sigma**2 * classifier_grad
        return mu_addition
    

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
            
            # set_trace()
            mu = c0 * (x_t - c1 * e_theta)
            # Modify e_theta based on classifier guidance
            if classifier is not None and desired_class is not None:
                mu_addition = self.get_mu_addition(x_t=x_t, sigma=sigma, classifier=classifier, desired_class=desired_class, s=s)
                mu += mu_addition                            

            x_next = mu + sigma * z
            assert x_next.requires_grad == True
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

class VariantFlowVAE(FlowVAE):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = VariantDiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )


    def sample(self, w, num_points, flexibility, truncate_std=None, classifier=None, desired_class=None, s=1):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility, classifier=classifier, desired_class=desired_class, s=1)
        return samples
    

    


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
    
    def predict_variant(self, x, guidance=True):
        # TODO: adapt above function to combine these later if it makes sense
        # Ensure x is a tensor and move it to the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        # Add a batch dimension if x doesn't have one
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Only track gradients if we're doing guidance.
        if guidance:
            outputs = self.model(x)
        else:
            with torch.no_grad():
                outputs = self.model(x)
        
        # If the model returns more than one output, we take the first one (logits)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
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
        # model = FlowVAE(self.ckpt['args']).to(self.args.device)
        model = VariantFlowVAE(self.ckpt['args']).to(self.args.device)
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

        # Now, conver them to pointcloud objects so they are formatted right
        self.pcs = [PointCloud(pc) for pc in self.pcs]
        return self.pcs
    

    def sample_variant(self, classifier=None, desired_class=None, s=1) -> list:
        self.pcs = []
        for i in range(self.num_batches):
            z = torch.randn([self.batch_size, self.ckpt['args'].latent_dim]).to(self.args.device)
            x = self.model.sample(z, self.args.sample_num_points, flexibility=self.ckpt['args'].flexibility, classifier=classifier, desired_class=desired_class, s=1)
            self.pcs.append(x.detach().cpu())
        self.pcs = torch.cat(self.pcs, dim=0)  # [:len(test_dset)] we don't need this right?
        
        if args.normalize is not None:
            # TODO: move normalize to be part of this class
            self.pcs = normalize_point_clouds(self.pcs, mode=args.normalize)

        # Now, conver them to pointcloud objects so they are formatted right
        self.pcs = [PointCloud(pc) for pc in self.pcs]
        return self.pcs
    

class PointCloud:
    def __init__(self, pc):
        self.pc = pc
        self.format_for_classifier()
        self.save_path = './pcs'
        # print(f'Shape after formatting in PC object: {self.pc.shape}')

    def format_for_classifier(self):
        self.pc = self.pc.transpose(0, 1).unsqueeze(0) # switching around first and second dim and then adding one in front

    def classify(self, classifier: Classifier):
        # TODO: implement classification code
        outputs, _ = classifier.predict(self.pc)
        _, predicted = torch.max(outputs.data, 1)
        predicted = classifier.mask[predicted[0].item()]
        return predicted
    
    def save(self, name):
        # TODO: Write out function that saves the point clouds in some format. 
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, name + '.npy')
        np.save(file_path, self.pc.numpy())
        print(f'Point cloud saved to {file_path}')
        

def experiment(s, num_clouds=10):
    classifier = Classifier(args)
    diffusion = Diffusion(args)    

    pcs = list()
    preds = list()
    for i in tqdm(range(num_clouds), 'Generating clouds'):
        pc = diffusion.sample_variant(
            classifier=classifier,
            desired_class=0,
            s=s
        )[0]
        pcs.append(pc)
        preds.append(pc.classify(classifier))

    print(f's: {s} | airplanes: {preds.count('airplane')}')


def main():
    pass


if __name__ == '__main__':
    for i in range(2):    
        classifier = Classifier(args)
        diffusion = Diffusion(args)
        pc = diffusion.sample_variant(
            classifier=classifier,
            desired_class=0,
            s=1
        )[0]
        pc.save(f'test_{i}')

    # s_vals = [1, 10, 100, 1000, 10000]
    # for s in s_vals:
    #     experiment(s, num_clouds = 500)







    # classifier = Classifier(args)
    # diffusion = Diffusion(args)
    # pcs = diffusion.sample()

 




    # x = pcs[0].pc
    # x.requires_grad = True
    # # x = torch.randn(1, 3, 2048, requires_grad=True)
    # print(f'Actual label: {pcs[0].classify(classifier)}')

    # outputs = classifier.predict_variant(x, guidance=True)
    # desired_class = 0 # TODO: add mask_inv method that maps value to key in classifier.mask dict
    # desired_output = outputs[:, desired_class] # THis is just a sort of squeeze and then selecting the right class
    # classifier_grad = torch.autograd.grad(outputs=desired_output.sum(), inputs=x)[0]
    # grad_list = classifier_grad.squeeze(0).flatten()
    # print(f'Total number of points: {len(grad_list)}')
    # print(f'Number of non-zero ele: {sum(grad_list == 0)}')
    # print(f'Percentage of zero ele: {(100 * sum(grad_list == 0) / len(grad_list)):.2f}%')
    # print(classifier_grad.shape)
