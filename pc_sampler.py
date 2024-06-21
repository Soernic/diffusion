
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


# TODO: REmove later - duct tape fix
t = None


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
parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/classifier_all_14k.pt')
parser.add_argument('--classifier_categories', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
parser.add_argument('--ret_traj', type=eval, default=False, choices=[True, False])

if __name__ == '__main__':
    args = parser.parse_args()
    seed_all(args.seed)



class VariantDiffusionPoint(DiffusionPoint):
    def __init__(self, net, var_sched: VarianceSchedule, ret_traj=False):
        super().__init__(net, var_sched)
        self.ret_traj = ret_traj

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
            classifier=None,
            desired_class=None,
            s=1,
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
            if not self.ret_traj:
                del traj[t]
        
        if self.ret_traj:
            return traj
        else:
            return traj[0]

class VariantFlowVAE(FlowVAE):
    def __init__(self, args, ret_traj):
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
            ),
            ret_traj=ret_traj
        )


    def sample(self, w, num_points, flexibility, truncate_std=None, classifier=None, desired_class=None, s=1, save_sample=False):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility, classifier=classifier, desired_class=desired_class, s=1)
        return samples
    

    


class Classifier:
    def __init__(self, args):
        
        self.classes = args.classifier_categories
        if self.classes == ['all']:
            self.classes = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock', 'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone', 'birdhouse', 'bookshelf']
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
        self.ret_traj = args.ret_traj
        self.path = args.ckpt
        self.batch_size = args.batch_size
        self.num_batches = args.num_batches
        self.model = self.load_model()

    
    def load_model(self):
        self.ckpt = torch.load(self.path, map_location=torch.device(self.args.device))
        # TODO: exchange for VariantFlowVAE()
        # model = FlowVAE(self.ckpt['args']).to(self.args.device)
        model = VariantFlowVAE(self.ckpt['args'], self.ret_traj).to(self.args.device)
        model.load_state_dict(self.ckpt['state_dict']) 
        return model
       

    def sample(self, classifier=None, desired_class=None, s=1) -> list:
        self.pcs = []
        if not self.ret_traj:
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
        
        else:
            for i in range(self.num_batches):
                z = torch.randn([self.batch_size, self.ckpt['args'].latent_dim]).to(self.args.device)
                traj = self.model.sample(z, self.args.sample_num_points, flexibility=self.ckpt['args'].flexibility, classifier=classifier, desired_class=desired_class, s=1)            
                # In this case traj is a dictonary
                traj_list = self.dict_to_list(traj)
            
            # Normalization? Probably not.. It would be different for each could and would have to be done for the entire trajectory
            self.pcs = [PointCloud(pc) for pc in traj_list]
            pc_batch = PointCloudBatch()
            for pc in self.pcs:
                pc_batch.append(pc)

            return pc_batch

    def dict_to_list(self, traj):
        return [traj[i].detach().cpu() for i in range(len(traj.keys()))]

    

class PointCloud:
    def __init__(self, pc):
        self.pc = pc
        # self.format_for_classifier()
        self.pc = self.reshape() # Standardizes shape to be [1, 3, 2048] which is what classifier wants. Don't use it twice though. It'll get mad.
        self.save_path = './pcs'
        # print(f'Shape after formatting in PC object: {self.pc.shape}')

    def format_for_classifier(self):
        self.pc = self.pc.transpose(0, 1).unsqueeze(0) # switching around first and second dim and then adding one in front

    def classify(self, classifier: Classifier):
        # pc = self.reshape()
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

    def rotate(self, axis, angle):
        # TODO: implement rotate function from animate.py here
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
        
        return np.dot(self.pc.squeeze(0).permute(1, 0), rotation_matrix.T)


    def copy(self):
        return PointCloud(self.pc)
    
    def reshape(self):
        """
        Shapes are a hassle. This function just formats any point cloud so the classifier can classify it as long as it has 2048 x 3 points in it.
        """
        flat_tensor = self.pc.view(-1)
        num_points = 2048 
        num_dims = 3
        reshaped = flat_tensor.view(num_points, num_dims)
        reshaped = reshaped.t()
        reshaped = reshaped.view(1, num_dims, num_points)
        return reshaped

        

class PointCloudBatch:
    """
    Helper class for animating a point cloud through its generation process
    You can start it empty or with a PointCloud object and then gradually update it through the sampling process. 
    With the format function you can ready it for animation
    """

    def __init__(self, gradients=False):
        self.batch_list = list() # Maybe I can concatenate them into a tensor instead?
        if gradients: 
            self.gradient_batch_list = list() # with tuples like [(PointCloud, gradients), (PointCloud, gradients), ...]

    def append(self, pc: PointCloud, gradients=None):
        if not gradients: 
            assert isinstance(pc, PointCloud)
            self.batch_list.append(pc)
        else:
            assert isinstance(pc, PointCloud)
            # assert isinstance(gradients, ...? what is that class?)
            # assert pc.shape == (1, 3, 2048)?
            self.gradient_batch_list.append((pc, gradients))

    def save(self, name):
        # TODO: save as npy file
        save_path = './trajectories'
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, name + '.npy')
        np.save(file_path, self.format())
        print(f'PointCloudBatch saved to {file_path}')

    # Just for animation
    # Function to update the plot for each frame
    @staticmethod
    def update(frame, point_clouds, scatter, ax):
        point_cloud = point_clouds[frame]
        scatter._offsets3d = (point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        
        # Rotate the view
        ax.view_init(elev=30, azim=45 + frame)
        
        return scatter,

    def animate(self, name):
        angle = np.pi / 2  # 90 degrees
        point_clouds = list(reversed([pc.rotate('x', angle) for pc in self.batch_list]))

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the initial point cloud
        initial_pc = point_clouds[0]

        # Create a scatter plot
        scatter = ax.scatter(initial_pc[:, 0], initial_pc[:, 1], initial_pc[:, 2],
                            c=initial_pc[:, 0], cmap='Blues', s=30, edgecolor='none', alpha=0.7)

        # Set up the axes limits
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

        # Create an animation
        ani = FuncAnimation(fig, self.update, frames=len(point_clouds), fargs=(point_clouds, scatter, ax), interval=100)

        # Save the animation as a GIF file
        ani.save(f'{name}.gif', writer='pillow', fps=33)

                


def experiment(s, num_clouds=10):
    classifier = Classifier(args)
    diffusion = Diffusion(args)    

    pcs = list()
    preds = list()
    for i in tqdm(range(num_clouds), 'Generating clouds'):
        pc = diffusion.sample(
            classifier=classifier,
            desired_class=0,
            s=s
        )[0]
        pcs.append(pc)
        preds.append(pc.classify(classifier))

    print(f's: {s} | airplanes: {preds.count('airplane')}')



if __name__ == '__main__':
    # s = 1
    # for i in range(3):    
    #     classifier = Classifier(args)
    #     diffusion = Diffusion(args)
    #     pc = diffusion.sample(
    #         classifier=classifier,
    #         desired_class=0,
    #         s=s
    #     )[0]
    #     label = pc.classify(classifier)
    #     pc.save(f'{label}_{s}_{i}_all')

    # s_vals = [1, 10, 100]
    # for s in s_vals:
    #     experiment(s, num_clouds = 50)

    for i in tqdm(range(5), 'generating gifs'):
        classifier = Classifier(args)
        diffusion = Diffusion(args)
        pc_batch = diffusion.sample()
        label = pc_batch.batch_list[0].classify(classifier)
        print(label)
        # pc_batch.animate(f'{i}_{label}')
        # print('success!')
        # pc = pc_batch.batch_list[0].copy()
        # label = pc.classify(classifier)
        # label = pc_batch.batch_list[0].classify(classifier)
        # # set_trace()
        # # set_trace()
        # assert isinstance(label, str)
        # # print(pc_batch)
        # pc_batch.save(f'test_{i}_{label}')





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
