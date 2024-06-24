
import argparse
import os
import torch

from tqdm.auto import tqdm
from typing import List # just type better type hints 
from pdb import set_trace # for debugging

from evaluation import *
from models.vae_flow import *
from models.vae_gaussian import *
from models.classifier_time_aware import * # this is the 6 dimensional points one
from utils.data import *
from utils.dataset import *
from utils.misc import *

import matplotlib.pyplot as plt
from pdb import *

from pc_sampler import Classifier, VariantDiffusionPoint, Diffusion, PointCloud, PointCloudBatch
import pandas as pd


def visualize_gradients(gradients, sigma):
    grad_values = gradients.detach().cpu().numpy().flatten()
    plt.hist(grad_values, bins=100)
    plt.xlabel('Gradient values')
    plt.ylabel('Frequency')
    plt.title('Gradient Value Distribution')
    plt.savefig('plots/sigma.png', dpi=300)

class ClassifierWithGradients(Classifier):
    def __init__(self, args):        
        self.classes = args.categories_classifier
        # print(self.classes)
        if self.classes == ['all']:
            self.classes = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock', 'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone', 'birdhouse', 'bookshelf']
            # raise ValueError('Stop it, this implementation only supports classifiers with 2 predictions, so "all" does not work. Default should be airplane, chair')
        
        self.device = args.device
        self.model = self.get_classifier(args.ckpt_classifier) 
        self.mask = {i: self.classes[i] for i in range(len(self.classes))} # translator for labels 0, 1, ... to actual labels airplane, chair, ...


    def get_classifier(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model = PointNetWithTimeEmbedding(k=len(self.classes), feature_transform=True).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        # set_trace()
        model.eval()  # Set the model to evaluation mode - does this affect gradient tracking?
        return model
    
    def predict_gradient(self, x, beta):
        x = self._format_x(x)
        with torch.enable_grad():
            outputs = self.model(x, beta)
        
        # outputs = self._get_first(outputs)
        return outputs
    
    def predict(self, x, beta=1e-4):
        # set_trace()
        x = self._format_x(x)
        beta = self._format_beta(beta)
        with torch.no_grad():
            outputs = self.model(x, beta)
        return outputs


    def _format_x(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return x
    
    def _format_beta(self, beta):
        if beta.size() == torch.Size([]):
            return beta.unsqueeze(0)
        return beta
    

    def _get_first(self, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs



    
class DiffusionWithGradients(Diffusion):
    def __init__(self, args, classifier):
        self.classifier = classifier
        self.s = args.gradient_scale
        self.device = args.device
        self.normalize = args.normalize
        super().__init__(args)


    def load_model(self):
        self.ckpt = torch.load(self.path, map_location=torch.device(self.args.device))
        model = TimeEmbeddingVariantFlowVAE(
            self.ckpt['args'], 
            self.ret_traj, 
            self.classifier, 
            self.s,
            self.device
            ).to(self.args.device)
        model.load_state_dict(self.ckpt['state_dict'])
        return model


    def sample(self, y=None):
        self.pcs = []

        for i in range(self.num_batches):
            z = torch.randn([self.batch_size, self.ckpt['args'].latent_dim]).to(self.args.device)
            x = self.model.sample(
                w=z, 
                num_points=self.args.sample_num_points,
                flexibility=self.ckpt['args'].flexibility,
                y=y
                )
            if self.ret_traj:
                assert self.batch_size == 1, "You can only use batch size 1 when sampling trajectories. Check the --ret_traj argument please."
                traj_list = self.dict_to_list(x)
            else:
                self.pcs.append(x.detach().cpu())
        
        if self.ret_traj:
            return self.ret_traj_(traj_list)
        else:
            #return self.pcs
            return self.ret_sample_()
        

    def ret_sample_(self):
        self.pcs = torch.cat(self.pcs, dim=0)
        if self.normalize is not None:
            self.pcs = self.normalize_(self.pcs, mode=self.normalize)
        # self.pcs = [GradientPointCloud(pc, device=self.device) for pc in self.pcs]
        #self.pcs = [PointCloud(pc) for pc in self.pcs]
        return self.pcs


    def ret_traj_(self, traj_list):
        raise NotImplementedError("This is close, but not quite there. The issue is that this function (ret_traj_) uses the regular PointCloud object. This will cause problems in the future, since the PointCloud.classify method which assumes the classifier takes only x as input, while the proper time-aware classifier also takes beta into account. So, this function must be modified such that it uses 1) the GradientPointCloud object instead, which calls the correct classifier and 2) passes the right values of beta throughout the trajectory, and not just random values. The default value of beta in GradientPointCloud is the final value of 1e-4, but this does not work in a trajectory setting.")
        self.pc_batch = PointCloudBatch()
        for pc in traj_list:
            self.pc_batch.append(PointCloud(pc))
        return self.pc_batch
    

    def normalize_(self, pcs, mode):
        if mode is None:
            return pcs
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


class TimeEmbeddingVariantFlowVAE(FlowVAE):
    def __init__(self, args, ret_traj, classifier, s, device):
        super().__init__(args)
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = TimeEmbeddingVariantDiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            ),
            ret_traj=ret_traj,
            classifier=classifier,
            s=s,
            device=device
        )

    def sample(self, w, num_points, flexibility, truncate_std=None, y=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)

        # Use flow to get z from isotropic gaussian
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)

        # Sample using TimeEmbeddingVariant of diff class
        samples = self.diffusion.sample(
            num_points=num_points,
            context=z,
            point_dim=3,
            flexibility=flexibility,
            y=y,
        )

        return samples        


class TimeEmbeddingVariantDiffusionPoint(VariantDiffusionPoint):
    def __init__(self, net, var_sched: VarianceSchedule, ret_traj=False, classifier=None, s=1, device=None):
        super().__init__(net, var_sched, ret_traj)
        self.classifier = classifier
        self.device = device
        self.s = s
        self.t = 0
        self.dataframe = pd.DataFrame(columns=np.arange(100))


    def get_mu_addition(self, x_t, sigma, y=None):
        gradients = self.classifier_gradients(x_t, sigma, y=y)
        return self.s * sigma**2 * gradients
    
    def classifier_gradients(self, x, sigma, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            beta = torch.ones(x_in.size()[0]).to(self.device) * sigma**2 # converting to tensor
            logits, _ = self.classifier.predict_gradient(x_in, beta)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            gradients = torch.autograd.grad(selected.sum(), x_in)[0]
            gradients = torch.clamp(gradients, min=-1.0, max=1.0)
            grad = gradients.detach().cpu().numpy().reshape(-1)
            self.dataframe[self.t] = grad
            self.t += 1
            
            #set_trace()
            
            # visualize_gradients(gradients, sigma)
            return gradients
        
    def sample(
            self,
            num_points,
            context,
            point_dim=3,
            flexibility=0.0,
            y=None,
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
            if self.classifier is not None and y is not None:
                mu += self.get_mu_addition(
                    x_t=x_t,
                    sigma=sigma,
                    y=y,
                )

            x_next = mu + sigma * z

            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not self.ret_traj:
                del traj[t]
        
        if self.ret_traj:
            return traj
        else:
            self.dataframe.to_csv("Gradient_dataframe_cl2.csv")
            return traj[0]



class GradientPointCloud(PointCloud):
    def __init__(self, pc, beta=None, device='cuda'):
        """
        Defaults to beta = beta_1.
        Specify beta for any other value.  
        """
        
        if beta is None:
            self.beta = torch.tensor(0).to(device)
        else:
            assert isinstance(beta, torch.Tensor)
            self.beta = beta

        self.pc = pc
        self.save_path = './pcs'


    def classify(self, classifier: ClassifierWithGradients):
        pc = self.pc.unsqueeze(0)
        # set_trace()
        assert pc.shape == torch.Size([1, 2048, 3])
        outputs, _ = classifier.predict(pc, self.beta)
        predicted = torch.argmax(outputs.data, 1)
        label = classifier.mask[predicted.item()]
        # set_trace()
        return label
        # _, predicted = torch.max(outputs.data, 1)
        predicted = classifier.mask[predicted[0].item()]
        return predicted

    def reshape(self, for_save=False):
        flat_tensor = self.pc.view(-1)
        num_points = 2048
        num_dims = 3

        if for_save:
            reshaped = flat_tensor.view(num_points, num_dims)
            reshaped = reshaped.t()
            reshaped = reshaped.view(1, num_dims, num_points)
            return reshaped
        else:
            reshaped = flat_tensor.view(num_points, num_dims)
            reshaped = reshaped.t()
            reshaped = reshaped.view(1, num_dims, num_points)
            return reshaped
            set_trace()
            return flat_tensor.view(num_points, num_dims).unsqueeze(0)
        
    
    def save(self, name):
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, name + '.npy')
        np.save(file_path, self.reshape(for_save=True).numpy())
        # print(f'Point cloud saved to {file_path}')














if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_base_800k.pt')
    parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/cl_2_max_100.pt')
    parser.add_argument('--categories', type=str_list, default=['airplane','chair'])
    parser.add_argument('--categories_classifier', type=str_list, default=['airplane','chair'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
    parser.add_argument('--ret_traj', type=eval, default=False, choices=[True, False])
    parser.add_argument('--gradient_scale', type=float, default=1) # noted "s" usually
    parser.add_argument('--desired_class', type=int, default=0) # This is airplane # [y] * batch_size converted to tensor or somehting like that? for multi-class.. idk
    parser.add_argument('--num_steps', type=int, default=100)
    args = parser.parse_args()
    seed_all(args.seed) # adding seed for consistency


    y = torch.ones(args.batch_size, dtype=torch.long).to(args.device) * args.desired_class
    # t = args.num_steps
    classifier = ClassifierWithGradients(args)
    diffusion = DiffusionWithGradients(args, classifier)

    # Adding somewhat independent classifier for better judgement on what a point cloud is. 
    args.ckpt_classifier = './relevant_checkpoints/cl_all_mean_100.pt'
    args.categories_classifier = ['all']
    less_biased_classifier = ClassifierWithGradients(args)

    labels_list = []

    for idx in tqdm(range(1)):
        pc_batch = diffusion.sample(y=y)
        # labels = [pc_batch[idx].classify(less_biased_classifier) for idx in range(len(pc_batch))]

        pc = pc_batch[0]
        set_trace()
        # pc.save(f'testing')
        # label = labels[0]
        # set_trace()
        # pc.save(f'mean_{args.gradient_scale}_{idx}_{labels[0]}')
        # pc_batch[0].save(f'{args.gradient_scale}_{idx}_{labels[0]}')
        # print(labels)
        #labels_list.extend(labels)
    # print(len(labels_list))

        

        # label = pc_batch.batch_list[0].classify(classifier)
        # pc_batch.animate(f'gradient_{idx}_{label}')


#    print(f'Ratio of airplanes vs. baseline ~38%: {(labels_list.count('airplane') / len(labels_list) * 100):.1f}')
