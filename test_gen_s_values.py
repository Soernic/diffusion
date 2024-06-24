import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

from pc_sampler import *

import pdb
from pc_with_gradients import *

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
parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_base_800k.pt')
parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/cl_all_max_100.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--classifier_categories', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=1)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()


# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
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
# logger.info('Loading model...')
# if ckpt['args'].model == 'gaussian':
#     model = GaussianVAE(ckpt['args']).to(args.device)
# elif ckpt['args'].model == 'flow':
#     model = FlowVAE(ckpt['args']).to(args.device)
# logger.info(repr(model))
# # if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
# model.load_state_dict(ckpt['state_dict'])

#
##### USE THIS FOR OUR SAMPLES
# Generate Point Clouds
# gen_pcs = []
# gen_pcs1 = []
# #while len(gen_pcs1) < len(ref_pcs):
# for i in tqdm(range(0, math.ceil(len(2000) / args.batch_size)), 'Generate'):
#     with torch.no_grad():
#         z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
#         x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
#         label = PointCloud(x).classify(Classifier(args))
#         if label == 'airplane': gen_pcs1.append(x.detach().cpu())
#         #pdb.set_trace()
#         if np.random.randn() > 2: print(len(gen_pcs))
#     # if len(gen_pcs) >= len(ref_pcs):
#     #     break
#########


logger.info('Loading model...')
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_base_800k.pt')
parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/cl_all_max_100.pt')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--categories_classifier', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_batches', type=int, default=4) # pcs generated = num_batches x batch_size
parser.add_argument('--ret_traj', type=eval, default=False, choices=[True, False])
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--desired_class', type=int, default=0) # This is airplane # [y] * batch_size converted to tensor or somehting like that? for multi-class.. idk

args = parser.parse_args()
seed_all(args.seed) # adding seed for consistency



# somehow define what we give the classifier
    # t = args.num_steps

ref_pcs = []
for i, data in enumerate(test_dset):
    if i >= args.test_size:
        break
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)

svals = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
#loop through svals, generate the point clouds and append them to gen_pcs, then compute all metrics and save them

for _, s_val in enumerate(svals):
    args.gradient_scale = s_val
    print(s_val)
    classifier = ClassifierWithGradients(args)
    diffusion = DiffusionWithGradients(args, classifier)
    gen_pcs = []
    #randomly 0 or 1 with probability 0.5 and legth of test_dset

    with torch.no_grad():
        y = torch.ones(args.batch_size, dtype=torch.long).to(args.device) * args.desired_class
        gen_pcs = diffusion.sample(y=y)
        #pdb.set_trace()

    # gen_pcs = torch.cat(gen_pcs, dim=0)[:400]
    # if args.normalize is not None:
    #     gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)
    # Compute metrics
    #pdb.set_trace()
    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
        results = {k:v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    for k, v in results.items():
        logger.info('%s: %.12f' % (k, v))


