import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from evaluation import EMD_CD
from models.classifier_time_aware import *
from models.autoencoder import *

from pdb import set_trace
from pc_sampler import PointCloud

# Arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--num_classes', type=int, default=55)
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=5000)
parser.add_argument('--sched_end_epoch', type=int, default=10000)

# Noising the data
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)

# Training
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_pointnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=np.inf)
parser.add_argument('--val_freq', type=float, default=200)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--noise_limit', type=float, default=0.6)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='PointNet_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = RandomRotate(180, ['pointcloud'], axis=1) if args.rotate else None
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')
train_dset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='train', scale_mode=args.scale_mode, transform=transform)
val_dset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='val', scale_mode=args.scale_mode, transform=transform)
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=0))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

# Model
logger.info('Building model...')
model = PointNet(k=args.num_classes, feature_transform=True).to(args.device)
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'])
# logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch, start_lr=args.lr, end_lr=args.end_lr)

# Criterion
criterion = torch.nn.CrossEntropyLoss()
cate_all = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock', 'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone', 'birdhouse', 'bookshelf']
cate_all = ['airplane', 'chair']
label_mapping = {cate_all[i]: i for i in range(len(cate_all))}

def get_alpha_alpha_bars(args):
    """
    Uses variance schedule and number of steps to get alphas and alpha_bars
    This way we can sample noisy pcs efficiently
    """
    beta_1 = args.beta_1
    beta_T = args.beta_T
    num_steps = args.num_steps

    betas = torch.linspace(beta_1, beta_T, steps=num_steps)
    betas = torch.cat([torch.zeros([1]), betas], dim=0)
    alphas = 1 - betas
    log_alphas = torch.log(alphas)
    for i in range(1, log_alphas.size(0)):
        log_alphas[i] += log_alphas[i - 1]
    alpha_bars = log_alphas.exp()
    return alphas, alpha_bars

alphas, alpha_bars = get_alpha_alpha_bars(args)

def noise_batch(x: torch.Tensor, t: int, alphas: torch.Tensor, alpha_bars: torch.Tensor):
    """
    Takes a batch and transforms it to its noisy version at time step t
    """
    
    # TODO: Sanity check here.. Is this actually correct?

    sqrt_alpha_bar_t = torch.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bars[t])
    epsilon = torch.randn_like(x)
    return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * epsilon

def add_time_step_feature(x: torch.Tensor, t: int):
    """
    Adds the time step as a feature to each point in the point cloud
    """

    # TODO: Sanity check here.. Is this actually correct?

    t_feature = torch.full((x.size(0), x.size(1), 1), t, device=x.device)
    return torch.cat((x, t_feature), dim=2)

def train(it):
    # Load data
    batch = next(train_iter)
    x = torch.tensor(batch['pointcloud']).to(args.device)
    x = x.transpose(1, 2)
    labels = [label_mapping[label] for label in batch['cate']]
    labels = torch.tensor(labels).to(args.device)

    # Add noise
    noise_limit = args.noise_limit * args.num_steps + 1 # Don't train on noisier stuff than this
    t = np.random.randint(0, int(noise_limit))

    # ## TODO: Temp code - remove afterwards
    # for t in np.arange(0, 100, 10):
    #     x_noisy = noise_batch(x, t, alphas, alpha_bars)
    #     PointCloud(x_noisy[0].unsqueeze(0).permute(0, 2, 1).cpu()).save(str(t))
    # ## TODO: Temp code - remove afterwards

    t = 0
    x_noisy = noise_batch(x, t, alphas, alpha_bars)



    # Add time step feature
    x_noisy = add_time_step_feature(x_noisy.transpose(1, 2), t).transpose(1, 2)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    outputs, trans_feat = model(x_noisy)
    loss = criterion(outputs, labels)
    if model.feature_transform:
        loss += 0.001 * model.feature_transform_regularizer(trans_feat)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    if int(it) % 50 == 0:
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):
    model.eval()
    correct = 0
    total = 0
    num_iterations = 1  # Set this to 3 to effectively 3x the validation iterations (.. only if sampling t from uniformt)
    with torch.no_grad():
        for _ in range(num_iterations):
            for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
                if args.num_val_batches > 0 and i >= args.num_val_batches:
                    break
                x = torch.tensor(batch['pointcloud']).float().to(args.device)
                x = x.transpose(1, 2)
                labels = [label_mapping[label] for label in batch['cate']]
                labels = torch.tensor(labels).to(args.device)

                # Add noise
                noise_limit = args.noise_limit * args.num_steps + 1
                t = np.random.randint(0, int(noise_limit))
                t = 1
                x_noisy = noise_batch(x, t, alphas, alpha_bars)

                # Add time step feature
                x_noisy = add_time_step_feature(x_noisy.transpose(1, 2), t).transpose(1, 2)

                outputs, _ = model(x_noisy)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info('[Val] Iter %04d | Accuracy %.2f%%' % (it, accuracy))
    writer.add_scalar('val/accuracy', accuracy, it)
    writer.flush()
    return accuracy


# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                accuracy = validate_loss(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, accuracy, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
