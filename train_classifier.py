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
# from pointnet.z_PointNetCls import *
from models.classifier import *
from models.autoencoder import *
# Arguments
from models.DGCNN import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--num_classes', type=int, default=2)  # Assuming 2 classes: airplane and chair
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])
# parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150 * 1000)
parser.add_argument('--sched_end_epoch', type=int, default=300 * 1000)

# Training
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_pointnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=np.inf)
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
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
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')
train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
    transform=transform,
)
val_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='val',
    scale_mode=args.scale_mode,
    transform=transform,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = PointNet(k=args.num_classes, feature_transform=True).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
# else:
#     model = PointNet(k=args.num_classes, feature_transform=True).to(args.device)
else:
    model = DGCNN(k=20, num_classes=args.num_classes)

logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Criterion
criterion = torch.nn.CrossEntropyLoss()
# Define a label mapping
if args.num_classes == 2: 
    label_mapping = {'airplane': 0, 'chair': 1}
    # label_mapping = {args.categories[i] for i in range(len(args.categories))}
elif args.num_classes == 55:
    cate_all = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock', 'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone', 'birdhouse', 'bookshelf']
    label_mapping = {cate_all[i]: i for i in range(len(cate_all))}
else:
    raise ValueError('Currently only accepts 2 or 55 classes. Choose one of those or be a man and implement it yourself.')


def train(it):
    # Load data
    batch = next(train_iter)
    x = torch.tensor(batch['pointcloud']).to(args.device)
    x = x.transpose(1, 2)
    # Convert string labels to numeric labels
    # print(f'Unique labels: {type(batch['cate'])}')
    labels = [label_mapping[label] for label in batch['cate']]
    labels = torch.tensor(labels).to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    outputs, trans_feat = model(x)
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


# Train, validate 

def validate_loss(it):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            if args.num_val_batches > 0 and i >= args.num_val_batches:
                break
            x = torch.tensor(batch['pointcloud']).float().to(args.device)
            
            # Verify input shape before transposing
            assert x.shape[1] == 2048 and x.shape[2] == 3, "Unexpected input shape, expected [batch_size, num_points, num_channels]"
            
            x = x.transpose(1, 2)  # Transpose to [batch_size, num_channels, num_points]

            # Verify input shape after transposing
            assert x.shape[1] == 3 and x.shape[2] == 2048, "Unexpected input shape, expected [batch_size, num_channels, num_points]"
            
            # Convert string labels to numeric labels
            labels = [label_mapping[label] for label in batch['cate']]
            labels = torch.tensor(labels).to(args.device)
            
            outputs, _ = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info('[Val] Iter %04d | Accuracy %.2f%%' % (it, accuracy))
    writer.add_scalar('val/accuracy', accuracy, it)
    writer.flush()
    return accuracy



def validate_inspect(it):
    model.eval()
    pointclouds = []
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = torch.tensor(batch['pointcloud']).float().to(args.device)
        
        # Verify input shape before transposing
        assert x.shape[1] == 2048 and x.shape[2] == 3, "Unexpected input shape, expected [batch_size, num_points, num_channels]"
        
        x = x.transpose(1, 2)  # Transpose to [batch_size, num_channels, num_points]

        # Verify input shape after transposing
        assert x.shape[1] == 3 and x.shape[2] == 2048, "Unexpected input shape, expected [batch_size, num_channels, num_points]"

        with torch.no_grad():
            outputs, _ = model(x)
            _, predicted = torch.max(outputs.data, 1)

        # Append original point clouds (transposed back to original shape) for inspection
        pointclouds.append(x.transpose(1, 2)[:args.num_inspect_pointclouds])

        if i >= args.num_inspect_batches:
            break   # Inspect only specified number of batches

    pointclouds = torch.cat(pointclouds, dim=0)

    # Ensure pointclouds are of shape [batch_size, num_points, 3]
    assert pointclouds.dim() == 3 and pointclouds.size(2) == 3, "Expected pointclouds to be of shape [batch_size, num_points, 3]"

    writer.add_mesh('val/pointcloud', pointclouds, global_step=it)
    writer.flush()


# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                accuracy = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, accuracy, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
