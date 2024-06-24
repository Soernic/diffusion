import os
import argparse

from utils.dataset import *
from utils.misc import *
from utils.data import *

parser = argparse.ArgumentParser()
parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])
parser.add_argument('--save_dir', type=str, default='./pcs/data_pcs')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
args = parser.parse_args()

pcs_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.normalize,
)
pcs = []
for i, data in enumerate(pcs_dset):
    pcs.append(data['pointcloud'].unsqueeze(0))
pcs = torch.cat(pcs, dim=0)
for _ in range(10):
    i = np.random.randint(1,9040)
    pc = pcs[i]
    pc = pc.transpose(-2, 1) # (1, 3, 2048)
    pc = np.expand_dims(pc, axis = 0)
    print(pc.shape)
    os.makedirs(args.save_dir, exist_ok=True)
    file_path = os.path.join(f'{args.save_dir}/{i}.npy')
    np.save(file_path, pc)