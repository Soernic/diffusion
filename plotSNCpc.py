import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

from utils.dataset import *
from utils.misc import *
from utils.data import *

# Assume ShapeNetCore and necessary imports are already done
# This code assumes that 'train_dset' has already been created as per your provided snippet
train_dset = ShapeNetCore(
    path="./data/shapenet.hdf5",
    cates=['airplane'],
    split='test',
    scale_mode='shape_unit'
)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot a point cloud with specified viewing angles and equal axis scales
def plot_point_cloud(point_cloud, title="Point Cloud", elev=30, azim=30, roll=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    # Plot the points
    ax.scatter(x, y, z, c='r', marker='o', alpha=0.5)
    ax.set_title(title)
    
    # Compute the limits
    max_range = np.max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]) / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    
    # Set the limits to be equal
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim, roll=roll)
    
    # Remove the axes
    ax.set_axis_off()
    
    plt.show()

# Assume ShapeNetCore and necessary imports are already done
# This code assumes that 'train_dset' has already been created as per your provided snippet
# Retrieve a sample point cloud from the dataset
# Here, we assume that the dataset has been correctly filtered to include only the desired categories (airplane or chair)
sample_point_cloud = train_dset[3]['pointcloud']  # Adjust index as necessary

# Plot the sample point cloud with specified viewing angles and equal axis scales
plot_point_cloud(sample_point_cloud, title="Sample Chair", elev=-35, azim=50)

