import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pdb import set_trace

# Function to load and process a point cloud from a file
def load_point_cloud(file_path):
    return np.load(file_path)

# Function to rotate the point cloud around a specified axis by a given angle
def rotate_point_cloud(point_cloud, axis, angle):
     #print(point_cloud.shape)
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
    
    return np.dot(point_cloud, rotation_matrix.T)

# Function to update the plot for each frame
def update(frame, point_clouds, scatter, ax):
    point_cloud = point_clouds[frame]
    scatter._offsets3d = (point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    
    # Rotate the view
    ax.view_init(elev=30, azim=45 + frame)
    
    return scatter,


def animate(path, name):
    # Load and rotate point clouds
    angle = np.pi / 2  # 90 degrees
    pc_batch = np.load(path)
    point_clouds = np.split(pc_batch, indices_or_sections=pc_batch.shape[0])
    point_clouds = [rotate_point_cloud(pc.squeeze(0).squeeze(1), 'x', angle) for pc in point_clouds]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2])

    # Set up the axes limits
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    # Create an animation
    ani = FuncAnimation(fig, update, frames=len(point_clouds), fargs=(point_clouds, scatter, ax), interval=100)


    # Save the animation as a GIF file
    ani.save(f'{name}.gif', writer='pillow', fps=33)

    # plt.show()

names = ['0_chair', '1_chair', '2_airplane', '3_airplane', '4_chair']
paths = [f'./trajectories/test_{name}.npy' for name in names]
# paths = ['./trajectories/test_0_chair.npy', './trajectories/test_1_chair.npy', './trajectories/test_2_airplane.npy', './trajectories/test_3_airplane.npy', './trajectories/test_4_chair.npy']
for path, name in zip(paths, names):
    animate(path, name)
