import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to load a point cloud from a file
def load_point_cloud(file_path):
    return np.load(file_path)

# Function to update the plot for each frame
def update(frame, point_clouds, scatter):
    point_cloud = point_clouds[frame]
    scatter._offsets3d = (point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    return scatter,

# Load point clouds
pc_batch = np.load('./trajectories/test.npy')
point_clouds = np.split(pc_batch, indices_or_sections=pc_batch.shape[0])
point_clouds = [pc.squeeze(0).squeeze(1) for pc in point_clouds]
print(point_clouds[0].shape)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2])

# Set up the axes limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Create an animation
ani = FuncAnimation(fig, update, frames=len(point_clouds), fargs=(point_clouds, scatter), interval=100)

# Save the animation as a GIF file
ani.save('point_cloud_animation.gif', writer='pillow', fps=10)

# plt.show()
