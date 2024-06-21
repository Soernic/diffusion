import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Load the point cloud data
data = np.load('./pcs/mean_1_0_chair.npy')  # replace this

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
gradient = data[:, 3]

# Filter out points with zero gradients
non_zero_indices = gradient != 0
x_non_zero = x[non_zero_indices]
y_non_zero = y[non_zero_indices]
z_non_zero = z[non_zero_indices]
grad_non_zero = gradient[non_zero_indices]

# Create a custom colormap that goes from blue (low) to red (high)
colors = [(0, 0, 1), (1, 0, 0)]  # B -> R
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'blue_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)



# # Plot both the original and the filtered point cloud side by side
fig = plt.figure(figsize=(12, 6))

# # Plot original point cloud
# ax1 = fig.add_subplot(121, projection='3d')
# sc1 = ax1.scatter(x, y, z, c=gradient, cmap=cm, marker='o')  # Set alpha for all points
# cbar1 = plt.colorbar(sc1, ax=ax1)
# cbar1.set_label('Gradient')
# ax1.set_title('Original Point Cloud')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# Plot filtered point cloud (only non-zero gradients)
ax2 = fig.add_subplot(121, projection='3d')
# Set different opacity for zero gradient points
sc2 = ax2.scatter(x_non_zero, y_non_zero, z_non_zero, c=grad_non_zero, cmap=cm, marker='o')
cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label('Gradient')
ax2.set_title('Filtered Point Cloud (Non-Zero Gradients)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

higher_indices = grad_non_zero > 0.5
x_higher = x_non_zero[higher_indices]
y_higher = y_non_zero[higher_indices]
z_higher = z_non_zero[higher_indices]
grad_higher = grad_non_zero[higher_indices]


ax3 = fig.add_subplot(122, projection='3d')
# Set different opacity for zero gradient points
sc3 = ax3.scatter(x_higher, y_higher, z_higher, c=grad_higher, cmap=cm, marker='o')
cbar3 = plt.colorbar(sc3, ax=ax3)
cbar3.set_label('Gradient')
ax3.set_title('Filtered Point Cloud (Gradients > 0.5)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')



plt.show()
