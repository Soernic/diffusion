import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
from pdb import set_trace

# Function to load and process a point cloud from a file
def load_point_cloud(file_path):
    return np.load(file_path)

# Function to rotate the point cloud around a specified axis by a given angle
def rotate_point_cloud(point_cloud, axis, angle):
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

def animate(path, name):
    # Load and rotate point clouds
    angle = np.pi / 2  # 90 degrees
    pc_batch = np.load(path)
    point_clouds = np.split(pc_batch, indices_or_sections=pc_batch.shape[0])
    point_clouds = [rotate_point_cloud(pc.squeeze(0).squeeze(1), 'x', angle) for pc in point_clouds]

    # Create a plotly figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Define the initial plot
    scatter = go.Scatter3d(
        x=point_clouds[0][:, 0],
        y=point_clouds[0][:, 1],
        z=point_clouds[0][:, 2],
        mode='markers',
        marker=dict(size=5, color=point_clouds[0][:, 0], colorscale='Viridis', opacity=0.8)
    )

    fig.add_trace(scatter)

    # Update frames for the animation
    frames = [go.Frame(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=pc[:, 2],
        mode='markers',
        marker=dict(size=5, color=pc[:, 0], colorscale='Viridis', opacity=0.8)
    )]) for pc in point_clouds]

    fig.frames = frames

    # Animation settings
    fig.update_layout(
        updatemenus=[{
            'buttons': [{'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'}],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        scene=dict(
            xaxis=dict(range=[-3, 3], backgroundcolor='rgb(20, 24, 54)', gridcolor='rgb(128, 128, 128)', zerolinecolor='rgb(255, 255, 255)'),
            yaxis=dict(range=[-3, 3], backgroundcolor='rgb(20, 24, 54)', gridcolor='rgb(128, 128, 128)', zerolinecolor='rgb(255, 255, 255)'),
            zaxis=dict(range=[-3, 3], backgroundcolor='rgb(20, 24, 54)', gridcolor='rgb(128, 128, 128)', zerolinecolor='rgb(255, 255, 255)'),
            aspectmode='cube'
        )
    )

    # Save the animation as an HTML file
    fig.write_html(f'{name}.html')

    # fig.show()

names = ['0_chair', '1_chair', '2_airplane', '3_airplane', '4_chair']
paths = [f'./trajectories/test_{name}.npy' for name in names]
for path, name in zip(paths, names):
    animate(path, name)
