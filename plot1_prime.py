import os
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Function to load and process point cloud
def load_point_cloud(file_path):
    point_cloud = np.load(file_path)
    return point_cloud

# Function to rotate the point cloud around the x-axis
def rotate_point_cloud(point_cloud, angle):
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    rotated_point_cloud = np.dot(rotation_matrix, point_cloud.reshape(3, -1))
    return rotated_point_cloud.reshape(point_cloud.shape)

# Function to visualize point cloud
def visualize_point_cloud(point_cloud, title, camera_scaling, height_adjustment):
    x = point_cloud[0, 0, :]
    y = point_cloud[0, 1, :]
    z = point_cloud[0, 2, :]

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=camera_scaling, y=camera_scaling + height_adjustment, z=-camera_scaling)
    )

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,
            colorscale='Redor',
            opacity=1
        )
    )

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    z_range = max(z) - min(z)
    max_range = max(x_range, y_range, z_range)
    aspect_ratio = {
        'x': x_range / max_range,
        'y': y_range / max_range,
        'z': z_range / max_range
    }

    layout = go.Layout(
        width=250,  # Adjusted width
        height=150,  # Adjusted height
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
            camera=camera,
            aspectmode='manual',
            aspectratio=aspect_ratio,
            bgcolor='rgba(255,255,255,1)',
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False
        ),
        title='',
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)'
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig

# Function to create and save grid of plots
def create_and_save_grid(main_folder, output_file, camera_scaling=0.7, height_adjustment=-0.5):
    subfolders = ['cl1', 'cl2', 'cl3']
    files = ['s_00000.npy', 's_00500.npy', 's_02000.npy', 's_05000.npy', 's_10000.npy']

    figs = {subfolder: [] for subfolder in subfolders}
    for subfolder in subfolders:
        for file in files:
            file_path = os.path.join(main_folder, subfolder, file)
            point_cloud = load_point_cloud(file_path)
            rotated_point_cloud = rotate_point_cloud(point_cloud, 0)  # Rotate -90 degrees around the x-axis to correct flipping
            fig = visualize_point_cloud(rotated_point_cloud, file, camera_scaling, height_adjustment)
            figs[subfolder].append(fig)

    # Create a 5x3 grid layout for the plots using make_subplots
    fig = make_subplots(rows=5, cols=3, specs=[[{'type': 'scatter3d'}]*3]*5,
                        horizontal_spacing=0.005, vertical_spacing=0.005)  # Decrease spacing

    # Add each plot to the corresponding subplot
    for col, subfolder in enumerate(subfolders, start=1):
        for row in range(1, 6):
            trace = figs[subfolder][row-1].data[0]
            fig.add_trace(trace, row=row, col=col)
            
            # Update the scene for each subplot to remove the background and axes
            scene_id = f'scene{(row-1)*3 + col}'
            fig.update_layout(
                **{scene_id: dict(
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
                    zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='', visible=False),
                    camera=dict(
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=camera_scaling, y=camera_scaling + height_adjustment, z=-camera_scaling)
                    ),
                    bgcolor='rgba(255,255,255,1)',
                    xaxis_showspikes=False,
                    yaxis_showspikes=False,
                    zaxis_showspikes=False
                )}
            )

    fig.update_layout(
        height=800,  # Adjust height to fit tighter
        width=1100,  # Adjust width to fit tighter
        showlegend=False
    )

    pio.write_image(fig, output_file, scale=3)  # Save with high DPI

# Main function to execute the script
if __name__ == '__main__':
    main_folder = './pcs'
    output_file = 'plot1_airplane.png'
    create_and_save_grid(main_folder, output_file)
