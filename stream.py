import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import os
from PIL import Image

# Function to load and process point cloud
def load_point_cloud(file_path):
    point_cloud = np.load(file_path)
    return point_cloud

# Function to visualize point cloud
def visualize_point_cloud(point_cloud, title, camera_scaling, height_adjustment):
    # Extract x, y, z coordinates
    x = point_cloud[0, 0, :]
    y = point_cloud[0, 1, :]
    z = point_cloud[0, 2, :]

    # Define camera settings for a further view
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=camera_scaling, y=camera_scaling + height_adjustment, z=-camera_scaling)  # Adjust camera scaling and height
    )

    # Create a 3D scatter plot using Plotly
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,                 # Increased point size
            color=y,                # Color by z value
            colorscale='Redor',     # Choose a colorscale
            opacity=1
        )
    )

    # Compute the aspect ratio
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
        width=600,  # Set the width of the figure
        height=450,  # Set the height of the figure
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                title=''
            ),
            camera=camera,
            aspectmode='manual',  # Set the aspect mode to manual
            aspectratio=aspect_ratio,  # Set the computed aspect ratio
            bgcolor='rgba(255,255,255,1)'  # Set background color to white
        ),
        title='',  # Explicitly set the title to an empty string
        paper_bgcolor='rgba(255,255,255,1)',  # Set the plot background to white
        plot_bgcolor='rgba(255,255,255,1)'  # Set the paper background to white
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig

# Function to save plot as PNG in the selected subfolder
def save_plot_as_png(fig, filename, folder):
    filepath = os.path.join(folder, filename)
    pio.write_image(fig, filepath)
    return filepath

# Streamlit UI
st.title("Point Cloud Viewer")
st.write('Select subfolder of "pcs"')

# Main folder path
main_folder = './pcs'

# List all subfolders in the main folder
subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

# Streamlit selectbox for subfolder selection
selected_subfolder = st.selectbox('Select a subfolder', subfolders)

# Path to the selected subfolder
subfolder_path = os.path.join(main_folder, selected_subfolder)

# Slider for camera scaling
camera_scaling = st.slider('Camera Scaling', min_value=0.1, max_value=5.0, value=1.2, step=0.1)

# Slider for height adjustment
height_adjustment = st.slider('Height Adjustment', min_value=-5.0, max_value=5.0, value=-0.9, step=0.1)

# Get list of .npy files in the selected subfolder
files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.npy')])

# File selection with multiselect
selected_files = st.multiselect("Select point cloud files", files)

figs = []
saved_files = []

if selected_files:
    for selected_file in selected_files:
        file_path = os.path.join(subfolder_path, selected_file)
        point_cloud = load_point_cloud(file_path)
        fig = visualize_point_cloud(point_cloud, selected_file, camera_scaling, height_adjustment)
        st.plotly_chart(fig)
        figs.append((fig, selected_file))  # Collect figures and filenames

# Button to save all plots as PNG
if st.button('Save all plots as PNG'):
    for fig, filename in figs:
        clean_filename = filename.replace('.npy', '')  # Remove .npy extension
        saved_path = save_plot_as_png(fig, f"{clean_filename}.png", subfolder_path)
        saved_files.append(saved_path)
    st.success('Plots saved successfully!')

# Display saved images as a 3x2 gallery
if saved_files:
    st.write('Downloaded Images:')
    cols = st.columns(3)  # Create 3 columns for the gallery
    for i, img_file in enumerate(saved_files):
        img = Image.open(img_file)
        cols[i % 3].image(img, use_column_width=True)
