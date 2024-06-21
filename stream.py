import streamlit as st
import numpy as np
import plotly.graph_objs as go
import os

# Function to load and process point cloud
def load_point_cloud(file_path):
    point_cloud = np.load(file_path)
    return point_cloud

# Function to visualize point cloud
def visualize_point_cloud(point_cloud, title):
    # Extract x, y, z coordinates
    x = point_cloud[0, 0, :]
    y = point_cloud[0, 1, :]
    z = point_cloud[0, 2, :]

    # Define camera settings for a closer view
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.75, y=0.4, z=-0.65)  # Adjust these values to set the desired zoom level
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
            colorscale='Blues',   # Choose a colorscale
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
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            camera=camera,
            aspectmode='manual',  # Set the aspect mode to manual
            aspectratio=aspect_ratio  # Set the computed aspect ratio
        ),
        title=title
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig


# Streamlit UI
st.title("Point Cloud Viewer")


# Get list of .npy files in the directory
directory = './pcs'
files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

# File selection with multiselect
selected_files = st.multiselect("Select point cloud files", files)

if selected_files:
    for selected_file in selected_files:
        file_path = os.path.join(directory, selected_file)
        point_cloud = load_point_cloud(file_path)
        fig = visualize_point_cloud(point_cloud, selected_file)
        st.plotly_chart(fig)
