import streamlit as st
import numpy as np
import plotly.graph_objs as go
import os

# Function to load and process point cloud
def load_point_cloud(file_path):
    point_cloud = np.load(file_path)
    return point_cloud

# Function to visualize point cloud
def visualize_point_cloud(point_cloud):
    # Extract x, y, z coordinates
    x = point_cloud[0, 0, :]
    y = point_cloud[0, 1, :]
    z = point_cloud[0, 2, :]

    # Define camera settings
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=0.8, z=-1.3)  # Adjust these values to set the desired angle
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

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=camera
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig


# Streamlit UI
st.title("Point Cloud Viewer")
st.write('test')

# Get list of .npy files in the directory
directory = './pcs'
files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

# File selection dropdown
selected_file = st.selectbox("Select a point cloud file", files)

if selected_file:
    file_path = os.path.join(directory, selected_file)
    point_cloud = load_point_cloud(file_path)
    fig = visualize_point_cloud(point_cloud)
    st.plotly_chart(fig)
