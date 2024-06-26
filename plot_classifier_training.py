import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os

# Define the file paths
rel_path = 'plots'
file_paths = ["classifier_training/cl_2_max.csv", 
              "classifier_training/cl_all_max.csv", 
              "classifier_training/cl_all_mean.csv"]
file_paths = [os.path.join(rel_path, file_path) for file_path in file_paths]

# Define labels for the legends
labels = ["2 classes, Max Pooling", "55 classes, Max Pooling", "55 classes, Mean Pooling"]

# Create a list to hold the traces
traces = []

# Loop through the files and create traces for the plot
for file_path, label in zip(file_paths, labels):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Create a trace
    trace = go.Scatter(x=data["Step"], y=data["Value"], mode='lines', name=label)
    traces.append(trace)

# Create the layout
layout = go.Layout(
    title="Classifier Validation Accuracies",
    xaxis=dict(title='Epochs'),
    yaxis=dict(title='Accuracy'),
    legend=dict(x=0.73, y=0.1),  # Move the legend to the bottom right
    margin=dict(l=50, r=20, t=50, b=50),  # Reduce the margins
    width=800,  # Set the width of the plot
    height=300   # Set the height of the plot to make it flatter
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Save the plot as an image file
pio.write_image(fig, file='classifier_training_accuracies.png', scale=3)
