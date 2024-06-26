import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Load the data
# file_path = 'plots/plot5/100_steps_16_64.csv'
file_path = 'plots/plot5/100_steps_16_32.csv'
df = pd.read_csv(file_path)

# Extract data from the dataframe
x = df['s_vals']
y1 = df[df.columns[1]]
y2 = df[df.columns[2]]
y3 = df[df.columns[3]]

# Define the traces for the plot with a more appealing color scheme
trace1 = go.Scatter(
    x=x,
    y=y1,
    mode='lines',
    name='Max Pooling, 2 classes, 100 steps',
    line=dict(color='#1f77b4')  # Updated color for line 1
)

trace2 = go.Scatter(
    x=x,
    y=y2,
    mode='lines',
    name='Max Pooling, 55 classes, 100 steps',
    line=dict(color='#ff7f0e')  # Updated color for line 2
)

trace3 = go.Scatter(
    x=x,
    y=y3,
    mode='lines',
    name='Mean Pooling, 55 classes, 100 steps',
    line=dict(color='#2ca02c')  # Updated color for line 3
)

# Define the layout
layout = go.Layout(
    title='Gradient scale impact on classifier\'s self-assessed "chair" generation percentage',
    xaxis=dict(
        title='Gradient scale (log2)',
        type='log',
        dtick=1,
        exponentformat='power',
        showgrid=True,  # Add grid lines
        gridcolor='LightGray',  # Grid color
        gridwidth=0.5  # Grid line width
    ),
    yaxis=dict(
        title='Percentage of "chair" classifications (self-assessed)',
        tickformat='.2%',
        showgrid=True,  # Add grid lines
        gridcolor='LightGray',  # Grid color
        gridwidth=0.5  # Grid line width
    ),
    legend=dict(
        x=0,
        y=0,  # Changed y to 0 to move legend to the lower left corner
        bgcolor='rgba(255,255,255,0.5)'
    ),
    plot_bgcolor='rgba(255,255,255,1)',
    paper_bgcolor='rgba(255,255,255,1)'
)

# Create the figure
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

# Save the figure
output_file = 'plot5.png'
pio.write_image(fig, output_file, scale=3)  # Save with high DPI
