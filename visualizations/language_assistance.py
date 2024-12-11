import plotly.graph_objects as go
from plotly import express as px
import time


# https://github.com/plotly/plotly.py/issues/3469
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("random.pdf")
time.sleep(1)


# Data
methods = ['Tactile Only', 'GPT2-Assisted', 'Ngram-Assisted']
accuracies = [73.69, 77.10, 80.51]

# Create the bar chart
fig = go.Figure(data=[
    go.Bar(
        x=methods,
        y=accuracies,
        text=[f'{acc:.2f}%' for acc in accuracies],
        textposition='auto',
        textfont=dict(size=32, color='white'),  # White text color added here
        marker_color=['#1f77b4', '#2ca02c', '#ff7f0e']
    )
])

# Update layout with larger text
fig.update_layout(
    # title={
    #     'text': 'Language Assisted Accuracy',
    #     'y':0.95,
    #     'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top',
    #     'font': dict(size=42)
    # },
    yaxis_title='Accuracy (%)',
    yaxis_range=[70, 85],
    showlegend=False,
    template='plotly_white',
    width=1024,
    height=512,
    # Increase font size for axis titles and tick labels
    xaxis=dict(
        tickfont=dict(size=32),
        titlefont=dict(size=32)
    ),
    yaxis=dict(
        tickfont=dict(size=42),
        titlefont=dict(size=42)
    )
)

# Show grid only for y-axis
fig.update_yaxes(showgrid=True)
fig.update_xaxes(showgrid=False)

# adjust margin
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

# Save the figure as JPG
fig.write_image("accuracy_comparison.pdf")