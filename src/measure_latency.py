import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from utils import read_key_data
from model import ModelFactory, train_model
import argparse
import time
from train import get_device


@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test

device = get_device()

model = ModelFactory.create_model('resnet', num_classes=30, device=device)
model.load_state_dict(torch.load('models/best_ResNet_model.pth', map_location=device))

shallowModel = ModelFactory.create_model('shallow_cnn', num_classes=30, device=device)
shallowModel.load_state_dict(torch.load('models/best_ShallowCNN_model.pth', map_location=device))

dummy_input = torch.randn(1, 1, 32, 32)

resnet_latency = measure_latency(model, dummy_input)
shallow_latency = measure_latency(shallowModel, dummy_input)

print(f"Resnet latency is: {resnet_latency} and Shallow CNN latency is {shallow_latency}")

import matplotlib.pyplot as plt
# Convert latencies to milliseconds
resnet_latency_ms = resnet_latency * 1000
shallow_cnn_latency_ms = shallow_latency * 1000

# Data for plotting
models = ['ResNet', 'Shallow CNN']
latencies = [resnet_latency_ms, shallow_cnn_latency_ms]

# Create bar plot
plt.bar(models, latencies, color=['blue', 'orange'])

# Add labels and title
plt.ylabel('Latency (ms)')
plt.title('Model Latency Comparison (without pruning)')

# Show values on top of bars
for i, latency in enumerate(latencies):
    plt.text(i, latency + 0.1, f'{latency:.2f} ms', ha='center')
plt.savefig('latency_comparison.svg', format='svg')
# Display plot
plt.show()
