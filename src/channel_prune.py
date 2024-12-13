import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs
from model import ModelFactory
from model_profile import get_model_size, MiB
from utils import read_key_data
from train import TactileDataset, evaluate, get_device, train_model
from sklearn.model_selection import  train_test_split

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        ##################### YOUR CODE STARTS HERE #####################
        importance = torch.norm(channel_weight)
        ##################### YOUR CODE ENDS HERE #####################
        importances.append(importance.view(1))
    return torch.cat(importances)

def get_next_module(model, layer_name):
    """
    Get the next module after the given layer name in the model.
    
    Parameters:
    - model (nn.Module): The model to search through.
    - layer_name (str): The name of the layer.
    
    Returns:
    - next_module (nn.Module): The next module after the given layer.
    """
    layers = list(model.named_modules())
    for i, (name, module) in enumerate(layers):
        if name == layer_name:
            return layers[i + 1][1]  # Return the next module after the current one
    return None

def get_next_conv(model, current_module):
    """
    Given a model and a current module, return the next Conv2d layer after the given module.
    
    Parameters:
    - model (nn.Module): The model containing the layers.
    - current_module (nn.Module): The current module (usually a BatchNorm2d layer).
    
    Returns:
    - nn.Conv2d: The next Conv2d layer after the current module.
    """
    # Flag to indicate if we have passed the current module
    found_current = False
    
    # Iterate through all modules in the model
    for name, module in model.named_modules():
        if found_current and isinstance(module, nn.Conv2d):
            return module  # Return the first Conv2d layer after the current one
        if module == current_module:
            found_current = True  # Set flag to True when we find the current module
    
    # If no Conv2d layer found after the current module, return None
    return None


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    ##################### YOUR CODE STARTS HERE #####################
    return int(round(channels*(1-prune_ratio)))
    ##################### YOUR CODE ENDS HERE #####################

@torch.no_grad()
def channel_prune_resnet(model: nn.Module,
                  prune_ratio: Union[List, float]) -> nn.Module:
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    n_conv = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv - 1)

    # we prune the convs in the backbone with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite
    # we only apply pruning to the backbone features
    
    all_convs = [m for m in model.named_modules() if isinstance(m[1], nn.Conv2d)]
    for m in all_convs:
        print(m[0])

    all_bns = [m for m in model.named_modules() if isinstance(m[1], nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    print(":hello")
    assert len(all_convs) == len(all_bns)
    print(prune_ratio)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        if i_ratio > 0  and "conv1" in all_convs[i_ratio][0]:
            print(all_convs[i_ratio][0])
            prev_conv = all_convs[i_ratio][1]
            prev_bn = all_bns[i_ratio][1]
            next_conv = all_convs[i_ratio + 1][1]
            print(prev_conv.weight.shape, next_conv.weight.shape)
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            # print(prev_conv.weight.shape, original_channels, next_conv.weight.shape)
            n_keep = get_num_channels_to_keep(original_channels, p_ratio)

            # prune the output of the previous conv and bn
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            if prev_conv.bias is not None:
                prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
            # prune the input of the next conv (hint: just one line of code)
            ##################### YOUR CODE STARTS HERE #####################
            next_conv.weight.set_(next_conv.weight.detach()[:,:n_keep,:,:])
            print(prev_conv.weight.shape[0]==next_conv.weight.shape[1])
            if isinstance(next_conv, nn.Conv2d) and next_conv.kernel_size == (1, 1):
                next_conv.in_channels = n_keep
            # next_conv.out_channels = next_conv.weight.size(0)

        ##################### YOUR CODE ENDS HERE #####################

    return model

@torch.no_grad()
def channel_prune(model: nn.Module,
                  prune_ratio: Union[List, float]) -> nn.Module:
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    n_conv = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv - 1)

    # we prune the convs in the backbone with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite
    # we only apply pruning to the backbone features
    
    all_convs = [m for m in model.named_modules() if isinstance(m[1], nn.Conv2d)]
    all_bns = [m for m in model.named_modules() if isinstance(m[1], nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    print(":hello")
    assert len(all_convs) == len(all_bns)
    print(prune_ratio)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        print(all_convs[i_ratio][0])
        prev_conv = all_convs[i_ratio][1]
        prev_bn = all_bns[i_ratio][1]
        next_conv = all_convs[i_ratio + 1][1]
        print(prev_conv.weight.shape, next_conv.weight.shape)
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        # print(prev_conv.weight.shape, original_channels, next_conv.weight.shape)
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        if prev_conv.bias is not None:
            prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
        # prune the input of the next conv (hint: just one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.set_(next_conv.weight.detach()[:,:n_keep,:,:])
        print(prev_conv.weight.shape[0]==next_conv.weight.shape[1])
        if isinstance(next_conv, nn.Conv2d) and next_conv.kernel_size == (1, 1):
            next_conv.in_channels = n_keep
        # next_conv.out_channels = next_conv.weight.size(0)

        ##################### YOUR CODE ENDS HERE #####################

    return model
# function to sort the channels from important to non-important


@torch.no_grad()
def apply_channel_sorting(model):
    model = copy.deepcopy(model)  # do not modify the original model
    all_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    for i_conv in range(len(all_convs) - 1):
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]

        # Compute importance based on the input channels of `next_conv`
        importance = get_input_channel_importance(next_conv.weight)
        sort_idx = torch.argsort(importance, descending=True)

        # Apply sorting to the previous conv and BN layers
        if prev_conv.weight.size(0) == sort_idx.size(0):  # Ensure size alignment
            prev_conv.weight.copy_(
                torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
            )
            for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )

        # Apply sorting to the input channels of `next_conv`
        if next_conv.weight.size(1) == sort_idx.size(0):  # Ensure size alignment
            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )
        else:
            raise ValueError(
                f"Mismatch: next_conv input channels ({next_conv.weight.size(1)}) != sort_idx size ({sort_idx.size(0)})"
            )

    return model




def preprocess_combined_data(data_dict):
    """
    Preprocess the combined data dictionary.

    Args:
        data_dict (dict): Combined dictionary with all tactile readings

    Returns:
        tuple: (processed samples, labels, label mapping)
    """
    samples = []
    labels = []
    label_to_idx = {key: idx for idx, key in enumerate(sorted(data_dict.keys()))}

    for key, tactile_readings in data_dict.items():
        for reading in tactile_readings:
            # Take min over temporal dimension
            processed = np.min(reading, axis=0)
            # Normalize to [0, 1]
            processed = (processed - processed.min()) / (processed.max() - processed.min())
            samples.append(processed[np.newaxis, :, :])  # Add channel dimension
            labels.append(label_to_idx[key])

    return np.array(samples), np.array(labels), label_to_idx

def combine_data_dicts(file_list, data_folder):
    """
    Combine multiple HDF5 files into a single data dictionary.

    Args:
        file_list (list): List of HDF5 filenames
        data_folder (str): Path to the data folder

    Returns:
        combined_dict (dict): Combined dictionary with all tactile readings
    """
    # Initialize combined dictionary using defaultdict to automatically handle new keys
    combined_dict = defaultdict(list)

    # Process each file
    for file in file_list:
        path = os.path.join(data_folder, file)
        data = read_key_data(path)

        # Extend the combined dictionary with new readings
        for key, readings in data.items():
            combined_dict[key].extend(readings)

    # Convert defaultdict back to regular dict
    return dict(combined_dict)

data_folder = 'recordings'
file_list = [f for f in os.listdir(data_folder)
             if f.endswith('layout_3.hdf5')]

combined_data = combine_data_dicts(file_list, data_folder)
samples, labels, label_to_idx = preprocess_combined_data(combined_data)
# Save class mapping
with open('class_mapping.json', 'w') as f:
    json.dump(label_to_idx, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    samples, labels, test_size=0.3, random_state=42, stratify=labels
)
 
test_dataset = TactileDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_dataset = TactileDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = get_device()

model = ModelFactory.create_model('shallow_cnn', num_classes=30, device=device)
# model = ModelFactory.create_model('resnet', num_classes=30, device=device)
model.load_state_dict(torch.load('models/best_ShallowCNN_model.pth', map_location=device))

sorted_model = apply_channel_sorting(model)

channel_pruning_ratio = 0.8

print(" * With sorting...")
pruned_model = channel_prune(sorted_model,channel_pruning_ratio)
#print(pruned_model)
pruned_model_accuracy = evaluate(pruned_model, test_loader, device)
print(f"pruned model has accuracy={pruned_model_accuracy:.2f}%")

num_finetune_epochs = 10
best_accuracy = 0
best_accuracy = train_model(pruned_model, train_loader,test_loader,device,num_epochs=num_finetune_epochs)
torch.save(pruned_model, "pruned_model_full.pth")
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
    return (t2 - t1) / n_test  # average latency

table_template = "{:<15} {:<15} {:<15} {:<15}"
print (table_template.format('', 'Original','Pruned','Reduction Ratio'))

# 1. measure the latency of the original model and the pruned model on CPU
#   which simulates inference on an edge device
dummy_input = torch.randn(1, 1, 32, 32).to('cpu')
pruned_model = pruned_model.to('cpu')
model = model.to('cpu')

pruned_latency = measure_latency(pruned_model, dummy_input)
original_latency = measure_latency(model, dummy_input)
print(table_template.format('Latency (ms)',
                            round(original_latency * 1000, 1),
                            round(pruned_latency * 1000, 1),
                            round(original_latency / pruned_latency, 1)))

# 2. measure the computation (MACs)
original_macs = get_model_macs(model, dummy_input)
pruned_macs = get_model_macs(pruned_model, dummy_input)
print(table_template.format('MACs (M)',
                            round(original_macs / 1e6),
                            round(pruned_macs / 1e6),
                            round(original_macs / pruned_macs, 1)))

original_size = get_model_size(model)
pruned_size = get_model_size(pruned_model)
print(table_template.format('Model Size (MiB)',
                            round(original_size / KiB),
                            round(pruned_size / KiB),
                            round(original_size / pruned_size, 1)))
# print(table_template.format('Param (M)',
#                             round(original_param / 1e6, 2),
#                             round(pruned_param / 1e6, 2),
#                             round(original_param / pruned_param, 1)))

