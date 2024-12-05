import os
import math
from collections import defaultdict
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from model import ModelFactory
from model_profile import get_model_size, MiB
from utils import read_key_data
from train import TactileDataset, evaluate, get_device

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
             if f.endswith('layout_1.hdf5') or f.endswith('2024-11-25_18-13-21.hdf5')]

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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = get_device()

#model = ModelFactory.create_model('shallow_cnn', num_classes=30, device=device)
model = ModelFactory.create_model('resnet', num_classes=30, device=device)
model.load_state_dict(torch.load('models/best_ResNet_model.pth', map_location=device))
#model.load_state_dict(torch.load('models/best_ShallowCNN_model.pth', map_location=device))

print(model)
dense_model_accuracy = evaluate(model, test_loader, device)
model_size = get_model_size(model)
print(f"Dense model has accuracy={dense_model_accuracy:.2f}%")
print(f"Dense model has size={model_size/MiB:.2f} MiB")




def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(num_elements * sparsity)

    importance = torch.abs(tensor)
    threshold = torch.kthvalue(torch.flatten(importance), num_zeros).values.item()

    mask = torch.gt(importance, threshold)
    tensor.mul_(mask)

    return mask

#for n, p in model.named_parameters():
#    print(n)

@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                      in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, device)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='\n')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]\n')
        accuracies.append(accuracy)
    return sparsities, accuracies

sparsities, accuracies = sensitivity_scan(model, test_loader, scan_step=0.1, scan_start=0.4, scan_end=1.0)

def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(5, int(math.ceil(len(accuracies) / 5)),figsize=(25,25))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=0.7, step=0.1))
            ax.set_ylim(40, 80)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'accuracy after pruning',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()
#
plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy)

sparsity_dict_shallow = {
    'features.0.weight': 0.0,
    'features.4.weight': 0.7,
    'classifier.0.weight': 0.9,
    'classifier.3.weight': 0.5,
}

sparsity_dict_resnet = {
    'conv1.weight': 0.0,
    'layer1.0.conv1.weight': 0.8,
    'layer1.0.conv2.weight': 0.95,
    'layer1.1.conv1.weight': 0.95,
    'layer1.1.conv2.weight': 0.95,
    'layer2.0.conv1.weight': 0.95,
    'layer2.0.conv2.weight': 0.95,
    'layer2.0.downsample.0.weight': 0.95,
    'layer2.1.conv1.weight': 0.95,
    'layer2.1.conv2.weight': 0.95,
    'layer3.0.conv1.weight': 0.95,
    'layer3.0.conv2.weight': 0.95,
    'layer3.0.downsample.0.weight': 0.95,
    'layer3.1.conv1.weight': 0.95,
    'layer3.1.conv2.weight': 0.95,
    'layer4.0.conv1.weight': 0.95,
    'layer4.0.conv2.weight': 0.95,
    'layer4.0.downsample.0.weight': 0.95,
    'layer4.1.conv1.weight': 0.95,
    'layer4.1.conv2.weight': 0.95,
    'fc.weight': 0.8
}

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

#pruner = FineGrainedPruner(model, sparsity_dict_resnet)
#sparse_model_size = get_model_size(model, count_nonzero_only=True)
#sparse_model_accuracy = evaluate(model, test_loader, device)
#print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}%")
#print(f"Sparse model has size={sparse_model_size/MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
